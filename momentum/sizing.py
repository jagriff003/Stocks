"""
Position sizing: how much of each name, given that we already know which names.

Selection and sizing are separable questions and this module only answers the
second.  Every scheme here takes the book the selector already chose and
decides the weights; none of them can add or drop a name.  That separation is
what makes the comparison clean — any difference in the results below is
sizing, because nothing else moved.

WHAT "EQUAL WEIGHT" ALREADY MEANT
   The existing engine recomputes an equal-weighted mean return on every
   segment, which is an implicit costless daily rebalance back to equal weight
   (see `_segment_return` in backtest.py).  The weighted schemes keep exactly
   that convention — daily rebalance back to the TARGET weights — because
   changing the rebalancing assumption at the same time as the weighting scheme
   would confound the two.  So what is tested here is target weights, not
   weight drift.  A buy-and-hold-the-weights variant is a different experiment
   and is not this one.

   The practical consequence: `scheme='equal'` must reproduce the baseline
   numbers to the last decimal.  If it does not, the harness is wrong and every
   other number it prints is meaningless.  `verify_equal_weight_parity` in
   scripts/analyze_position_sizing.py checks precisely this, and the analysis
   refuses to report anything if it fails.

THE SCHEMES
   equal              1/N.  The baseline.

   inverse_vol        w ∝ 1/σ, σ from trailing daily returns.  The standard
                      answer, and the one most likely to help: it stops a
                      single high-vol name from dominating the book's risk
                      even though it holds only 1/N of its capital.

   score_proportional w ∝ the composite score, floored at zero and rescaled.
                      This is the interesting one, because it tests something
                      the selection step cannot: whether the ranker's score
                      carries information ABOUT MAGNITUDE, not just order.  If
                      the ranker is only ordinally meaningful — rank 1 beats
                      rank 5, but the score gap does not say by how much — this
                      scheme should be indistinguishable from equal weight.

   vol_target         inverse-vol weights, then the whole book scaled so its
                      predicted volatility hits a target.  Capped at fully
                      invested: no leverage, so it can only ever de-risk.  That
                      cap is a real constraint, not a modelling convenience —
                      it means the scheme systematically gives up return in
                      calm markets and can only pay for it in volatile ones.

WHY THE CAPS MATTER MORE THAN THE SCHEMES
   With a book of 5-8 names, an uncapped inverse-vol weighting can put 40% into
   the single lowest-vol name, which is a concentration decision wearing a risk
   management costume.  `max_weight` is what stops that, and it binds often
   enough that it deserves to be read as part of the scheme rather than as a
   safety rail.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

SCHEMES = ("equal", "inverse_vol", "score_proportional", "vol_target")


@dataclass
class SizingConfig:
    """
    How the book is weighted.

    scheme
        One of SCHEMES.  'equal' is the baseline and is bit-identical to the
        pre-sizing engine.

    vol_window
        Trailing sessions used to estimate each name's daily volatility.  60 is
        about a quarter — long enough to be stable, short enough to notice a
        regime change.  Names with fewer than `min_vol_obs` observations fall
        back to equal weight rather than to a guessed volatility.

    target_vol
        Annualized portfolio volatility aimed at, for scheme='vol_target'.
        0.15 is roughly the long-run volatility of the S&P.

    max_weight
        Cap on any single position, applied after the scheme and followed by a
        renormalization of the rest.  None disables it.  With a 5-name book,
        anything below 0.20 would force equal weight and make the scheme inert.

    max_gross
        Cap on total invested fraction.  1.0 means no leverage; vol_target can
        sit below it but never above.
    """
    scheme: str = "equal"
    vol_window: int = 60
    min_vol_obs: int = 20
    target_vol: float = 0.15
    max_weight: Optional[float] = 0.35
    max_gross: float = 1.0

    def __post_init__(self):
        if self.scheme not in SCHEMES:
            raise ValueError(
                f"Unknown sizing scheme {self.scheme!r}; expected one of "
                f"{SCHEMES}")

    def label(self) -> str:
        if self.scheme == "equal":
            return "equal"
        bits = [self.scheme]
        if self.scheme == "vol_target":
            bits.append(f"tgt{self.target_vol:.0%}")
        if self.max_weight is not None:
            bits.append(f"cap{self.max_weight:.0%}")
        return " ".join(bits)


def _cap_and_renormalize(w: pd.Series, max_weight: Optional[float]) -> pd.Series:
    """
    Apply a per-position cap, pushing the excess onto the uncapped names.

    Iterative, because capping one name raises the others and can push a second
    over.  The subtlety is that a capped name must STAY capped: redistributing
    onto everything including the names already at the cap makes the weights
    oscillate — cap C, which lifts B over, cap B, which lifts C over again —
    and the result depends on which iteration the loop happens to stop at.
    The fix is to freeze capped names and share the remaining budget only among
    those still free.

    If the cap cannot be satisfied at all (max_weight * n <= 1), equal weight is
    the only feasible answer and is returned directly.
    """
    if max_weight is None or w.empty:
        return w
    n = len(w)
    if max_weight * n <= 1.0 + 1e-12:
        return pd.Series(1.0 / n, index=w.index)

    w = w.clip(lower=0.0)
    if w.sum() <= 0:
        return pd.Series(1.0 / n, index=w.index)
    w = w / w.sum()

    capped = pd.Series(False, index=w.index)
    for _ in range(n):
        free = ~capped
        budget = 1.0 - max_weight * int(capped.sum())
        if not free.any() or budget <= 0:
            break
        free_sum = float(w[free].sum())
        if free_sum <= 0:
            w[free] = budget / int(free.sum())
        else:
            w[free] = w[free] * budget / free_sum
        w[capped] = max_weight
        newly = free & (w > max_weight + 1e-12)
        if not newly.any():
            break
        capped |= newly
    return w


def _trailing_vol(symbols: List[str], close: pd.DataFrame, date,
                  window: int, min_obs: int) -> pd.Series:
    """
    Annualized trailing volatility per symbol, as of `date` inclusive.

    Uses only data up to and including `date`, which is the signal date — the
    fill happens the next session, so this is point-in-time correct and does
    not peek.
    """
    cols = [s for s in symbols if s in close.columns]
    if not cols:
        return pd.Series(dtype=float)
    hist = close.loc[:date, cols].tail(window + 1)
    rets = hist.pct_change().dropna(how="all")
    vol = rets.std() * np.sqrt(252)
    vol[rets.count() < min_obs] = np.nan
    return vol


def compute_weights(symbols: List[str],
                    date,
                    close: pd.DataFrame,
                    config: SizingConfig,
                    scores: Optional[pd.DataFrame] = None) -> pd.Series:
    """
    Target weights for `symbols` as of `date`.

    Returns a Series indexed by symbol.  It sums to 1.0 for every scheme except
    'vol_target', which may sum to less when it de-risks — that shortfall is
    held in cash and earns nothing, which is the cost the scheme pays.

    Any scheme that cannot be computed for a name — no volatility estimate, no
    score — falls back to equal weight across the whole book rather than
    guessing a value for that one name.  A partial estimate silently mixed with
    invented numbers is worse than no estimate.
    """
    symbols = [s for s in symbols]
    if not symbols:
        return pd.Series(dtype=float)

    n = len(symbols)
    equal = pd.Series(1.0 / n, index=symbols)

    if config.scheme == "equal":
        return equal

    if config.scheme == "score_proportional":
        if scores is None or date not in scores.index:
            return equal
        row = scores.loc[date].reindex(symbols)
        if row.isna().any():
            return equal
        # Scores can be negative; shifting to a positive floor preserves the
        # ORDER but changes the RATIOS, which is the whole point of the
        # scheme.  Anchoring the floor at the book's own minimum keeps the
        # spread interpretable: the weakest held name gets a small but nonzero
        # weight rather than an arbitrary one.
        shifted = row - row.min() + row.abs().mean() * 0.25
        if shifted.sum() <= 0 or not np.isfinite(shifted.sum()):
            return equal
        return _cap_and_renormalize(shifted / shifted.sum(), config.max_weight)

    vol = _trailing_vol(symbols, close, date, config.vol_window,
                        config.min_vol_obs)
    vol = vol.reindex(symbols)
    if vol.isna().any() or (vol <= 0).any():
        return equal

    inv = 1.0 / vol
    w = _cap_and_renormalize(inv / inv.sum(), config.max_weight)

    if config.scheme == "inverse_vol":
        return w

    # vol_target: scale the book toward a volatility target.
    #
    # The portfolio vol is approximated as the weighted average of the
    # constituent vols, which ignores correlation and therefore OVERSTATES the
    # book's true volatility — a 5-name momentum book is far from perfectly
    # correlated.  The scheme will de-risk more often than a correlation-aware
    # version would.  That is a known bias and is stated rather than hidden;
    # correcting it needs the covariance matrix, which is a different and
    # noisier estimate on a 5-name book.
    port_vol = float((w * vol).sum())
    if port_vol <= 0:
        return equal
    scale = min(config.target_vol / port_vol, config.max_gross)
    return w * scale


def weights_for_history(holdings: pd.Series,
                        close: pd.DataFrame,
                        config: SizingConfig,
                        scores: Optional[pd.DataFrame] = None) -> pd.Series:
    """Target weights for each date in a holdings series.  Convenience only."""
    return pd.Series(
        {d: compute_weights(list(h), d, close, config, scores)
         for d, h in holdings.items()})
