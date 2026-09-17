"""
Weight-aware simulation: what the book earns when weights are allowed to drift.

`backtest._segment_return` computes the equal-weighted mean of its constituents'
returns on every segment, which silently assumes the book is rebalanced back to
equal weight every single day, for free.  That convention is load-bearing for
comparability with every historical result in FINDINGS and must not change.  It
is also wrong about two things that now matter:

  - It hands the backtest a free rebalancing bonus.  Daily rebalancing into a
    mean-reverting cross-section earns something, and real books do not do it.
  - It makes "how much rebalancing do I actually need?" unaskable, because the
    answer is assumed in the return calculation.

This module is the parallel path.  It tracks real weights, lets them drift with
prices between rebalances, and prices the trading it takes to reset them.  It
does NOT replace `simulate_portfolio`; it exists beside it so the two can be
compared, and so questions about weighting have somewhere to be asked.

VALIDATION
   Under `RebalanceConfig(policy="daily")` this reproduces `simulate_portfolio`'s
   GROSS return stream exactly - that is the point of keeping the policy, and
   `validate_against_legacy` asserts it.  Net returns differ by construction
   under that policy, because the legacy path charges nothing for the daily
   rebalance it assumes and this one charges for every trade it makes.  That
   difference is a finding, not a defect.

WEIGHT-BASED REGIME OVERLAY
   The live overlay de-risks by SLOT COUNT: in an elevated regime it keeps
   `elevated_top_n` momentum names and fills the rest from the defensive sleeve.
   That mechanism does not scale to a wide book, because only three defensive
   tickers exist - a 25-name book can be at most 12% defensive no matter what
   the rule says.  `defensive_weight_targets` expresses the same intent as a
   WEIGHT instead, so a wide book can hold 30% defensive across three names.
   It is the only form of the overlay that can work at scale.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .config import ExecutionConfig
from .metrics import calculate_performance_metrics, calculate_turnover_metrics


@dataclass
class RebalanceConfig:
    """
    When to push weights back to target.

    policy
      'daily'       every session.  Reproduces the legacy gross convention;
                    kept for validation, not as a serious proposal.
      'on_rotation' only when the target book changes.  The natural default:
                    you are already trading, so the reset is nearly free.
      'never'       never.  True buy-and-hold; weights run wherever they run.
                    New names still enter at equal weight to the book average.
      'periodic'    every `period_days` sessions.
      'band'        when any holding's weight deviates from its target by more
                    than `band` in relative terms (0.25 = 25% off target).
    """
    policy: str = "on_rotation"
    period_days: int = 63
    band: float = 0.25

    def __post_init__(self):
        valid = {"daily", "on_rotation", "never", "periodic", "band"}
        if self.policy not in valid:
            raise ValueError(f"Unknown rebalance policy: {self.policy!r}")
        if self.policy == "periodic" and self.period_days < 1:
            raise ValueError("periodic rebalancing needs period_days >= 1")
        if self.policy == "band" and not 0 < self.band < 10:
            raise ValueError("band must be a relative deviation in (0, 10)")

    @property
    def label(self) -> str:
        if self.policy == "periodic":
            return f"periodic:{self.period_days}d"
        if self.policy == "band":
            return f"band:{self.band:.0%}"
        return self.policy


@dataclass
class DriftResult:
    returns: pd.Series
    gross_returns: pd.Series
    weights: pd.DataFrame
    holdings_history: List[Dict]
    total_cost: float
    n_rebalances: int
    metrics: Dict[str, float]
    turnover: Dict[str, float]

    @property
    def max_weight(self) -> pd.Series:
        """Largest single position over time - what drift actually did."""
        return self.weights.max(axis=1)


def _priced(symbols: Sequence[str], p0: pd.Series, p1: pd.Series):
    """Symbols with usable prices at both ends, and their simple returns."""
    out = {}
    for s in symbols:
        a, b = p0.get(s, np.nan), p1.get(s, np.nan)
        if pd.notna(a) and pd.notna(b) and a != 0:
            out[s] = float(b / a - 1)
    return out


def _target_weights(symbols: Sequence[str],
                    overrides: Optional[Dict[str, float]]) -> Dict[str, float]:
    """
    Desired weights for a book: equal by default, or as overridden.

    Overrides need not sum to one and need not name every holding; whatever
    weight they claim is honoured and the remainder is split equally among the
    names they do not mention.  That is what lets a regime rule say "30% across
    the defensive sleeve" without also having to restate the rest of the book.
    """
    symbols = list(symbols)
    if not symbols:
        return {}
    if not overrides:
        return {s: 1.0 / len(symbols) for s in symbols}

    named = {s: float(w) for s, w in overrides.items() if s in symbols}
    claimed = sum(named.values())
    rest = [s for s in symbols if s not in named]

    if claimed > 1.0:                      # scale back rather than go short
        named = {s: w / claimed for s, w in named.items()}
        claimed = 1.0
    if rest:
        share = (1.0 - claimed) / len(rest)
        named.update({s: share for s in rest})
    else:
        total = sum(named.values()) or 1.0
        named = {s: w / total for s, w in named.items()}
    return named


def _drift(weights: Dict[str, float], rets: Dict[str, float],
           port_ret: float) -> Dict[str, float]:
    """Carry weights forward one leg, renormalized by the book's own return."""
    scale = 1.0 + port_ret
    if scale <= 0:
        return weights
    return {s: w * (1.0 + rets.get(s, 0.0)) / scale for s, w in weights.items()}


def _turnover_cost(old: Dict[str, float], new: Dict[str, float],
                   execution: ExecutionConfig) -> float:
    """
    Slippage on the weight actually traded.

    One-way turnover is half the summed absolute weight change; both sides pay,
    so the cost is the full sum times the one-way rate.  Unlike the legacy
    name-count model this prices a pure rebalance - which is exactly the thing
    the legacy path gets for free.
    """
    names = set(old) | set(new)
    traded = sum(abs(new.get(s, 0.0) - old.get(s, 0.0)) for s in names)
    return traded * execution.slippage_frac


def simulate_with_drift(targets: pd.Series,
                        close: pd.DataFrame,
                        open_: pd.DataFrame,
                        execution: Optional[ExecutionConfig] = None,
                        rebalance: Optional[RebalanceConfig] = None,
                        weight_targets: Optional[pd.Series] = None,
                        ) -> DriftResult:
    """
    Simulate `targets` with real weights, on the next_open fill convention.

    Parameters
    ----------
    targets : Series of date -> list of symbols
    weight_targets : Series of date -> dict of symbol -> weight, optional
        Desired weights on a rebalance.  Missing dates and missing symbols fall
        back to equal weight; see `_target_weights`.
    """
    execution = execution or ExecutionConfig()
    rebalance = rebalance or RebalanceConfig()
    if execution.execute_at != "next_open":
        raise ValueError("simulate_with_drift implements next_open only")

    dates = list(targets.index)
    weights: Dict[str, float] = {}
    prev_overrides: Optional[Dict[str, float]] = None
    held: List[str] = []
    total_cost = 0.0
    n_rebal = 0
    since_reset = 0

    records: List[Dict] = []
    weight_rows: Dict[pd.Timestamp, Dict[str, float]] = {}
    holdings_history: List[Dict] = []

    def wants_reset(w: Dict[str, float], tgt: Dict[str, float],
                    override_changed: bool) -> bool:
        # A change in the weight target is a change in the RISK DIAL, and the
        # risk dial must not wait for the selection clock.  This is the same
        # lesson backtest.py records for Track A: when exposure could only move
        # at a scheduled rebalance, only 25% of crisis-flagged days were
        # actually defensive.  Without this branch a 126-day hold would let a
        # crisis pass with the overlay stuck off, and the sweep would report
        # that long holds have bad drawdown when what it measured was an
        # overlay that could not engage.
        if override_changed:
            return True
        if rebalance.policy == "daily":
            return True
        if rebalance.policy == "never":
            return False
        if rebalance.policy == "on_rotation":
            return False          # handled by the book-change branch
        if rebalance.policy == "periodic":
            return since_reset >= rebalance.period_days
        # band
        return any(abs(w.get(s, 0.0) - t) > rebalance.band * t
                   for s, t in tgt.items() if t > 0)

    for i in range(1, len(dates)):
        d_prev, d = dates[i - 1], dates[i]
        if d_prev not in close.index or d not in close.index:
            continue

        signal = list(targets.loc[d_prev])
        overrides = (weight_targets.get(d_prev)
                     if weight_targets is not None else None)
        close_prev, close_now = close.loc[d_prev], close.loc[d]
        cost = 0.0

        if signal != held:
            open_now = open_.loc[d]
            if held:
                r1_by = _priced(held, close_prev, open_now)
                if rebalance.policy == "daily":
                    # 'daily' means rebalanced at every close, so the book
                    # carried into this open starts equal-weighted.  That is
                    # precisely the legacy fiction, and reproducing it here is
                    # what makes the validation exact.  Under every other policy
                    # these weights are whatever they drifted to - you cannot
                    # trade the old book before the open.
                    w1 = _target_weights(list(r1_by), None)
                else:
                    w1 = {s: weights.get(s, 0.0) for s in r1_by}
                    tot = sum(w1.values())
                    w1 = ({s: w / tot for s, w in w1.items()} if tot > 0
                          else _target_weights(list(r1_by), None))
                r1 = sum(w1[s] * r1_by[s] for s in r1_by)
                w1 = _drift(w1, r1_by, r1)
            else:
                r1, w1 = 0.0, {}

            r2_by = _priced(signal, open_now, close_now)
            new_w = _target_weights(list(r2_by), overrides)
            cost = _turnover_cost(w1, new_w, execution)
            total_cost += cost
            n_rebal += 1
            since_reset = 0

            r2 = sum(new_w[s] * r2_by[s] for s in r2_by)
            gross = (1 + r1) * (1 + r2) - 1 if held else r2
            weights = _drift(new_w, r2_by, r2)
            held = signal
            holdings_history.append({"Date": d, "Holdings": list(held)})

        else:
            if not held:
                continue
            r_by = _priced(held, close_prev, close_now)
            if not r_by:
                continue
            w = {s: weights.get(s, 0.0) for s in r_by}
            tot = sum(w.values())
            w = ({s: x / tot for s, x in w.items()} if tot > 0
                 else _target_weights(list(r_by), None))

            tgt = _target_weights(list(r_by), overrides)
            if wants_reset(w, tgt, overrides != prev_overrides):
                cost = _turnover_cost(w, tgt, execution)
                total_cost += cost
                n_rebal += 1
                since_reset = 0
                w = tgt
            else:
                since_reset += 1

            gross = sum(w[s] * r_by[s] for s in r_by)
            weights = _drift(w, r_by, gross)

        prev_overrides = overrides
        weight_rows[d] = dict(weights)
        records.append({"Date": d, "Net": gross - cost, "Gross": gross})

    if not records:
        raise ValueError("Simulation produced no return observations")

    frame = pd.DataFrame(records).set_index("Date")
    net, gross_s = frame["Net"], frame["Gross"]
    metrics = calculate_performance_metrics(
        net, risk_free_rate=execution.risk_free_rate)
    turnover = calculate_turnover_metrics(holdings_history, metrics["years"])

    return DriftResult(
        returns=net,
        gross_returns=gross_s,
        weights=pd.DataFrame.from_dict(weight_rows, orient="index").fillna(0.0),
        holdings_history=holdings_history,
        total_cost=total_cost,
        n_rebalances=n_rebal,
        metrics=metrics,
        turnover=turnover,
    )


def validate_against_legacy(targets, close, open_, execution,
                            tol: float = 1e-10) -> float:
    """
    Assert the drift path reproduces the legacy GROSS stream under daily reset.

    Net cannot match and is not checked: the legacy path assumes a free daily
    rebalance, this one charges for it.  Quantifying that gap is one of the
    reasons this module exists.
    """
    from .backtest import simulate_portfolio

    legacy = simulate_portfolio(targets, close, open_, execution=execution)
    drift = simulate_with_drift(targets, close, open_, execution=execution,
                                rebalance=RebalanceConfig(policy="daily"))
    a = legacy.gross_returns
    b = drift.gross_returns.reindex(a.index)
    gap = float((a - b).abs().max())
    if not gap < tol:
        raise AssertionError(
            f"Drift simulator does not reproduce the legacy gross stream "
            f"under policy='daily': max abs diff {gap:.3e} (tolerance {tol:.0e})")
    return gap


def defensive_weight_targets(regimes: pd.Series,
                             defensive: Sequence[str],
                             weight_by_regime: Dict[str, float],
                             ) -> pd.Series:
    """
    Regime overlay expressed as a defensive WEIGHT rather than a slot count.

    `weight_by_regime` maps a regime label to the share of the book the
    defensive sleeve should carry, e.g. {'ELEVATED': 0.5, 'CRISIS': 1.0}.  The
    weight is split equally across whichever defensive names are in the book.
    Regimes absent from the mapping get no override, which means equal weight.

    This is the form of the overlay that survives a wide book.  The slot-count
    version caps out at three names, so a 25-name book cannot exceed 12%
    defensive however elevated the regime; a weight target has no such ceiling.
    """
    defensive = list(defensive)
    out = {}
    for date, regime in regimes.items():
        share = weight_by_regime.get(regime)
        if not share:
            continue
        per = float(share) / len(defensive) if defensive else 0.0
        out[date] = {s: per for s in defensive}
    return pd.Series(out, dtype=object)
