"""
Synthetic delistings, for bounding the survivorship exposure of a pool.

THE PROBLEM THIS EXISTS FOR

`random_pool.csv` was built in 2026 from names that are liquid *today*.  Every
name in it survived.  The Track J score buys pullbacks in strong performers, so
every dip in that sample was followed by a recovery — because the names whose
dips were terminal are not there to be bought.  That is the single largest
reason not to believe a 24% CAGR.

It cannot be fixed here.  Fixing it needs point-in-time data carrying delisted
securities (CRSP, Norgate, Sharadar); yfinance simply has no row for a company
that stopped trading in 2014.  What this module does instead is **bound** it:
put plausible delisting names into the panel, let the strategy see and trade
them under its normal rules, and measure how much edge survives.

The deliverable is a breakeven, in the same spirit as the breakeven-bps figure:
*what annual delisting rate would it take to erase this?*  If the answer is far
above any plausible rate for $10M+ ADV names, the result stands.  If it is
inside the plausible range, the wide-pool result is not usable.

DELISTING IS NOT ONE EVENT, AND THE BRANCHES PULL OPPOSITE WAYS

Modelling every delisting as -100% would badly overstate the damage and produce
a bound nobody should believe.  The real mixture:

  merger / acquisition   cash or acquirer stock, usually at a PREMIUM.  The
                         most common exit for a healthy company, and a
                         *tailwind* for a momentum signal.
  going private          cashed out, near the market price.
  compliance delisting   price under $1, market-cap or filing failures.  Shares
                         are NOT cancelled — they move to OTC and keep trading
                         at a wide spread.  Large loss, not a zero.
  Chapter 11             equity usually cancelled or massively diluted.
  Chapter 7              equity is zero.

So the generator takes a branch mixture, and the merger branch is deliberately
the largest for a liquid pool.  Get the mixture roughly right and the bound
means something; assume a single catastrophic branch and it does not.

THE LOSS IS MOSTLY BEFORE THE EVENT

A name takes months to travel from a deficiency notice to an actual delisting —
the main exchange rules allow 180 days to regain compliance, often extendable.
A strategy rebalancing every 14 to 42 days with a liquidity and price screen
gets many chances to leave first.  So the generator applies a **decline path**
ahead of the cause-based branches rather than a single terminal shock, and the
screen is left switched on.  Whether the screen catches them is the question
being asked, not an assumption being made.

WHAT THIS IS NOT

It is a bound built on assumed rates and an assumed mixture, not a measurement.
Its output is a sensitivity curve, and the honest way to read it is "the edge
survives up to rate X", never "the edge is Y after correcting for delisting".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class DelistingBranch:
    """One way a listing can end."""
    name: str
    weight: float           # share of delistings taking this branch
    terminal_return: float  # return booked on the final day
    decline: float          # cumulative drift applied over `decline_days` before it
    decline_days: int = 126


@dataclass
class DelistingConfig:
    """
    Assumptions.  All of them.  Nothing here is estimated from the data being
    tested, and the point of the exercise is to sweep the first field.

    The default mixture leans heavily on mergers because the pool being
    augmented is screened for $10M+ daily dollar volume — at that size the
    dominant reason a listing ends is that somebody bought the company, not
    that it failed.  A microcap pool would want the opposite weights.
    """

    annual_rate: float = 0.03      # share of listings ending per year

    branches: List[DelistingBranch] = field(default_factory=lambda: [
        DelistingBranch("merger", 0.50, terminal_return=0.20, decline=0.00,
                        decline_days=0),
        DelistingBranch("going_private", 0.10, terminal_return=0.05,
                        decline=0.00, decline_days=0),
        # The declines are deep on purpose.  A compliance delisting happens
        # because the price went through the exchange's minimum, so a name that
        # merely halves does not delist — it has to collapse.  The first
        # version used -60%/-75%/-85%, which left names trading in the tens of
        # dollars and made the price screen unfireable by construction.
        DelistingBranch("compliance_otc", 0.22, terminal_return=-0.50,
                        decline=-0.85, decline_days=189),
        DelistingBranch("chapter_11", 0.15, terminal_return=-0.90,
                        decline=-0.92, decline_days=189),
        DelistingBranch("chapter_7", 0.03, terminal_return=-1.00,
                        decline=-0.95, decline_days=189),
    ])

    #: Minimum history a synthetic name gets before it is allowed to die, so it
    #: can clear the model's `min_data_days` gate and actually be pickable.
    min_life_days: int = 400

    #: Donor returns are circularly shifted by at least this much, so a
    #: synthetic name is not a near-duplicate of its donor sitting in the same
    #: cross-section on the same dates.
    min_shift_days: int = 252

    def __post_init__(self):
        total = sum(b.weight for b in self.branches)
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"branch weights sum to {total}, not 1.0")
        if self.annual_rate < 0:
            raise ValueError("annual_rate must be non-negative")

    def branch_for(self, u: float) -> DelistingBranch:
        """Pick a branch from a uniform draw."""
        acc = 0.0
        for b in self.branches:
            acc += b.weight
            if u <= acc:
                return b
        return self.branches[-1]

    @property
    def mean_terminal(self) -> float:
        return sum(b.weight * b.terminal_return for b in self.branches)


def n_synthetic(n_real: int, n_days: int, config: DelistingConfig) -> int:
    """
    How many synthetic listings to add.

    A pool of `n_real` survivors observed over `n_days` implies that, at an
    annual ending rate `r`, roughly `n_real * r * years` further listings
    existed at some point and ended.  Those are the rows a survivorship-free
    panel would carry and this one does not.
    """
    years = n_days / 252.0
    return int(round(n_real * config.annual_rate * years))


def _donor_paths(close: pd.DataFrame, open_: pd.DataFrame,
                 volume: pd.DataFrame, donor: str
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Donor daily returns, open/close ratio and volume, NaNs made harmless."""
    c = close[donor].astype(float)
    ret = c.pct_change().to_numpy()
    ratio = (open_[donor].astype(float) / c).to_numpy()
    vol = volume[donor].astype(float).to_numpy() if donor in volume.columns \
        else np.full(len(c), np.nan)
    ret = np.nan_to_num(ret, nan=0.0, posinf=0.0, neginf=0.0)
    ratio = np.where(np.isfinite(ratio), ratio, 1.0)
    return ret, ratio, vol


def build_synthetic_panel(close: pd.DataFrame, open_: pd.DataFrame,
                          volume: pd.DataFrame,
                          config: Optional[DelistingConfig] = None,
                          seed: int = 0,
                          prefix: str = "ZDL",
                          immortal: bool = False
                          ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
                                     pd.DataFrame]:
    """
    Augment a panel with synthetic listings that end.

    `immortal=True` is the CONTROL ARM, and the study is not interpretable
    without it.  Injecting names does two things at once: it adds delisting
    risk, and it makes the pool bigger — and a bigger pool means the top-K cut
    is a more extreme percentile, which raises the measured edge on its own.
    Run with `immortal=True` and the identical seed to get the same names, the
    same donors and the same count, with no decline, no terminal return and no
    truncation. The difference between the two arms is the cost of delisting
    with the pool-size effect divided out.

    Each synthetic name is built by cloning a real name's RETURNS with a large
    circular shift — so it has realistic volatility, clustering and fat tails
    without being a duplicate of any name trading beside it — then having a
    decline applied and a terminal return booked, after which its prices are
    NaN and it is gone from the cross-section exactly as a delisted name is.

    Returns (close, open_, volume, manifest).  The manifest carries one row per
    synthetic name with its donor, start, delisting date and branch, so the
    simulation can be audited and so the script can report what the strategy
    actually held.
    """
    config = config or DelistingConfig()
    rng = np.random.default_rng(seed)

    idx = close.index
    n_days = len(idx)
    donors = [c for c in close.columns if close[c].notna().sum() > config.min_life_days]
    if not donors:
        raise ValueError("no donor columns with enough history")

    count = n_synthetic(len(close.columns), n_days, config)
    if count == 0:
        manifest = pd.DataFrame(columns=["Symbol", "Donor", "Start", "Delist",
                                         "Branch", "TerminalReturn"])
        return close.copy(), open_.copy(), volume.copy(), manifest

    new_close, new_open, new_vol, rows = {}, {}, {}, []

    for i in range(count):
        donor = donors[rng.integers(len(donors))]
        ret, ratio, vol = _donor_paths(close, open_, volume, donor)

        shift = int(rng.integers(config.min_shift_days, max(
            config.min_shift_days + 1, n_days - config.min_shift_days)))
        ret = np.roll(ret, shift)
        ratio = np.roll(ratio, shift)
        vol = np.roll(vol, shift)

        # Delisting date: uniform over the part of the panel that leaves room
        # for a full life first.  Names that would die before they are old
        # enough to be pickable teach us nothing.
        earliest = config.min_life_days
        if earliest >= n_days - 5:
            continue
        d_end = int(rng.integers(earliest, n_days))
        branch = config.branch_for(float(rng.random()))

        path = ret.copy()

        if not immortal:
            # Pre-delisting decline, spread over the window ahead of the event.
            if branch.decline_days > 0 and branch.decline < 0:
                start = max(0, d_end - branch.decline_days)
                n = d_end - start
                if n > 0:
                    per_day = (1.0 + branch.decline) ** (1.0 / n) - 1.0
                    path[start:d_end] = (1.0 + path[start:d_end]) * (1.0 + per_day) - 1.0

            path[d_end] = branch.terminal_return

        # Start at the DONOR's own price level, not at a normalized 100.
        #
        # This was a real defect in the first version.  Every synthetic name
        # began at $100, so a 75% decline left it at $25 — comfortably above a
        # $5 floor — and the price screen could never fire no matter how badly
        # the name was doing.  The measurement that came out of it ("96% were
        # still tradable the day before death") was therefore an artifact of
        # the generator, not a finding about the screen.  Real delisting
        # candidates fall through the listing minimums, which is the whole
        # mechanism the screen is supposed to exploit.
        donor_px = close[donor].dropna()
        start_px = float(donor_px.iloc[0]) if len(donor_px) else 100.0
        prices = start_px * np.cumprod(1.0 + np.clip(path, -0.99, None))
        if not immortal:
            prices[d_end + 1:] = np.nan
        # Terminal return of exactly -100% leaves a zero price, which is not a
        # tradable quote; the position is booked at the terminal return and the
        # name then vanishes.  Guard against a literal zero so nothing downstream
        # divides by it.
        prices = np.where(prices <= 1e-8, np.nan, prices)

        sym = f"{prefix}{i:04d}"
        new_close[sym] = prices
        new_open[sym] = prices * np.where(np.isfinite(ratio), ratio, 1.0)
        v = vol.astype(float).copy()
        if not immortal:
            v[d_end + 1:] = np.nan
        new_vol[sym] = v

        first = int(np.argmax(np.isfinite(prices)))
        rows.append({
            "Symbol": sym, "Donor": donor,
            "Start": idx[first],
            "Delist": pd.NaT if immortal else idx[d_end],
            "Branch": "survivor_control" if immortal else branch.name,
            "TerminalReturn": np.nan if immortal else branch.terminal_return,
        })

    add_c = pd.DataFrame(new_close, index=idx)
    add_o = pd.DataFrame(new_open, index=idx)
    add_v = pd.DataFrame(new_vol, index=idx)

    return (pd.concat([close, add_c], axis=1),
            pd.concat([open_, add_o], axis=1),
            pd.concat([volume.reindex(index=idx), add_v], axis=1),
            pd.DataFrame(rows))


def audit(manifest: pd.DataFrame, close: pd.DataFrame,
          config: DelistingConfig) -> Dict[str, float]:
    """
    Assertable facts about what was generated, for the validation gate.

    Every synthetic name must: exist before its delisting date, not exist after
    it, and live long enough to have been pickable.  A generator that silently
    produced names with no history, or names that keep trading past their own
    delisting, would make the whole bound meaningless while still returning
    plausible numbers.
    """
    if manifest.empty:
        return {"count": 0}

    alive_before, dead_after, too_short = 0, 0, 0
    for _, r in manifest.iterrows():
        col = close[r["Symbol"]]
        before = col.loc[:r["Delist"]]
        after = col.loc[r["Delist"]:].iloc[1:]
        if before.notna().sum() > 0:
            alive_before += 1
        if after.notna().sum() == 0:
            dead_after += 1
        if before.notna().sum() < config.min_life_days * 0.5:
            too_short += 1

    return {
        "count": len(manifest),
        "alive_before": alive_before,
        "dead_after": dead_after,
        "too_short": too_short,
        "mean_terminal": float(manifest["TerminalReturn"].mean()),
    }
