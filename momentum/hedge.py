"""
Track K hedge layer: hand slots of the book to real assets when they are
beating the stock book, and take them back when they stop.

WHAT IT IS

A layer over ANY stock book, expressed only through that book's return stream.
It never looks inside the book, so the same code runs over a Ken French
momentum decile in 1946, an ETF proxy in 2008 and Track J's own simulated
returns in 2022.  Shrinking the book pro-rata across its sleeves is exactly
scaling its return by the stock weight, which is what makes that possible.

HOW IT DECIDES — CONTINUOUS COMPETITION, NOT A REGIME CLASSIFIER

At each decision date, every hedge asset is scored on its trailing return over
`lookback` periods.  An asset QUALIFIES when both hold:

    relative   its trailing return beats the stock book's by `enter_margin`
    absolute   its trailing return beats cash's        (`require_beats_cash`)

Qualifiers are ranked by how far they beat the stock book, and the best take
slots of `slot` weight each up to `max_hedge`.  Cash is itself a hedge asset:
it qualifies whenever the stock book trails cash, which is "danger in the
regular market" with nowhere better to go.

Nothing forecasts inflation.  In an inflationary regime Treasuries trail and
drop out on their own; in a deflationary crash they lead and qualify.  That is
the whole reason for choosing this over a classifier (TODO 0l, agreed
2026-09-23).

THE ONE NON-PRICE-RETURN INPUT: STOCK-BOND CORRELATION

Duration assets are eligible only while the trailing correlation between the
stock book and the bond is below `corr_gate`.  A positive stock-bond
correlation is the market's own statement that the two are driven by the same
inflation shock (1946-48, 1970s, 2022), in which case the bond is not a hedge
however its trend reads.  `corr_gate=None` switches the gate off.

THE DANGER GATE (added 2026-09-23, after the first Tier 3 run)

The competition alone hedged 74% of all months: with seven candidates, one of
them beats the stock book over three months by chance most of the time, and the
calm decades paid for it (2011-19: -8.2pp CAGR, Sharpe 0.83 -> 0.41).
`danger_lookback` makes the slots available only while the stock book's own
trailing return is below cash's — absolute momentum on the book, the second
half of dual momentum, and James's "the data says there's danger in the
regular market".  This was NOT in the pre-registered default and is reported
as a post-hoc variant.

THE REGIME TRIGGER (option a, agreed 2026-09-23)

Tier 1 found that on Track J the layer must not run continuously: it costs
~5pp and sells Track J's recoveries.  As a PREPARATION model it should speak
only when the environment looks like the one it prepares for.  `regime_assets`
/ `regime_min` open the slots only while at least that many of those assets
beat the stock book by `enter_margin` and beat cash; `regime_corr_asset` /
`regime_corr_above` additionally require the stock-bond correlation (stock book
against that bond, over `corr_window`) to be above the threshold — the positive
correlation of 1946-48, the 1970s and 2022.  Both are AND-ed with any danger
gate.  Unset, nothing changes.

HARVEST: CATCH THE OUTSIZED RUN, THEN HAND THE SLOT BACK (James, 2026-09-23)

The hypothesis: in currency and rate stress, direct real-asset exposure runs
far harder than the stock book falls (stocks -20% while oil +50%).  The layer
should catch that early, sell into it once it has delivered what such events
deliver, and return the capital to stocks to await the recovery — not ride
the asset back down waiting for its trend to break.

Two ways to say "it has delivered", both optional and both a shape check,
because events like this are rare enough that no threshold is estimable:

    harvest_gain   gain since entry has reached this
    harvest_z      the asset's trailing `lookback` return sits this many
                   standard deviations above its own expanding history —
                   "topping expectations" for the asset.  An asset already
                   this extreme is not ENTERED either: buying it then is late.

    harvest_trail  (James, agreed after Tier 3) the asset has fallen this far
                   from its peak since entry.  Tier 3 found fixed targets
                   sell the fat tail that makes an episode; a trailing stop
                   sells only AFTER the top, so a run is allowed to continue.

    harvest_rel_lookback
                   (post-hoc, Tier 2, 2026-09-23) the STOCK BOOK has beaten
                   the asset over this shorter window.  Slow and decisive to
                   enter, fast to hand back: after a V-shaped crash the 63-day
                   comparison keeps favouring hedges for months into the
                   rebound (2020: -46pp), while a 10-21 day comparison sees
                   stocks recovering within weeks.  This is "go back to stocks
                   to await recovery" stated relative to the stock book, and
                   unlike the others it applies to a cash slot too.

A harvested asset is blocked for `harvest_blackout` periods so the rule does
not buy the top straight back.  With `harvest_to_stocks`, its slot stays with
the stock book for the blackout rather than passing to the next candidate.

HYSTERESIS AND EXITS

An asset already held stays while it beats the stock book by more than
`-exit_margin` rather than `enter_margin`, so a marginal asset does not flip in
and out each period.  Track A measured that whipsaw cost 2.53pp at daily
evaluation; this is the knob that pays for responsiveness.  Separately,
`fast_exit_lookback` ejects a held asset whose own shorter-window return has
fallen below cash's — the per-asset trend break, independent of the regime.

TIMING AND CADENCE

Weights decided at row t use only rows <= t and earn from row t+`exec_lag`.
On daily data `exec_lag=1` is trading at the signal close, which is not quite
achievable; `exec_lag=2` trades at the next session's close and gives the day
in between to the old allocation — the conservative reading.

`decide_every` / `decide_phase` evaluate only every N-th row and carry the
book (and the hysteresis state) in between.  The agreed sweep is 1/5/10/20
sessions (TODO 0l).  Every phase of a cadence should be run, because this repo
has already found a 14pp CAGR spread from rotation phase alone (TODO 0i).
Between decisions the weights are held at target, i.e. a costless daily
rebalance — the same convention every backtest here uses (FINDINGS caveat 3).  `hedge_weights`
is asserted free of look-ahead by truncation in tests/test_hedge.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class HedgeConfig:
    lookback: int = 3                     # periods; months in Tier 3
    slot: float = 0.25                    # weight per qualifying asset
    max_hedge: float = 0.75               # agreed cap, TODO 0l
    enter_margin: float = 0.0             # must beat stocks by this to enter
    exit_margin: float = 0.0              # held asset stays while excess > -exit_margin
    require_beats_cash: bool = True
    cash_asset: Optional[str] = "CASH"    # None: cash is never a hedge slot
    duration_assets: Tuple[str, ...] = ("UST10",)
    corr_window: int = 36                 # periods
    corr_gate: Optional[float] = 0.0      # None: no gate
    fast_exit_lookback: Optional[int] = None
    danger_lookback: Optional[int] = None  # None: slots always open
    danger_margin: float = 0.0            # stocks must trail cash by this
    regime_assets: Tuple[str, ...] = ()   # the trigger's real-asset set
    regime_min: int = 0                   # how many must beat stocks (and cash) to fire
    regime_corr_asset: Optional[str] = None     # bond for the stock-bond condition
    regime_corr_above: Optional[float] = None   # fire only while that correlation is above this
    harvest_gain: Optional[float] = None  # exit once gain since entry >= this
    harvest_z: Optional[float] = None     # exit (and never enter) at this z of trailing return
    harvest_z_min_history: int = 60       # periods of own history before z is trusted
    harvest_blackout: int = 6             # periods a harvested asset is locked out
    harvest_to_stocks: bool = True        # blacked-out slots stay with the stock book
    harvest_trail: Optional[float] = None  # exit after this fall from the peak since entry
    harvest_rel_lookback: Optional[int] = None  # exit once stocks beat the asset over this window
    decide_every: int = 1                 # evaluate every N rows
    decide_phase: int = 0                 # ...starting at this row offset
    exec_lag: int = 1                     # weights decided at t earn from t+exec_lag
    cost_bps: float = 10.0                # per unit of one-way turnover


def trailing_return(returns: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Compounded return over the last `lookback` rows, NaN until that much history exists."""
    return np.exp(np.log1p(returns).rolling(lookback, min_periods=lookback).sum()) - 1


def regime_open(stock: pd.Series, assets: pd.DataFrame, cfg: HedgeConfig,
                tr: Optional[pd.DataFrame] = None,
                tr_stock: Optional[pd.Series] = None) -> np.ndarray:
    """
    True where the regime trigger is satisfied (all True when it is not set).
    NaN history counts as NOT firing: a trigger with no history stays quiet.
    """
    ok = np.ones(len(stock), dtype=bool)
    if cfg.regime_min and cfg.regime_assets:
        if tr is None:
            tr = trailing_return(assets, cfg.lookback)
        if tr_stock is None:
            tr_stock = trailing_return(stock.to_frame("s"), cfg.lookback)["s"]
        cash = tr[cfg.cash_asset] if cfg.cash_asset in tr else 0.0
        names = [a for a in cfg.regime_assets if a in assets.columns]
        beats = pd.DataFrame({a: (tr[a].sub(tr_stock) > cfg.enter_margin) & (tr[a] > cash)
                              for a in names})
        ok &= (beats.sum(axis=1) >= cfg.regime_min).values
    if cfg.regime_corr_asset is not None and cfg.regime_corr_above is not None:
        c = stock.rolling(cfg.corr_window, min_periods=cfg.corr_window).corr(
            assets[cfg.regime_corr_asset])
        ok &= (c > cfg.regime_corr_above).fillna(False).values
    return ok


def hedge_weights(stock: pd.Series, assets: pd.DataFrame,
                  config: HedgeConfig,
                  decide_at: Optional[Sequence[pd.Timestamp]] = None) -> pd.DataFrame:
    """
    Target weights decided at each row, columns STOCKS plus every asset.
    Row t uses data through t only and is meant to earn row t+1.

    `decide_at`, when given, replaces the positional `decide_every` cadence
    with explicit decision dates — Track J's rotation Tuesdays, which drift
    around holidays and so are not every N sessions.
    """
    cfg = config
    names = list(assets.columns)
    if cfg.cash_asset is not None and cfg.cash_asset not in names:
        raise ValueError(f"cash asset {cfg.cash_asset!r} not among the assets")
    n_slots = int(np.floor(cfg.max_hedge / cfg.slot + 1e-9))

    tr = trailing_return(assets, cfg.lookback)
    tr_stock = trailing_return(stock.to_frame("s"), cfg.lookback)["s"]
    fast = (trailing_return(assets, cfg.fast_exit_lookback)
            if cfg.fast_exit_lookback else None)
    if cfg.danger_lookback:
        if cfg.cash_asset is None:
            raise ValueError("the danger gate compares the stock book with cash")
        d_stock = trailing_return(stock.to_frame("s"), cfg.danger_lookback)["s"]
        d_cash = trailing_return(assets[[cfg.cash_asset]], cfg.danger_lookback)[cfg.cash_asset]
        # NaN history -> not in danger, so no hedge rather than an unearned one
        open_slots = (d_stock < d_cash - cfg.danger_margin).values
    else:
        open_slots = np.ones(len(stock), dtype=bool)
    open_slots = open_slots & regime_open(stock, assets, cfg, tr, tr_stock)
    gated = pd.DataFrame(False, index=assets.index, columns=names)
    if cfg.corr_gate is not None:
        for a in cfg.duration_assets:
            if a in names:
                c = stock.rolling(cfg.corr_window, min_periods=cfg.corr_window).corr(assets[a])
                # no correlation history yet -> not eligible, rather than silently ungated
                gated[a] = ~(c < cfg.corr_gate)

    # wealth index per asset, for gain since entry; z of the trailing return
    # against the asset's own expanding history, both through row t only
    wealth = (1 + assets.fillna(0.0)).cumprod()
    if cfg.harvest_rel_lookback:
        rel_a = trailing_return(assets, cfg.harvest_rel_lookback)
        rel_s = trailing_return(stock.to_frame("s"), cfg.harvest_rel_lookback)["s"]
    if cfg.harvest_z is not None:
        mu = tr.expanding(min_periods=cfg.harvest_z_min_history).mean()
        sd = tr.expanding(min_periods=cfg.harvest_z_min_history).std()
        z = (tr - mu) / sd
    else:
        z = None

    cash = cfg.cash_asset
    W = np.zeros((len(stock), len(names)))
    held: set = set()
    entry: Dict[str, float] = {}
    peak: Dict[str, float] = {}
    blocked_until: Dict[str, int] = {}
    every = max(int(cfg.decide_every), 1)
    on_date = (np.asarray(stock.index.isin(pd.DatetimeIndex(decide_at)))
               if decide_at is not None else None)
    for i, t in enumerate(stock.index):
        for a in held:
            peak[a] = max(peak[a], wealth[a].iat[i])
        deciding = on_date[i] if on_date is not None else (i - cfg.decide_phase) % every == 0
        if not deciding:
            if i > 0:
                W[i] = W[i - 1]
            continue
        rs = tr_stock.iat[i]
        if np.isnan(rs) or not open_slots[i]:
            held, entry, peak = set(), {}, {}
            continue

        for a in sorted(held):
            if cfg.harvest_rel_lookback and rel_s.iat[i] > rel_a[a].iat[i]:
                blocked_until[a] = i + cfg.harvest_blackout
                held.discard(a)
                entry.pop(a)
                peak.pop(a)
                continue
            if a == cash:
                continue
            gain = wealth[a].iat[i] / entry[a] - 1
            topped = z is not None and z[a].iat[i] >= cfg.harvest_z
            trailed = (cfg.harvest_trail is not None
                       and wealth[a].iat[i] / peak[a] - 1 <= -cfg.harvest_trail)
            if (cfg.harvest_gain is not None and gain >= cfg.harvest_gain) or topped or trailed:
                blocked_until[a] = i + cfg.harvest_blackout   # rows i..i+blackout-1 are out
                held.discard(a)
                entry.pop(a)
                peak.pop(a)
        blocked = {a for a, u in blocked_until.items() if i < u}
        room = n_slots - (len(blocked) if cfg.harvest_to_stocks else 0)

        rc = tr[cash].iat[i] if cash is not None else 0.0
        scored = []
        for j, a in enumerate(names):
            ra = tr[a].iat[i]
            if np.isnan(ra) or gated[a].iat[i] or a in blocked:
                continue
            if (z is not None and a != cash and a not in held
                    and z[a].iat[i] >= cfg.harvest_z):
                continue
            excess = ra - rs
            margin = -cfg.exit_margin if a in held else cfg.enter_margin
            if not excess > margin:
                continue
            if cfg.require_beats_cash and a != cash and not ra > rc:
                continue
            if fast is not None and a in held and a != cash:
                fa, fc = fast[a].iat[i], (fast[cash].iat[i] if cash else 0.0)
                if not fa > fc:
                    continue
            scored.append((excess, j, a))
        scored.sort(reverse=True)
        chosen = scored[:max(room, 0)]
        new_held = {a for _, _, a in chosen}
        entry = {a: entry.get(a, wealth[a].iat[i]) for a in new_held}
        peak = {a: peak.get(a, wealth[a].iat[i]) for a in new_held}
        held = new_held
        for _, j, _ in chosen:
            W[i, j] = cfg.slot

    out = pd.DataFrame(W, index=stock.index, columns=names)
    out.insert(0, "STOCKS", 1.0 - out.sum(axis=1))
    return out


def apply_weights(stock: pd.Series, assets: pd.DataFrame, weights: pd.DataFrame,
                  cost_bps: float, exec_lag: int = 1) -> pd.DataFrame:
    """
    Realised returns: weights at row t earn row t+exec_lag.  Rebalanced to target each
    period, so turnover is the change in target weights.  Returns a frame with
    `gross`, `cost`, `net`, `turnover`, `hedge_share`, on the EARNING rows.
    """
    rets = pd.concat([stock.rename("STOCKS"), assets], axis=1)[weights.columns]
    w = weights.shift(exec_lag)                        # decided at t, earns t+exec_lag
    live = w.notna().all(axis=1) & rets["STOCKS"].notna()
    held_ret = rets.where(w > 0)
    # an asset given weight must have a return in the period it is held
    missing = (w > 0) & held_ret.isna()
    if missing.any().any():
        bad = missing.stack()
        raise AssertionError(f"weight on an asset with no return: {bad[bad].index[:5].tolist()}")
    gross = (w * rets.fillna(0.0)).sum(axis=1)
    turnover = w.diff().abs().sum(axis=1) / 2           # one-way
    cost = turnover * cost_bps / 1e4
    out = pd.DataFrame({"gross": gross, "cost": cost, "net": gross - cost,
                        "turnover": turnover,
                        "hedge_share": 1.0 - w["STOCKS"]})
    return out[live]


def run_hedge(stock: pd.Series, assets: pd.DataFrame,
              config: HedgeConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """(weights by decision date, realised returns by earning date)."""
    w = hedge_weights(stock, assets, config)
    return w, apply_weights(stock, assets, w, config.cost_bps, config.exec_lag)
