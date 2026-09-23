"""
Track K live: the preparation model's reading, from the daily market store.

WHAT TRACK K IS (TODO 0l, option a, agreed 2026-09-23)

A model that prepares for a regime not in Track J's record — inflation or
currency debasement, where real assets run while stocks stall.  It runs
alongside Track J and does not override it.  Most of the time it says nothing.
When its regime trigger fires, it recommends handing slots of the book to the
real assets that are beating the market, and Track J shrinks pro-rata.

Measured before it was allowed to report (scripts/analyze_hedge_trigger.py):
it fires in 3% of months over 1927-2026 outside the inflation episodes, fires
in 1973-74, 1977-81 and 2021-22, and — obeyed mechanically on Track J
2012-2026 — costs 0.3-0.7pp of CAGR with Sharpe unchanged.  It misses 1946-48,
when only two of its four assets existed.  It is a discretionary input, and
the decision log records what was done with it.

THE CONFIGURATION IS HERE AND ONLY HERE

`LIVE_CONFIG` is what the report runs and what the trigger study validated;
the study imports it rather than restating it.

SIGNALS ON THE BACKTEST SYMBOLS, TRADES IN THE LIVE ONES

The trigger was validated on DBC and TIP.  The live instruments are PDBC and
SCHP (same exposures; PDBC avoids the K-1).  Signals stay on the validated
series; recommendations are expressed in the live tickers.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .hedge import HedgeConfig, hedge_weights, regime_open, trailing_return
from .marketstore import INSTRUMENTS, REFERENCE

LIVE_CONFIG = HedgeConfig(
    lookback=63, enter_margin=0.10, slot=0.25, max_hedge=0.75,
    corr_window=252, corr_gate=0.0, duration_assets=("UST10", "UST_L"),
    regime_assets=("GOLD", "SILVER", "CMDTY", "ENERGY"), regime_min=2,
    regime_corr_asset="UST10", regime_corr_above=0.0,
    decide_every=5, exec_lag=2, cost_bps=10.0)

SIGNAL_SYMBOL = {i.role: i.backtest for i in INSTRUMENTS}
LIVE_SYMBOL = {i.role: i.live for i in INSTRUMENTS}


def assets_from_store(store: pd.DataFrame) -> pd.DataFrame:
    """Role-named daily returns on SPY's calendar, gaps after listing as 0."""
    cal = store[REFERENCE].dropna().index
    a = pd.DataFrame({role: store[sym] for role, sym in SIGNAL_SYMBOL.items()}).reindex(cal)
    for c in a.columns:
        first = a[c].first_valid_index()
        if first is not None:
            a.loc[first:, c] = a.loc[first:, c].fillna(0.0)
    return a


def reading_at(market: pd.Series, assets: pd.DataFrame, date: pd.Timestamp,
               cfg: HedgeConfig = LIVE_CONFIG) -> pd.DataFrame:
    """Per-asset diagnostics on one date: the numbers behind the decision."""
    tr = trailing_return(assets, cfg.lookback).loc[date]
    tm = trailing_return(market.to_frame("m"), cfg.lookback)["m"].loc[date]
    cash = tr[cfg.cash_asset]
    rows = []
    for role in assets.columns:
        r = tr[role]
        rows.append({"role": role, "signal": SIGNAL_SYMBOL[role], "live": LIVE_SYMBOL[role],
                     "3m return": r, "vs SPY": r - tm, "beats SPY by margin": r - tm > cfg.enter_margin,
                     "beats cash": (r > cash) if role != cfg.cash_asset else True,
                     "trigger asset": role in cfg.regime_assets})
    return pd.DataFrame(rows).set_index("role")


def trigger_state(market: pd.Series, assets: pd.DataFrame, date: pd.Timestamp,
                  cfg: HedgeConfig = LIVE_CONFIG) -> Dict:
    diag = reading_at(market, assets, date, cfg)
    trig = diag[diag["trigger asset"]]
    n = int((trig["beats SPY by margin"] & trig["beats cash"]).sum())
    corr = market.rolling(cfg.corr_window, min_periods=cfg.corr_window).corr(
        assets[cfg.regime_corr_asset]).loc[date]
    firing = bool(pd.Series(regime_open(market, assets, cfg), index=market.index).loc[date])
    return {"date": date, "firing": firing, "assets_qualifying": n, "needed": cfg.regime_min,
            "stock_bond_corr": float(corr), "market_3m": float(
                trailing_return(market.to_frame("m"), cfg.lookback)["m"].loc[date])}


def recommendations(store: pd.DataFrame, rotations: Sequence[pd.Timestamp],
                    cfg: HedgeConfig = LIVE_CONFIG) -> Dict[str, Dict]:
    """
    Track K at the last rotation on or before the latest session, and on the
    latest session as if it were a decision date.  Weights are by LIVE ticker.
    """
    market = store[REFERENCE].dropna()
    assets = assets_from_store(store)
    latest = market.index[-1]
    rot = [pd.Timestamp(d) for d in rotations if pd.Timestamp(d) <= latest]
    if not rot:
        raise ValueError("no rotation date on or before the latest session")
    last = max(d for d in rot if d in market.index)
    out = {}
    for label, date, decide in (("last_rotation", last, rot),
                                ("current", latest, sorted(set(rot) | {latest}))):
        w = hedge_weights(market, assets, cfg, decide_at=decide).loc[date]
        hedge = {LIVE_SYMBOL[r]: float(v) for r, v in w.drop("STOCKS").items() if v > 0}
        out[label] = {"date": date, "hedge_share": float(1.0 - w["STOCKS"]), "weights": hedge,
                      "trigger": trigger_state(market, assets, date, cfg),
                      "diagnostics": reading_at(market, assets, date, cfg)}
    return out


def combine(trackj_weights: Dict[str, float], hedge_share: float,
            trackk_weights: Dict[str, float]) -> pd.DataFrame:
    """Track J scaled by (1 - hedge share) plus Track K's slots, by ticker."""
    rows: Dict[str, Dict] = {}
    for s, w in trackj_weights.items():
        rows.setdefault(s, {"Track J": 0.0, "Track K": 0.0})["Track J"] += w * (1.0 - hedge_share)
    for s, w in trackk_weights.items():
        rows.setdefault(s, {"Track J": 0.0, "Track K": 0.0})["Track K"] += w
    df = pd.DataFrame(rows).T
    df["weight"] = df["Track J"] + df["Track K"]
    return df.sort_values("weight", ascending=False)
