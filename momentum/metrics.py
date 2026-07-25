"""
Performance and turnover metrics.

Metrics return floats, not preformatted strings.  The pre-refactor version
formatted to strings inside the calculation and then parsed them back out with
`float(s.replace('%',''))` in the experiment runner, which silently truncated
every comparison to two decimal places.  Format at the point of display.

Turnover is a first-class metric here, reported next to Sharpe rather than
buried.  Tracks A-C all trade the same signal more often, and "more often" is
only worth it if the return improvement survives the cost of getting there.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def calculate_performance_metrics(returns: pd.Series,
                                  risk_free_rate: float = 0.045,
                                  periods_per_year: int = TRADING_DAYS) -> Dict[str, float]:
    """
    Core performance metrics from a series of periodic returns.

    Returns floats: total_return, cagr, volatility, sharpe_ratio, sortino_ratio,
    max_drawdown, calmar_ratio, hit_rate, num_periods, years.
    """
    returns = pd.Series(returns).dropna().astype(float)
    if returns.empty:
        raise ValueError("Cannot compute metrics on an empty return series")

    wealth = (1 + returns).cumprod()
    total_return = wealth.iloc[-1] - 1
    years = len(returns) / periods_per_year

    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else np.nan
    volatility = returns.std() * np.sqrt(periods_per_year)

    sharpe = (cagr - risk_free_rate) / volatility if volatility > 0 else 0.0

    downside = returns[returns < 0]
    downside_vol = downside.std() * np.sqrt(periods_per_year) if len(downside) > 1 else np.nan
    sortino = ((cagr - risk_free_rate) / downside_vol
               if downside_vol and downside_vol > 0 else np.nan)

    drawdown = wealth / wealth.expanding().max() - 1
    max_drawdown = drawdown.min()

    calmar = cagr / abs(max_drawdown) if max_drawdown < 0 else np.nan

    return {
        "total_return": float(total_return),
        "cagr": float(cagr),
        "volatility": float(volatility),
        "sharpe_ratio": float(sharpe),
        "sortino_ratio": float(sortino) if pd.notna(sortino) else np.nan,
        "max_drawdown": float(max_drawdown),
        "calmar_ratio": float(calmar) if pd.notna(calmar) else np.nan,
        "hit_rate": float((returns > 0).mean()),
        "num_periods": int(len(returns)),
        "years": float(years),
    }


def calculate_turnover_metrics(holdings_history: List[Dict],
                               years: float) -> Dict[str, float]:
    """
    How much trading the strategy actually does.

    Parameters
    ----------
    holdings_history : list of dicts with keys 'Date' and 'Holdings'
        One entry per portfolio change (not per day).
    years : float
        Length of the test period, for annualizing.

    Returns
    -------
    dict with:
      num_changes          portfolio changes over the period
      changes_per_year
      trades_per_year      individual buy/sell legs per year — the number that
                           maps to your actual order count
      avg_turnover         mean fraction of the portfolio replaced per change
      annual_turnover      fraction of portfolio value traded per year
      avg_hold_days        mean calendar days a position was held
      median_hold_days
    """
    if not holdings_history:
        return {k: 0.0 for k in ("num_changes", "changes_per_year", "trades_per_year",
                                 "avg_turnover", "annual_turnover",
                                 "avg_hold_days", "median_hold_days")}

    turnovers: List[float] = []
    n_trades = 0
    entry_dates: Dict[str, pd.Timestamp] = {}
    hold_lengths: List[float] = []

    prev: set = set()
    for entry in holdings_history:
        current = set(entry["Holdings"])
        date = pd.Timestamp(entry["Date"])

        sold = prev - current
        bought = current - prev

        if prev:
            # Fraction of the portfolio replaced, from the outgoing side.
            turnovers.append(len(sold) / len(prev))
        n_trades += len(sold) + len(bought)

        for sym in sold:
            if sym in entry_dates:
                hold_lengths.append((date - entry_dates.pop(sym)).days)
        for sym in bought:
            entry_dates[sym] = date

        prev = current

    # Positions still open at the end contribute their partial hold.
    if entry_dates:
        last_date = pd.Timestamp(holdings_history[-1]["Date"])
        hold_lengths.extend((last_date - d).days for d in entry_dates.values())

    years = max(years, 1e-9)
    avg_turnover = float(np.mean(turnovers)) if turnovers else 0.0

    return {
        "num_changes": len(holdings_history),
        "changes_per_year": len(holdings_history) / years,
        "trades_per_year": n_trades / years,
        "avg_turnover": avg_turnover,
        "annual_turnover": avg_turnover * len(holdings_history) / years,
        "avg_hold_days": float(np.mean(hold_lengths)) if hold_lengths else 0.0,
        "median_hold_days": float(np.median(hold_lengths)) if hold_lengths else 0.0,
    }


def returns_by_vix_band(returns: pd.Series, vix: pd.Series,
                        bands: Optional[List[float]] = None,
                        risk_free_rate: float = 0.045) -> pd.DataFrame:
    """
    Segment strategy performance by absolute VIX band.

    This is the analysis that produced the Sharpe staircase motivating Track A
    (3.30 / 0.88 / -0.99 / -1.48 across <15 / 15-20 / 20-25 / 25+).  Keeping it
    as a function means the graduated ladder can be validated against a
    re-derived staircase rather than against remembered numbers.

    Sharpe within a band is annualized from that band's days only; treat it as a
    conditional risk-adjusted return, not a number you could have earned by
    holding for a year.
    """
    bands = bands or [0, 15, 20, 25, np.inf]
    labels = []
    for i in range(len(bands) - 1):
        lo, hi = bands[i], bands[i + 1]
        labels.append(f"{lo:g}+" if np.isinf(hi) else f"{lo:g}-{hi:g}")

    returns = pd.Series(returns).dropna().astype(float)
    vix_aligned = pd.Series(vix).reindex(returns.index).ffill()

    band_series = pd.cut(vix_aligned, bins=bands, labels=labels, right=False)

    rows = []
    for label in labels:
        mask = band_series == label
        band_returns = returns[mask]
        if len(band_returns) < 2:
            continue

        mean_daily = band_returns.mean()
        ann_return = (1 + mean_daily) ** TRADING_DAYS - 1
        ann_vol = band_returns.std() * np.sqrt(TRADING_DAYS)
        sharpe = (ann_return - risk_free_rate) / ann_vol if ann_vol > 0 else np.nan

        rows.append({
            "VIX Band": label,
            "Days": len(band_returns),
            "Share of Days": len(band_returns) / len(returns),
            "Ann. Return": ann_return,
            "Ann. Volatility": ann_vol,
            "Sharpe": sharpe,
            "Hit Rate": (band_returns > 0).mean(),
            "Worst Day": band_returns.min(),
        })

    return pd.DataFrame(rows)


def format_metrics(metrics: Dict[str, float]) -> Dict[str, str]:
    """Display formatting, applied at the edge rather than in the calculation."""
    pct_keys = {"total_return", "cagr", "volatility", "max_drawdown",
                "hit_rate", "avg_turnover", "annual_turnover"}
    out = {}
    for key, value in metrics.items():
        if value is None or (isinstance(value, float) and np.isnan(value)):
            out[key] = "n/a"
        elif key in pct_keys:
            out[key] = f"{value:.2%}"
        elif isinstance(value, float):
            out[key] = f"{value:.2f}"
        else:
            out[key] = str(value)
    return out
