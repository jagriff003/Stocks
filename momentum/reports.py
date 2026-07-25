"""
Reporting outputs: per-stock performance, correlation matrices, holdings.

These are diagnostics rather than strategy logic — nothing here feeds back into
selection.  Kept separate so the strategy modules stay free of formatting and
CSV-writing concerns.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .metrics import TRADING_DAYS, calculate_performance_metrics


def individual_stock_performance(close: pd.DataFrame,
                                 recent_years: int = 3,
                                 risk_free_rate: float = 0.045) -> pd.DataFrame:
    """
    Per-ticker performance, plus volatility-scaled stop levels.

    The stop columns are sized off *recent* volatility rather than the full
    history, because a stop calibrated to a stock's 2010-era volatility will be
    either useless or constantly triggered today.
    """
    recent_cutoff = close.index[-1] - pd.DateOffset(years=recent_years)
    rows = []

    for stock in close.columns:
        returns = close[stock].pct_change().dropna()
        if returns.empty:
            continue

        m = calculate_performance_metrics(returns, risk_free_rate=risk_free_rate)

        recent_returns = close[stock].loc[recent_cutoff:].pct_change().dropna()
        source = recent_returns if len(recent_returns) > 0 else returns
        recent_ann_vol = source.std() * np.sqrt(TRADING_DAYS)
        daily_vol = recent_ann_vol / np.sqrt(TRADING_DAYS)

        rows.append({
            "Stock": stock,
            "Total Return": m["total_return"],
            "CAGR": m["cagr"],
            "Volatility": m["volatility"],
            "Sharpe Ratio": m["sharpe_ratio"],
            "Max Drawdown": m["max_drawdown"],
            "Number of Days": m["num_periods"],
            f"Recent {recent_years}Y Vol": recent_ann_vol,
            "Daily Vol": daily_vol,
            "2x Stop %": daily_vol * 2.0,
            "2.5x Stop %": daily_vol * 2.5,
        })

    return pd.DataFrame(rows).sort_values("CAGR", ascending=False)


def correlation_matrices(close: pd.DataFrame,
                         periods: Optional[List[int]] = None
                         ) -> Dict[int, pd.DataFrame]:
    """
    Trailing correlation matrices over the given lookbacks.

    Worth reading alongside a rebalance: four names that each rank well but
    correlate at 0.9 are one position, not four, and the equal-weight
    construction will not tell you that.
    """
    periods = periods or [20, 50, 200]
    returns = close.pct_change().dropna()

    out = {}
    for period in periods:
        if len(returns) < period:
            continue
        out[period] = returns.iloc[-period:].corr()
    return out


def summarize_correlations(matrix: pd.DataFrame, period: int, top: int = 10) -> str:
    """Human-readable summary of one correlation matrix."""
    mask = np.triu(np.ones_like(matrix, dtype=bool), k=1)
    pairs = matrix.where(mask).stack()

    lines = [
        f"\n=== {period}-DAY CORRELATION MATRIX SUMMARY ===",
        f"Period: Last {period} trading sessions",
        f"Mean:   {pairs.mean():.3f}   Median: {pairs.median():.3f}",
        f"Min:    {pairs.min():.3f}   Max:    {pairs.max():.3f}   "
        f"Std: {pairs.std():.3f}",
        f"\nTop {top} highest correlations:",
    ]
    for (a, b), val in pairs.sort_values(ascending=False).head(top).items():
        lines.append(f"  {a} - {b}: {val:.3f}")

    lines.append(f"\nTop {top} lowest correlations:")
    for (a, b), val in pairs.sort_values().head(top).items():
        lines.append(f"  {a} - {b}: {val:.3f}")

    return "\n".join(lines)


def portfolio_concentration(holdings: List[str],
                            sectors: Dict[str, str]) -> pd.DataFrame:
    """
    Sector breakdown of a portfolio.

    Read this alongside `portfolio_correlation`, never on its own.  Sector
    labels are a poor proxy for shared risk in both directions: V and STT are
    both "Financials" yet correlate at 0.02, while IAU and NEM sit in different
    sectors and correlate at 0.79.  A sector-only concentration warning will
    reliably flag the wrong things.
    """
    rows = [{"Symbol": s, "Sector": sectors.get(s, "Unknown")} for s in holdings]
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame

    counts = (frame.groupby("Sector").size()
              .rename("Positions").reset_index())
    counts["Weight"] = counts["Positions"] / len(holdings)
    return counts.sort_values("Weight", ascending=False)


def portfolio_correlation(holdings: List[str], close: pd.DataFrame,
                          window: int = 50) -> pd.DataFrame:
    """
    Pairwise correlation within the current book.

    This is the honest concentration measure: what actually moves together, not
    what shares a label.
    """
    held = [s for s in holdings if s in close.columns]
    if len(held) < 2:
        return pd.DataFrame(columns=["Symbol A", "Symbol B", "Correlation"])

    returns = close[held].pct_change().dropna(axis=0, how="all").iloc[-window:]
    corr = returns.corr().rename_axis(index="Symbol A", columns="Symbol B")

    mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
    pairs = (corr.where(mask).stack(future_stack=True)
                 .rename("Correlation").reset_index()
                 .dropna(subset=["Correlation"]))
    return pairs.sort_values("Correlation", ascending=False).reset_index(drop=True)


def format_report_frame(frame: pd.DataFrame,
                        pct_columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Format a numeric report frame for display, leaving the source untouched."""
    out = frame.copy()
    pct_columns = pct_columns or [
        c for c in out.columns
        if any(k in c for k in ("Return", "CAGR", "Vol", "Drawdown", "Stop", "Rate"))
    ]
    for col in pct_columns:
        if col in out.columns and pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].map(lambda v: f"{v:.2%}" if pd.notna(v) else "n/a")
    return out
