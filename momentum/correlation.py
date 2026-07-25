"""
Correlation-aware selection and universe screening.

The ranker scores each stock in isolation, so nothing in the strategy can see
that two of its four picks are the same bet wearing different tickers.  With
equal weights and only four slots, two names correlating at 0.9 is half the book
in one position.  The correlation matrices this repo already exports were the
manual workaround; this module folds that judgment into the algorithm.

Two independent mechanisms:

  `select_diversified`   medium-term filter applied at selection time
  `screen_universe`      long-term diagnostic for the quarterly screen

Both compute correlations from trailing data only.  Using the full-sample
correlation matrix — which is what the exported CSVs contain — would leak future
information into historical selections and quietly inflate every backtest.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import CorrelationConfig


# --------------------------------------------------------------------------
# Selection-time filter
# --------------------------------------------------------------------------

@dataclass
class SelectionTrace:
    """Why each candidate was taken or skipped — for diagnostics, not logic."""
    selected: List[str]
    rejected: List[Tuple[str, str, float]]   # (candidate, conflicts_with, corr)
    threshold: float
    relaxed: bool = False


def effective_threshold(corr: pd.DataFrame, config: CorrelationConfig) -> float:
    """
    The correlation cap in force for this date.

    For 'absolute' this is just the configured constant.  For 'relative' it is a
    quantile of the current pairwise distribution, so the cap rises in regimes
    where everything moves together and falls when dispersion returns.
    """
    if config.method == "absolute":
        return config.max_correlation

    if config.method != "relative":
        raise ValueError(f"Unknown correlation method: {config.method!r}")

    if corr is None or corr.empty or corr.shape[0] < 3:
        return config.max_correlation

    mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
    pairs = corr.values[mask]
    pairs = pairs[~np.isnan(pairs)]
    if pairs.size == 0:
        return config.max_correlation

    return float(np.quantile(pairs, config.max_percentile))


def select_diversified(ranked: pd.Series,
                       corr: Optional[pd.DataFrame],
                       config: CorrelationConfig,
                       top_n: int,
                       exempt: Sequence[str] = ()) -> SelectionTrace:
    """
    Walk down the ranked list taking the best candidate that adds a new bet.

    The top-ranked name is always taken — the filter never overrides the ranker's
    first choice, only its later ones.  Each subsequent candidate is accepted if
    its correlation with every already-accepted holding is at or below the
    threshold.

    Parameters
    ----------
    ranked : Series
        Candidates sorted best-first, index = symbols.
    corr : DataFrame or None
        Trailing correlation matrix.  None disables filtering.
    config : CorrelationConfig
    top_n : int
    exempt : sequence of str
        Symbols the filter ignores (the defensive sleeve, normally).

    Returns
    -------
    SelectionTrace
    """
    candidates = list(ranked.index)
    if not config.enabled or corr is None or corr.empty:
        return SelectionTrace(candidates[:top_n], [], np.nan)

    threshold = effective_threshold(corr, config)
    exempt_set = set(exempt)

    selected: List[str] = []
    rejected: List[Tuple[str, str, float]] = []

    for symbol in candidates:
        if len(selected) >= top_n:
            break

        if symbol in exempt_set or symbol not in corr.index:
            selected.append(symbol)
            continue

        conflict = None
        for held in selected:
            if held in exempt_set or held not in corr.columns:
                continue
            rho = corr.at[symbol, held] if held in corr.columns else np.nan
            if pd.notna(rho) and rho > threshold:
                conflict = (held, float(rho))
                break

        if conflict is None:
            selected.append(symbol)
        else:
            rejected.append((symbol, conflict[0], conflict[1]))

    relaxed = False
    if len(selected) < top_n:
        if config.on_infeasible == "relax":
            # Backfill with the best rejected names, in original rank order.
            relaxed = True
            for symbol, _, _ in rejected:
                if len(selected) >= top_n:
                    break
                if symbol not in selected:
                    selected.append(symbol)
            # Still short (thin universe): take anything remaining.
            for symbol in candidates:
                if len(selected) >= top_n:
                    break
                if symbol not in selected:
                    selected.append(symbol)
        elif config.on_infeasible == "hold_fewer":
            if len(selected) < config.min_positions:
                relaxed = True
                for symbol, _, _ in rejected:
                    if len(selected) >= config.min_positions:
                        break
                    if symbol not in selected:
                        selected.append(symbol)
        else:
            raise ValueError(f"Unknown on_infeasible: {config.on_infeasible!r}")

    return SelectionTrace(selected, rejected, threshold, relaxed)


class RollingCorrelation:
    """
    Trailing correlation matrices, computed on demand and cached per date.

    A rebalance needs the matrix for one date at a time, and there are only a few
    hundred rebalances, so computing lazily beats materializing a full rolling
    panel.  Caching matters because the same date is queried more than once when
    several configs share a run.
    """

    def __init__(self, close: pd.DataFrame, window: int):
        self.returns = close.pct_change()
        self.window = window
        self._cache: Dict[pd.Timestamp, Optional[pd.DataFrame]] = {}

    def at(self, date) -> Optional[pd.DataFrame]:
        date = pd.Timestamp(date)
        if date in self._cache:
            return self._cache[date]

        window_returns = self.returns.loc[:date].iloc[-self.window:]
        if len(window_returns) < max(10, self.window // 2):
            matrix = None
        else:
            # Require most of the window present per column, else the pairwise
            # estimate is built from too few overlapping observations to mean
            # anything.
            usable = window_returns.columns[
                window_returns.notna().sum() >= max(10, self.window // 2)
            ]
            matrix = window_returns[usable].corr() if len(usable) >= 2 else None

        self._cache[date] = matrix
        return matrix


# --------------------------------------------------------------------------
# Universe screening
# --------------------------------------------------------------------------

def redundant_pairs(close: pd.DataFrame,
                    config: Optional[CorrelationConfig] = None,
                    as_of=None) -> pd.DataFrame:
    """
    Universe members that are near-duplicates of each other.

    Directly actionable at the quarterly screen: a pair above the threshold is
    two slots' worth of universe delivering one bet's worth of diversification,
    so one of them can be swapped for something that adds an independent
    exposure.
    """
    config = config or CorrelationConfig()
    returns = close.pct_change()
    if as_of is not None:
        returns = returns.loc[:pd.Timestamp(as_of)]
    returns = returns.iloc[-config.long_term_window:]

    corr = returns.corr()
    # yfinance names both axes 'Ticker'; stacking that gives a MultiIndex with
    # two identically-named levels, which reset_index refuses to flatten.
    corr = corr.rename_axis(index="Symbol A", columns="Symbol B")

    mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
    pairs = (corr.where(mask)
                 .stack(future_stack=True)
                 .rename("Correlation")
                 .reset_index()
                 .dropna(subset=["Correlation"]))

    pairs = pairs.sort_values("Correlation", ascending=False).reset_index(drop=True)
    pairs["Flagged"] = pairs["Correlation"] >= config.redundant_threshold
    return pairs


def redundancy_by_symbol(close: pd.DataFrame,
                         config: Optional[CorrelationConfig] = None,
                         as_of=None) -> pd.DataFrame:
    """
    Per-ticker redundancy: how much each name duplicates the rest of the universe.

    `Mean Correlation` is the better screening statistic than any single pair —
    a stock can be below the pair threshold against everything yet still be
    broadly redundant, adding nothing the universe doesn't already cover.
    """
    config = config or CorrelationConfig()
    returns = close.pct_change()
    if as_of is not None:
        returns = returns.loc[:pd.Timestamp(as_of)]
    returns = returns.iloc[-config.long_term_window:]

    # Mask the diagonal so a ticker's perfect self-correlation does not count
    # toward its redundancy.  Build a fresh frame rather than writing into
    # `.values`, which pandas 3 exposes read-only.
    corr = returns.corr()
    corr = corr.mask(np.eye(corr.shape[0], dtype=bool))

    rows = []
    for symbol in corr.columns:
        series = corr[symbol].dropna()
        if series.empty:
            continue
        closest = series.idxmax()
        rows.append({
            "Symbol": symbol,
            "Mean Correlation": series.mean(),
            "Max Correlation": series.max(),
            "Closest Peer": closest,
            "Above Threshold": int((series >= config.redundant_threshold).sum()),
        })

    return (pd.DataFrame(rows)
            .sort_values("Mean Correlation", ascending=False)
            .reset_index(drop=True))


def effective_bets(close: pd.DataFrame,
                   config: Optional[CorrelationConfig] = None,
                   as_of=None) -> Dict[str, float]:
    """
    How many independent bets the universe actually contains.

    Computed from the eigenvalue spread of the correlation matrix: a universe of
    N perfectly uncorrelated assets has N equal eigenvalues and N effective bets;
    one where everything moves together has a single dominant eigenvalue and one
    effective bet.  A 52-name universe delivering 6 effective bets is a much
    narrower portfolio than its ticker count suggests.
    """
    config = config or CorrelationConfig()
    returns = close.pct_change()
    if as_of is not None:
        returns = returns.loc[:pd.Timestamp(as_of)]
    returns = returns.iloc[-config.long_term_window:]
    # Drop empty sessions before dropping incomplete tickers, or a single
    # all-NaN row takes the whole universe with it.
    returns = returns.dropna(axis=0, how="all").dropna(axis=1, how="any")

    if returns.shape[1] < 2:
        return {"n_assets": returns.shape[1], "effective_bets": float(returns.shape[1]),
                "top_eigenvalue_share": np.nan}

    corr = returns.corr().values
    eigenvalues = np.linalg.eigvalsh(corr)
    eigenvalues = np.clip(eigenvalues, 1e-12, None)
    weights = eigenvalues / eigenvalues.sum()

    # Exponential of Shannon entropy — the standard "effective number" measure.
    entropy = -(weights * np.log(weights)).sum()

    return {
        "n_assets": int(corr.shape[0]),
        "effective_bets": float(np.exp(entropy)),
        "top_eigenvalue_share": float(weights.max()),
    }


def screen_universe(close: pd.DataFrame,
                    sectors: Optional[Dict[str, str]] = None,
                    config: Optional[CorrelationConfig] = None,
                    as_of=None) -> Dict[str, object]:
    """
    Full quarterly screening report.

    Returns a dict of frames rather than printing, so callers can export or
    render however they like.
    """
    config = config or CorrelationConfig()

    pairs = redundant_pairs(close, config, as_of)
    if sectors and not pairs.empty:
        pairs["Sector A"] = pairs["Symbol A"].map(sectors)
        pairs["Sector B"] = pairs["Symbol B"].map(sectors)
        pairs["Same Sector"] = pairs["Sector A"] == pairs["Sector B"]

    return {
        "redundant_pairs": pairs,
        "by_symbol": redundancy_by_symbol(close, config, as_of),
        "effective_bets": effective_bets(close, config, as_of),
        "window": config.long_term_window,
        "threshold": config.redundant_threshold,
    }
