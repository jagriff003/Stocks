"""
End-to-end strategy run: prices in, `PortfolioResult` out.

One function, `run_strategy`, so every caller — the live script, the experiment
runner, the Track A/B/C simulations — goes through the same path.  Divergent
copies of "score, blend, select, simulate" is exactly how a live model and its
backtest drift apart without anyone noticing.
"""

from __future__ import annotations

from typing import Optional, Tuple

import pandas as pd

from .backtest import (PortfolioResult, build_target_portfolios,
                       selection_for_date, simulate_portfolio)
from .config import ModelConfig
from .data import PriceData
from .signals import apply_velocity_blend, calculate_composite_scores


def compute_scores(prices: PriceData, config: ModelConfig,
                   underlying_path: Optional[str] = None
                   ) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Composite scores, before and after the velocity blend.

    Returns
    -------
    ranking_scores : DataFrame used for ranking (blended if velocity is enabled)
    base_scores    : level-only scores, kept for the min_level_threshold floor
    detail         : the full dict from `calculate_composite_scores`
    """
    detail = calculate_composite_scores(
        prices.close, prices.spy, config.scoring, underlying_path=underlying_path
    )

    base_scores = detail["composite"]
    ranking_scores = base_scores

    if config.velocity is not None:
        ranking_scores = apply_velocity_blend(base_scores, config.velocity)

    return ranking_scores, base_scores, detail


def run_strategy(prices: PriceData, config: ModelConfig,
                 underlying_path: Optional[str] = None,
                 verbose: bool = False,
                 ranking_override: Optional[pd.DataFrame] = None,
                 slippage_by_symbol: Optional[pd.Series] = None
                 ) -> PortfolioResult:
    """
    Score, select and simulate under `config`.

    The result's `metrics` are net of the slippage in `config.execution`, and
    its `turnover` block reports what the strategy actually had to trade to get
    them.  Read those together: a variant that improves CAGR by trading three
    times as often has not necessarily improved anything.

    ranking_override
        Use this score panel for RANKING instead of the composite, while
        leaving every other part of the system — eligibility, the level floor,
        the regime overlay, the correlation filter, the hold clock, execution —
        exactly as configured.  This is what makes a ranker comparison fair:
        one input changes and nothing else can.  `base_scores` still comes from
        the real composite, so the `min_level_threshold` floor keeps meaning
        what it means and the two arms face an identical eligible pool.

    slippage_by_symbol
        Optional per-symbol one-way slippage as a fraction.  None keeps the
        flat rate, which is what every historical result was computed under.
    """
    ranking_scores, base_scores, _ = compute_scores(prices, config, underlying_path)

    if ranking_override is not None:
        ranking_scores = ranking_override.reindex(
            index=ranking_scores.index, columns=ranking_scores.columns)

    targets, rebalance_history = build_target_portfolios(
        ranking_scores,
        price_columns=list(prices.close.columns),
        top_n=config.top_n,
        min_data_days=config.min_data_days,
        hold_days=config.hold_days,
        vix_data=prices.vix,
        vix_config=config.vix,
        base_composite_scores=base_scores,
        velocity_config=config.velocity,
        correlation_config=config.correlation,
        graduated_config=config.graduated_vix,
        exit_config=config.exits,
        close=prices.close,
        rank_offset=config.rank_offset,
        rank_offset_scope=config.rank_offset_scope,
        monitor_symbols=config.monitor_symbols,
        verbose=verbose,
    )

    result = simulate_portfolio(
        targets, prices.close, prices.open_, execution=config.execution,
        sizing=config.sizing, scores=ranking_scores,
        slippage_by_symbol=slippage_by_symbol,
    )
    result.rebalance_history = rebalance_history
    return result


def current_selection(prices: PriceData, config: ModelConfig,
                      ranking_scores: Optional[pd.DataFrame] = None,
                      base_scores: Optional[pd.DataFrame] = None,
                      as_of=None,
                      underlying_path: Optional[str] = None):
    """
    The selection the model would make right now, off the rotation clock.

    The live book rotates on `config.hold_days`, so between rotations the held
    names and the currently top-ranked names diverge.  Reporting only the held
    book hides that divergence; reporting only the current ranking invites
    trading it, which is a different (and untested) strategy.  Both are printed
    so the gap is visible and the decision to act on it is deliberate.

    Pass `ranking_scores`/`base_scores` if they are already computed — scoring
    is the expensive step and there is no reason to repeat it.

    Returns `(record, ranked)` as `selection_for_date`, or None.
    """
    if ranking_scores is None or base_scores is None:
        ranking_scores, base_scores, _ = compute_scores(
            prices, config, underlying_path)

    return selection_for_date(
        ranking_scores,
        price_columns=list(prices.close.columns),
        date=as_of,
        top_n=config.top_n,
        min_data_days=config.min_data_days,
        hold_days=config.hold_days,
        vix_data=prices.vix,
        vix_config=config.vix,
        base_composite_scores=base_scores,
        velocity_config=config.velocity,
        correlation_config=config.correlation,
        graduated_config=config.graduated_vix,
        # Exits are an intra-hold rotation rule, so they play no part in a
        # single-date selection.  Passing None also skips building the daily
        # rank panel, which is pure cost here.
        exit_config=None,
        close=prices.close,
        rank_offset=config.rank_offset,
        rank_offset_scope=config.rank_offset_scope,
        monitor_symbols=config.monitor_symbols,
    )
