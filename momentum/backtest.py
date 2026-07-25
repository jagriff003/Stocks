"""
Portfolio construction and simulation.

Two responsibilities, deliberately separated:

  `build_target_portfolios`  — what the model *wants* to hold, per day.
  `simulate_portfolio`       — what that costs once you actually have to trade it.

Splitting them is what makes Tracks A-C tractable.  Track A changes the regime
input to selection, Track B changes when a position leaves the target, Track C
changes when one enters — and none of them need to touch the execution and cost
machinery, so their results stay comparable to each other and to the baseline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import (CorrelationConfig, ExecutionConfig, GraduatedVixConfig,
                     VelocityConfig, VixRegimeConfig)
from .correlation import RollingCorrelation, select_diversified
from .metrics import (calculate_performance_metrics, calculate_turnover_metrics)
from .regime import (CRISIS, ELEVATED, NORMAL, calculate_vix_regime,
                     graduated_regime)


@dataclass
class PortfolioResult:
    """Everything one simulation produced."""
    returns: pd.Series                      # daily returns, net of costs
    gross_returns: pd.Series                # before slippage
    holdings: pd.Series                     # date -> list of held symbols
    holdings_history: List[Dict] = field(repr=False, default_factory=list)
    rebalance_history: List[Dict] = field(repr=False, default_factory=list)
    total_cost: float = 0.0
    metrics: Dict[str, float] = field(default_factory=dict)
    turnover: Dict[str, float] = field(default_factory=dict)

    @property
    def summary(self) -> Dict[str, float]:
        return {**self.metrics, **self.turnover, "total_cost": self.total_cost}


# --------------------------------------------------------------------------
# Target portfolio construction
# --------------------------------------------------------------------------

def build_target_portfolios(composite_scores: pd.DataFrame,
                            price_columns: List[str],
                            top_n: int = 4,
                            min_data_days: int = 200,
                            hold_days: int = 14,
                            vix_data: Optional[pd.Series] = None,
                            vix_config: Optional[VixRegimeConfig] = None,
                            base_composite_scores: Optional[pd.DataFrame] = None,
                            velocity_config: Optional[VelocityConfig] = None,
                            correlation_config: Optional[CorrelationConfig] = None,
                            graduated_config: Optional[GraduatedVixConfig] = None,
                            close: Optional[pd.DataFrame] = None,
                            verbose: bool = False):
    """
    Decide what the model wants to hold on each date.

    Selection is relative: the top `top_n` by composite score out of whatever is
    currently available.  There is no absolute score bar, deliberately — the
    strategy leans into the best currently-available opportunity rather than
    defaulting to cash on an arbitrary threshold.  Going defensive is the
    regime overlay's job, not the ranker's.

    When `correlation_config.enabled`, selection walks down the ranking taking
    the best candidate that is not already duplicated by something held.  The
    top-ranked name is always taken; the filter only ever redirects later picks
    to the next best alternative.  `close` is required in that case, to estimate
    trailing correlations.

    Rebalances every `hold_days` *calendar* days (so ~10 trading days for the
    14-day setting), which is the pre-existing convention.

    Returns
    -------
    targets : Series indexed by date, values are lists of symbols
    rebalance_history : list of dicts, one per rebalance
    """
    dates = composite_scores.index[min_data_days:]

    rolling_corr = None
    vix_levels = None
    if correlation_config is not None and correlation_config.enabled:
        if close is None:
            raise ValueError("correlation filtering requires `close` prices")
        rolling_corr = RollingCorrelation(close, correlation_config.window)
        if correlation_config.apply_above_vix is not None:
            if vix_data is None:
                raise ValueError(
                    "correlation_config.apply_above_vix requires vix_data; "
                    "set it to None to filter in all regimes"
                )
            vix_levels = pd.Series(vix_data).reindex(composite_scores.index).ffill()

    vix_regime = None
    if vix_data is not None and vix_config is not None:
        regime_series, _ = calculate_vix_regime(vix_data, vix_config)
        vix_regime = regime_series.reindex(composite_scores.index).ffill()

    # --- Track A: graduated absolute-level ladder ---
    ladder = None
    use_ladder = graduated_config is not None and graduated_config.enabled
    if use_ladder:
        if vix_data is None:
            raise ValueError("graduated VIX ladder requires vix_data")
        ladder = graduated_regime(vix_data, graduated_config)
        ladder = ladder.reindex(composite_scores.index).ffill().bfill()
        ladder["momentum_slots"] = ladder["momentum_slots"].astype(int)

    # Cumulative count of non-null scores per stock, so the "enough history"
    # filter is a lookup instead of a re-scan of the whole column per rebalance.
    history_counts = composite_scores.notna().cumsum()

    targets: Dict[pd.Timestamp, List[str]] = {}
    rebalance_history: List[Dict] = []

    current_portfolio: List[str] = []
    current_ranked: Optional[pd.Series] = None
    last_rebalance_date: Optional[pd.Timestamp] = None
    last_slots: Optional[int] = None

    def correlation_open(date) -> bool:
        """Is the correlation filter armed on this date?"""
        if rolling_corr is None:
            return False
        if correlation_config.apply_above_vix is None:
            return True
        level = vix_levels.get(date) if vix_levels is not None else None
        return (level is not None and pd.notna(level)
                and level >= correlation_config.apply_above_vix)

    def pick_momentum(ranked: pd.Series, date, n: int
                      ) -> Tuple[List[str], Optional[object]]:
        """Top `n` by rank, correlation-filtered when the gate is open."""
        if n <= 0:
            return [], None
        if not correlation_open(date):
            return ranked.head(n).index.tolist(), None

        exempt_syms = ()
        if correlation_config is not None and not correlation_config.apply_to_defensive:
            if graduated_config is not None:
                exempt_syms = graduated_config.defensive_symbols
            elif vix_config is not None:
                exempt_syms = vix_config.defensive_symbols

        trace = select_diversified(ranked, rolling_corr.at(date),
                                   correlation_config, n, exempt=exempt_syms)
        return trace.selected, trace

    def build_from_ladder(ranked: pd.Series, date, n_slots: int) -> List[str]:
        """Momentum picks for the current band, defensive fill for the rest."""
        n_slots = max(0, min(n_slots, top_n, len(ranked)))
        picks, _ = pick_momentum(ranked, date, n_slots)
        fill = [s for s in graduated_config.defensive_symbols
                if s in price_columns and s not in picks]
        return picks + fill[:max(0, top_n - len(picks))]

    for date in dates:
        due = (last_rebalance_date is None
               or (date - last_rebalance_date).days >= hold_days)

        if due:
            valid_scores = composite_scores.loc[date].dropna()
            counts = history_counts.loc[date]

            valid_stocks = []
            for stock in valid_scores.index:
                if counts.get(stock, 0) < min_data_days:
                    continue

                # Hard floor on the LEVEL score: a stock in complete freefall is
                # skipped no matter how strong its velocity bounce looks.
                if (velocity_config is not None
                        and base_composite_scores is not None
                        and stock in base_composite_scores.columns):
                    level = base_composite_scores.loc[date, stock]
                    if pd.notna(level) and level < velocity_config.min_level_threshold:
                        continue

                valid_stocks.append(stock)

            if len(valid_stocks) >= top_n:
                ranked = valid_scores[valid_stocks].sort_values(ascending=False)
                trace = None

                if use_ladder:
                    # Ranking is recomputed on the hold clock; how much of it
                    # gets used is decided daily, below.
                    slots = int(ladder.at[date, "momentum_slots"])
                    new_portfolio = build_from_ladder(ranked, date, slots)
                    regime = graduated_config.band_labels[int(ladder.at[date, "band"])]
                    last_slots = slots
                else:
                    regime = NORMAL
                    if vix_regime is not None:
                        r = vix_regime.get(date)
                        if r is not None and not pd.isna(r):
                            regime = r

                    if regime == CRISIS and vix_config is not None:
                        available = [s for s in vix_config.crisis_symbols
                                     if s in price_columns]
                        new_portfolio = available or pick_momentum(ranked, date, top_n)[0]

                    elif regime == ELEVATED and vix_config is not None:
                        n_momentum = min(vix_config.elevated_top_n, len(valid_stocks))
                        momentum_picks, trace = pick_momentum(ranked, date, n_momentum)
                        fill = [s for s in vix_config.defensive_symbols
                                if s in price_columns and s not in momentum_picks]
                        new_portfolio = (momentum_picks
                                         + fill[:max(0, top_n - n_momentum)])

                    else:
                        new_portfolio, trace = pick_momentum(ranked, date, top_n)

                record = {
                    "Date": date,
                    "Selected_Stocks": new_portfolio,
                    "Scores": valid_scores.reindex(new_portfolio).to_dict(),
                    "Regime": regime,
                    "Trigger": "rebalance",
                }
                if trace is not None:
                    record["Corr_Threshold"] = trace.threshold
                    record["Corr_Rejected"] = [
                        f"{sym}~{peer}:{rho:.2f}" for sym, peer, rho in trace.rejected
                    ]
                    record["Corr_Relaxed"] = trace.relaxed
                rebalance_history.append(record)

                current_portfolio = new_portfolio
                current_ranked = ranked
                last_rebalance_date = date

                if verbose:
                    print(f"Rebalanced on {date:%Y-%m-%d} [{regime}]: "
                          f"{', '.join(new_portfolio)}")

        # --- Track A: daily exposure decision, off the rebalance clock ---
        #
        # The old design could only change exposure at a scheduled rebalance,
        # which is why only 25% of crisis-flagged days were actually holding
        # defensive positions — the rest were riding pre-crisis picks waiting
        # for the clock. Re-deciding daily is the fix. Selection still moves on
        # the hold clock; only the risk dial moves daily.
        if (use_ladder and graduated_config.evaluate_daily
                and current_ranked is not None and not due):
            slots = int(ladder.at[date, "momentum_slots"])
            if last_slots is None or slots != last_slots:
                new_portfolio = build_from_ladder(current_ranked, date, slots)
                if new_portfolio != current_portfolio:
                    band_label = graduated_config.band_labels[
                        int(ladder.at[date, "band"])]
                    rebalance_history.append({
                        "Date": date,
                        "Selected_Stocks": new_portfolio,
                        "Scores": current_ranked.reindex(new_portfolio).to_dict(),
                        "Regime": band_label,
                        "Trigger": "regime_shift",
                    })
                    current_portfolio = new_portfolio
                    if verbose:
                        print(f"Regime shift {date:%Y-%m-%d} [{band_label}]: "
                              f"{slots}/{top_n} momentum — "
                              f"{', '.join(new_portfolio)}")
                last_slots = slots

        targets[date] = list(current_portfolio)

    return pd.Series(targets, name="target"), rebalance_history


# --------------------------------------------------------------------------
# Execution
# --------------------------------------------------------------------------

def _segment_return(symbols: List[str],
                    start_prices: pd.Series,
                    end_prices: pd.Series) -> float:
    """
    Equal-weighted return of `symbols` between two price vectors.

    Note: weights are reset to equal on every segment, which is how the
    pre-refactor model computed returns.  That implicitly assumes a costless
    daily rebalance back to equal weight.  It slightly understates the return
    contribution of a runaway winner inside a hold period.  Changing it would
    break comparability with every historical result, so it stays — but it is a
    known approximation, not an intended feature.
    """
    rets = []
    for sym in symbols:
        p0 = start_prices.get(sym, np.nan)
        p1 = end_prices.get(sym, np.nan)
        if pd.notna(p0) and pd.notna(p1) and p0 != 0:
            rets.append(p1 / p0 - 1)
    return float(np.mean(rets)) if rets else 0.0


def _trade_cost(old: List[str], new: List[str], execution: ExecutionConfig) -> float:
    """
    Slippage cost of moving from `old` to `new`, as a fraction of portfolio value.

    The fraction of the book that changes hands is paid twice — once selling the
    outgoing names, once buying the incoming ones.
    """
    if not new:
        return 0.0
    if not old:
        # Initial build: buy side only, on the full book.
        return execution.slippage_frac

    changed = len(set(old) - set(new)) / len(old)
    return 2.0 * changed * execution.slippage_frac


def simulate_portfolio(targets: pd.Series,
                       close: pd.DataFrame,
                       open_: Optional[pd.DataFrame] = None,
                       execution: Optional[ExecutionConfig] = None) -> PortfolioResult:
    """
    Turn a series of target portfolios into a realized return stream.

    The execution convention matters more than it looks.  The pre-refactor model
    ranked on close(T) and booked the return from close(T) to close(T+1) — it
    filled at the same close that produced the signal, which is not achievable.
    In elevated-VIX regimes the open-to-close swing runs several percent, so this
    is not a rounding error; it is a systematic overstatement of every result
    the strategy has ever produced.

    execution.execute_at
      'next_open'  signal from close(T), fill at open(T+1).  Realistic default.
                   The transition day is split: the old book is held from
                   close(T) to open(T+1), the new book from open(T+1) to
                   close(T+1).
      'next_close' fill at close(T+1) — a full extra day of lag.
      'same_close' fill at close(T).  Unachievable; retained only to quantify
                   how much of the historical CAGR came from assuming it.
    """
    execution = execution or ExecutionConfig()
    mode = execution.execute_at

    if mode not in ("next_open", "next_close", "same_close"):
        raise ValueError(f"Unknown execute_at: {mode!r}")
    if mode == "next_open" and open_ is None:
        raise ValueError("execute_at='next_open' requires open prices")

    dates = list(targets.index)
    held: List[str] = []
    total_cost = 0.0

    records: List[Dict] = []
    holdings_history: List[Dict] = []

    for i in range(1, len(dates)):
        d_prev, d = dates[i - 1], dates[i]
        if d_prev not in close.index or d not in close.index:
            continue

        signal = list(targets.loc[d_prev])
        close_prev, close_now = close.loc[d_prev], close.loc[d]
        cost = 0.0

        if mode == "same_close":
            if signal != held:
                cost = _trade_cost(held, signal, execution)
                held = signal
                if held:
                    holdings_history.append({"Date": d_prev, "Holdings": list(held)})
            if not held:
                continue
            gross = _segment_return(held, close_prev, close_now)

        elif mode == "next_open":
            if signal != held:
                if held:
                    r1 = _segment_return(held, close_prev, open_.loc[d])
                    r2 = _segment_return(signal, open_.loc[d], close_now)
                    gross = (1 + r1) * (1 + r2) - 1
                else:
                    # Building the first book: no overnight leg to carry.
                    gross = _segment_return(signal, open_.loc[d], close_now)
                cost = _trade_cost(held, signal, execution)
                held = signal
                holdings_history.append({"Date": d, "Holdings": list(held)})
            else:
                if not held:
                    continue
                gross = _segment_return(held, close_prev, close_now)

        else:  # next_close — the whole day is held on the old book
            if held:
                gross = _segment_return(held, close_prev, close_now)
            else:
                gross = None
            if signal != held:
                cost = _trade_cost(held, signal, execution)
                held = signal
                if held:
                    holdings_history.append({"Date": d, "Holdings": list(held)})
            if gross is None:
                continue

        total_cost += cost
        records.append({
            "Date": d,
            "Portfolio_Return": gross - cost,
            "Gross_Return": gross,
            "Cost": cost,
            "Holdings": list(held),
        })

    if not records:
        raise ValueError("Simulation produced no return observations")

    frame = pd.DataFrame(records).set_index("Date")
    net = frame["Portfolio_Return"]
    gross_series = frame["Gross_Return"]

    metrics = calculate_performance_metrics(net, risk_free_rate=execution.risk_free_rate)
    turnover = calculate_turnover_metrics(holdings_history, metrics["years"])

    return PortfolioResult(
        returns=net,
        gross_returns=gross_series,
        holdings=frame["Holdings"],
        holdings_history=holdings_history,
        total_cost=total_cost,
        metrics=metrics,
        turnover=turnover,
    )
