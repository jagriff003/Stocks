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
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import (CorrelationConfig, ExecutionConfig, ExitConfig,
                     GraduatedVixConfig, VelocityConfig, VixRegimeConfig)
from .correlation import RollingCorrelation, select_diversified
from .exits import (ExitState, choose_replacements, daily_ranks, evaluate_exits,
                    evaluate_swaps)
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

def _apply_rank_offset(ranked: pd.Series, offset: int, n: int) -> pd.Series:
    """
    Drop the top `offset` names before picking `n`.

    The offset is clamped so it can never leave the book short: if skipping the
    leaders would mean fewer than `n` candidates remain, the offset shrinks to
    whatever still fits, down to zero.  That matters because the alternative —
    holding three names because the fourth got skipped — would confound the
    test with a position-sizing change, and any comparison against the baseline
    would then be measuring two things at once.
    """
    if offset <= 0 or n <= 0:
        return ranked
    usable = max(0, min(offset, len(ranked) - n))
    return ranked.iloc[usable:]


class _PortfolioBuilder:
    """
    Selection state for one configuration, shared by the backtest loop and the
    live "what would it pick today" call.

    This exists so there is exactly one implementation of "rank, filter, pick".
    `build_target_portfolios` walks it over history; `select_on` runs it for a
    single date.  A second copy of the selection rules would drift from this
    one, and the live signal disagreeing with its own backtest is the specific
    failure this package is organized to prevent.
    """

    def __init__(self, composite_scores: pd.DataFrame,
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
                 exit_config: Optional[ExitConfig] = None,
                 close: Optional[pd.DataFrame] = None,
                 rank_offset: int = 0,
                 rank_offset_scope: str = "all",
                 monitor_symbols: Sequence[str] = ()):
        self.scores = composite_scores
        # Scored for comparison, never selectable.  Excluded in `eligible`
        # rather than by masking the score panel, so the name still takes part
        # in cross-sectional normalization and still shows up in the ranking a
        # caller inspects.
        self.monitor_symbols = set(monitor_symbols)
        self.price_columns = price_columns
        self.top_n = top_n
        self.min_data_days = min_data_days
        self.hold_days = hold_days
        self.vix_config = vix_config
        self.base_scores = base_composite_scores
        self.velocity_config = velocity_config
        self.correlation_config = correlation_config
        self.graduated_config = graduated_config
        self.exit_config = exit_config
        self.rank_offset = int(rank_offset)

        if rank_offset_scope not in ("all", "normal"):
            raise ValueError(f"Unknown rank_offset_scope: {rank_offset_scope!r}")
        self.rank_offset_scope = rank_offset_scope

        self.rolling_corr = None
        self.vix_levels = None
        if correlation_config is not None and correlation_config.enabled:
            if close is None:
                raise ValueError("correlation filtering requires `close` prices")
            self.rolling_corr = RollingCorrelation(close, correlation_config.window)
            if correlation_config.apply_above_vix is not None:
                if vix_data is None:
                    raise ValueError(
                        "correlation_config.apply_above_vix requires vix_data; "
                        "set it to None to filter in all regimes"
                    )
                self.vix_levels = pd.Series(vix_data).reindex(
                    composite_scores.index).ffill()

        self.vix_regime = None
        if vix_data is not None and vix_config is not None:
            regime_series, _ = calculate_vix_regime(vix_data, vix_config)
            self.vix_regime = regime_series.reindex(composite_scores.index).ffill()

        # --- Track A: graduated absolute-level ladder ---
        self.ladder = None
        self.use_ladder = graduated_config is not None and graduated_config.enabled
        if self.use_ladder:
            if vix_data is None:
                raise ValueError("graduated VIX ladder requires vix_data")
            ladder = graduated_regime(vix_data, graduated_config)
            ladder = ladder.reindex(composite_scores.index).ffill().bfill()
            ladder["momentum_slots"] = ladder["momentum_slots"].astype(int)
            self.ladder = ladder

        # Cumulative count of non-null scores per stock, so the "enough history"
        # filter is a lookup instead of a re-scan of the whole column per rebalance.
        self.history_counts = composite_scores.notna().cumsum()

        # --- Track B: daily per-stock rank monitoring ---
        self.ranks = None
        self.use_exits = exit_config is not None and exit_config.enabled
        if self.use_exits:
            self.ranks = daily_ranks(
                composite_scores, self.history_counts, min_data_days,
                base_scores=base_composite_scores,
                min_level=(velocity_config.min_level_threshold
                           if velocity_config is not None else None),
            )

    # -- selection primitives -------------------------------------------

    def correlation_open(self, date) -> bool:
        """Is the correlation filter armed on this date?"""
        if self.rolling_corr is None:
            return False
        if self.correlation_config.apply_above_vix is None:
            return True
        level = self.vix_levels.get(date) if self.vix_levels is not None else None
        return (level is not None and pd.notna(level)
                and level >= self.correlation_config.apply_above_vix)

    def offset_for(self, regime: str) -> int:
        """
        How far down the ranking to start on this date.

        With scope='normal' the offset stands down whenever the regime overlay
        has already cut momentum exposure: those books are half defensive
        already, and skipping the leaders on top of that is a different bet from
        the one being tested.
        """
        if self.rank_offset <= 0:
            return 0
        if self.rank_offset_scope == "normal" and regime != NORMAL:
            return 0
        return self.rank_offset

    def pick_momentum(self, ranked: pd.Series, date, n: int, offset: int = 0
                      ) -> Tuple[List[str], Optional[object]]:
        """Top `n` after skipping `offset`, correlation-filtered when armed."""
        if n <= 0:
            return [], None
        pool = _apply_rank_offset(ranked, offset, n)
        if not self.correlation_open(date):
            return pool.head(n).index.tolist(), None

        exempt_syms = ()
        cfg = self.correlation_config
        if cfg is not None and not cfg.apply_to_defensive:
            if self.graduated_config is not None:
                exempt_syms = self.graduated_config.defensive_symbols
            elif self.vix_config is not None:
                exempt_syms = self.vix_config.defensive_symbols

        trace = select_diversified(pool, self.rolling_corr.at(date),
                                   cfg, n, exempt=exempt_syms)
        return trace.selected, trace

    def build_from_ladder(self, ranked: pd.Series, date, n_slots: int,
                          offset: int = 0) -> List[str]:
        """Momentum picks for the current band, defensive fill for the rest."""
        n_slots = max(0, min(n_slots, self.top_n, len(ranked)))
        picks, _ = self.pick_momentum(ranked, date, n_slots, offset)
        fill = [s for s in self.graduated_config.defensive_symbols
                if s in self.price_columns and s not in picks]
        return picks + fill[:max(0, self.top_n - len(picks))]

    def eligible(self, date) -> Tuple[pd.Series, List[str]]:
        """Scores on `date`, and the subset with enough history and level."""
        valid_scores = self.scores.loc[date].dropna()
        counts = self.history_counts.loc[date]

        valid_stocks = []
        for stock in valid_scores.index:
            if stock in self.monitor_symbols:
                continue
            if counts.get(stock, 0) < self.min_data_days:
                continue

            # Hard floor on the LEVEL score: a stock in complete freefall is
            # skipped no matter how strong its velocity bounce looks.
            if (self.velocity_config is not None
                    and self.base_scores is not None
                    and stock in self.base_scores.columns):
                level = self.base_scores.loc[date, stock]
                if pd.notna(level) and level < self.velocity_config.min_level_threshold:
                    continue

            valid_stocks.append(stock)

        return valid_scores, valid_stocks

    def select_on(self, date) -> Optional[Dict]:
        """
        The full rebalance decision for one date.

        Returns a rebalance-history record, or None if the universe cannot fill
        the book on that date.  `_ranked` rides along for the caller's daily
        re-evaluation; it is stripped before the record is published.
        """
        valid_scores, valid_stocks = self.eligible(date)
        if len(valid_stocks) < self.top_n:
            return None

        ranked = valid_scores[valid_stocks].sort_values(ascending=False)
        trace = None
        slots = None

        if self.use_ladder:
            # Ranking is recomputed on the hold clock; how much of it gets used
            # is decided daily, by the caller.
            slots = int(self.ladder.at[date, "momentum_slots"])
            regime = self.graduated_config.band_labels[
                int(self.ladder.at[date, "band"])]
            new_portfolio = self.build_from_ladder(
                ranked, date, slots, self.offset_for(regime))
        else:
            regime = NORMAL
            if self.vix_regime is not None:
                r = self.vix_regime.get(date)
                if r is not None and not pd.isna(r):
                    regime = r

            offset = self.offset_for(regime)

            if regime == CRISIS and self.vix_config is not None:
                available = [s for s in self.vix_config.crisis_symbols
                             if s in self.price_columns]
                new_portfolio = available or self.pick_momentum(
                    ranked, date, self.top_n, offset)[0]

            elif regime == ELEVATED and self.vix_config is not None:
                # Clamped to top_n as well: without it a book smaller than
                # elevated_top_n would GROW in an elevated regime (top_n=1 with
                # elevated_top_n=2 held two names), which is the opposite of
                # what the overlay is for.  A no-op at the live 2-of-4 setting;
                # it only bites when sweeping book size below elevated_top_n.
                n_momentum = min(self.vix_config.elevated_top_n, self.top_n,
                                 len(valid_stocks))
                momentum_picks, trace = self.pick_momentum(
                    ranked, date, n_momentum, offset)
                fill = [s for s in self.vix_config.defensive_symbols
                        if s in self.price_columns and s not in momentum_picks]
                new_portfolio = (momentum_picks
                                 + fill[:max(0, self.top_n - n_momentum)])

            else:
                new_portfolio, trace = self.pick_momentum(
                    ranked, date, self.top_n, offset)

        # Monitors are excluded from `ranked` so they can never be selected by
        # any path, but the whole point of carrying one is to SEE it.  Report
        # its score and the rank it would have held, without ever letting it
        # into the candidate list.
        monitor_view = {}
        for sym in self.monitor_symbols:
            score = valid_scores.get(sym)
            if score is not None and pd.notna(score):
                would_be = int((ranked > score).sum()) + 1
                monitor_view[sym] = {"score": float(score),
                                     "would_rank": would_be,
                                     "of": len(ranked) + 1}

        record = {
            "Date": date,
            "Selected_Stocks": new_portfolio,
            "Scores": valid_scores.reindex(new_portfolio).to_dict(),
            "Regime": regime,
            "Trigger": "rebalance",
            "Monitors": monitor_view,
            "_ranked": ranked,
            "_slots": slots,
        }
        if trace is not None:
            record["Corr_Threshold"] = trace.threshold
            record["Corr_Rejected"] = [
                f"{sym}~{peer}:{rho:.2f}" for sym, peer, rho in trace.rejected
            ]
            record["Corr_Relaxed"] = trace.relaxed
        return record

    # -- the walk over history ------------------------------------------

    def run(self, verbose: bool = False):
        """Walk every date, applying the hold clock and the daily overlays."""
        composite_scores = self.scores
        top_n = self.top_n
        dates = composite_scores.index[self.min_data_days:]

        targets: Dict[pd.Timestamp, List[str]] = {}
        rebalance_history: List[Dict] = []

        exit_state = ExitState()
        current_portfolio: List[str] = []
        current_ranked: Optional[pd.Series] = None
        last_rebalance_date: Optional[pd.Timestamp] = None
        last_slots: Optional[int] = None

        for date in dates:
            due = (last_rebalance_date is None
                   or (date - last_rebalance_date).days >= self.hold_days)

            if due:
                record = self.select_on(date)
                if record is not None:
                    ranked = record.pop("_ranked")
                    slots = record.pop("_slots")
                    new_portfolio = record["Selected_Stocks"]
                    rebalance_history.append(record)

                    if self.use_exits:
                        exit_state.forget([s for s in current_portfolio
                                           if s not in new_portfolio])
                        exit_state.note_entries(new_portfolio, date)

                    current_portfolio = new_portfolio
                    current_ranked = ranked
                    last_rebalance_date = date
                    if slots is not None:
                        last_slots = slots

                    if verbose:
                        print(f"Rebalanced on {date:%Y-%m-%d} [{record['Regime']}]: "
                              f"{', '.join(new_portfolio)}")

            # --- Track A: daily exposure decision, off the rebalance clock ---
            #
            # The old design could only change exposure at a scheduled rebalance,
            # which is why only 25% of crisis-flagged days were actually holding
            # defensive positions — the rest were riding pre-crisis picks waiting
            # for the clock. Re-deciding daily is the fix. Selection still moves on
            # the hold clock; only the risk dial moves daily.
            if (self.use_ladder and self.graduated_config.evaluate_daily
                    and current_ranked is not None and not due):
                slots = int(self.ladder.at[date, "momentum_slots"])
                if last_slots is None or slots != last_slots:
                    band_label = self.graduated_config.band_labels[
                        int(self.ladder.at[date, "band"])]
                    new_portfolio = self.build_from_ladder(
                        current_ranked, date, slots, self.offset_for(band_label))
                    if new_portfolio != current_portfolio:
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

            # --- Track B: per-stock rank exits, off the buy clock ---
            #
            # 68% of holdings drop out of the top 4 before their hold ends, at a
            # median of day 3 of ~10, and only 17% recover. This exits on the
            # position's own decay rather than waiting for the shared timer.
            #
            # Note this does NOT reduce market exposure — it rotates into a
            # better-ranked name. Track A showed that de-risking forfeits the
            # overnight premium, which is why 'immediate' is the default
            # replacement and the defensive variant is expected to lose.
            if self.use_exits and current_portfolio and not due:
                defensive = (self.graduated_config.defensive_symbols
                             if self.graduated_config
                             else (self.vix_config.defensive_symbols
                                   if self.vix_config else []))
                candidates = (composite_scores.loc[date]
                              .where(self.history_counts.loc[date] >= self.min_data_days)
                              .dropna()
                              .sort_values(ascending=False))

                leaving: List[str] = []
                replacements: List[str] = []
                trigger = None

                if self.exit_config.mode == "score_gap":
                    # Trigger on the opportunity, not the decay: swap only when a
                    # challenger beats the incumbent by more than the cost hurdle.
                    swaps = evaluate_swaps(current_portfolio, candidates, date,
                                           self.exit_config, exit_state,
                                           protected=defensive,
                                           available=self.price_columns)
                    if swaps:
                        leaving = [out for out, _ in swaps]
                        replacements = [inc for _, inc in swaps]
                        trigger = "score_swap"
                else:
                    leaving = evaluate_exits(current_portfolio, self.ranks.loc[date],
                                             date, self.exit_config, exit_state,
                                             protected=defensive)
                    if leaving:
                        replacements = choose_replacements(
                            len(leaving), candidates,
                            exclude=set(current_portfolio),
                            config=self.exit_config, defensive=defensive,
                            available=self.price_columns,
                        )
                        trigger = "rank_exit"

                if leaving:
                    keep = [s for s in current_portfolio if s not in leaving]
                    new_portfolio = keep + replacements
                    if new_portfolio != current_portfolio:
                        rebalance_history.append({
                            "Date": date,
                            "Selected_Stocks": new_portfolio,
                            "Scores": candidates.reindex(new_portfolio).to_dict(),
                            "Regime": trigger,
                            "Trigger": trigger,
                            "Exited": leaving,
                            "Added": replacements,
                        })
                        exit_state.forget(leaving)
                        exit_state.note_entries(replacements, date)
                        current_portfolio = new_portfolio

                        if verbose:
                            print(f"{trigger} {date:%Y-%m-%d}: out {', '.join(leaving)}"
                                  f" -> in {', '.join(replacements) or '(none)'}")

            targets[date] = list(current_portfolio)

        return pd.Series(targets, name="target"), rebalance_history


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
                            exit_config: Optional[ExitConfig] = None,
                            close: Optional[pd.DataFrame] = None,
                            rank_offset: int = 0,
                            rank_offset_scope: str = "all",
                            monitor_symbols: Sequence[str] = (),
                            verbose: bool = False):
    """
    Decide what the model wants to hold on each date.

    Selection is relative: the top `top_n` by composite score out of whatever is
    currently available.  There is no absolute score bar, deliberately — the
    strategy leans into the best currently-available opportunity rather than
    defaulting to cash on an arbitrary threshold.  Going defensive is the
    regime overlay's job, not the ranker's.

    `rank_offset` starts that count further down the ranking: 0 takes ranks
    1-4, 1 takes 2-5, and so on.  It is clamped so the book is always full —
    see `_apply_rank_offset`.  `rank_offset_scope` decides whether it also
    applies when the regime overlay has already cut momentum exposure ('all')
    or only in the normal band ('normal').

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
    builder = _PortfolioBuilder(
        composite_scores, price_columns, top_n=top_n,
        min_data_days=min_data_days, hold_days=hold_days,
        vix_data=vix_data, vix_config=vix_config,
        base_composite_scores=base_composite_scores,
        velocity_config=velocity_config,
        correlation_config=correlation_config,
        graduated_config=graduated_config,
        exit_config=exit_config, close=close,
        rank_offset=rank_offset, rank_offset_scope=rank_offset_scope,
        monitor_symbols=monitor_symbols,
    )
    return builder.run(verbose=verbose)


def selection_for_date(composite_scores: pd.DataFrame,
                       price_columns: List[str],
                       date=None,
                       **builder_kwargs
                       ) -> Optional[Tuple[Dict, pd.Series]]:
    """
    What the model would select on one date, ignoring the hold clock.

    This is the off-cycle question: the book is on a 14-day timer, so on any
    given day the held names and the currently best-ranked names are two
    different lists, and only one of them is in `rebalance_history`.  Running
    the same `_PortfolioBuilder` for a single date answers it without a second
    implementation of the selection rules.

    `date` defaults to the last scored date; a date that is not itself a
    scoring date resolves back to the most recent one on or before it.

    Returns
    -------
    (record, ranked) : the rebalance-style record, and the full eligible
        ranking behind it, so a caller can show what the offset skipped.
        None when the universe could not fill the book on that date.
    """
    builder = _PortfolioBuilder(composite_scores, price_columns, **builder_kwargs)

    index = composite_scores.index
    if date is None:
        when = index[-1]
    else:
        eligible_dates = index[index <= pd.Timestamp(date)]
        if len(eligible_dates) == 0:
            raise ValueError(f"No scored dates on or before {date}")
        when = eligible_dates[-1]

    record = builder.select_on(when)
    if record is None:
        return None

    ranked = record.pop("_ranked")
    record.pop("_slots")
    record["Trigger"] = "as_of"
    return record, ranked


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
