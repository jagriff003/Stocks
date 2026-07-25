"""
Track B — per-stock rank exits, decoupled from the buy clock.

The strategy buys and sells on the same 14-day timer, which is a historical
accident rather than a design choice.  A stock's signal decays on its own
schedule: 68% of holdings drop out of the top 4 before their hold ends, at a
median of day 3 of ~10, and only 17% recover.  That leaves roughly 6.5 trading
days per instance holding something the model has already stopped liking.

What this module does NOT do is de-risk.  Track A established that reducing
market exposure forfeits the overnight premium — 63% of the return stream to
avoid 36% of the variance — so an exit that moves to cash or bonds inherits that
penalty.  A rank exit rotates into a better-ranked name and stays invested.
Keeping those two ideas separate is the whole point of decoupling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import ExitConfig


def daily_ranks(composite_scores: pd.DataFrame,
                history_counts: pd.DataFrame,
                min_data_days: int,
                base_scores: Optional[pd.DataFrame] = None,
                min_level: Optional[float] = None) -> pd.DataFrame:
    """
    Rank of every stock on every date, 1 = best.

    Eligibility is applied before ranking, using the same rules selection uses,
    so "rank 6" here means the same thing it would at a rebalance.  Ranking the
    raw score frame instead would let stocks without enough history occupy
    ranks, and a holding would appear to decay simply because ineligible names
    drifted above it.
    """
    eligible = composite_scores.where(history_counts >= min_data_days)

    if base_scores is not None and min_level is not None:
        eligible = eligible.where(base_scores >= min_level)

    return eligible.rank(axis=1, ascending=False, method="min")


@dataclass
class ExitState:
    """Running state for the daily exit check."""
    breach_streak: Dict[str, int] = field(default_factory=dict)
    entry_date: Dict[str, pd.Timestamp] = field(default_factory=dict)
    exits_this_month: int = 0
    current_month: Optional[tuple] = None

    def note_entries(self, symbols: Sequence[str], date) -> None:
        for sym in symbols:
            self.entry_date.setdefault(sym, date)

    def forget(self, symbols: Sequence[str]) -> None:
        for sym in symbols:
            self.breach_streak.pop(sym, None)
            self.entry_date.pop(sym, None)

    def roll_month(self, date) -> None:
        key = (date.year, date.month)
        if key != self.current_month:
            self.current_month = key
            self.exits_this_month = 0


def evaluate_exits(holdings: Sequence[str],
                   ranks_today: pd.Series,
                   date,
                   config: ExitConfig,
                   state: ExitState,
                   protected: Sequence[str] = ()) -> List[str]:
    """
    Which holdings should be exited today.

    A holding is exited when its rank has been at or beyond
    `exit_rank_threshold` for `consecutive_days` consecutive sessions.  Both
    conditions matter: the rank buffer stops a stock that slipped from 4th to
    5th from being sold into what is usually noise, and the day count stops a
    single bad print from triggering.

    `protected` symbols (the defensive sleeve) are never rank-exited — they are
    held for a reason unrelated to momentum rank, so their rank is meaningless.
    """
    if not config.enabled:
        return []

    state.roll_month(pd.Timestamp(date))
    protected_set = set(protected)
    exits: List[str] = []

    for symbol in holdings:
        if symbol in protected_set:
            continue

        rank = ranks_today.get(symbol, np.nan)
        if pd.isna(rank):
            # No eligible score today — treat as a breach; a stock that has
            # dropped out of the ranking entirely has certainly stopped
            # qualifying, and leaving it in place would be a silent hold.
            rank = np.inf

        if rank >= config.exit_rank_threshold:
            state.breach_streak[symbol] = state.breach_streak.get(symbol, 0) + 1
        else:
            state.breach_streak[symbol] = 0
            continue

        if state.breach_streak[symbol] < config.consecutive_days:
            continue

        if config.min_hold_days > 0:
            entered = state.entry_date.get(symbol)
            if entered is not None and (pd.Timestamp(date) - entered).days < config.min_hold_days:
                continue

        if (config.max_exits_per_month > 0
                and state.exits_this_month >= config.max_exits_per_month):
            continue

        exits.append(symbol)
        state.exits_this_month += 1
        state.breach_streak.pop(symbol, None)

    return exits


def evaluate_swaps(holdings: Sequence[str],
                   scores_today: pd.Series,
                   date,
                   config: ExitConfig,
                   state: ExitState,
                   protected: Sequence[str] = (),
                   available: Sequence[str] = ()) -> List[Tuple[str, str]]:
    """
    Swap out a holding only when a materially better buy exists.

    This triggers on the OPPORTUNITY rather than on the holding's decay, which
    is the economically coherent framing: a round trip costs real money, so the
    question is not "has this position slipped in a relative ranking" but "is
    something else enough better to be worth paying to switch".

    Rank-based exits fail precisely because they answer the first question.
    Rank is relative — a holding drops from 4th to 6th when others rise — so the
    trigger fires on information that says nothing about the holding's own
    prospects, and measured forward return after trigger is +0.39%.

    The open risk, which the caller should measure rather than assume: by the
    time a challenger's score exceeds the incumbent's by `min_score_gap`, the
    challenger may already have made its move. The model is known to be late to
    its own winners — 60% of new entrants had already spent 10+ of the prior 40
    sessions in the top 8 before being bought. A swap rule inherits that
    lateness. `swap_quality_analysis` measures whether it does.

    Returns a list of (outgoing, incoming) pairs.
    """
    if not config.enabled or config.mode != "score_gap":
        return []

    state.roll_month(pd.Timestamp(date))
    protected_set = set(protected)
    available_set = set(available) if available else None

    held = [s for s in holdings if s not in protected_set]
    if not held:
        return []

    candidates = scores_today.drop(labels=[s for s in holdings
                                           if s in scores_today.index],
                                   errors="ignore")
    if available_set is not None:
        candidates = candidates[candidates.index.isin(available_set)]
    candidates = candidates.dropna().sort_values(ascending=False)
    if candidates.empty:
        return []

    swaps: List[Tuple[str, str]] = []
    taken: set = set()

    # Weakest holding first — that is the one a challenger has to beat.
    held_scores = scores_today.reindex(held).astype(float)
    for symbol in held_scores.sort_values(na_position="first").index:
        incumbent = held_scores.get(symbol, np.nan)
        incumbent = -np.inf if pd.isna(incumbent) else float(incumbent)

        challenger = next((c for c in candidates.index if c not in taken), None)
        if challenger is None:
            break

        gap = float(candidates[challenger]) - incumbent
        if gap <= config.min_score_gap:
            state.breach_streak[symbol] = 0
            continue

        state.breach_streak[symbol] = state.breach_streak.get(symbol, 0) + 1
        if state.breach_streak[symbol] < config.consecutive_days:
            continue

        if config.min_hold_days > 0:
            entered = state.entry_date.get(symbol)
            if entered is not None and (pd.Timestamp(date) - entered).days < config.min_hold_days:
                continue

        if (config.max_exits_per_month > 0
                and state.exits_this_month >= config.max_exits_per_month):
            break

        swaps.append((symbol, challenger))
        taken.add(challenger)
        state.exits_this_month += 1
        state.breach_streak.pop(symbol, None)

    return swaps


def swap_quality_analysis(rebalance_history: List[Dict],
                          close: pd.DataFrame,
                          horizons: Sequence[int] = (5, 10, 21)) -> pd.DataFrame:
    """
    Did the incoming name actually beat the one it replaced?

    Tests the caveat that matters: a swap rule is only worth running if the
    signal arrives before the opportunity is gone. If challengers underperform
    the incumbents they displaced, the rule is systematically buying after the
    move — which is exactly the lateness Track C documents on the entry side.

    Returns one row per swap per horizon, with the forward return of both legs.
    """
    rows = []
    for record in rebalance_history:
        if record.get("Trigger") not in ("rank_exit", "score_swap"):
            continue

        date = pd.Timestamp(record["Date"])
        if date not in close.index:
            continue
        start = close.index.get_loc(date)

        outgoing = record.get("Exited", [])
        incoming = record.get("Added", [])

        for out_sym, in_sym in zip(outgoing, incoming):
            for horizon in horizons:
                end = start + horizon
                if end >= len(close.index):
                    continue
                end_date = close.index[end]

                def fwd(sym):
                    if sym not in close.columns:
                        return np.nan
                    p0, p1 = close.at[date, sym], close.at[end_date, sym]
                    if pd.isna(p0) or pd.isna(p1) or p0 == 0:
                        return np.nan
                    return p1 / p0 - 1

                out_ret, in_ret = fwd(out_sym), fwd(in_sym)
                if pd.isna(out_ret) or pd.isna(in_ret):
                    continue

                rows.append({
                    "Date": date,
                    "Horizon": horizon,
                    "Out": out_sym,
                    "In": in_sym,
                    "Out Return": out_ret,
                    "In Return": in_ret,
                    "Edge": in_ret - out_ret,
                })

    return pd.DataFrame(rows)


def choose_replacements(n: int,
                        ranked_today: pd.Series,
                        exclude: Sequence[str],
                        config: ExitConfig,
                        defensive: Sequence[str],
                        available: Sequence[str]) -> List[str]:
    """
    What fills the vacated slots.

    Deliberately a separate decision from the exit itself — the spec is right
    that "this position has decayed" and "here is what to buy instead" are
    independent questions, and bundling them hides which one is adding value.
    """
    if n <= 0:
        return []

    excluded = set(exclude)

    if config.replacement == "defer":
        return []

    if config.replacement == "defensive":
        return [s for s in defensive if s in available and s not in excluded][:n]

    if config.replacement != "immediate":
        raise ValueError(f"Unknown replacement mode: {config.replacement!r}")

    picks = []
    for symbol in ranked_today.index:
        if len(picks) >= n:
            break
        if symbol in excluded:
            continue
        picks.append(symbol)
    return picks


# --------------------------------------------------------------------------
# Diagnostics — rebuilding the rank-decay evidence
# --------------------------------------------------------------------------

def rank_decay_analysis(rebalance_history: List[Dict],
                        ranks: pd.DataFrame,
                        close: pd.DataFrame,
                        top_n: int = 4,
                        hold_days: int = 14,
                        exit_rank_threshold: int = 6,
                        consecutive_days: int = 2) -> Dict[str, object]:
    """
    Measure what holding a decayed position actually costs.

    Prior analysis established the FREQUENCY of rank dropout but not its return
    impact, because it had rank data and no prices.  This computes both, per
    holding instance:

      - when the position first dropped below the exit trigger
      - the return actually earned from that point to the end of the hold
      - whether the rank recovered before the hold ended

    The return from trigger to hold-end is the quantity that matters.  If it is
    reliably negative, exiting early is worth its transaction cost; if it is
    noisy around zero, the frequency statistics were describing a real pattern
    with no money in it.
    """
    instances = []

    for i, rebalance in enumerate(rebalance_history):
        start = pd.Timestamp(rebalance["Date"])
        end = (pd.Timestamp(rebalance_history[i + 1]["Date"])
               if i + 1 < len(rebalance_history) else close.index[-1])

        window = ranks.loc[start:end]
        if window.empty:
            continue

        for symbol in rebalance["Selected_Stocks"]:
            if symbol not in ranks.columns or symbol not in close.columns:
                continue

            series = window[symbol]
            if series.empty:
                continue

            breach = (series >= exit_rank_threshold).astype(int)
            # First date where the breach has persisted long enough to trigger.
            streak = breach.groupby((breach != breach.shift()).cumsum()).cumsum()
            triggered = streak[streak >= consecutive_days]
            trigger_date = triggered.index[0] if len(triggered) else None

            hold_start_price = close.at[series.index[0], symbol]
            hold_end_price = close.at[series.index[-1], symbol]
            full_return = (hold_end_price / hold_start_price - 1
                           if pd.notna(hold_start_price) and pd.notna(hold_end_price)
                           else np.nan)

            record = {
                "Rebalance": start,
                "Symbol": symbol,
                "Hold Days": len(series),
                "Entry Rank": series.iloc[0],
                "Min Rank": series.min(),
                "Max Rank": series.max(),
                "Dropped Out": bool((series > top_n).any()),
                "First Dropout Day": (int(np.argmax((series > top_n).to_numpy())) + 1
                                      if (series > top_n).any() else np.nan),
                "Triggered": trigger_date is not None,
                "Trigger Date": trigger_date,
                "Full Hold Return": full_return,
            }

            if trigger_date is not None:
                trigger_price = close.at[trigger_date, symbol]
                if pd.notna(trigger_price) and pd.notna(hold_end_price):
                    record["Return After Trigger"] = hold_end_price / trigger_price - 1
                    record["Days After Trigger"] = int(
                        (series.index[-1] - trigger_date).days)
                record["Recovered"] = bool(
                    (series.loc[trigger_date:] <= top_n).any())

            instances.append(record)

    frame = pd.DataFrame(instances)
    if frame.empty:
        return {"instances": frame, "summary": {}}

    triggered = frame[frame["Triggered"]]
    after = triggered["Return After Trigger"].dropna()

    summary = {
        "n_instances": len(frame),
        "pct_dropped_out": float(frame["Dropped Out"].mean()),
        "median_dropout_day": float(frame["First Dropout Day"].median()),
        "n_triggered": len(triggered),
        "pct_triggered": float(frame["Triggered"].mean()),
        "pct_recovered": float(triggered["Recovered"].mean())
                         if "Recovered" in triggered else np.nan,
        "mean_return_after_trigger": float(after.mean()) if len(after) else np.nan,
        "median_return_after_trigger": float(after.median()) if len(after) else np.nan,
        "pct_negative_after_trigger": float((after < 0).mean()) if len(after) else np.nan,
        "mean_days_after_trigger": float(
            triggered["Days After Trigger"].dropna().mean())
            if "Days After Trigger" in triggered else np.nan,
    }

    return {"instances": frame, "summary": summary}
