"""
Live model run — produces the current signal and the standard exports.

    python scripts/run_live.py
    python scripts/run_live.py --legacy      reproduce pre-refactor behaviour
    python scripts/run_live.py --no-plots    skip charts (for scheduled runs)

To change the ticker universe, edit `universe.csv` in the repo root.
To change model parameters, edit `build_config()` below.

Every run snapshots the universe and the config to `snapshots/`, dated.  That is
what makes a future live-vs-backtest reconciliation answerable from the record
instead of from memory.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from momentum.config import (ExecutionConfig, ModelConfig, ScoringConfig,
                             VelocityConfig, VixRegimeConfig, snapshot_config)
from momentum.data import export_price_data, load_data
from momentum.health import (HealthConfig, compute_health, current_state,
                             defensive_exposure, defensive_line,
                             defensive_state, status_line)
from momentum.experiments import legacy_config, production_config
from momentum.reports import (correlation_matrices, individual_stock_performance,
                              portfolio_concentration, portfolio_correlation,
                              summarize_correlations)
from momentum.strategy import compute_scores, current_selection, run_strategy
from momentum.restrictions import check_symbols
from momentum.restrictions import describe as describe_restrictions
from momentum.universe import (current_symbols, sector_map,
                               snapshot_current_universe)


def build_config(legacy: bool = False) -> ModelConfig:
    """
    The live model configuration.  Edit here.

    Current settings and where they came from:
      zscore_window=126     chosen by the z-score window comparison
      velocity 0.7/0.3      chosen by the velocity sweep — but see the note
                            below; that sweep ran against a scale mismatch and
                            is being re-run
      VIX z 1.5/2.5         chosen by the VIX regime comparison; superseded by
                            the graduated ladder once Track A lands
      execute_at next_open  realistic fill; see ExecutionConfig
    """
    if legacy:
        return legacy_config(notes="pre-refactor reproduction")

    return production_config(
        notes="post-refactor baseline: fixed blend scale, realistic execution"
    )


def _print_book(record, prices, indent="    "):
    """One selected book: names, scores, and the risk context that matters."""
    symbols = record["Selected_Stocks"]
    for i, (stock, score) in enumerate(record["Scores"].items(), 1):
        score_txt = f"{score:>7.3f}" if pd.notna(score) else "      -"
        print(f"{indent}{i}. {stock:<6} score {score_txt}")

    # Correlation first — it is the measure that reflects shared risk.  Sector
    # labels are a poor proxy in both directions and are reported only as
    # context.
    pairs = portfolio_correlation(symbols, prices.close, window=50)
    if not pairs.empty:
        worst = pairs.iloc[0]
        print(f"{indent}corr (50d): max {worst['Symbol A']}/{worst['Symbol B']} "
              f"{worst['Correlation']:.2f}, mean {pairs['Correlation'].mean():.2f}")
        if worst["Correlation"] >= 0.70:
            print(f"{indent}  ^ {worst['Symbol A']} and {worst['Symbol B']} are "
                  f"effectively one position.")

    mons = record.get("Monitors") or {}
    for sym, info in mons.items():
        note = ""
        if info["would_rank"] <= 4:
            note = "  <- would be IN the book if it were tradable"
        elif info["would_rank"] <= 10:
            note = "  <- ranking high; breadth is narrowing"
        print(f"{indent}monitor {sym}: score {info['score']:>6.3f}, "
              f"would rank {info['would_rank']} of {info['of']}{note}")

    conc = portfolio_concentration(symbols, sector_map())
    if not conc.empty and (conc["Positions"] > 1).any():
        top = conc.iloc[0]
        print(f"{indent}sector: {top['Weight']:.0%} {top['Sector']}"
              + (" (label only — see corr above)"
                 if not pairs.empty and pairs["Correlation"].max() < 0.5
                 else ""))

    if record.get("Corr_Rejected"):
        print(f"{indent}corr filter redirected: "
              f"{', '.join(record['Corr_Rejected'])}"
              + ("  [relaxed]" if record.get("Corr_Relaxed") else ""))


CONTEXT_LEGS = ["UUP", "DBC", "LQD", "RSP", "IWM", "SPY", "HYG"]
CONTEXT_SERIES = ["RSP_SPY", "HYG_LQD", "IWM_SPY", "AD_LINE", "UUP", "DBC"]


def build_context(prices, start, use_breadth=True):
    """
    Build the context series on a SEPARATE panel from the model's.

    Deliberately separate.  These series are displayed and never scored — every
    one of them tested at the noise floor as a monitor and they were materially
    negative in combination (all monitors: +0.02pp of CAGR, drawdown out to
    -20.84% from -18.80%).  Downloading them into the model's own panel would
    put them into the cross-sectional normalization and quietly change the
    scores, which is exactly the outcome the testing said to avoid.  Keeping
    two panels makes that mistake impossible rather than merely unlikely.
    """
    import yfinance as yf
    from momentum.data import PriceData
    from momentum.synthetic import BreadthSpec, RatioSpec, augment

    pool = []
    pool_path = REPO_ROOT / "random_pool.csv"
    if use_breadth and pool_path.exists():
        pool = pd.read_csv(pool_path)["symbol"].dropna().astype(str).tolist()

    syms = sorted(set(CONTEXT_LEGS) | set(pool))
    frames = []
    for i in range(0, len(syms), 400):
        d = yf.download(syms[i:i + 400], start=start, interval="1d",
                        auto_adjust=True, progress=False, threads=True,
                        group_by="column")
        frames.append(d["Close"] if isinstance(d.columns, pd.MultiIndex) else d)
    close = pd.concat(frames, axis=1)
    close = close.loc[:, ~close.columns.duplicated()]

    monitors = list(CONTEXT_SERIES)
    ratios = [RatioSpec("RSP_SPY", "RSP", "SPY"),
              RatioSpec("HYG_LQD", "HYG", "LQD"),
              RatioSpec("IWM_SPY", "IWM", "SPY")]
    ratios = [r for r in ratios
              if r.numerator in close.columns and r.denominator in close.columns]
    breadth = []
    live_pool = [s for s in pool if s in close.columns]
    if len(live_pool) >= 20:
        breadth = [BreadthSpec("AD_LINE", live_pool)]

    panel = PriceData(close=close, open_=close, spy=prices.spy, vix=prices.vix)
    return augment(panel, ratios=ratios, breadth=breadth,
                   monitor_symbols=monitors), len(live_pool)


def plot_context(ctx_close, names, window=200, lookback=504):
    """Small multiples: each signal against its own trend, shaded by state."""
    import matplotlib.pyplot as plt

    have = [n for n in names if n in ctx_close.columns
            and ctx_close[n].notna().sum() > window + 20]
    if not have:
        return None
    ncol = 3
    nrow = (len(have) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(14, 3.1 * nrow), squeeze=False)
    from momentum.context import SIGNALS

    for i, nm in enumerate(have):
        ax = axes[i // ncol][i % ncol]
        s = ctx_close[nm].dropna().iloc[-lookback:]
        ma = ctx_close[nm].dropna().rolling(window).mean().reindex(s.index)
        ax.plot(s.index, s.values, linewidth=1.4)
        ax.plot(ma.index, ma.values, linewidth=1.0, linestyle="--",
                color="grey")
        ax.fill_between(s.index, s.values, ma.values,
                        where=(s.values >= ma.values), alpha=0.18,
                        color="tab:green", interpolate=True)
        ax.fill_between(s.index, s.values, ma.values,
                        where=(s.values < ma.values), alpha=0.18,
                        color="tab:red", interpolate=True)
        state = "ABOVE" if s.iloc[-1] >= (ma.iloc[-1] or 0) else "BELOW"
        sig = SIGNALS.get(nm)
        title = (sig.label if sig else nm)
        ax.set_title(f"{title}\n{state} its {window}d avg", fontsize=9)
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=7)
    for j in range(len(have), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle("Context — displayed only, never scored "
                 "(green = above trend, red = below)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def print_rotation_sets(result, prices, config, ranking_scores, base_scores,
                        as_of=None):
    """
    The two sets, at the bottom of the run so no scrolling is needed.

    SET 1 is the aligned rotation: what the `hold_days` clock last selected, and
    therefore what the book should actually be holding.  SET 2 is a fresh
    selection on the latest available close, which is what the model would pick
    if it were rotating today.

    Running off-cycle is exactly when these two diverge, and the divergence is
    the point of showing both.  It is not a trade list — acting on SET 2 between
    rotations is a different strategy from the one that was backtested, and the
    rank-exit and score-swap suites both tested versions of that idea and
    rejected them.  Read it as information about how stale the aligned book is.
    """
    print("\n" + "=" * 78)
    print("ROTATION SETS")
    print("=" * 78)

    history = result.rebalance_history or []
    scheduled = [(i, r) for i, r in enumerate(history)
                 if r.get("Trigger") == "rebalance"]
    latest_data = prices.close.index[-1]

    # --- SET 1: the aligned rotation ---
    if not scheduled:
        print("\n  [1] ALIGNED ROTATION: none in the sample.")
    else:
        aligned_pos, aligned = scheduled[-1]
        rotated = aligned["Date"]
        due = rotated + pd.Timedelta(days=config.hold_days)
        age = (latest_data - rotated).days

        print(f"\n  [1] ALIGNED ROTATION — on the {config.hold_days}-day clock")
        print(f"      rotated {rotated:%Y-%m-%d} ({age}d ago), "
              f"regime {str(aligned['Regime']).upper()}")
        print(f"      next scheduled rotation on or after {due:%Y-%m-%d} "
              f"({(due - pd.Timestamp(date.today())).days:+d}d from today)")
        _print_book(aligned, prices, indent="        ")

        # Intra-hold triggers (rank exits, regime shifts) move the book off the
        # aligned set.  Both are disabled in the production config, so this is a
        # guard against silently reporting a stale set if one is ever enabled.
        if history[-1] is not aligned:
            drifted = history[-1]
            print(f"      NOTE: {len(history) - aligned_pos - 1} "
                  f"intra-hold change(s) since; effective book as of "
                  f"{drifted['Date']:%Y-%m-%d} is "
                  f"{', '.join(drifted['Selected_Stocks'])}")

    # --- SET 2: fresh selection on the latest close ---
    fresh = current_selection(prices, config,
                              ranking_scores=ranking_scores,
                              base_scores=base_scores,
                              as_of=as_of)
    if fresh is None:
        print("\n  [2] CURRENT SET: universe could not fill the book.")
        return

    record, ranked = fresh
    print(f"\n  [2] CURRENT SET — fresh selection as of {record['Date']:%Y-%m-%d} "
          f"close, regime {str(record['Regime']).upper()}")
    if record["Date"] != latest_data:
        print(f"      (latest price date is {latest_data:%Y-%m-%d})")
    if record["Date"].date() == date.today():
        # yfinance keeps a partially-formed bar for a session in progress, and
        # only drops a row that is empty for every ticker. Scoring off an
        # intraday price is not the same signal the close will produce, so say
        # so rather than let the set look settled.
        print("      CAUTION: this is today's bar — if the session is still "
              "open these are intraday prices.")
        print("      Use --as-of YYYY-MM-DD to pin it to a completed session.")
    _print_book(record, prices, indent="        ")

    if config.rank_offset:
        skipped = [s for s in ranked.index[:config.rank_offset]]
        print(f"      rank_offset={config.rank_offset} "
              f"({config.rank_offset_scope}) skipped: {', '.join(skipped)}")

    if scheduled:
        held = scheduled[-1][1]["Selected_Stocks"]
        now = record["Selected_Stocks"]
        out = [s for s in held if s not in now]
        into = [s for s in now if s not in held]
        if not out and not into:
            print("      unchanged from the aligned rotation.")
        else:
            print(f"      drift vs aligned: out {', '.join(out) or '(none)'}"
                  f"  /  in {', '.join(into) or '(none)'}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the live momentum model")
    parser.add_argument("--legacy", action="store_true",
                        help="reproduce pre-refactor behaviour exactly")
    parser.add_argument("--no-plots", action="store_true",
                        help="skip charts, for scheduled/headless runs")
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--no-cache", action="store_true",
                        help="force a fresh download")
    parser.add_argument("--drop-unsettled", action="store_true",
                        help="exclude today's session when the market has not "
                             "closed yet, instead of warning about it")
    parser.add_argument("--no-context", action="store_true",
                        help="skip the context panel and its chart")
    parser.add_argument("--no-breadth", action="store_true",
                        help="skip the advance/decline line (avoids the ~750 "
                             "symbol pool download)")
    parser.add_argument("--chart-days", type=int, default=252,
                        help="sessions shown in the held-book chart")
    parser.add_argument("--show-rebalances", type=int, default=20,
                        help="how many recent rebalances to print (0 = none)")
    parser.add_argument("--corr-window", type=int, default=50,
                        help="which correlation window to display")
    parser.add_argument("--corr-threshold", type=float, default=0.70,
                        help="only show pairs at or above this |correlation|")
    parser.add_argument("--as-of", default=None,
                        help="date for the CURRENT SET, e.g. 2026-09-02; "
                             "defaults to the latest available session")
    args = parser.parse_args()

    config = build_config(legacy=args.legacy)
    today = date.today()

    # Compliance gate, before anything is computed or printed.  This raises
    # rather than warns: a restricted name reaching a recommendation is a
    # failure regardless of what it would have earned, and the only safe
    # response is to stop rather than print a book with a caveat attached.
    check_symbols(current_symbols(), "universe.csv")

    print("=" * 78)
    print(f"MOMENTUM MODEL — LIVE RUN  {today.isoformat()}")
    if args.legacy:
        print("MODE: legacy (pre-refactor reproduction)")
    print("=" * 78)

    # --- point-in-time record (Request #1) ---
    uni_path = snapshot_current_universe(as_of=today)
    cfg_path = snapshot_config(config, as_of=today,
                               label="legacy" if args.legacy else "live")
    print(f"\nSnapshots written:\n  {uni_path.name}\n  {cfg_path.name}")
    print()
    print(describe_restrictions())

    symbols = current_symbols()
    prices = load_data(symbols, start_date=args.start,
                       use_cache=not args.no_cache,
                       drop_unsettled=args.drop_unsettled)

    # The panel's end date decides which session the book is built from, so it
    # belongs in the header next to the run date rather than buried in the load
    # output.  When the two differ, the run is not the rebalance-day run.
    print(f"\nSignal session: {prices.index[-1]:%Y-%m-%d}  (run date {today:%Y-%m-%d})")

    # --- price panel export (Request #2) ---
    print("\n=== EXPORTING PRICE DATA ===")
    export_price_data(prices)

    # --- scores ---
    print("\n=== CALCULATING COMPOSITE SCORES ===")
    ranking_scores, base_scores, detail = compute_scores(
        prices, config, underlying_path=str(REPO_ROOT / "rsi_ma_underlying_measures.csv")
    )

    # --- backtest / signal ---
    print("\n=== RUNNING STRATEGY ===")
    result = run_strategy(prices, config, verbose=False)
    hist = result.rebalance_history
    tail = hist[-args.show_rebalances:] if args.show_rebalances else []
    print(f"  {len(hist)} rebalances since {hist[0]['Date']:%Y-%m-%d}; "
          f"showing last {len(tail)}:")
    for rec in tail:
        print(f"    {rec['Date']:%Y-%m-%d} [{rec['Regime']:<8}] "
              f"{', '.join(rec['Selected_Stocks'])}")

    m, t = result.metrics, result.turnover
    print("\n=== STRATEGY RESULTS ===")
    print("  (backtest on today's universe; see the caveats note at the end)")

    # Each metric carries its direction and what it is FOR.  A column of numbers
    # with no interpretation is where over-reading starts: Calmar 1.19 means
    # nothing to a reader who has to remember whether high is good and what it
    # trades off against.
    rows = [
        ("Total Return", f"{m['total_return']:>10.2%}", "higher",
         "growth of $1 over the whole record; scales with length, not skill"),
        ("CAGR", f"{m['cagr']:>10.2%}", "higher",
         "annualized return. Inflated here by survivorship and a top-decile "
         "window — read deltas, not the level"),
        ("Volatility", f"{m['volatility']:>10.2%}", "lower",
         "annualized dispersion. ~17% is equity-like; below that is the "
         "defensive sleeve working"),
        ("Sharpe Ratio", f"{m['sharpe_ratio']:>10.2f}", "higher",
         "return per unit of total risk. >1.0 is good for a long-only "
         "equity book"),
        ("Sortino Ratio", f"{m['sortino_ratio']:>10.2f}", "higher",
         "same, counting only downside moves. Above Sharpe means the "
         "volatility is mostly upside"),
        ("Max Drawdown", f"{m['max_drawdown']:>10.2%}", "smaller loss",
         "worst peak-to-trough. THE number this model exists to control — "
         "but one episode, so weakly estimated"),
        ("Calmar Ratio", f"{m['calmar_ratio']:>10.2f}", "higher",
         "CAGR per unit of max drawdown. >1.0 means a year of return "
         "exceeds the worst hole"),
        ("Trading Days", f"{m['num_periods']:>10,}", "n/a",
         "sample size. ~3,700 days is only ~265 independent 14-day holds"),
    ]
    for name, val, direction, why in rows:
        arrow = {"higher": "^ higher better", "lower": "v lower better",
                 "smaller loss": "v smaller loss better"}.get(direction, "")
        print(f"  {name:<14}{val}   {arrow}")
        print(f"  {'':<14}{'':>10}   {why}")

    print("\n  --- turnover: what it cost to get the above ---")
    print(f"  Trades / year  {t['trades_per_year']:>10.1f}   v lower better — "
          f"each is a real order and real slippage")
    print(f"  Avg hold       {t['avg_hold_days']:>10.1f} days  context for the "
          f"{config.hold_days}-day rotation clock")
    print(f"  Median hold    {t['median_hold_days']:>10.1f} days  well below avg "
          f"means a few names are held much longer")
    print(f"  Annual turnover{t['annual_turnover']:>10.1%}   v lower better — "
          f"fraction of the book replaced per year")
    print(f"  Slippage paid  {result.total_cost:>10.2%}   cumulative drag "
          f"already deducted from CAGR above")

    # --- exports ---
    print("\n=== EXPORTING RESULTS ===")
    out = REPO_ROOT

    perf = pd.DataFrame({
        "Date": result.returns.index,
        "Portfolio_Return": result.returns.values,
        "Gross_Return": result.gross_returns.values,
        "Holdings": result.holdings.values,
    })
    perf.to_csv(out / "rsi_ma_portfolio_performance.csv", index=False)
    ranking_scores.to_csv(out / "rsi_ma_composite_scores.csv")
    base_scores.to_csv(out / "rsi_ma_composite_scores_level_only.csv")
    pd.DataFrame(result.rebalance_history).to_csv(
        out / "rsi_ma_rebalance_history.csv", index=False)

    perf_by_stock = individual_stock_performance(prices.close)
    perf_by_stock.to_csv(out / "rsi_ma_individual_stock_performance.csv", index=False)

    # One line when everything worked; the full list only when it did not.
    # A run that exports the same five files every time does not need five
    # lines to say so — but a missing one must be impossible to overlook.
    expected = ("rsi_ma_portfolio_performance.csv", "rsi_ma_composite_scores.csv",
                "rsi_ma_composite_scores_level_only.csv",
                "rsi_ma_rebalance_history.csv",
                "rsi_ma_individual_stock_performance.csv")
    missing = [n for n in expected if not (out / n).exists()]
    if missing:
        print(f"  *** {len(missing)} of {len(expected)} EXPORTS FAILED ***")
        for n in expected:
            print(f"    {'MISSING' if n in missing else 'ok     '}  {n}")
    else:
        print(f"  {len(expected)} files exported OK -> {out}")

    # --- correlations ---
    print("\n=== CORRELATION ===")
    mats = correlation_matrices(prices.close)
    for period, matrix in mats.items():          # every window still exported
        matrix.to_csv(out / f"rsi_ma_correlation_{period}d.csv")
    if args.corr_window in mats:
        print(summarize_correlations(mats[args.corr_window], args.corr_window,
                                     threshold=args.corr_threshold))
    else:
        print(f"  (no {args.corr_window}d matrix; have "
              f"{sorted(mats)})")
    print(f"  all {len(mats)} windows exported to CSV")

    # --- context signals (displayed, never scored) ---
    ctx = None
    if not args.no_context:
        try:
            ctx, n_pool = build_context(prices, args.start,
                                        use_breadth=not args.no_breadth)
            from momentum.context import report as context_report
            print()
            print(context_report(ctx.close, prices.spy, CONTEXT_SERIES,
                                 model_returns=result.returns,
                                 horizon=config.hold_days))
            if n_pool:
                print(f"  (A/D computed over {n_pool} pool names)")
        except Exception as exc:   # context must never break the live run
            print(f"\n(Context skipped: {type(exc).__name__}: {exc})")

    # --- health monitor ---
    #
    # Above the rotation sets rather than below them: if the model is in a
    # sustained shortfall against the market, that is context for reading the
    # picks, not a footnote to them.  ALARM means investigate, never trade —
    # see momentum/health.py on why the historical record does not support
    # treating this as predictive.
    try:
        health_config = HealthConfig(
            risk_free_rate=config.execution.risk_free_rate)
        health = compute_health(result.returns, prices.spy, health_config)
        state = current_state(health, health_config)
        print()
        print("=" * 78)
        print(status_line(state, health_config))
        exposure = defensive_exposure(result.holdings, health_config)
        print(defensive_line(defensive_state(exposure, health_config),
                             health_config))
        if state.get("state") == "ALARM":
            print("  -> investigate: scripts/monitor_health.py for the record, "
                  "then screen_universe.py")
        print("=" * 78)
    except Exception as exc:          # a monitor must never break the live run
        print()
        print(f"(Health monitor skipped: {exc})")

    # --- the two sets, last so they need no scrolling ---
    print_rotation_sets(result, prices, config, ranking_scores,
                        base_scores, as_of=args.as_of)

    # --- charts ---
    if not args.no_plots:
        try:
            import matplotlib.pyplot as plt

            wealth = (1 + result.returns).cumprod()
            fig, axes = plt.subplots(2, 1, figsize=(12, 8))
            axes[0].plot(wealth.index, wealth.values, linewidth=1.6)
            axes[0].set_title("Momentum strategy — cumulative return "
                              f"({config.execution.execute_at} execution, "
                              f"{config.execution.slippage_bps:g} bps slippage)")
            axes[0].set_ylabel("Growth of $1")
            axes[0].set_yscale("log")
            axes[0].grid(alpha=0.3)

            axes[1].plot(result.returns.index, result.returns.values * 100,
                         linewidth=0.6, alpha=0.8)
            axes[1].set_title("Daily returns (%)")
            axes[1].set_xlabel("Date")
            axes[1].grid(alpha=0.3)

            plt.tight_layout()

            # --- held names, restored ---
            #
            # The pre-refactor script drew this (plot_momentum_portolio, legacy
            # line 663) and the refactor dropped it.  It was never reported as
            # broken because a chart that is simply absent produces no error.
            #
            # Rebased to 100 at the window start rather than drawn on raw
            # price: the legacy version plotted absolute prices on one axis, so
            # a $600 name and a $40 name shared a scale and only the expensive
            # one was legible.  Rebasing is what makes the lines comparable,
            # which is the whole point of putting them together.
            held = list(result.holdings.iloc[-1]) if len(result.holdings) else []
            if held:
                lookback = min(args.chart_days, len(prices.close))
                sub = prices.close[held].iloc[-lookback:].dropna(axis=1, how="all")
                if not sub.empty:
                    fig2, ax = plt.subplots(figsize=(13, 6))
                    for sym in sub.columns:
                        s = sub[sym].dropna()
                        if s.empty:
                            continue
                        ax.plot(s.index, s / s.iloc[0] * 100.0, linewidth=1.5,
                                label=f"{sym}  {s.iloc[-1] / s.iloc[0] - 1:+.1%}")
                    spy_w = prices.spy.reindex(sub.index).ffill().dropna()
                    if not spy_w.empty:
                        ax.plot(spy_w.index, spy_w / spy_w.iloc[0] * 100.0,
                                linewidth=2.0, linestyle="--", color="black",
                                alpha=0.55,
                                label=f"SPY  {spy_w.iloc[-1] / spy_w.iloc[0] - 1:+.1%}")
                    ax.axhline(100, color="grey", linewidth=0.8, alpha=0.5)
                    ax.set_title(f"Held book over the last {lookback} sessions "
                                 f"— rebased to 100, SPY dashed")
                    ax.set_ylabel("Rebased (100 = start)")
                    ax.grid(alpha=0.3)
                    ax.legend(loc="best", fontsize=9)
                    fig2.tight_layout()

            if ctx is not None:
                plot_context(ctx.close, CONTEXT_SERIES)

            plt.show()
        except Exception as exc:      # a headless box should not fail the run
            print(f"\n(Charts skipped: {exc})")

    print("\n=== RUN COMPLETE ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
