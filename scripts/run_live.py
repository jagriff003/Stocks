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
from momentum.experiments import legacy_config, production_config
from momentum.reports import (correlation_matrices, individual_stock_performance,
                              portfolio_concentration, portfolio_correlation,
                              summarize_correlations)
from momentum.strategy import compute_scores, current_selection, run_strategy
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
    parser.add_argument("--as-of", default=None,
                        help="date for the CURRENT SET, e.g. 2026-09-02; "
                             "defaults to the latest available session")
    args = parser.parse_args()

    config = build_config(legacy=args.legacy)
    today = date.today()

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

    symbols = current_symbols()
    prices = load_data(symbols, start_date=args.start,
                       use_cache=not args.no_cache)

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
    result = run_strategy(prices, config, verbose=True)

    m, t = result.metrics, result.turnover
    print("\n=== STRATEGY RESULTS ===")
    print(f"  Total Return   {m['total_return']:>10.2%}")
    print(f"  CAGR           {m['cagr']:>10.2%}")
    print(f"  Volatility     {m['volatility']:>10.2%}")
    print(f"  Sharpe Ratio   {m['sharpe_ratio']:>10.2f}")
    print(f"  Sortino Ratio  {m['sortino_ratio']:>10.2f}")
    print(f"  Max Drawdown   {m['max_drawdown']:>10.2%}")
    print(f"  Calmar Ratio   {m['calmar_ratio']:>10.2f}")
    print(f"  Trading Days   {m['num_periods']:>10,}")
    print("\n  --- turnover ---")
    print(f"  Trades / year  {t['trades_per_year']:>10.1f}")
    print(f"  Avg hold       {t['avg_hold_days']:>10.1f} days")
    print(f"  Median hold    {t['median_hold_days']:>10.1f} days")
    print(f"  Annual turnover{t['annual_turnover']:>10.1%}")
    print(f"  Slippage paid  {result.total_cost:>10.2%} cumulative")

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

    for name in ("rsi_ma_portfolio_performance.csv", "rsi_ma_composite_scores.csv",
                 "rsi_ma_composite_scores_level_only.csv",
                 "rsi_ma_rebalance_history.csv",
                 "rsi_ma_individual_stock_performance.csv"):
        print(f"- {name}")

    # --- correlations ---
    print("\n=== CORRELATION MATRICES ===")
    for period, matrix in correlation_matrices(prices.close).items():
        path = out / f"rsi_ma_correlation_{period}d.csv"
        matrix.to_csv(path)
        print(summarize_correlations(matrix, period))
        print(f"Exported to: {path.name}")

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
            plt.show()
        except Exception as exc:      # a headless box should not fail the run
            print(f"\n(Charts skipped: {exc})")

    print("\n=== RUN COMPLETE ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
