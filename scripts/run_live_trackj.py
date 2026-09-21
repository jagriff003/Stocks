"""
The Track J book: what the pullback model would hold right now.

This is the live runner for the candidate model. It is a SEPARATE script from
`run_live.py` on purpose — that one runs the current production model on the
46-name universe, and it must keep working untouched while both are in use. No
shared state, no shared config, no flag that switches one into the other. Run
either, or both, on the same day.

    python scripts/run_live.py            # the model you trade today
    python scripts/run_live_trackj.py     # the candidate

WHAT IT PRINTS

  SET 1  the ALIGNED book — what the model holds given its rotation clock. This
         is the one to trade. It only changes on a rebalance date.
  SET 2  the FRESH ranking on the latest close, which is what the model WOULD
         buy if today were a rebalance. Between rotations these diverge, and
         trading Set 2 off-cycle is a different (untested) strategy. Both are
         printed so the gap is visible and acting on it is deliberate rather
         than accidental — same reasoning as `run_live.py`.

WHAT IT RECORDS

Every run writes a dated pool snapshot. That is the point-in-time record which
is the only thing that can ever close the survivorship question (TODO 0d), and
it is worthless retroactively — so it starts now, before the model is traded,
not after.

SAFETY

The restricted list is applied by symbol, industry and issuer name, and then
asserted again on the final panel. A restricted name reaching the book raises
and the script exits non-zero. Per RUNBOOK, that must never be suppressed.

Run:  python scripts/run_live_trackj.py
      python scripts/run_live_trackj.py --no-cache      # force fresh prices
      python scripts/run_live_trackj.py --as-of 2026-09-15
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from momentum.backtest import build_target_portfolios
from momentum.config import snapshot_config
from momentum.data import PriceData
from momentum.experiments import production_config
from momentum.liquidity import (LiquidityConfig, apply_tradable, dollar_volume,
                                slippage_panel, tradable_mask)
from momentum.pool_snapshot import coverage, snapshot_pool
from momentum.restrictions import check_symbols
from momentum.restrictions import describe as describe_restrictions
from momentum.reversal import ReversalConfig, build_terms, composite
from momentum.strategy import compute_scores
from momentum.universe import defensive_symbols

from momentum.backtest import simulate_portfolio
from momentum.health import (HealthConfig, compute_health, current_state,
                             defensive_exposure, defensive_line,
                             defensive_state, status_line)
from momentum.metrics import calculate_performance_metrics
from momentum.reports import (correlation_matrices, individual_stock_performance,
                              summarize_correlations)

from scripts.analyze_reversal_backtest import load_volume
from scripts.analyze_reversal_ic import NON_EQUITY, load_pool_panel, pool_symbols
# Imported rather than reimplemented: one context panel, one degradation
# tracker, one held-book chart.  A second copy would drift from the original
# and the two runners would quietly stop being comparable.
from scripts.run_live import (CONTEXT_SERIES, Degradations, build_context,
                              plot_context)

POOL_FILE = "random_pool.csv"


def build_config(top_n: int, hold: int):
    """
    The Track J production configuration, as decided 2026-09-20.

    Every departure from `production_config()` is a recorded decision:
      vix=None            the overlay costs 3.4pp CAGR on this score
      top_n=8             plateau rather than peak on the parameter grid
      hold_days=40        8 whole weeks, so the rebalance weekday is fixed
      correlation kept    gated above VIX 25; verified better than off or always
    The level floor is switched off at the call site by passing base=None.
    """
    return replace(production_config(), top_n=top_n, hold_days=hold, vix=None)


def main() -> int:
    p = argparse.ArgumentParser(description="Run the Track J pullback model")
    p.add_argument("--top-n", type=int, default=8)
    p.add_argument("--hold", type=int, default=40)
    p.add_argument("--start", default="2010-01-01")
    p.add_argument("--min-adv", type=float, default=10e6)
    p.add_argument("--min-price", type=float, default=5.0)
    p.add_argument("--account", type=float, default=100_000.0)
    p.add_argument("--as-of", default=None,
                   help="pretend today is this date (YYYY-MM-DD)")
    p.add_argument("--no-cache", action="store_true",
                   help="force a fresh download; use this for a real run")
    p.add_argument("--no-snapshot", action="store_true",
                   help="skip the pool snapshot (do not use on a real run)")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--save-charts", metavar="DIR", default=None,
                   help="write charts as PNGs instead of opening windows")
    p.add_argument("--no-context", action="store_true")
    p.add_argument("--no-breadth", action="store_true")
    p.add_argument("--chart-days", type=int, default=252)
    p.add_argument("--show-rebalances", type=int, default=20)
    p.add_argument("--corr-window", type=int, default=50)
    p.add_argument("--corr-threshold", type=float, default=0.70)
    args = p.parse_args()

    if args.save_charts:
        import matplotlib
        matplotlib.use("Agg")

    degraded = Degradations()
    cfg = build_config(args.top_n, args.hold)
    rcfg = ReversalConfig()
    lcfg = LiquidityConfig(account_notional=args.account)
    defensive = set(defensive_symbols())
    today = date.fromisoformat(args.as_of) if args.as_of else date.today()

    print("=" * 100)
    print("TRACK J — PULLBACK MODEL — LIVE BOOK")
    print(f"Run {datetime.now():%Y-%m-%d %H:%M:%S}   as-of {today}")
    print(f"score pullback = z(-flip) + z(range_pos)   top_n {cfg.top_n}   "
          f"hold {cfg.hold_days}d   overlay off")
    print("=" * 100)
    print("\n*** CANDIDATE MODEL — NOT THE PRODUCTION BOOK. ***")
    print("*** `python scripts/run_live.py` is the model currently traded. ***")

    print("\n" + describe_restrictions())

    # --- panel ---
    syms = pool_symbols(POOL_FILE)
    raw = sorted(set(pd.read_csv(REPO_ROOT / POOL_FILE)["symbol"].dropna())
                 - NON_EQUITY)
    prices = load_pool_panel(raw + sorted(defensive), start=args.start,
                             use_cache=not args.no_cache)
    allowed = set(syms) | defensive
    keep = [c for c in prices.close.columns if c in allowed]
    prices = PriceData(close=prices.close[keep], open_=prices.open_[keep],
                       spy=prices.spy, vix=prices.vix)
    volume = load_volume(raw + sorted(defensive), start=args.start,
                         use_cache=not args.no_cache)
    volume = volume.reindex(index=prices.close.index,
                            columns=prices.close.columns)

    # Hard stop: nothing restricted may reach a panel the model ranks.
    check_symbols(list(prices.close.columns), "Track J live panel")

    if args.as_of:
        prices = PriceData(close=prices.close.loc[:args.as_of],
                           open_=prices.open_.loc[:args.as_of],
                           spy=prices.spy, vix=prices.vix)
        volume = volume.loc[:args.as_of]

    session = prices.close.index[-1]
    print(f"\nSignal session: {session:%Y-%m-%d}   "
          f"({prices.close.shape[1]} names in panel)")
    if (pd.Timestamp(today) - session).days > 4:
        print("*** WARNING: the panel is more than four days stale. Do NOT "
              "trade this book;\n    re-run with --no-cache. ***")

    # --- screen and score ---
    tmask = tradable_mask(prices.close, volume, args.min_adv, args.min_price,
                          lcfg)
    ranking, base, _ = compute_scores(prices, cfg)
    terms = build_terms(prices.close, rcfg)
    pull = composite(terms, replace(rcfg, turn_weight=-1.0,
                                    strength_weight=0.0, room_weight=1.0),
                     "flip").reindex_like(ranking)
    scored = apply_tradable(pull, tmask, defensive)

    tradable_today = int(tmask.loc[session].sum())
    print(f"Tradable today: {tradable_today} names "
          f"(ADV >= ${args.min_adv/1e6:.0f}M, price >= ${args.min_price:.0f})")

    # --- the rotation clock ---
    # `base_composite_scores=None` switches off the level floor, per TODO 0h.2.
    targets, history = build_target_portfolios(
        scored,
        price_columns=list(prices.close.columns),
        top_n=cfg.top_n, min_data_days=cfg.min_data_days,
        hold_days=cfg.hold_days,
        vix_data=prices.vix, vix_config=cfg.vix,
        base_composite_scores=None,
        velocity_config=cfg.velocity,
        correlation_config=cfg.correlation,
        graduated_config=cfg.graduated_vix, exit_config=cfg.exits,
        close=prices.close, rank_offset=cfg.rank_offset,
        rank_offset_scope=cfg.rank_offset_scope,
        monitor_symbols=cfg.monitor_symbols,
    )

    # --- simulate, so the metrics and charts describe THIS configuration ---
    slip_panel = slippage_panel(prices.close, volume, lcfg, top_n=cfg.top_n)
    result = simulate_portfolio(targets, prices.close, prices.open_,
                                execution=cfg.execution, sizing=cfg.sizing,
                                scores=scored, slippage_by_symbol=slip_panel)
    result.rebalance_history = history

    print("\n=== REBALANCE HISTORY ===")
    tail = history[-args.show_rebalances:] if args.show_rebalances else []
    print(f"  {len(history)} rotations since {history[0]['Date']:%Y-%m-%d}; "
          f"showing last {len(tail)}:")
    for rec in tail:
        print(f"    {rec['Date']:%Y-%m-%d} "
              f"{', '.join(rec['Selected_Stocks'])}")

    m, tno = result.metrics, result.turnover
    gross = calculate_performance_metrics(
        result.gross_returns, risk_free_rate=cfg.execution.risk_free_rate)
    print("\n=== STRATEGY RESULTS ===")
    print("  Backtest of THIS configuration on the current pool. Levels are")
    print("  inflated by survivorship (TODO 0d), and this is ONE rotation phase")
    print("  of %d -- phase is worth ~10pp of CAGR. Read deltas, not levels."
          % cfg.hold_days)
    rows = [
        ("CAGR", "%10.2f%%" % (100 * m["cagr"]), "^ higher",
         "gross of costs %.2f%%; the phase median is the honest figure"
         % (100 * gross["cagr"])),
        ("Volatility", "%10.2f%%" % (100 * m["volatility"]), "v lower",
         "runs hotter than the 46-name model -- 8 names out of ~610"),
        ("Sharpe Ratio", "%10.2f" % m["sharpe_ratio"], "^ higher",
         "the constraint: must not be worse than what it replaces"),
        ("Sortino Ratio", "%10.2f" % m["sortino_ratio"], "^ higher",
         "above Sharpe means the volatility is mostly upside"),
        ("Max Drawdown", "%10.2f%%" % (100 * m["max_drawdown"]), "v smaller",
         "the fortitude number: what holding this will demand of you"),
        ("Calmar Ratio", "%10.2f" % m["calmar_ratio"], "^ higher",
         "CAGR per unit of worst hole"),
        ("Trading Days", "%10d" % m["num_periods"], "n/a",
         "~%d independent %d-day holds"
         % (m["num_periods"] // cfg.hold_days, cfg.hold_days)),
    ]
    for name, val, arrow, why in rows:
        print("  %-14s%s   %s" % (name, val, arrow))
        print("  %-14s%10s   %s" % ("", "", why))

    print("\n  --- turnover: what it cost to get the above ---")
    print("  Trades / year  %10.1f   v lower better" % tno["trades_per_year"])
    print("  Avg hold       %10.1f days  against a %d-session clock"
          % (tno["avg_hold_days"], cfg.hold_days))
    print("  Annual turnover%10.1f%%   v lower better"
          % (100 * tno["annual_turnover"]))
    print("  Slippage paid  %10.2f%%   per-name liquidity model, already "
          "deducted above" % (100 * result.total_cost))

    # --- exports ---
    print("\n=== EXPORTING RESULTS ===")
    out = REPO_ROOT
    exports = {
        "trackj_portfolio_performance.csv": pd.DataFrame({
            "Date": result.returns.index,
            "Portfolio_Return": result.returns.values,
            "Gross_Return": result.gross_returns.values,
            "Holdings": result.holdings.values}),
        "trackj_rebalance_history.csv": pd.DataFrame(history),
        "trackj_individual_stock_performance.csv":
            individual_stock_performance(prices.close),
    }
    written = []
    try:
        for name, frame in exports.items():
            frame.to_csv(out / name, index=False)
            written.append(name)
        scored.to_csv(out / "trackj_scores.csv")
        written.append("trackj_scores.csv")
    except Exception as exc:
        print("  *** export failed: %s: %s ***" % (type(exc).__name__, exc))
        degraded.note("exports", exc)
    missing = [n for n in written if not (out / n).exists()]
    if missing:
        print("  *** %d of %d EXPORTS FAILED ***" % (len(missing), len(written)))
        for n in missing:
            print("    MISSING  %s" % n)
        degraded.note("exports", RuntimeError("missing: %s" % missing))
    else:
        print("  %d files exported OK -> %s" % (len(written), out))

    # --- correlation and sector mix of the HELD book ---
    print("\n=== CORRELATION ===")
    try:
        held_now = list(targets.loc[session]) if session in targets.index else []
        cols = [h for h in held_now if h in prices.close.columns]
        mats = correlation_matrices(prices.close[cols]) if len(cols) > 1 else {}
        for period, matrix in mats.items():
            matrix.to_csv(out / ("trackj_correlation_%dd.csv" % period))
        if args.corr_window in mats:
            print(summarize_correlations(mats[args.corr_window],
                                         args.corr_window,
                                         threshold=args.corr_threshold))
        print("  %d windows exported for the held book" % len(mats))

        secs = pd.read_csv(REPO_ROOT / POOL_FILE).set_index("symbol")["sector"]
        mix = pd.Series([secs.get(h) for h in held_now]).dropna().value_counts()
        if len(mix):
            print("\n  Sector mix of the held book:")
            for sec, n in mix.items():
                flag = "   <-- MAJORITY" if n / len(held_now) >= 0.5 else ""
                print("    %-26s%d/%d%s" % (sec, n, len(held_now), flag))
            if mix.iloc[0] / len(held_now) >= 0.5:
                print("  The correlation filter constrains CORRELATION, not")
                print("  SECTOR, and is gated above VIX 25 (TODO 0h.4).")
    except Exception as exc:
        print("  (correlation skipped: %s: %s)" % (type(exc).__name__, exc))
        degraded.note("correlation", exc)

    # --- context (displayed, never scored) ---
    ctx = None
    if not args.no_context:
        try:
            ctx, n_pool = build_context(prices, args.start,
                                        use_breadth=not args.no_breadth)
            from momentum.context import report as context_report
            print()
            print(context_report(ctx.close, prices.spy, CONTEXT_SERIES,
                                 model_returns=result.returns,
                                 horizon=cfg.hold_days))
            if n_pool:
                print("  (A/D computed over %d pool names)" % n_pool)
        except Exception as exc:
            print("\n(Context skipped: %s: %s)" % (type(exc).__name__, exc))
            degraded.note("context panel", exc)

    # --- health monitor ---
    try:
        hcfg = HealthConfig(risk_free_rate=cfg.execution.risk_free_rate)
        health = compute_health(result.returns, prices.spy, hcfg)
        state = current_state(health, hcfg)
        print()
        print("=" * 100)
        print(status_line(state, hcfg))
        exposure = defensive_exposure(result.holdings, hcfg)
        print(defensive_line(defensive_state(exposure, hcfg), hcfg))
        print("  NOTE: these thresholds were calibrated on the 46-name model.")
        print("  Displayed for this one, not validated for it (TODO 3).")
        print("=" * 100)
    except Exception as exc:
        print("\n(Health monitor skipped: %s)" % exc)
        degraded.note("health monitor", exc)

    held = list(targets.loc[session]) if session in targets.index else []
    last_rebal = None
    for h in history:
        d = pd.Timestamp(h["Date"])
        if d <= session:
            last_rebal = d
    sessions_held = (len(prices.close.loc[last_rebal:session]) - 1
                     if last_rebal is not None else None)

    close_now = prices.close.loc[session]
    adv_now = dollar_volume(prices.close, volume, lcfg.adv_window).loc[session]
    score_now = scored.loc[session]
    slip_now = slippage_panel(prices.close, volume, lcfg,
                              top_n=cfg.top_n).loc[session] * 10_000

    def show(names, title):
        print(f"\n  {title}")
        hdr = (f"    {'symbol':<8}{'price':>10}{'score':>8}{'rank':>6}"
               f"{'$vol (M)':>11}{'est bps':>9}")
        print(hdr)
        print("    " + "-" * (len(hdr) - 4))
        ranks = score_now.rank(ascending=False, method="min")
        for n in names:
            print(f"    {n:<8}{close_now.get(n, float('nan')):>10.2f}"
                  f"{score_now.get(n, float('nan')):>8.2f}"
                  f"{ranks.get(n, float('nan')):>6.0f}"
                  f"{adv_now.get(n, float('nan'))/1e6:>11.0f}"
                  f"{slip_now.get(n, float('nan')):>9.1f}")

    print("\n" + "=" * 100)
    print("SET 1 — THE ALIGNED BOOK  (this is the one to trade)")
    print("=" * 100)
    if last_rebal is not None:
        idx = prices.close.index
        pos = idx.get_loc(last_rebal) + cfg.hold_days
        if pos < len(idx):
            nxt, exact = idx[pos], True
        else:
            # The rotation is in the future, so it is not in the panel yet.
            # Project it on business days: an estimate, and labelled as one,
            # because it does not account for market holidays.
            remaining = cfg.hold_days - (sessions_held or 0)
            nxt = pd.bdate_range(session, periods=remaining + 1)[-1]
            exact = False
        print(f"  last rotation {last_rebal:%Y-%m-%d} ({last_rebal:%A}), "
              f"held {sessions_held} of {cfg.hold_days} sessions")
        print(f"  next rotation {nxt:%Y-%m-%d} ({nxt:%A})"
              f"{'' if exact else '  [estimated; business days, ignores holidays]'}")
        if not exact:
            print(f"                {remaining} sessions from now")
    show(held, f"{len(held)} positions, equal weight "
               f"(${args.account/max(1,len(held)):,.0f} each)")

    print("\n" + "=" * 100)
    print("SET 2 — FRESH RANKING on the latest close")
    print("=" * 100)
    print("  What the model WOULD buy if today were a rotation date. Trading")
    print("  this off-cycle is a different and untested strategy.")
    fresh = list(score_now.dropna().sort_values(ascending=False)
                 .head(cfg.top_n).index)
    show(fresh, "top %d by score" % cfg.top_n)

    overlap = set(held) & set(fresh)
    print(f"\n  Overlap with the aligned book: {len(overlap)} of {len(held)}"
          f"{' — ' + ', '.join(sorted(overlap)) if overlap else ''}")

    # --- the point-in-time record ---
    if not args.no_snapshot:
        path = snapshot_pool(
            tradable=tmask.loc[session],
            dollar_volume=adv_now, price=close_now, scores=score_now,
            as_of=today, label="trackj")
        print(f"\n  Pool snapshot written: {path.name}")
        cov = coverage()
        print(f"  Point-in-time record now spans {len(cov)} snapshot(s)"
              f"{' — too short to answer anything yet' if len(cov) < 8 else ''}")
        snapshot_config(cfg, as_of=today, label="trackj")

    # --- charts ---
    if not args.no_plots:
        try:
            import matplotlib.pyplot as plt

            wealth = (1 + result.returns).cumprod()
            fig, axes = plt.subplots(2, 1, figsize=(12, 8))
            axes[0].plot(wealth.index, wealth.values, linewidth=1.6)
            axes[0].set_title("Track J pullback - cumulative return "
                              "(%s, per-name liquidity costs)"
                              % cfg.execution.execute_at)
            axes[0].set_ylabel("Growth of $1")
            axes[0].set_yscale("log")
            axes[0].grid(alpha=0.3)
            axes[1].plot(result.returns.index, result.returns.values * 100,
                         linewidth=0.6, alpha=0.8)
            axes[1].set_title("Daily returns (%)")
            axes[1].set_xlabel("Date")
            axes[1].grid(alpha=0.3)
            plt.tight_layout()

            held_fig = None
            if held:
                lookback = min(args.chart_days, len(prices.close))
                sub = prices.close[[h for h in held if h in prices.close.columns]]
                sub = sub.iloc[-lookback:].dropna(axis=1, how="all")
                if not sub.empty:
                    held_fig, ax = plt.subplots(figsize=(13, 6))
                    for sym in sub.columns:
                        s = sub[sym].dropna()
                        if s.empty:
                            continue
                        ax.plot(s.index, s / s.iloc[0] * 100.0, linewidth=1.5,
                                label="%s  %+.1f%%"
                                      % (sym, 100 * (s.iloc[-1] / s.iloc[0] - 1)))
                    spy_w = prices.spy.reindex(sub.index).ffill().dropna()
                    if not spy_w.empty:
                        ax.plot(spy_w.index, spy_w / spy_w.iloc[0] * 100.0,
                                linewidth=2.0, linestyle="--", color="black",
                                alpha=0.55,
                                label="SPY  %+.1f%%"
                                      % (100 * (spy_w.iloc[-1] / spy_w.iloc[0] - 1)))
                    ax.axhline(100, color="grey", linewidth=0.8, alpha=0.5)
                    ax.set_title("Held book over the last %d sessions "
                                 "- rebased to 100, SPY dashed" % lookback)
                    ax.set_ylabel("Rebased (100 = start)")
                    ax.grid(alpha=0.3)
                    ax.legend(loc="best", fontsize=9)
                    held_fig.tight_layout()

            ctx_fig = (plot_context(ctx.close, CONTEXT_SERIES)
                       if ctx is not None else None)

            if args.save_charts:
                outdir = Path(args.save_charts)
                outdir.mkdir(parents=True, exist_ok=True)
                stamp = today.isoformat()
                print("\nCharts written to %s:" % outdir)
                for name, figure in (("performance", fig),
                                     ("held-book", held_fig),
                                     ("context", ctx_fig)):
                    if figure is None:
                        continue
                    path = outdir / ("%s_trackj_%s.png" % (stamp, name))
                    figure.savefig(path, dpi=110, bbox_inches="tight")
                    print("  %s" % path.name)
            else:
                plt.show()
            plt.close("all")
        except Exception as exc:
            print("\n(Charts skipped: %s: %s)" % (type(exc).__name__, exc))
            degraded.note("charts", exc)

    print("\n" + "=" * 100)
    print("BEFORE TRADING THIS")
    print("=" * 100)
    print("  - It is a CANDIDATE. Out of sample it is not statistically")
    print("    distinguishable from the current model (t = 1.34 by window).")
    print("  - Survivorship is unquantified and accepted as a known risk.")
    print("  - Phase risk is ~10pp of CAGR; you get one phase and cannot know")
    print("    in advance whether it is a good one (TODO 0i).")
    print("  - There is no exit rule. A position is held the full 40 sessions")
    print("    however it behaves (TODO 0g).")
    degraded.report()
    print(f"\nCompleted {datetime.now():%Y-%m-%d %H:%M:%S}")
    return 2 if degraded else 0


if __name__ == "__main__":
    sys.exit(main())
