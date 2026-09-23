"""
The Track J book: what the pullback model would hold right now.

This is the live runner for the candidate model. It is a SEPARATE script from
`run_live.py` on purpose — that one runs the current production model on the
46-name universe, and it must keep working untouched while both are in use. No
shared state, no shared config, no flag that switches one into the other. Run
either, or both, on the same day.

    python scripts/run_live.py            # the model you trade today
    python scripts/run_live_trackj.py     # the candidate

THE ROTATION CALENDAR

Four sleeves of equal capital, one rotating every two weeks on a TUESDAY, so
the book is built from Tuesday's close and traded at Wednesday's open. Each
sleeve therefore holds eight weeks.

The calendar is anchored to a production-model rotation date, which puts both
models on the same Tuesdays — they can be run and traded together. A session
count cannot do this: 40 sessions is eight weeks only when no holiday falls
inside the cycle, and Track J's rotations drifted Tue/Mon/Mon/Tue/Mon/Mon before
this. See `momentum/schedule.py`.

WHAT IT PRINTS

  SET 1  the COMBINED book across all four sleeves, with each name's weight set
         by how many sleeves hold it. Names held by more than one sleeve carry
         more weight, which is deliberate: names that keep re-qualifying
         returned 31-35% annualised against 14% for single-sleeve names.
  SET 2  what the NEXT sleeve to rotate would buy on the latest close. It is not
         today's trade unless today is a rotation date, and trading it off-cycle
         is a different (untested) strategy.

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

from momentum.backtest import _PortfolioBuilder, build_target_portfolios
from momentum.config import snapshot_config
from momentum.data import PriceData
from momentum.experiments import production_config
from momentum.liquidity import (LiquidityConfig, apply_tradable, dollar_volume,
                                realized_vol, slippage_panel, tradable_mask)
from momentum.pool_snapshot import coverage, snapshot_pool
from momentum.schedule import (TUESDAY, current_sleeves, next_rotation,
                               rotation_dates, sleeve_assignment)
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
      correlation         absolute 0.70, every rebalance (swept 0.60-0.80)

    `hold_days` survives only for the eligibility gate inside
    `_PortfolioBuilder`; the live rotation is driven by the calendar in
    `momentum/schedule.py`, not by a session count.
    The level floor is switched off at the call site by passing base=None.
    """
    from momentum.config import CorrelationConfig
    cfg = replace(production_config(), top_n=top_n, hold_days=hold, vix=None)
    # Correlation filter: ABSOLUTE 0.70, applied EVERY rebalance.
    #
    # The inherited setting was a relative 85th-percentile threshold gated above
    # VIX 25, i.e. no diversification constraint at all in a normal regime. On a
    # book of 8 drawn from ~610 that let the model hold five semiconductors.
    # Swept 0.60-0.80: 0.70 is the ridge, not a spike, and it improves CAGR,
    # Sharpe, drawdown and Calmar at unchanged turnover.
    return replace(cfg, correlation=replace(
        cfg.correlation, enabled=True, apply_above_vix=None,
        method="absolute", max_correlation=0.70))


def main() -> int:
    p = argparse.ArgumentParser(description="Run the Track J pullback model")
    p.add_argument("--top-n", type=int, default=8,
                   help="names per sleeve")
    p.add_argument("--tranches", type=int, default=4,
                   help="staggered sleeves; 4 is production")
    p.add_argument("--every-weeks", type=int, default=2,
                   help="weeks between rotations; one sleeve rotates each time")
    p.add_argument("--anchor", default="2026-09-15",
                   help="a known rotation date to align the phase to. The "
                        "default is a production-model rotation, so both "
                        "models rotate on the same Tuesdays.")
    p.add_argument("--hold", type=int, default=40,
                   help="session-count hold, used only by the backtest "
                        "reference block; the live book uses the calendar")
    p.add_argument("--start", default="2010-01-01")
    p.add_argument("--min-adv", type=float, default=10e6)
    p.add_argument("--min-price", type=float, default=5.0)
    p.add_argument("--account", type=float, default=100_000.0)
    p.add_argument("--as-of", default=None,
                   help="pretend today is this date (YYYY-MM-DD)")
    p.add_argument("--no-cache", action="store_true",
                   help="force a fresh download; use this for a real run")
    p.add_argument("--no-snapshot", action="store_true",
                   help="skip the pool snapshot. FOR TESTING ONLY -- the "
                        "snapshot is the point-in-time record that closes "
                        "TODO 0d and it cannot be reconstructed later.")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--save-charts", metavar="DIR", default=None,
                   help="write charts as PNGs instead of opening windows")
    p.add_argument("--show", action="store_true",
                   help="with --save-charts, ALSO open the charts (interactive use; "
                        "a scheduled run must not pass this, or it blocks)")
    p.add_argument("--no-context", action="store_true")
    p.add_argument("--no-breadth", action="store_true")
    p.add_argument("--chart-days", type=int, default=252)
    p.add_argument("--show-rebalances", type=int, default=20)
    p.add_argument("--corr-window", type=int, default=50)
    p.add_argument("--corr-threshold", type=float, default=0.70)
    args = p.parse_args()

    if args.save_charts and not args.show:
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

    # --- per-name state as of the signal session ---
    close_now = prices.close.loc[session]
    score_now = scored.loc[session]
    ranks_now = score_now.rank(ascending=False, method="min")
    adv_now = dollar_volume(prices.close, volume, lcfg.adv_window).loc[session]
    slip_now = slippage_panel(prices.close, volume, lcfg,
                              top_n=cfg.top_n).loc[session] * 10_000

    # Daily volatility, for sizing a stop and for reading a drawdown.
    #
    # A move expressed in percent is a different thing for every name: -8% is
    # noise on a 5%/day semiconductor and a thesis break on a 1%/day utility.
    vol_now = realized_vol(prices.close, lcfg.vol_window).loc[session]

    def show(names, title):
        print()
        print(f"  {title}")
        hdr = (f"    {'symbol':<8}{'price':>10}{'score':>8}{'rank':>6}"
               f"{'vol/day':>9}{'$vol (M)':>11}{'est bps':>9}")
        print(hdr)
        print("    " + "-" * (len(hdr) - 4))
        for n in names:
            print(f"    {n:<8}{close_now.get(n, float('nan')):>10.2f}"
                  f"{score_now.get(n, float('nan')):>8.2f}"
                  f"{ranks_now.get(n, float('nan')):>6.0f}"
                  f"{float(vol_now.get(n, float('nan'))):>9.2%}"
                  f"{adv_now.get(n, float('nan'))/1e6:>11.0f}"
                  f"{slip_now.get(n, float('nan')):>9.1f}")

    # --- the rotation calendar ---
    #
    # Calendar, not session count. `hold_days=40` was chosen because 40
    # sessions is eight weeks, but that only holds when no market holiday falls
    # inside the cycle -- each one pushes the rotation a weekday, so it drifted
    # Tue/Mon/Mon/Tue/Mon/Mon. Anchoring to a weekday makes the schedule
    # publishable in advance, and anchoring to a PRODUCTION rotation date makes
    # both models rotate on the same Tuesdays so they can be run and traded
    # together.
    rot = rotation_dates(prices.close.index, weekday=TUESDAY,
                         every_weeks=args.every_weeks,
                         anchor=pd.Timestamp(args.anchor))
    if len(rot) == 0:
        print("\n  *** no rotation dates on this panel ***")
        return 3
    assign = sleeve_assignment(rot, args.tranches)
    sleeves = current_sleeves(rot, args.tranches, as_of=session)

    builder = _PortfolioBuilder(
        scored, price_columns=list(prices.close.columns),
        top_n=cfg.top_n, min_data_days=cfg.min_data_days,
        hold_days=cfg.hold_days, vix_data=prices.vix, vix_config=cfg.vix,
        base_composite_scores=None, velocity_config=cfg.velocity,
        correlation_config=cfg.correlation, graduated_config=cfg.graduated_vix,
        exit_config=None, close=prices.close, rank_offset=cfg.rank_offset,
        rank_offset_scope=cfg.rank_offset_scope,
        monitor_symbols=cfg.monitor_symbols)

    sleeve_books = {}
    for j in sorted(sleeves):
        rec = builder.select_on(sleeves[j])
        sleeve_books[j] = list(rec["Selected_Stocks"]) if rec else []

    from collections import Counter
    counts = Counter()
    for names in sleeve_books.values():
        counts.update(names)
    held = sorted(counts, key=lambda n: (-counts[n], n))

    nxt_date, nxt_sleeve, projected = next_rotation(
        rot, args.tranches, prices.close.index, weekday=TUESDAY,
        every_weeks=args.every_weeks, as_of=session)

    # --- simulate the ACTUAL configuration: k sleeves on the calendar ---
    #
    # Not `build_target_portfolios`, which runs a session-count clock. This
    # rebuilds each sleeve's history from the same rotation calendar the live
    # book uses, so the metrics below describe the thing being traded rather
    # than a neighbouring configuration.
    print("=== SIMULATING THE SLEEVES ===", flush=True)
    slip_sleeve = slippage_panel(prices.close, volume, lcfg,
                                 top_n=cfg.top_n * args.tranches)
    sleeve_rets, sleeve_hold, sleeve_cost = [], [], 0.0
    for j in range(args.tranches):
        mine = [d for d in rot if assign[d] == j]
        picks = {}
        for d in mine:
            rec = builder.select_on(d)
            if rec:
                picks[d] = list(rec["Selected_Stocks"])
        if not picks:
            continue
        tgt = pd.Series({d: picks[d] for d in sorted(picks)}, dtype=object)
        tgt = tgt.reindex(prices.close.index).ffill().dropna()
        r = simulate_portfolio(tgt, prices.close, prices.open_,
                               execution=cfg.execution, sizing=cfg.sizing,
                               scores=scored, slippage_by_symbol=slip_sleeve)
        sleeve_rets.append(r.returns.rename(f"s{j}"))
        sleeve_hold.append(r.holdings)
        sleeve_cost += r.total_cost
        print(f"    sleeve {j}: {len(picks)} rotations, "
              f"{len(r.returns)} days", flush=True)

    if not sleeve_rets:
        print("  *** no sleeve produced a return stream ***")
        return 3

    combined = pd.concat(sleeve_rets, axis=1).mean(axis=1).dropna()
    m = calculate_performance_metrics(
        combined, risk_free_rate=cfg.execution.risk_free_rate)

    print("=== STRATEGY RESULTS ===")
    print(f"  {args.tranches} sleeves, equal capital, rotating every "
          f"{args.every_weeks} weeks on a Tuesday.")
    print("  Levels are inflated by survivorship (TODO 0d) and by the fact that")
    print("  this configuration was selected on the full sample. Read deltas.")
    rows = [
        ("CAGR", "%10.2f%%" % (100 * m["cagr"]), "^ higher",
         "tranched, so this is near the phase MEDIAN rather than one draw"),
        ("Volatility", "%10.2f%%" % (100 * m["volatility"]), "v lower",
         "hotter than the 46-name model: 8 names per sleeve out of ~610"),
        ("Sharpe Ratio", "%10.2f" % m["sharpe_ratio"], "^ higher",
         "the constraint: must not be worse than what it replaces"),
        ("Sortino Ratio", "%10.2f" % m["sortino_ratio"], "^ higher",
         "above Sharpe means the volatility is mostly upside"),
        ("Max Drawdown", "%10.2f%%" % (100 * m["max_drawdown"]), "v smaller",
         "the fortitude number: what holding this will demand of you"),
        ("Calmar Ratio", "%10.2f" % m["calmar_ratio"], "^ higher",
         "CAGR per unit of worst hole"),
        ("Trading Days", "%10d" % m["num_periods"], "n/a",
         "sample size"),
    ]
    for name, val, arrow, why in rows:
        print("  %-14s%s   %s" % (name, val, arrow))
        print("  %-14s%10s   %s" % ("", "", why))
    print("  %-14s%10.2f%%   cumulative slippage across all sleeves"
          % ("Slippage", 100 * sleeve_cost))

    # --- union holdings, for the health monitor and the charts ---
    union = {}
    for hs in sleeve_hold:
        for d, names in hs.items():
            union.setdefault(d, set()).update(names)
    holdings = pd.Series({d: sorted(v) for d, v in sorted(union.items())},
                         dtype=object)

    # --- exports ---
    print("=== EXPORTING RESULTS ===")
    out = REPO_ROOT
    written = []
    try:
        pd.DataFrame({"Date": combined.index,
                      "Portfolio_Return": combined.values}).to_csv(
            out / "trackj_portfolio_performance.csv", index=False)
        written.append("trackj_portfolio_performance.csv")
        pd.DataFrame([{"Date": d, "Sleeve": assign[d],
                       "Holdings": " ".join(
                           (builder.select_on(d) or {}).get(
                               "Selected_Stocks", []))}
                      for d in rot[-24:]]).to_csv(
            out / "trackj_rebalance_history.csv", index=False)
        written.append("trackj_rebalance_history.csv")
        scored.to_csv(out / "trackj_scores.csv")
        written.append("trackj_scores.csv")
        individual_stock_performance(prices.close).to_csv(
            out / "trackj_individual_stock_performance.csv", index=False)
        written.append("trackj_individual_stock_performance.csv")
    except Exception as exc:
        print("  *** export failed: %s: %s ***" % (type(exc).__name__, exc))
        degraded.note("exports", exc)
    missing = [n for n in written if not (out / n).exists()]
    if missing:
        print("  *** %d EXPORTS FAILED ***" % len(missing))
        degraded.note("exports", RuntimeError(str(missing)))
    else:
        print("  %d files exported OK -> %s" % (len(written), out))

    # --- correlation and sector mix of the COMBINED book ---
    print("=== CORRELATION ===")
    try:
        cols = [h for h in held if h in prices.close.columns]
        mats = correlation_matrices(prices.close[cols]) if len(cols) > 1 else {}
        for period, matrix in mats.items():
            matrix.to_csv(out / ("trackj_correlation_%dd.csv" % period))
        if args.corr_window in mats:
            print(summarize_correlations(mats[args.corr_window],
                                         args.corr_window,
                                         threshold=args.corr_threshold))
        print("  %d windows exported for the combined book" % len(mats))
        print("  NOTE: the 0.70 cap is applied WITHIN a sleeve. Across sleeves")
        print("  a correlated pair can reappear, which is the price of")
        print("  staggering and is not a defect.")
        secs = pd.read_csv(REPO_ROOT / POOL_FILE).set_index("symbol")["sector"]
        mix = pd.Series([secs.get(h) for h in held]).dropna().value_counts()
        if len(mix):
            print("  Sector mix of the combined book:")
            for sec, n in mix.items():
                flag = "   <-- MAJORITY" if n / len(held) >= 0.5 else ""
                print("    %-26s%d/%d%s" % (sec, n, len(held), flag))
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
                                 model_returns=combined,
                                 horizon=args.tranches * args.every_weeks * 5))
            if n_pool:
                print("  (A/D computed over %d pool names)" % n_pool)
        except Exception as exc:
            print("(Context skipped: %s: %s)" % (type(exc).__name__, exc))
            degraded.note("context panel", exc)

    # --- health monitor ---
    try:
        hcfg = HealthConfig(risk_free_rate=cfg.execution.risk_free_rate)
        health = compute_health(combined, prices.spy, hcfg)
        state = current_state(health, hcfg)
        print()
        print("=" * 100)
        print(status_line(state, hcfg))
        print(defensive_line(defensive_state(
            defensive_exposure(holdings, hcfg), hcfg), hcfg))
        print("  NOTE: these thresholds were calibrated on the 46-name model.")
        print("  Displayed for this one, not validated for it (TODO 3).")
        print("=" * 100)
    except Exception as exc:
        print("(Health monitor skipped: %s)" % exc)
        degraded.note("health monitor", exc)

    print("\n" + "=" * 100)
    print(f"SET 1 — THE BOOK  ({args.tranches} sleeves, one rotating every "
          f"{args.every_weeks} weeks)")
    print("=" * 100)
    print(f"  Rotation calendar: every {args.every_weeks} weeks on a TUESDAY, "
          f"anchored to {args.anchor}.")
    print(f"  Trade at Wednesday's open. Each sleeve holds "
          f"{args.tranches * args.every_weeks} weeks.")
    if nxt_date is not None:
        print(f"\n  NEXT ROTATION: {nxt_date:%Y-%m-%d} ({nxt_date:%A}) — "
              f"sleeve {nxt_sleeve}"
              f"{'  [projected on the calendar]' if projected else ''}")
        print(f"  Only that sleeve trades. The other "
              f"{args.tranches - 1} are untouched.")

    print(f"\n  Sleeves, oldest book first:")
    for j in sorted(sleeve_books, key=lambda k: sleeves[k]):
        age = int((prices.close.index > sleeves[j]).sum())
        marker = "  <- rotates next" if j == nxt_sleeve else ""
        print(f"    sleeve {j}: selected {sleeves[j]:%Y-%m-%d} "
              f"({age} sessions ago){marker}")
        print(f"              {', '.join(sleeve_books[j]) or '(empty)'}")

    print(f"\n  COMBINED BOOK — {len(held)} names, "
          f"{args.tranches * cfg.top_n} sleeve-slots")
    print(f"  A name held by more than one sleeve carries more weight. That is "
          f"deliberate:\n  names that keep re-qualifying returned 31-35% "
          f"annualised against 14% for\n  single-sleeve names.")
    hdr = (f"    {'symbol':<8}{'sleeves':>9}{'weight':>9}{'$ at 100k':>11}"
           f"{'price':>10}{'rank now':>10}{'vol/day':>9}{'bps':>7}")
    print(hdr)
    print("    " + "-" * (len(hdr) - 4))
    total_slots = args.tranches * cfg.top_n
    for n in held:
        w = counts[n] / total_slots
        print(f"    {n:<8}{counts[n]:>9}{w:>9.1%}"
              f"{args.account * w:>11,.0f}"
              f"{close_now.get(n, float('nan')):>10.2f}"
              f"{ranks_now.get(n, float('nan')):>10.0f}"
              f"{float(vol_now.get(n, float('nan'))):>9.2%}"
              f"{slip_now.get(n, float('nan')):>7.1f}")

    dupes = sum(1 for n in held if counts[n] > 1)
    print(f"\n    {dupes} of {len(held)} names are held by more than one "
          f"sleeve.")

    print("\n" + "=" * 100)
    print("SET 2 — WHAT THE NEXT SLEEVE WOULD BUY, on the latest close")
    print("=" * 100)
    print("  What the rotating sleeve would buy if today were its rotation")
    print("  date. It is NOT today's trade unless today is a rotation date —")
    print("  trading this off-cycle is a different and untested strategy.")
    # The DIVERSIFIED selection, not the raw top-N.
    #
    # Showing the raw ranking here would be actively misleading: with an
    # absolute 0.70 filter running every rebalance, the top of the ranking is
    # routinely rejected for redundancy, and today's raw top 8 is eight
    # semiconductors the model would never buy together. Set 2 has to answer
    # "what would it buy", so it runs the same `select_diversified` the
    # rebalance does.
    from momentum.correlation import RollingCorrelation, select_diversified

    ranked_now = score_now.dropna().sort_values(ascending=False)
    raw_top = list(ranked_now.head(cfg.top_n).index)

    fresh, rejected = raw_top, []
    if cfg.correlation is not None and cfg.correlation.enabled:
        try:
            rc = RollingCorrelation(prices.close, cfg.correlation.window)
            # Exactly what the rotation passes: with the VIX overlay off the
            # builder exempts nothing but `exempt_symbols`.  This used to pass
            # the defensive sleeve, so on a day IAU/SHY/TLT ranked high SET 2
            # could show a book the rotation would not buy (fixed 2026-09-23).
            trace = select_diversified(ranked_now, rc.at(session),
                                       cfg.correlation, cfg.top_n,
                                       exempt=list(cfg.correlation.exempt_symbols))
            fresh = list(trace.selected)
            rejected = list(trace.rejected)
        except Exception as exc:
            print("  (diversified selection unavailable: %s: %s)"
                  % (type(exc).__name__, exc))
            degraded.note("fresh selection", exc)

    show(fresh, "what it would BUY today (rank order, after the %s filter)"
                % ("absolute %.2f" % cfg.correlation.max_correlation
                   if cfg.correlation is not None else "no"))

    skipped = [s for s in raw_top if s not in fresh]
    if skipped:
        print("\n    raw top %d by score: %s" % (cfg.top_n, ", ".join(raw_top)))
        print("    rejected for redundancy: %s" % ", ".join(skipped))
    if rejected:
        print("\n    why each was skipped (correlation with a name already taken):")
        for sym, against, rho in rejected[:10]:
            print("      %-8s vs %-8s  rho %.2f" % (sym, against, rho))

    overlap = set(held) & set(fresh)
    print(f"\n  Overlap with the aligned book: {len(overlap)} of {len(held)}"
          f"{' — ' + ', '.join(sorted(overlap)) if overlap else ''}")

    # --- machine-readable books, for scripts/run_live_combined.py ---
    #
    # Both recommendations James asked for (2026-09-23): the book as of the
    # LAST rotation (what should be held now) and the book as it would be if
    # the next sleeve rotated on today's close (what the model says now).
    try:
        import json
        past_rot = [d for d in rot if d <= session]
        last_rot = past_rot[-1] if past_rot else None
        rotated = next((j for j in sleeves if sleeves[j] == last_rot), None)
        now_counts = Counter(counts)
        if nxt_sleeve is not None and nxt_sleeve in sleeve_books:
            now_counts.subtract(sleeve_books[nxt_sleeve])
            now_counts.update(fresh)
        book = {
            "written": datetime.now().isoformat(timespec="seconds"),
            "as_of": today.isoformat(),
            "signal_session": f"{session:%Y-%m-%d}",
            "anchor": args.anchor, "every_weeks": args.every_weeks,
            "tranches": args.tranches, "top_n": cfg.top_n, "slots": total_slots,
            "last_rotation": {
                "date": f"{last_rot:%Y-%m-%d}" if last_rot is not None else None,
                "sleeve": rotated,
                "bought": sleeve_books.get(rotated, []),
                "sleeves": {str(j): {"selected": f"{sleeves[j]:%Y-%m-%d}",
                                     "names": sleeve_books[j]} for j in sleeve_books},
                "weights": {n: counts[n] / total_slots for n in held},
            },
            "current": {
                "next_rotation": f"{nxt_date:%Y-%m-%d}" if nxt_date is not None else None,
                "next_sleeve": nxt_sleeve, "projected": bool(projected),
                "would_buy": list(fresh),
                "weights": {n: c / total_slots for n, c in now_counts.items() if c > 0},
            },
        }
        live = REPO_ROOT / "live"
        live.mkdir(exist_ok=True)
        (live / "trackj_book.json").write_text(json.dumps(book, indent=2))
        print(f"\n  Books written for the combined report: live/trackj_book.json")
    except Exception as exc:
        print("  *** book export failed: %s: %s ***" % (type(exc).__name__, exc))
        degraded.note("book export", exc)

    # --- the point-in-time record ---
    if args.no_snapshot:
        # Loud, because the cost of skipping is invisible today and permanent.
        # The snapshots are the only asset that can ever answer the
        # survivorship question, and a rotation that goes unrecorded is a hole
        # in that record forever -- there is no way to reconstruct which names
        # were tradable on a past date once the data provider has moved on.
        print()
        print("!" * 100)
        print("  NO POOL SNAPSHOT WRITTEN (--no-snapshot).")
        print("  If this was a real run, that rotation is now permanently")
        print("  missing from the point-in-time record (TODO 0d). Re-run")
        print("  without the flag.")
        print("!" * 100)
    else:
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

            wealth = (1 + combined).cumprod()
            fig, axes = plt.subplots(2, 1, figsize=(12, 8))
            axes[0].plot(wealth.index, wealth.values, linewidth=1.6)
            axes[0].set_title("Track J pullback, %d sleeves - cumulative return"
                              % args.tranches)
            axes[0].set_ylabel("Growth of $1")
            axes[0].set_yscale("log")
            axes[0].grid(alpha=0.3)
            axes[1].plot(combined.index, combined.values * 100,
                         linewidth=0.6, alpha=0.8)
            axes[1].set_title("Daily returns (%)")
            axes[1].grid(alpha=0.3)
            plt.tight_layout()

            held_fig = None
            if held:
                lookback = min(args.chart_days, len(prices.close))
                sub = prices.close[[h for h in held
                                    if h in prices.close.columns]]
                sub = sub.iloc[-lookback:].dropna(axis=1, how="all")
                if not sub.empty:
                    held_fig, ax = plt.subplots(figsize=(13, 7))
                    for sym in sub.columns:
                        s = sub[sym].dropna()
                        if s.empty:
                            continue
                        lw = 1.0 + 0.9 * (counts[sym] - 1)
                        ax.plot(s.index, s / s.iloc[0] * 100.0, linewidth=lw,
                                label="%s x%d  %+.1f%%"
                                      % (sym, counts[sym],
                                         100 * (s.iloc[-1] / s.iloc[0] - 1)))
                    spy_w = prices.spy.reindex(sub.index).ffill().dropna()
                    if not spy_w.empty:
                        ax.plot(spy_w.index, spy_w / spy_w.iloc[0] * 100.0,
                                linewidth=2.2, linestyle="--", color="black",
                                alpha=0.6, label="SPY")
                    ax.axhline(100, color="grey", linewidth=0.8, alpha=0.5)
                    ax.set_title("Combined book, last %d sessions - rebased to "
                                 "100 (line width = sleeves holding it)"
                                 % lookback)
                    ax.grid(alpha=0.3)
                    ax.legend(loc="best", fontsize=7, ncol=2)
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
            if not args.save_charts or args.show:
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
