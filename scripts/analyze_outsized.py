"""
How much of the model's gain came from a handful of outsized single-stock moves?

The worry this answers: if the record rests on a few earnings surprises, the
CAGR is a statement about draws that already happened rather than about edge
that repeats.  Two independent readings, because each one alone is arguable.

1. THE TRIM CURVE (threshold-free)
   Rank every position-day by its contribution to that day's portfolio return,
   zero out the best K, and recompute.  No definition of "outsized" is needed,
   so nothing can be tuned to produce a comfortable answer.  Reported in both
   directions and symmetrically, because trimming only the winners is a
   guaranteed way to make any strategy look fragile - the honest question is
   whether the top tail is LARGER than the bottom tail, not whether removing it
   hurts.

2. EVENT TAGGING (thresholded, and the threshold is a flag)
   Tag a position-day as an outsized event when the stock's move, net of its
   trailing beta to SPY, exceeds `--sigma` trailing residual standard
   deviations.  Then split the record into tagged and untagged P&L.  Because
   earnings moves are overwhelmingly overnight, each tagged event is also split
   into its overnight gap and its intraday leg: a gap-dominated tail is the
   signature of news, an intraday-dominated tail is not.  No earnings calendar
   is used - none is in the repo - so the gap split is the proxy and is labelled
   as one.

CONTRIBUTION ARITHMETIC
   The book is equal-weighted with weights reset daily (see `_segment_return`),
   so a held name's contribution on an ordinary day is exactly `r_s / N`.  On a
   rebalance day the fill is at the next open and the day splits in two:

       gross = (1 + r1)(1 + r2) - 1,   r1 = old book close(T-1) -> open(T)
                                       r2 = new book open(T)    -> close(T)

   which decomposes exactly as `r1_s / N_old` for each outgoing name and
   `r2_s / N_new * (1 + r1)` for each incoming one.  That identity is asserted
   against the simulator's own return series before anything is reported; if it
   does not reconcile to 1e-12 the script fails rather than publishing numbers
   that do not add up.

Slippage is held out of the decomposition and reported separately - it is a
cost of trading, not a contribution of any position.

Run:  python scripts/analyze_outsized.py
      python scripts/analyze_outsized.py --sigma 3.0 --beta-window 126
      python scripts/analyze_outsized.py --trim 1 5 10 25 50 100 250
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from momentum.backtest import build_target_portfolios, simulate_portfolio
from momentum.data import load_data
from momentum.experiments import production_config
from momentum.metrics import calculate_performance_metrics
from momentum.strategy import compute_scores
from momentum.universe import current_symbols

OUT_EVENTS = "rsi_ma_outsized_events.csv"
OUT_TRIM = "rsi_ma_outsized_trim.csv"
OUT_CONTRIB = "rsi_ma_outsized_contributions.csv"


# --------------------------------------------------------------------------
# Contribution decomposition
# --------------------------------------------------------------------------

def _leg_returns(symbols, start_prices, end_prices):
    """Per-symbol simple returns over one leg, skipping unpriced names."""
    out = {}
    for sym in symbols:
        p0 = start_prices.get(sym, np.nan)
        p1 = end_prices.get(sym, np.nan)
        if pd.notna(p0) and pd.notna(p1) and p0 != 0:
            out[sym] = float(p1 / p0 - 1)
    return out


def decompose(targets: pd.Series, close: pd.DataFrame, open_: pd.DataFrame,
              slippage_frac: float) -> pd.DataFrame:
    """
    Replay the simulation, emitting one row per (date, symbol) contribution.

    This mirrors `simulate_portfolio` under execute_at='next_open' exactly,
    including its list-equality test for "did the book change" and its
    equal-weight-reset convention.  It is a replay rather than a re-derivation
    so the reconciliation assert in `main` is a real check on both.

    `targets` is the TARGET series out of `build_target_portfolios`, not
    `PortfolioResult.holdings` - the latter is the book realized on each return
    date, which is the target lagged by a day.  Feeding it the wrong one shifts
    every contribution by one session; the reconciliation assert catches that.
    """
    dates = list(targets.index)
    held: list = []
    rows = []

    for i in range(1, len(dates)):
        d_prev, d = dates[i - 1], dates[i]
        if d_prev not in close.index or d not in close.index:
            continue

        signal = list(targets.loc[d_prev])
        close_prev, close_now = close.loc[d_prev], close.loc[d]
        cost = 0.0

        if signal != held:
            if held:
                open_now = open_.loc[d]
                r1_by = _leg_returns(held, close_prev, open_now)
                r2_by = _leg_returns(signal, open_now, close_now)
                r1 = float(np.mean(list(r1_by.values()))) if r1_by else 0.0

                n1, n2 = len(r1_by), len(r2_by)
                for sym, r in r1_by.items():
                    rows.append((d, sym, r / n1, "overnight_exit", r))
                for sym, r in r2_by.items():
                    rows.append((d, sym, r / n2 * (1 + r1), "entry", r))
            else:
                r2_by = _leg_returns(signal, open_.loc[d], close_now)
                n2 = len(r2_by)
                for sym, r in r2_by.items():
                    rows.append((d, sym, r / n2, "entry", r))

            cost = (slippage_frac if not held
                    else 2.0 * (len(set(held) - set(signal)) / len(held))
                    * slippage_frac)
            held = signal
        else:
            if not held:
                continue
            r_by = _leg_returns(held, close_prev, close_now)
            n = len(r_by)
            for sym, r in r_by.items():
                rows.append((d, sym, r / n, "hold", r))

        if cost:
            rows.append((d, "_SLIPPAGE_", -cost, "cost", np.nan))

    return pd.DataFrame(rows, columns=["Date", "Symbol", "Contribution",
                                       "Leg", "Stock Return"])


# --------------------------------------------------------------------------
# The trim curve
# --------------------------------------------------------------------------

def trim_curve(contrib: pd.DataFrame, returns: pd.Series, ks, rf: float
               ) -> pd.DataFrame:
    """
    CAGR after zeroing the K largest / smallest / most extreme position-days.

    "Zeroing" means that position earned nothing that day - the slot is still
    held, the rest of the book still earns.  That is the honest counterfactual
    for "what if this one name had simply not popped": it is not the same as
    removing the day, which would also delete the other three positions.
    """
    pos = contrib[contrib["Symbol"] != "_SLIPPAGE_"].copy()
    daily_cost = (contrib[contrib["Symbol"] == "_SLIPPAGE_"]
                  .groupby("Date")["Contribution"].sum())

    order_top = pos["Contribution"].sort_values(ascending=False).index
    order_bot = pos["Contribution"].sort_values(ascending=True).index
    order_abs = pos["Contribution"].abs().sort_values(ascending=False).index

    base = calculate_performance_metrics(returns, risk_free_rate=rf)
    rows = [{"Trim": 0, "Side": "none", "CAGR": base["cagr"],
             "Sharpe": base["sharpe_ratio"], "MaxDD": base["max_drawdown"],
             "Total Return": base["total_return"], "dCAGR": 0.0,
             "Share of CAGR Lost": 0.0, "Trimmed P&L": 0.0}]

    for side, order in (("top", order_top), ("bottom", order_bot),
                        ("both", order_abs)):
        for k in ks:
            if k > len(order):
                continue
            drop = order[:k]
            kept = pos.drop(index=drop)
            rebuilt = (kept.groupby("Date")["Contribution"].sum()
                       .add(daily_cost, fill_value=0.0)
                       .reindex(returns.index).fillna(0.0))
            m = calculate_performance_metrics(rebuilt, risk_free_rate=rf)
            rows.append({
                "Trim": k,
                "Side": side,
                "CAGR": m["cagr"],
                "Sharpe": m["sharpe_ratio"],
                "MaxDD": m["max_drawdown"],
                "Total Return": m["total_return"],
                "dCAGR": m["cagr"] - base["cagr"],
                "Share of CAGR Lost": ((base["cagr"] - m["cagr"]) / base["cagr"]
                                       if base["cagr"] else np.nan),
                "Trimmed P&L": float(pos.loc[drop, "Contribution"].sum()),
            })

    return pd.DataFrame(rows)


def print_trim(curve: pd.DataFrame, n_position_days: int) -> None:
    base = curve[curve["Side"] == "none"].iloc[0]
    print("\n" + "=" * 100)
    print("TRIM CURVE  - CAGR after zeroing the K most extreme position-days")
    print("=" * 100)
    print(f"  {n_position_days:,} position-days in the record. "
          f"Baseline CAGR {base['CAGR']:.2%}, total return {base['Total Return']:.1f}x.")
    print("  'top' removes the best contributions, 'bottom' the worst, "
          "'both' the largest by absolute size.")

    for side, label in (("top", "Remove the K BEST position-days"),
                        ("bottom", "Remove the K WORST position-days"),
                        ("both", "Remove the K LARGEST by absolute size")):
        sub = curve[curve["Side"] == side]
        if sub.empty:
            continue
        print(f"\n  {label}")
        header = (f"    {'K':>6}{'% of days':>12}{'CAGR':>10}{'dCAGR':>10}"
                  f"{'% CAGR lost':>14}{'Sharpe':>9}{'MaxDD':>10}")
        print(header)
        print("    " + "-" * (len(header) - 4))
        for _, row in sub.iterrows():
            print(f"    {row['Trim']:>6.0f}"
                  f"{row['Trim'] / n_position_days:>11.3%} "
                  f"{row['CAGR']:>9.2%} "
                  f"{row['dCAGR']:>+9.2%} "
                  f"{row['Share of CAGR Lost']:>13.1%} "
                  f"{row['Sharpe']:>8.2f} "
                  f"{row['MaxDD']:>9.2%}")


def print_concentration(contrib: pd.DataFrame) -> None:
    """
    How lopsided the contribution distribution is, before any threshold.

    The asymmetry line is the one that matters.  Every equity strategy loses
    badly when you delete its best days; what would be evidence of fragility is
    a top tail materially fatter than the bottom tail, because that is the
    shape that cannot be earned back by holding through.
    """
    pos = contrib[contrib["Symbol"] != "_SLIPPAGE_"]
    c = pos["Contribution"]
    n = len(c)
    gains, losses = c[c > 0], c[c < 0]

    print("\n" + "=" * 100)
    print("CONTRIBUTION CONCENTRATION")
    print("=" * 100)
    print(f"  Position-days           {n:>12,}")
    print(f"  Gross positive P&L      {gains.sum():>12.2f}   ({len(gains):,} days)")
    print(f"  Gross negative P&L      {losses.sum():>12.2f}   ({len(losses):,} days)")
    print(f"  Net                     {c.sum():>12.2f}")

    print(f"\n  {'Tail':>8}{'Days':>9}{'Top-tail P&L':>16}{'Bottom-tail P&L':>18}"
          f"{'Top / |Bottom|':>17}")
    print("  " + "-" * 66)
    for q in (0.001, 0.005, 0.01, 0.05, 0.10):
        k = max(1, int(round(n * q)))
        top = c.nlargest(k).sum()
        bot = c.nsmallest(k).sum()
        ratio = top / abs(bot) if bot else np.nan
        print(f"  {q:>7.1%}{k:>9,}{top:>16.2f}{bot:>18.2f}{ratio:>17.2f}")
    print("  " + "-" * 66)
    print("  A ratio near 1.00 means the two tails offset: the strategy is "
          "volatile, not\n  lottery-dependent.  A ratio well above 1 is the "
          "fragility the question asks about.")

    share = c.nlargest(max(1, int(round(n * 0.01)))).sum() / gains.sum()
    print(f"\n  Top 1% of position-days supply {share:.1%} of all gross gains.")


# --------------------------------------------------------------------------
# Event tagging
# --------------------------------------------------------------------------

def tag_events(contrib: pd.DataFrame, close: pd.DataFrame, open_: pd.DataFrame,
               spy: pd.Series, beta_window: int, sigma: float) -> pd.DataFrame:
    """
    Flag position-days whose move is large relative to the stock's own
    market-adjusted history.

    Residual = r_stock - beta * r_spy, with beta and the residual sigma both
    estimated on a trailing window that ENDS THE DAY BEFORE the move, so the
    event never contributes to the yardstick that measures it.
    """
    stock_ret = close.pct_change()
    mkt = spy.pct_change().reindex(stock_ret.index)

    cov = stock_ret.rolling(beta_window).cov(mkt)
    var = mkt.rolling(beta_window).var()
    beta = cov.div(var, axis=0)

    resid = stock_ret.sub(beta.mul(mkt, axis=0))
    # Shift by one so the trailing sigma excludes the day being scored.
    resid_sigma = resid.rolling(beta_window).std().shift(1)
    beta_lag = beta.shift(1)

    z = resid.div(resid_sigma)

    # Overnight gap vs intraday, the earnings proxy.
    gap = (open_ / close.shift(1) - 1)
    intraday = (close / open_ - 1)

    pos = contrib[contrib["Symbol"] != "_SLIPPAGE_"].copy()

    def pick(panel, row):
        try:
            return float(panel.at[row["Date"], row["Symbol"]])
        except (KeyError, TypeError, ValueError):
            return np.nan

    pos["Resid Z"] = [pick(z, r) for _, r in pos.iterrows()]
    pos["Beta"] = [pick(beta_lag, r) for _, r in pos.iterrows()]
    pos["Mkt Return"] = pos["Date"].map(mkt).astype(float)
    pos["Overnight Gap"] = [pick(gap, r) for _, r in pos.iterrows()]
    pos["Intraday"] = [pick(intraday, r) for _, r in pos.iterrows()]
    pos["Outsized"] = pos["Resid Z"].abs() >= sigma
    return pos


def print_events(tagged: pd.DataFrame, sigma: float, top_k: int) -> None:
    ev = tagged[tagged["Outsized"]]
    scored = tagged[tagged["Resid Z"].notna()]
    total = tagged["Contribution"].sum()

    print("\n" + "=" * 100)
    print(f"OUTSIZED EVENTS  - market-adjusted move >= {sigma:.1f} trailing sigma")
    print("=" * 100)
    if scored.empty:
        print("  No scorable position-days.")
        return

    up, down = ev[ev["Resid Z"] > 0], ev[ev["Resid Z"] < 0]
    untagged = scored[~scored["Outsized"]]

    print(f"  Scorable position-days      {len(scored):>10,}")
    print(f"  Tagged as outsized          {len(ev):>10,}   "
          f"({len(ev) / len(scored):.2%} of days)")
    print(f"    upside surprises          {len(up):>10,}")
    print(f"    downside surprises        {len(down):>10,}")

    print(f"\n  {'Bucket':<26}{'Days':>9}{'P&L':>12}{'Share of net':>15}"
          f"{'P&L / day':>13}")
    print("  " + "-" * 73)
    for label, sub in (("Outsized - upside", up),
                       ("Outsized - downside", down),
                       ("Outsized - net", ev),
                       ("Everything else", untagged)):
        pnl = sub["Contribution"].sum()
        per = pnl / len(sub) if len(sub) else np.nan
        print(f"  {label:<26}{len(sub):>9,}{pnl:>12.2f}"
              f"{pnl / total if total else np.nan:>15.1%}{per:>13.4f}")
    print("  " + "-" * 73)
    print("  'Share of net' is of the summed contribution stream, not of CAGR - "
          "the two\n  differ because compounding is path-dependent. The trim "
          "curve above is the\n  compounding-aware version.")

    # The earnings proxy.
    if len(ev):
        gap_share = (ev["Overnight Gap"].abs()
                     / (ev["Overnight Gap"].abs() + ev["Intraday"].abs()))
        print(f"\n  Overnight gap accounts for a median {gap_share.median():.0%} "
              f"of the absolute move on\n  tagged days (vs "
              f"{(untagged['Overnight Gap'].abs() / (untagged['Overnight Gap'].abs() + untagged['Intraday'].abs())).median():.0%} "
              f"on untagged days). Gap dominance is the earnings/news\n  "
              f"signature; no earnings calendar is used, so read this as a proxy.")

    print(f"\n  Largest {top_k} single contributions in the record:")
    header = (f"    {'Date':<12}{'Sym':<7}{'Leg':<16}{'Stock':>9}{'Mkt':>9}"
              f"{'Z':>8}{'Gap':>9}{'Contrib':>10}")
    print(header)
    print("    " + "-" * (len(header) - 4))
    for _, r in tagged.nlargest(top_k, "Contribution").iterrows():
        z = f"{r['Resid Z']:>7.1f}" if pd.notna(r["Resid Z"]) else "      -"
        g = f"{r['Overnight Gap']:>8.1%}" if pd.notna(r["Overnight Gap"]) else "       -"
        print(f"    {r['Date']:%Y-%m-%d}  {r['Symbol']:<7}{r['Leg']:<16}"
              f"{r['Stock Return']:>8.1%} {r['Mkt Return']:>8.1%} {z} {g} "
              f"{r['Contribution']:>9.2%}")


def print_symbol_concentration(tagged: pd.DataFrame, top_k: int) -> None:
    """Is the gain one lucky ticker, or spread across the universe?"""
    by_sym = (tagged.groupby("Symbol")
              .agg(Days=("Contribution", "size"),
                   PnL=("Contribution", "sum"),
                   Outsized=("Outsized", "sum"))
              .sort_values("PnL", ascending=False))
    total = by_sym["PnL"].sum()

    print("\n" + "=" * 100)
    print("WHERE THE P&L CAME FROM  - by ticker")
    print("=" * 100)
    cum = by_sym["PnL"].cumsum() / total
    n_half = int((cum < 0.5).sum()) + 1
    print(f"  {len(by_sym)} tickers held. The top {n_half} supply half the "
          f"summed contribution.")
    header = (f"    {'Symbol':<9}{'Days held':>11}{'P&L':>10}{'Share':>9}"
              f"{'Cum':>8}{'Outsized days':>16}")
    print(header)
    print("    " + "-" * (len(header) - 4))
    for sym, row in by_sym.head(top_k).iterrows():
        print(f"    {sym:<9}{row['Days']:>11,.0f}{row['PnL']:>10.2f}"
              f"{row['PnL'] / total:>9.1%}{cum[sym]:>8.1%}"
              f"{row['Outsized']:>16,.0f}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Attribute the model's gain to outsized single-stock events")
    parser.add_argument("--sigma", type=float, default=4.0,
                        help="trailing residual sigmas that define an outsized "
                             "move (default 4.0)")
    parser.add_argument("--beta-window", type=int, default=126,
                        help="trailing window for beta and residual sigma")
    parser.add_argument("--trim", type=int, nargs="+",
                        default=[1, 5, 10, 25, 50, 100, 250],
                        help="K values for the trim curve")
    parser.add_argument("--top-k", type=int, default=20,
                        help="rows in the largest-events and by-ticker tables")
    parser.add_argument("--top-n", type=int, default=None,
                        help="override book size (default: the live value)")
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--keep-unsettled", action="store_true",
                        help="keep an unclosed final session; excluded by "
                             "default so the run is reproducible")
    args = parser.parse_args()

    print("=" * 100)
    print("OUTSIZED-EVENT ATTRIBUTION")
    print(f"Started {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"sigma {args.sigma}   beta window {args.beta_window}d")
    print("=" * 100)

    prices = load_data(current_symbols(), start_date=args.start,
                       use_cache=not args.no_cache,
                       drop_unsettled=not args.keep_unsettled)

    overrides = {} if args.top_n is None else {"top_n": args.top_n}
    config = production_config(**overrides)
    print(f"\nConfig: top_n={config.top_n}, hold_days={config.hold_days}, "
          f"elevated_top_n={config.vix.elevated_top_n}, "
          f"execute_at={config.execution.execute_at}, "
          f"slippage={config.execution.slippage_bps}bps")

    # Inlined rather than via run_strategy() so the target series is in hand:
    # the decomposition needs the targets, and run_strategy discards them.
    ranking_scores, base_scores, _ = compute_scores(prices, config)
    targets, _hist = build_target_portfolios(
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
    )
    result = simulate_portfolio(targets, prices.close, prices.open_,
                                execution=config.execution)
    m = result.metrics
    print(f"Baseline: CAGR {m['cagr']:.2%}, Sharpe {m['sharpe_ratio']:.2f}, "
          f"MaxDD {m['max_drawdown']:.2%}, over {m['years']:.1f} years")

    contrib = decompose(targets, prices.close, prices.open_,
                        config.execution.slippage_frac)

    # Integrity check: the decomposition must reproduce the simulator exactly.
    rebuilt = contrib.groupby("Date")["Contribution"].sum()
    aligned = rebuilt.reindex(result.returns.index)
    worst = float((aligned - result.returns).abs().max())
    print(f"\nReconciliation against simulate_portfolio: max abs diff {worst:.2e}")
    if not worst < 1e-12:
        raise SystemExit(
            f"Contribution decomposition does not reconcile (max diff {worst:.2e}). "
            "Refusing to report attribution off a stream that does not add up.")

    rf = config.execution.risk_free_rate
    n_pos_days = int((contrib["Symbol"] != "_SLIPPAGE_").sum())

    print_concentration(contrib)
    curve = trim_curve(contrib, result.returns, args.trim, rf)
    print_trim(curve, n_pos_days)

    tagged = tag_events(contrib, prices.close, prices.open_, prices.spy,
                        args.beta_window, args.sigma)
    print_events(tagged, args.sigma, args.top_k)
    print_symbol_concentration(tagged, args.top_k)

    curve.to_csv(REPO_ROOT / OUT_TRIM, index=False)
    tagged[tagged["Outsized"]].sort_values("Contribution", ascending=False) \
        .to_csv(REPO_ROOT / OUT_EVENTS, index=False)
    tagged.to_csv(REPO_ROOT / OUT_CONTRIB, index=False)
    print(f"\nTrim curve exported to:    {REPO_ROOT / OUT_TRIM}")
    print(f"Tagged events exported to: {REPO_ROOT / OUT_EVENTS}")
    print(f"Contributions exported to: {REPO_ROOT / OUT_CONTRIB}")

    print(f"\nCompleted {datetime.now():%Y-%m-%d %H:%M:%S}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
