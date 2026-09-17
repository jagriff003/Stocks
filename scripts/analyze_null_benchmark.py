"""
What is the ranker actually worth?  A null-portfolio benchmark.

Every result in FINDINGS is measured against SPY.  SPY answers a decision
question - should this be run instead of indexing - and it answers it well.  It
cannot answer an attribution question, because the universe is 52 names screened
*today* on a momentum filter and applied backwards to 2010.  Some of the return
belongs to that universe and would arrive without any ranking at all.  Nothing
in this repo has ever separated the two.

This does, with four arms:

    arm                selection   overlay   isolates
    ---------------------------------------------------------------
    ranked_overlay     ranked      on        the model as it stands
    random_overlay     random      on        what the overlay alone earns
    ranked_plain       ranked      off       the ranker alone
    random_plain       random      off       the universe alone

`random_plain` is the real baseline.  `ranked_overlay` minus `random_overlay` is
what the ranking is worth on top of the overlay it already has.  Two reference
rows are carried alongside: SPY buy-and-hold (the decision bar, kept because it
answers a question the arms do not) and an equal-weighted hold of the whole
momentum universe (the most literal statement of "what did picking get you").

HOW THE NULL IS CONSTRUCTED
   The random arms replace the *ranking scores* with iid normal draws carrying
   the same NaN pattern, then run the unchanged `build_target_portfolios`.
   Everything else is held identical: the history gate reads `notna().cumsum()`
   on that same pattern, the `min_level_threshold` floor still reads the REAL
   level scores, and the hold clock, regime overlay, correlation filter and
   defensive fill are untouched.  So the random arms draw from exactly the
   eligible pool the ranker draws from, and differ only in which member of that
   pool they take.  That is the comparison worth making; drawing from all 52
   ignoring eligibility would be a weaker null and a softer test.

WHY GROSS RETURNS ARE REPORTED TOO
   An iid draw has no persistence, so a random book rotates close to 100% every
   cycle where the ranked book frequently re-picks a name it already holds.  The
   random arms therefore pay materially more slippage *for being random*, which
   would flatter the model for reasons unrelated to skill.  Gross CAGR removes
   that confound entirely; net CAGR is what either would actually earn.  Read
   both, and read the turnover column next to them.

Every threshold is a flag.

Run:  python scripts/analyze_null_benchmark.py
      python scripts/analyze_null_benchmark.py --trials 1000 --seed 7
      python scripts/analyze_null_benchmark.py --trials 100     # quick look
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
from momentum.universe import current_symbols, defensive_symbols

OUT_TRIALS = "rsi_ma_null_trials.csv"
OUT_SUMMARY = "rsi_ma_null_summary.csv"


def make_targets(scores, prices, cfg, base_scores):
    """`build_target_portfolios` under `cfg`, for whatever scores are handed in."""
    return build_target_portfolios(
        scores,
        price_columns=list(prices.close.columns),
        top_n=cfg.top_n,
        min_data_days=cfg.min_data_days,
        hold_days=cfg.hold_days,
        vix_data=prices.vix,
        vix_config=cfg.vix,
        base_composite_scores=base_scores,
        velocity_config=cfg.velocity,
        correlation_config=cfg.correlation,
        graduated_config=cfg.graduated_vix,
        exit_config=cfg.exits,
        close=prices.close,
        rank_offset=cfg.rank_offset,
        rank_offset_scope=cfg.rank_offset_scope,
    )[0]


def run_arm(scores, prices, cfg, base_scores):
    """One simulation, returned as a flat metrics dict."""
    targets = make_targets(scores, prices, cfg, base_scores)
    r = simulate_portfolio(targets, prices.close, prices.open_,
                           execution=cfg.execution)
    gross = calculate_performance_metrics(
        r.gross_returns, risk_free_rate=cfg.execution.risk_free_rate)
    return {
        "CAGR": r.metrics["cagr"],
        "Gross CAGR": gross["cagr"],
        "Sharpe": r.metrics["sharpe_ratio"],
        "MaxDD": r.metrics["max_drawdown"],
        "Calmar": r.metrics["calmar_ratio"],
        "Volatility": r.metrics["volatility"],
        "Trades/Year": r.turnover["trades_per_year"],
        "Annual Turnover": r.turnover["annual_turnover"],
    }


def random_scores(template: pd.DataFrame, rng) -> pd.DataFrame:
    """
    An iid score panel with the template's exact NaN pattern.

    Preserving the pattern is what keeps the eligibility gate identical: the
    history counter is `notna().cumsum()` over this very panel, so a different
    pattern would silently change which names are eligible on which dates and
    the null would no longer be comparable.
    """
    draw = rng.standard_normal(template.shape)
    return pd.DataFrame(draw, index=template.index,
                        columns=template.columns).where(template.notna())


def buy_and_hold_universe(prices, cfg, base_scores, ranking_scores, symbols):
    """
    Equal-weighted hold of the whole momentum universe on the same clock.

    Implemented through the same simulator so fills, costs and the return
    convention match every other row.  The book is every named symbol that has
    a score on the date, so it grows as names come into history - which is the
    honest version of "own the universe" given the panel starts staggered.
    """
    held = [s for s in symbols if s in ranking_scores.columns]
    sub = ranking_scores[held]
    dates = ranking_scores.index[cfg.min_data_days:]
    targets = pd.Series(
        {d: sorted(sub.columns[sub.loc[d].notna()].tolist()) for d in dates},
        name="target")
    targets = targets[targets.apply(len) > 0]
    r = simulate_portfolio(targets, prices.close, prices.open_,
                           execution=cfg.execution)
    gross = calculate_performance_metrics(
        r.gross_returns, risk_free_rate=cfg.execution.risk_free_rate)
    return {
        "CAGR": r.metrics["cagr"], "Gross CAGR": gross["cagr"],
        "Sharpe": r.metrics["sharpe_ratio"], "MaxDD": r.metrics["max_drawdown"],
        "Calmar": r.metrics["calmar_ratio"],
        "Volatility": r.metrics["volatility"],
        "Trades/Year": r.turnover["trades_per_year"],
        "Annual Turnover": r.turnover["annual_turnover"],
    }, r.returns.index


def spy_buy_and_hold(prices, cfg, index) -> dict:
    """SPY over the model's own date range - the decision bar, kept on purpose."""
    spy = prices.spy.reindex(index).ffill()
    rets = spy.pct_change().dropna()
    m = calculate_performance_metrics(
        rets, risk_free_rate=cfg.execution.risk_free_rate)
    return {
        "CAGR": m["cagr"], "Gross CAGR": m["cagr"],
        "Sharpe": m["sharpe_ratio"], "MaxDD": m["max_drawdown"],
        "Calmar": m["calmar_ratio"], "Volatility": m["volatility"],
        "Trades/Year": 0.0, "Annual Turnover": 0.0,
    }


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

METRICS = ["CAGR", "Gross CAGR", "Sharpe", "MaxDD", "Calmar", "Volatility",
           "Trades/Year", "Annual Turnover"]


def distribution_row(name: str, trials: pd.DataFrame) -> dict:
    row = {"Arm": name, "Trials": len(trials)}
    for m in METRICS:
        row[m] = trials[m].mean()
        row[f"{m} sd"] = trials[m].std()
    return row


def percentile_of(value: float, sample: pd.Series) -> float:
    return float((sample < value).mean())


def print_reference(rows: list) -> None:
    print("\n" + "=" * 100)
    print("REFERENCE ROWS")
    print("=" * 100)
    header = (f"{'Row':<26}{'CAGR':>9}{'Gross':>9}{'Sharpe':>9}{'MaxDD':>10}"
              f"{'Calmar':>9}{'Vol':>9}{'Trd/Yr':>9}")
    print(header)
    print("-" * len(header))
    for name, m in rows:
        print(f"{name:<26}{m['CAGR']:>8.2%} {m['Gross CAGR']:>8.2%} "
              f"{m['Sharpe']:>8.2f} {m['MaxDD']:>9.2%} {m['Calmar']:>8.2f} "
              f"{m['Volatility']:>8.2%} {m['Trades/Year']:>8.1f}")


def print_arms(live: dict, plain: dict, dists: dict, trials: dict) -> None:
    print("\n" + "=" * 100)
    print("THE FOUR ARMS")
    print("=" * 100)
    header = (f"{'Arm':<22}{'CAGR':>9}{'Gross':>9}{'Sharpe':>9}{'MaxDD':>10}"
              f"{'Trd/Yr':>9}{'Turnover':>11}")
    print(header)
    print("-" * len(header))

    def line(label, m, sd=None):
        s = (f"{label:<22}{m['CAGR']:>8.2%} {m['Gross CAGR']:>8.2%} "
             f"{m['Sharpe']:>8.2f} {m['MaxDD']:>9.2%} "
             f"{m['Trades/Year']:>8.1f} {m['Annual Turnover']:>10.1%}")
        print(s)
        if sd is not None:
            print(f"{'  +/- 1 sd':<22}{sd['CAGR']:>8.2%} {sd['Gross CAGR']:>8.2%} "
                  f"{sd['Sharpe']:>8.2f} {sd['MaxDD']:>9.2%}")

    line("ranked_overlay", live)
    line("random_overlay", dists["random_overlay"],
         {k: dists["random_overlay"][f"{k} sd"] for k in METRICS})
    line("ranked_plain", plain)
    line("random_plain", dists["random_plain"],
         {k: dists["random_plain"][f"{k} sd"] for k in METRICS})
    print("-" * len(header))
    print("  random arms are the MEAN over trials; the sd line is the spread "
          "across draws.")

    print("\n" + "=" * 100)
    print("WHERE THE LIVE MODEL SITS IN THE NULL DISTRIBUTION")
    print("=" * 100)
    for arm, ref, label in (
            ("random_overlay", live, "ranked_overlay vs random_overlay"),
            ("random_plain", plain, "ranked_plain   vs random_plain")):
        t = trials[arm]
        for metric in ("CAGR", "Gross CAGR", "Sharpe"):
            pct = percentile_of(ref[metric], t[metric])
            bar = "#" * int(round(pct * 40))
            print(f"  {label:<34}{metric:<12}{pct:>7.1%}  |{bar:<40}|")
        print()

    print("=" * 100)
    print("WHAT EACH COMPONENT IS WORTH  (CAGR, net and gross)")
    print("=" * 100)
    pairs = [
        ("Ranking, with the overlay on",
         live["CAGR"] - dists["random_overlay"]["CAGR"],
         live["Gross CAGR"] - dists["random_overlay"]["Gross CAGR"]),
        ("Ranking, with the overlay off",
         plain["CAGR"] - dists["random_plain"]["CAGR"],
         plain["Gross CAGR"] - dists["random_plain"]["Gross CAGR"]),
        ("The overlay, on ranked picks",
         live["CAGR"] - plain["CAGR"],
         live["Gross CAGR"] - plain["Gross CAGR"]),
        ("The overlay, on random picks",
         dists["random_overlay"]["CAGR"] - dists["random_plain"]["CAGR"],
         dists["random_overlay"]["Gross CAGR"] - dists["random_plain"]["Gross CAGR"]),
    ]
    print(f"  {'Component':<36}{'net':>10}{'gross':>10}")
    print("  " + "-" * 54)
    for label, net, gross in pairs:
        print(f"  {label:<36}{net:>+9.2%}{gross:>+9.2%}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Null-portfolio benchmark: what is the ranker worth?")
    parser.add_argument("--trials", type=int, default=500,
                        help="random draws per random arm (default 500)")
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--top-n", type=int, default=None,
                        help="override book size (default: the live value)")
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--keep-unsettled", action="store_true")
    args = parser.parse_args()

    print("=" * 100)
    print("NULL-PORTFOLIO BENCHMARK")
    print(f"Started {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"{args.trials} trials per random arm, seed {args.seed}")
    print("=" * 100)

    prices = load_data(current_symbols(), start_date=args.start,
                       use_cache=not args.no_cache,
                       drop_unsettled=not args.keep_unsettled)

    overrides = {} if args.top_n is None else {"top_n": args.top_n}
    cfg = production_config(**overrides)
    cfg_plain = production_config(vix=None, **overrides)
    print(f"\nConfig: top_n={cfg.top_n}, hold_days={cfg.hold_days}, "
          f"execute_at={cfg.execution.execute_at}, "
          f"slippage={cfg.execution.slippage_bps}bps")

    ranking_scores, base_scores, _ = compute_scores(prices, cfg)

    live = run_arm(ranking_scores, prices, cfg, base_scores)
    plain = run_arm(ranking_scores, prices, cfg_plain, base_scores)
    print(f"\nranked_overlay  CAGR {live['CAGR']:.2%}  "
          f"(gross {live['Gross CAGR']:.2%})")
    print(f"ranked_plain    CAGR {plain['CAGR']:.2%}  "
          f"(gross {plain['Gross CAGR']:.2%})")

    defensive = set(defensive_symbols())
    momentum_syms = [s for s in current_symbols() if s not in defensive]
    ew, ew_index = buy_and_hold_universe(prices, cfg, base_scores,
                                         ranking_scores, momentum_syms)
    spy = spy_buy_and_hold(prices, cfg, ew_index)

    rng = np.random.default_rng(args.seed)
    records = {"random_overlay": [], "random_plain": []}
    print(f"\nRunning {args.trials} null trials per arm "
          f"(~{args.trials * 2 * 1.6 / 60:.0f} min)...")
    for i in range(args.trials):
        rnd = random_scores(ranking_scores, rng)
        records["random_overlay"].append(run_arm(rnd, prices, cfg, base_scores))
        records["random_plain"].append(run_arm(rnd, prices, cfg_plain, base_scores))
        if (i + 1) % 50 == 0:
            m = np.mean([r["CAGR"] for r in records["random_plain"]])
            print(f"  {i + 1:>5}/{args.trials}   "
                  f"random_plain mean CAGR so far {m:.2%}")

    trials = {k: pd.DataFrame(v) for k, v in records.items()}
    dists = {k: distribution_row(k, v) for k, v in trials.items()}

    print_reference([("SPY buy and hold", spy),
                     ("Equal-weight universe", ew)])
    print_arms(live, plain, dists, trials)

    print("\n" + "=" * 100)
    print("NULL DISTRIBUTION PERCENTILES  (CAGR)")
    print("=" * 100)
    print(f"  {'Arm':<18}" + "".join(f"{f'p{p}':>10}" for p in
                                     (1, 5, 25, 50, 75, 95, 99)) + f"{'max':>10}")
    print("  " + "-" * 88)
    for arm, t in trials.items():
        qs = t["CAGR"].quantile([.01, .05, .25, .50, .75, .95, .99])
        print(f"  {arm:<18}" + "".join(f"{q:>9.2%} " for q in qs)
              + f"{t['CAGR'].max():>9.2%}")

    out = pd.concat([t.assign(Arm=k) for k, t in trials.items()])
    out.to_csv(REPO_ROOT / OUT_TRIALS, index=False)
    summary = pd.DataFrame(
        [{"Arm": "ranked_overlay", **live}, {"Arm": "ranked_plain", **plain},
         {"Arm": "equal_weight_universe", **ew}, {"Arm": "spy", **spy}]
        + [dists["random_overlay"], dists["random_plain"]])
    summary.to_csv(REPO_ROOT / OUT_SUMMARY, index=False)
    print(f"\nTrials exported to:  {REPO_ROOT / OUT_TRIALS}")
    print(f"Summary exported to: {REPO_ROOT / OUT_SUMMARY}")

    print(f"\nCompleted {datetime.now():%Y-%m-%d %H:%M:%S}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
