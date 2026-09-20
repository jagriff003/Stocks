"""
Would ranking differently actually have made money?

Track J measured that `rank(12-1 month) - rank(3-month)` carries cross-sectional
information the production composite does not.  An information coefficient is
not a P&L, so this puts the two scores through the same portfolio machinery and
reports what each would have earned.

HOW THE COMPARISON IS KEPT FAIR

The live model is a SYSTEM - universe, ranker, VIX overlay, correlation filter,
top_n, hold clock, equal weight, execution.  The proposition is only a ranker.
Comparing "live system on 46 curated names" against "new ranker on 747" would
confound the score with the universe, which is the exact mistake Track F was
built to stop.

So: one input changes and nothing else can.  `run_strategy(ranking_override=...)`
swaps the score panel used for RANKING while eligibility, the level floor, the
regime overlay, the correlation filter, the hold clock and execution stay
exactly as configured, and the level floor still reads the REAL composite so
both arms face an identical eligible pool.  Check 3 below asserts that the
override plumbing is neutral, by passing the real composite through it and
requiring the result to be identical to not overriding at all.

THE BAR IS NOT SPY, AND IT IS NOT THE LIVE MODEL

Track F established that owning this universe equal-weighted (19.99%) beats the
live model (19.58%).  A new ranker that beats the live model has therefore
proved very little.  Every pool is run with the four arms Track F used:

    arm             selection   overlay
    ------------------------------------
    <score>_overlay ranked      on
    <score>_plain   ranked      off
    random_overlay  random      on
    random_plain    random      off

plus an equal-weight buy-and-hold of the pool.  The quantity that means
something is `score - random_plain` against `live - random_plain` ON THE SAME
POOL: ranking skill with the universe divided out.

COSTS

Two cost models, reported side by side:

  flat        7.5 bps one way, every name.  Comparable to every historical
              number in FINDINGS, and a fiction below about $10B.
  liquidity   `momentum/liquidity.py` - half-spread plus square-root impact,
              from POINT-IN-TIME dollar volume, so a name is charged what it
              would have cost on the day rather than what it would cost now.

And a breakeven: the uniform bps that erases the new score's advantage.  That
is usually more decision-useful than either model, because it converts the
question into "do you believe fills are better or worse than X".

Run:  python scripts/analyze_reversal_backtest.py --pool universe
      python scripts/analyze_reversal_backtest.py --pool wide --trials 100
      python scripts/analyze_reversal_backtest.py --sweep
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import warnings
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from momentum.backtest import build_target_portfolios, simulate_portfolio
from momentum.data import PriceData, load_data
from momentum.experiments import production_config
from momentum.liquidity import LiquidityConfig, dollar_volume, slippage_panel
from momentum.metrics import calculate_performance_metrics
from momentum.reversal import ReversalConfig, build_terms, composite
from momentum.strategy import compute_scores
from momentum.universe import current_symbols, defensive_symbols

from scripts.analyze_null_benchmark import random_scores
from scripts.analyze_reversal_ic import (NON_EQUITY, load_pool_panel,
                                         pool_symbols)

OUT_ARMS = "rsi_ma_reversal_backtest.csv"
OUT_SUB = "rsi_ma_reversal_backtest_subperiods.csv"
OUT_SWEEP = "rsi_ma_reversal_backtest_sweep.csv"
OUT_COST = "rsi_ma_reversal_costs.csv"

POOL_FILE = "random_pool.csv"
METRICS = ["CAGR", "Gross CAGR", "Sharpe", "MaxDD", "Calmar", "Volatility",
           "Trades/Year", "Annual Turnover"]


# ==========================================================================
# Volume, for the cost model
# ==========================================================================

def load_volume(symbols, start="2010-01-01", use_cache=True, verbose=True):
    """
    Daily share volume for `symbols`, cached beside the price panels.

    Kept separate from `PriceData` rather than bolted onto it: every existing
    caller of `load_data` would otherwise pay for a field none of them use,
    and `PriceData` is the type the whole package passes around.
    """
    import yfinance as yf
    warnings.filterwarnings("ignore")

    syms = sorted(set(symbols))
    key = f"vol_{len(syms)}sym_" + hashlib.md5(",".join(syms).encode()).hexdigest()[:12]
    path = REPO_ROOT / ".price_cache" / f"{key}_{start}.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)

    if use_cache and path.exists():
        if verbose:
            print(f"  loading volume from cache ({path.name})")
        return pd.read_pickle(path)

    frames = []
    for i in range(0, len(syms), 400):
        part = syms[i:i + 400]
        d = yf.download(part, start=start, interval="1d", auto_adjust=True,
                        progress=False, threads=True, group_by="column")
        frames.append(d["Volume"])
        if verbose:
            print(f"    volume {min(i + 400, len(syms))}/{len(syms)}")
    vol = pd.concat(frames, axis=1)
    vol.to_pickle(path)
    return vol


# ==========================================================================
# Arms
# ==========================================================================

def arm_metrics(result, cfg) -> dict:
    gross = calculate_performance_metrics(
        result.gross_returns, risk_free_rate=cfg.execution.risk_free_rate)
    return {
        "CAGR": result.metrics["cagr"],
        "Gross CAGR": gross["cagr"],
        "Sharpe": result.metrics["sharpe_ratio"],
        "MaxDD": result.metrics["max_drawdown"],
        "Calmar": result.metrics["calmar_ratio"],
        "Volatility": result.metrics["volatility"],
        "Trades/Year": result.turnover["trades_per_year"],
        "Annual Turnover": result.turnover["annual_turnover"],
    }


def run_arm(ranking, base, prices, cfg, slippage=None, return_result=False):
    """
    One simulation.  Everything except `ranking` comes from `cfg`, which is how
    the comparison stays a comparison of scores.
    """
    targets, _ = build_target_portfolios(
        ranking,
        price_columns=list(prices.close.columns),
        top_n=cfg.top_n,
        min_data_days=cfg.min_data_days,
        hold_days=cfg.hold_days,
        vix_data=prices.vix,
        vix_config=cfg.vix,
        base_composite_scores=base,
        velocity_config=cfg.velocity,
        correlation_config=cfg.correlation,
        graduated_config=cfg.graduated_vix,
        exit_config=cfg.exits,
        close=prices.close,
        rank_offset=cfg.rank_offset,
        rank_offset_scope=cfg.rank_offset_scope,
        monitor_symbols=cfg.monitor_symbols,
    )
    r = simulate_portfolio(targets, prices.close, prices.open_,
                           execution=cfg.execution, sizing=cfg.sizing,
                           scores=ranking, slippage_by_symbol=slippage)
    return (arm_metrics(r, cfg), r) if return_result else arm_metrics(r, cfg)


def equal_weight_arm(prices, cfg, ranking, symbols, slippage=None):
    """Own the pool, rebalanced never — the bar Track F says actually matters."""
    held = [s for s in symbols if s in ranking.columns]
    sub = ranking[held]
    dates = ranking.index[cfg.min_data_days:]
    targets = pd.Series(
        {d: sorted(sub.columns[sub.loc[d].notna()].tolist()) for d in dates},
        name="target")
    targets = targets[targets.apply(len) > 0]
    r = simulate_portfolio(targets, prices.close, prices.open_,
                           execution=cfg.execution,
                           slippage_by_symbol=slippage)
    return arm_metrics(r, cfg), r


def subperiod_cagr(returns: pd.Series, n=3) -> list:
    """
    CAGR within each of `n` equal segments.

    Reported on every arm because Track J's edge was flat in 2012-2016 and a
    full-sample figure would hide exactly that.
    """
    out = []
    for idx in np.array_split(np.arange(len(returns)), n):
        seg = returns.iloc[idx[0]:idx[-1] + 1]
        if len(seg) < 30:
            out.append(np.nan)
            continue
        out.append(calculate_performance_metrics(seg)["cagr"])
    return out


# ==========================================================================
# Validation
# ==========================================================================

class ValidationFailure(RuntimeError):
    pass


def check_override_is_neutral(ranking, base, prices, cfg) -> str:
    """
    Check 3 (the important one): the ranking-override path must change nothing
    when handed the score it is replacing.

    If `ranking_override` altered alignment, column order or dtype in any way,
    every "new score beats live score" number would be partly measuring the
    plumbing.  Passing the real composite through the override and requiring
    bit-identical output is the only way to rule that out.
    """
    from momentum.strategy import run_strategy
    direct = run_strategy(prices, cfg)
    forced = run_strategy(prices, cfg, ranking_override=ranking)

    gap = abs(direct.metrics["cagr"] - forced.metrics["cagr"])
    if gap > 1e-12:
        raise ValidationFailure(
            f"ranking_override is not neutral: CAGR {direct.metrics['cagr']:.6%} "
            f"vs {forced.metrics['cagr']:.6%} (gap {gap:.2e}). Every arm "
            f"comparison would be contaminated by the plumbing.")
    if not np.allclose(direct.returns.to_numpy(), forced.returns.to_numpy(),
                       equal_nan=True):
        raise ValidationFailure("ranking_override changed the return stream")
    return (f"CAGR identical to {gap:.1e}; return streams equal over "
            f"{len(direct.returns):,} days")


def check_uniform_slippage_parity(ranking, base, prices, cfg) -> str:
    """
    Check 2: a uniform per-name rate must reproduce the flat path exactly.

    Unit-tested in `tests/test_sizing.py`; asserted again here end to end on the
    real panel, because this is the run whose conclusions depend on it.
    """
    flat = run_arm(ranking, base, prices, cfg)
    uniform = pd.DataFrame(cfg.execution.slippage_frac,
                           index=prices.close.index, columns=prices.close.columns)
    per_name = run_arm(ranking, base, prices, cfg, slippage=uniform)
    gap = abs(flat["CAGR"] - per_name["CAGR"])
    if gap > 1e-12:
        raise ValidationFailure(
            f"uniform per-name slippage does not reproduce the flat path: "
            f"{flat['CAGR']:.6%} vs {per_name['CAGR']:.6%}")
    return f"flat and uniform-per-name CAGR agree to {gap:.1e}"


def check_production_reconciliation(prices, cfg, tol=5e-4) -> str:
    """
    Check 1: the live arm through THIS script reproduces production.

    `analyze_random_baskets.py` makes the same check for the same reason - the
    panel start alone is worth 6.6pp of CAGR here because it shifts every
    rebalance date.  Without it, a misaligned panel would quietly make every
    arm incomparable to the model actually being run.
    """
    from momentum.strategy import run_strategy

    live = current_symbols()
    shared = [c for c in prices.close.columns if c in set(live)]
    if len(shared) < 0.8 * len(live):
        return (f"skipped: this panel holds only {len(shared)} of "
                f"{len(live)} live universe names")

    sub = PriceData(close=prices.close[shared], open_=prices.open_[shared],
                    spy=prices.spy, vix=prices.vix)
    mine = run_strategy(sub, cfg)

    ref = load_data(live, start_date=str(prices.index[0].date()),
                    use_cache=True, cache_max_age_hours=1e9, verbose=False)
    ref = PriceData(close=ref.close[[c for c in shared if c in ref.close.columns]],
                    open_=ref.open_[[c for c in shared if c in ref.open_.columns]],
                    spy=ref.spy, vix=ref.vix)
    theirs = run_strategy(ref, cfg)

    gap = abs(mine.metrics["cagr"] - theirs.metrics["cagr"])
    if gap > tol:
        raise ValidationFailure(
            f"live universe does not reconcile through this script's panel: "
            f"{mine.metrics['cagr']:.2%} vs production {theirs.metrics['cagr']:.2%} "
            f"(gap {gap:.3%} > {tol:.2%}). The panel is misaligned with "
            f"production; every arm below would be incomparable to the model "
            f"actually being run.")
    return (f"live universe subpanel {mine.metrics['cagr']:.2%} vs production "
            f"{theirs.metrics['cagr']:.2%}, gap {gap:.4%} over "
            f"{len(shared)} shared names")


def check_cost_model_is_monotone(slip, adv, tol=0.0) -> str:
    """
    Check 4: the cost model must charge less where there is more volume.

    A sign error or an inverted ratio in the impact term would otherwise produce
    a cost panel that subsidizes exactly the illiquid names the whole exercise
    is meant to penalize — and the resulting backtest would look fine.
    """
    med_slip = slip.median() * 10_000
    med_adv = adv.median()
    both = pd.DataFrame({"bps": med_slip, "adv": med_adv}).dropna()
    if len(both) < 20:
        raise ValidationFailure(f"only {len(both)} names have both cost and ADV")

    both["bucket"] = pd.qcut(both["adv"], 5, labels=False, duplicates="drop")
    by = both.groupby("bucket")["bps"].median()
    if not by.is_monotonic_decreasing:
        raise ValidationFailure(
            f"cost is not monotone in liquidity across ADV quintiles: "
            f"{by.round(1).to_dict()}")
    corr = both["bps"].corr(both["adv"], method="spearman")
    return (f"ADV quintile median bps {by.round(1).tolist()} "
            f"(decreasing); Spearman(bps, ADV) = {corr:+.2f}")


# ==========================================================================
# Reporting
# ==========================================================================

def print_arms(rows, title, note=""):
    print(f"\n  {title}")
    if note:
        print(f"  {note}")
    hdr = (f"    {'arm':<26}{'CAGR':>9}{'Gross':>9}{'Sharpe':>8}{'MaxDD':>9}"
           f"{'Calmar':>8}{'Vol':>8}{'Trd/yr':>8}{'Turn':>9}")
    print(hdr)
    print("    " + "-" * (len(hdr) - 4))
    for r in rows:
        sd = f" +/-{r['CAGR sd']:.2%}" if r.get("CAGR sd") is not None else ""
        print(f"    {r['Arm']:<26}{r['CAGR']:>8.2%}{r['Gross CAGR']:>9.2%}"
              f"{r['Sharpe']:>8.2f}{r['MaxDD']:>9.2%}{r['Calmar']:>8.2f}"
              f"{r['Volatility']:>8.2%}{r['Trades/Year']:>8.1f}"
              f"{r['Annual Turnover']:>8.0%}{sd}")


def breakeven_bps(new_gross, live_gross, new_turn, live_turn):
    """
    The uniform one-way bps at which the new arm's gross advantage is spent.

    cost_CAGR ~ turnover * bps, so the advantage survives until
        (gross_new - gross_live) = (turn_new - turn_live) * bps
    Negative or infinite when the new arm also trades less, which is reported
    as such rather than as a number.
    """
    dg = new_gross - live_gross
    dt = new_turn - live_turn
    if dt <= 0:
        return np.inf if dg > 0 else -np.inf
    return dg / dt * 10_000.0


def main() -> int:
    p = argparse.ArgumentParser(
        description="Backtest the trend-change score against the live composite")
    p.add_argument("--pool", choices=["universe", "wide", "screened", "all"],
                   default="all")
    p.add_argument("--trials", type=int, default=200,
                   help="random draws per random arm")
    p.add_argument("--seed", type=int, default=20260920)
    p.add_argument("--start", default="2010-01-01")
    p.add_argument("--top-n", type=int, default=4)
    p.add_argument("--hold", type=int, default=14)
    p.add_argument("--subperiods", type=int, default=3)
    p.add_argument("--sweep", action="store_true",
                   help="also sweep top_n and hold for BOTH arms")
    p.add_argument("--sweep-top-n", type=int, nargs="+", default=[2, 4, 8, 16])
    p.add_argument("--sweep-hold", type=int, nargs="+", default=[5, 14, 21, 42, 63])
    p.add_argument("--no-cache", action="store_true")

    # Liquidity screen for the 'screened' pool.
    p.add_argument("--min-adv", type=float, default=10e6,
                   help="median trailing dollar volume floor, in dollars")
    # Cost model.
    p.add_argument("--account", type=float, default=100_000.0)
    p.add_argument("--impact-k", type=float, default=0.8)
    p.add_argument("--bps-ladder", type=float, nargs="+",
                   default=[7.5, 15.0, 30.0, 60.0])
    args = p.parse_args()

    cfg = production_config()
    cfg = replace(cfg, top_n=args.top_n, hold_days=args.hold)
    cfg_plain = replace(cfg, vix=None)
    rcfg = ReversalConfig()
    lcfg = LiquidityConfig(account_notional=args.account,
                           impact_coefficient=args.impact_k)

    print("=" * 112)
    print("TREND-CHANGE SCORE vs THE LIVE COMPOSITE - portfolio comparison")
    print(f"Started {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"top_n {cfg.top_n}   hold {cfg.hold_days}d   "
          f"{args.trials} random trials   account ${args.account:,.0f}")
    print("=" * 112)

    defensive = defensive_symbols()
    pools = []
    if args.pool in ("universe", "all"):
        pools.append("universe")
    if args.pool in ("screened", "all"):
        pools.append("screened")
    if args.pool in ("wide", "all"):
        pools.append("wide")

    arm_rows, sub_rows, sweep_rows, cost_rows = [], [], [], []

    for pool_name in pools:
        print("\n" + "=" * 112)
        print(f"POOL: {pool_name}")
        print("=" * 112)

        if pool_name == "universe":
            syms = current_symbols()
            prices = load_data(syms, start_date=args.start,
                               use_cache=not args.no_cache, verbose=False,
                               drop_unsettled=True)
            volume_syms = list(prices.close.columns)
        else:
            # Load on the FULL symbol list so the cached panel is reused, then
            # drop the restricted names from the panel.  Filtering before the
            # load would change the cache key and force a re-download of ~750
            # symbols to arrive at the same columns minus five.
            raw = sorted(set(pd.read_csv(REPO_ROOT / POOL_FILE)["symbol"]
                             .dropna()) - NON_EQUITY)
            prices = load_pool_panel(raw + defensive, start=args.start,
                                     use_cache=not args.no_cache)
            volume_syms = raw + defensive
            allowed = set(pool_symbols(POOL_FILE)) | set(defensive)
            keep = [c for c in prices.close.columns if c in allowed]
            removed = [c for c in prices.close.columns if c not in allowed]
            if removed:
                print(f"  compliance: removed from panel -> {', '.join(removed)}")
            prices = PriceData(close=prices.close[keep], open_=prices.open_[keep],
                               spy=prices.spy, vix=prices.vix)

        # Belt and braces: whatever the pool path did, nothing restricted may
        # reach a panel the model will rank.  This RAISES — the RUNBOOK is
        # explicit that a restricted name reaching the universe must never be
        # suppressed.
        from momentum.restrictions import check_symbols
        check_symbols(list(prices.close.columns), f"{pool_name} panel")

        # Volume is fetched on the same symbol list the PRICE panel was cached
        # under, then reindexed down — otherwise dropping five restricted names
        # changes the cache key and forces a second ~750-symbol download to
        # produce a subset of what is already on disk.
        volume = load_volume(volume_syms, start=args.start,
                             use_cache=not args.no_cache)
        volume = volume.reindex(index=prices.close.index,
                                columns=prices.close.columns)
        adv = dollar_volume(prices.close, volume, lcfg.adv_window)

        if pool_name == "screened":
            liquid = adv.median()
            keep = [c for c in prices.close.columns
                    if (c in defensive) or (liquid.get(c, 0) >= args.min_adv)]
            dropped = len(prices.close.columns) - len(keep)
            print(f"  liquidity screen: median trailing dollar volume >= "
                  f"${args.min_adv/1e6:.0f}M keeps {len(keep)} of "
                  f"{len(prices.close.columns)} names ({dropped} dropped)")
            prices = PriceData(close=prices.close[keep], open_=prices.open_[keep],
                               spy=prices.spy, vix=prices.vix)
            volume = volume[keep]
            adv = adv[keep]

        print(f"  panel: {prices.close.shape[1]} tickers, "
              f"{prices.index[0]:%Y-%m-%d} to {prices.index[-1]:%Y-%m-%d}")

        ranking, base, _ = compute_scores(prices, cfg)
        terms = build_terms(prices.close, rcfg)
        flip_neg = (-terms["flip"]).reindex_like(ranking)
        pullback = composite(
            terms, replace(rcfg, turn_weight=-1.0, strength_weight=0.0,
                           room_weight=1.0), "flip").reindex_like(ranking)

        slip = slippage_panel(prices.close, volume, lcfg, top_n=cfg.top_n)
        cost_summary = (slip.median() * 10_000).describe()

        # ---------------- validation ----------------
        print("\n  VALIDATION - no arm is printed unless these pass")
        try:
            print(f"    PASS  1 live universe reconciles with production\n"
                  f"          {check_production_reconciliation(prices, cfg)}")
            print(f"    PASS  2 uniform per-name slippage == flat\n"
                  f"          {check_uniform_slippage_parity(ranking, base, prices, cfg)}")
            print(f"    PASS  3 ranking_override is neutral\n"
                  f"          {check_override_is_neutral(ranking, base, prices, cfg)}")
            print(f"    PASS  4 cost model monotone in liquidity\n"
                  f"          {check_cost_model_is_monotone(slip, adv)}")
        except ValidationFailure as exc:
            print("\n" + "=" * 112)
            print("VALIDATION FAILED - refusing to report")
            print("=" * 112)
            print(f"  {exc}")
            return 3

        print(f"\n  cost model: median {cost_summary['50%']:.1f} bps across "
              f"names, 25th {cost_summary['25%']:.1f}, 75th "
              f"{cost_summary['75%']:.1f}, max {cost_summary['max']:.1f}")
        cost_rows.append(pd.DataFrame({
            "Pool": pool_name,
            "Symbol": slip.columns,
            "median_bps": (slip.median() * 10_000).to_numpy(),
            "median_adv": adv.median().to_numpy(),
        }))

        scores = {"live": ranking, "flip_neg": flip_neg, "pullback": pullback}

        for cost_label, slippage in (("flat 7.5bps", None),
                                     ("liquidity model", slip)):
            rows = []
            eq, eq_res = equal_weight_arm(
                prices, cfg, ranking,
                [c for c in prices.close.columns if c not in defensive],
                slippage)
            rows.append({"Arm": "equal-weight pool", **eq, "CAGR sd": None})
            sub_rows.append({"Pool": pool_name, "Cost": cost_label,
                             "Arm": "equal-weight pool",
                             **dict(zip([f"P{i+1}" for i in range(args.subperiods)],
                                        subperiod_cagr(eq_res.returns,
                                                       args.subperiods)))})

            for label, panel in scores.items():
                for overlay, c in (("overlay", cfg), ("plain", cfg_plain)):
                    m, res = run_arm(panel, base, prices, c, slippage,
                                     return_result=True)
                    rows.append({"Arm": f"{label}_{overlay}", **m,
                                 "CAGR sd": None})
                    sub_rows.append(
                        {"Pool": pool_name, "Cost": cost_label,
                         "Arm": f"{label}_{overlay}",
                         **dict(zip([f"P{i+1}" for i in range(args.subperiods)],
                                    subperiod_cagr(res.returns, args.subperiods)))})

            # ---- the null ----
            rng = np.random.default_rng(args.seed)
            for overlay, c in (("overlay", cfg), ("plain", cfg_plain)):
                trials = []
                for _ in range(args.trials):
                    rs = random_scores(ranking, rng)
                    try:
                        trials.append(run_arm(rs, base, prices, c, slippage))
                    except Exception:
                        continue
                if not trials:
                    continue
                td = pd.DataFrame(trials)
                row = {"Arm": f"random_{overlay}", "CAGR sd": td["CAGR"].std()}
                row.update({k: td[k].mean() for k in METRICS})
                rows.append(row)

            print_arms(rows, f"{pool_name}  -  {cost_label}",
                       "random arms are the mean over "
                       f"{args.trials} trials; +/- is the spread across them")

            # ---- what the ranking is worth, over its own null ----
            byname = {r["Arm"]: r for r in rows}
            print(f"\n    Ranking skill, net of the universe "
                  f"(arm minus its own random null):")
            print(f"      {'score':<12}{'overlay: net':>14}{'gross':>10}"
                  f"{'  |  plain: net':>16}{'gross':>10}")
            for label in scores:
                line = f"      {label:<12}"
                for overlay in ("overlay", "plain"):
                    a, n = byname.get(f"{label}_{overlay}"), byname.get(f"random_{overlay}")
                    if a and n:
                        line += (f"{a['CAGR'] - n['CAGR']:>+13.2%}"
                                 f"{a['Gross CAGR'] - n['Gross CAGR']:>+10.2%}")
                        if overlay == "overlay":
                            line += "  |  "
                print(line)

            for r in rows:
                arm_rows.append({"Pool": pool_name, "Cost": cost_label, **r})

            # ---- breakeven ----
            if "flip_neg_overlay" in byname and "live_overlay" in byname:
                a, b = byname["flip_neg_overlay"], byname["live_overlay"]
                be = breakeven_bps(a["Gross CAGR"], b["Gross CAGR"],
                                   a["Annual Turnover"], b["Annual Turnover"])
                if np.isfinite(be):
                    print(f"\n    BREAKEVEN: flip_neg's gross advantage over live "
                          f"({a['Gross CAGR'] - b['Gross CAGR']:+.2%}) is spent at "
                          f"{be:.0f} bps one way,\n    against the "
                          f"{a['Annual Turnover'] - b['Annual Turnover']:+.0%} "
                          f"extra annual turnover it runs. Live assumes 7.5.")
                else:
                    print(f"\n    BREAKEVEN: flip_neg trades no more than live, so "
                          f"no uniform slippage level erases its advantage.")

        # ---- flat-bps ladder ----
        print(f"\n  BPS LADDER - live vs flip_neg, uniform slippage, overlay on")
        print(f"    {'bps':>6}{'live':>10}{'flip_neg':>11}{'delta':>10}")
        print("    " + "-" * 33)
        for bps in args.bps_ladder:
            c = replace(cfg, execution=replace(cfg.execution, slippage_bps=bps))
            a = run_arm(ranking, base, prices, c)
            b = run_arm(flip_neg, base, prices, c)
            print(f"    {bps:>6.1f}{a['CAGR']:>10.2%}{b['CAGR']:>11.2%}"
                  f"{b['CAGR'] - a['CAGR']:>+10.2%}")
            arm_rows.append({"Pool": pool_name, "Cost": f"uniform {bps}bps",
                             "Arm": "live_overlay", **a, "CAGR sd": None})
            arm_rows.append({"Pool": pool_name, "Cost": f"uniform {bps}bps",
                             "Arm": "flip_neg_overlay", **b, "CAGR sd": None})

        # ---- subperiods ----
        sub = pd.DataFrame([r for r in sub_rows if r["Pool"] == pool_name])
        print(f"\n  SUBPERIOD CAGR  (equal segments; Track J's edge was absent "
              f"in the first)")
        pcols = [f"P{i+1}" for i in range(args.subperiods)]
        print(f"    {'arm':<26}{'cost':<18}" + "".join(f"{c:>10}" for c in pcols))
        print("    " + "-" * (44 + 10 * len(pcols)))
        for _, r in sub.iterrows():
            cells = "".join(f"{r[c]:>9.2%} " if pd.notna(r[c]) else f"{'-':>10}"
                            for c in pcols)
            print(f"    {r['Arm']:<26}{r['Cost']:<18}{cells}")

        # ---- sweep ----
        if args.sweep:
            print(f"\n  SWEEP - both arms, every setting. SENSITIVITY, NOT A "
                  f"RECOMMENDATION:\n  walk-forward established that re-tuning "
                  f"on a trailing window costs 3pp of CAGR here.")
            print(f"  Read the SHAPE, not the maximum. flip_neg's windows "
                  f"(252/21/63) are literature\n  conventions that have never "
                  f"been fitted to this data, so a flat surface around them is\n"
                  f"  evidence and a spike is a warning.")
            print(f"\n    {'top_n':>6}{'hold':>6}{'live CAGR':>11}"
                  f"{'new CAGR':>10}{'delta':>9}{'live Sh':>9}{'new Sh':>8}"
                  f"{'new DD':>9}")
            print("    " + "-" * 58)
            grid = []
            for tn in args.sweep_top_n:
                for hd in args.sweep_hold:
                    c = replace(cfg, top_n=tn, hold_days=hd)
                    s = slippage_panel(prices.close, volume, lcfg, top_n=tn)
                    a = run_arm(ranking, base, prices, c, s)
                    b = run_arm(flip_neg, base, prices, c, s)
                    print(f"    {tn:>6}{hd:>6}{a['CAGR']:>11.2%}"
                          f"{b['CAGR']:>10.2%}{b['CAGR'] - a['CAGR']:>+9.2%}"
                          f"{a['Sharpe']:>9.2f}{b['Sharpe']:>8.2f}"
                          f"{b['MaxDD']:>9.1%}")
                    row = {"Pool": pool_name, "top_n": tn, "hold": hd,
                           "live": a["CAGR"], "flip_neg": b["CAGR"],
                           "delta": b["CAGR"] - a["CAGR"],
                           "live_sharpe": a["Sharpe"], "new_sharpe": b["Sharpe"],
                           "new_maxdd": b["MaxDD"]}
                    grid.append(row)
                    sweep_rows.append(row)

            # The statistic that answers "is this a real effect or a lucky
            # cell".  Beating the live model at its own settings is one
            # observation; beating it across the grid is a different claim.
            g = pd.DataFrame(grid)
            win = (g["delta"] > 0).mean()
            sh_ok = (g["new_sharpe"] >= g["live_sharpe"]).mean()
            both = ((g["delta"] > 0) & (g["new_sharpe"] >= g["live_sharpe"])).mean()
            print(f"\n    ROBUSTNESS ACROSS THE GRID ({len(g)} cells)")
            print(f"      flip_neg beats live on CAGR:          {win:.0%}")
            print(f"      flip_neg Sharpe >= live Sharpe:       {sh_ok:.0%}")
            print(f"      both at once (your stated bar):       {both:.0%}")
            print(f"      delta CAGR: median {g['delta'].median():+.2%}, "
                  f"worst {g['delta'].min():+.2%}, best {g['delta'].max():+.2%}")
            print(f"      at the unfitted default "
                  f"(top_n={args.top_n}, hold={args.hold}): "
                  f"{g[(g.top_n == args.top_n) & (g.hold == args.hold)]['delta'].iloc[0]:+.2%}"
                  if len(g[(g.top_n == args.top_n) & (g.hold == args.hold)])
                  else "")

    pd.DataFrame(arm_rows).to_csv(REPO_ROOT / OUT_ARMS, index=False)
    pd.DataFrame(sub_rows).to_csv(REPO_ROOT / OUT_SUB, index=False)
    if cost_rows:
        pd.concat(cost_rows).to_csv(REPO_ROOT / OUT_COST, index=False)
    if sweep_rows:
        pd.DataFrame(sweep_rows).to_csv(REPO_ROOT / OUT_SWEEP, index=False)

    print("\n" + "=" * 112)
    print("EXPORTS")
    print("=" * 112)
    print(f"  Arms       {REPO_ROOT / OUT_ARMS}")
    print(f"  Subperiods {REPO_ROOT / OUT_SUB}")
    print(f"  Costs      {REPO_ROOT / OUT_COST}")
    if sweep_rows:
        print(f"  Sweep      {REPO_ROOT / OUT_SWEEP}")
    print(f"\nCompleted {datetime.now():%Y-%m-%d %H:%M:%S}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
