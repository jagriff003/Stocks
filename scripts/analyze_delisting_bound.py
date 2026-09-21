"""
How much delisting would it take to erase the Track J edge?

TODO 0d.  The Track J backtest earns 24.35% CAGR at 0.65 Sharpe on 635 liquid
names against 14.81% for owning the pool.  The largest reason not to believe it
is that the pool was built in 2026 from names that still exist, and the score
buys pullbacks — so every dip in the sample was followed by a recovery, because
the dips that were terminal are not in the data.

This does not fix that.  It bounds it: synthetic listings that END are injected
at a given annual rate, the strategy trades them under its normal rules with the
liquidity and price screens left ON, and the edge is re-measured.  Sweeping the
rate produces a curve, and the number to take away is the **breakeven rate** —
the annual delisting rate at which the advantage over owning the pool is gone.

WHY THE SCREENS STAY ON

The open question is not "what would delistings cost a strategy that blindly
held them" — it is whether a liquidity and price screen, re-applied every
rebalance, ducks most of them.  Switching the screens off would answer a
question nobody asked and would overstate the damage.  Both arms face the same
augmented pool, so the comparison stays fair either way.

READ THE CURVE, NOT A POINT

Every input is an assumption: the rate, the branch mixture, the shape of the
pre-delisting decline.  A single "corrected" CAGR would be false precision.
The output is a sensitivity, and the honest statement is "the edge survives up
to rate X", with X compared against plausible rates for names of this size.

Run:  python scripts/analyze_delisting_bound.py
      python scripts/analyze_delisting_bound.py --rates 0 0.02 0.05 0.10
      python scripts/analyze_delisting_bound.py --top-n 8 --hold 42
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from momentum.data import PriceData
from momentum.delisting import (DelistingConfig, audit, build_synthetic_panel,
                                n_synthetic)
from momentum.experiments import production_config
from momentum.liquidity import (LiquidityConfig, apply_tradable,
                                dollar_volume, slippage_panel, tradable_mask)
from momentum.reversal import ReversalConfig, build_terms, composite
from momentum.strategy import compute_scores
from momentum.universe import defensive_symbols

from scripts.analyze_reversal_backtest import (equal_weight_arm, load_volume,
                                               run_arm, subperiod_cagr)
from scripts.analyze_reversal_ic import NON_EQUITY, load_pool_panel, pool_symbols

OUT = "rsi_ma_delisting_bound.csv"
OUT_HELD = "rsi_ma_delisting_held.csv"
POOL_FILE = "random_pool.csv"


class ValidationFailure(RuntimeError):
    pass


def screen_report(mask: pd.DataFrame, manifest: pd.DataFrame,
                  close: pd.DataFrame) -> dict:
    """
    Did the per-date screen actually duck the synthetic delistings?

    This is the question the whole exercise turns on, so it is measured rather
    than asserted.  For each injected listing: was it still tradable on the day
    before it died, and if not, how long before the end did it stop being
    tradable?  A screen that works excludes most of them well ahead of the
    event, because the cause-based branches decline for 189 sessions first.
    """
    if manifest.empty:
        return {"n": 0}

    still_tradable, ever, ducked = 0, 0, []
    for _, r in manifest.iterrows():
        sym = r["Symbol"]
        if sym not in mask.columns:
            continue
        col = mask[sym]
        if bool(col.any()):
            ever += 1
        i = close.index.get_loc(r["Delist"])
        if i > 0 and bool(col.iloc[i - 1]):
            still_tradable += 1
        else:
            prior = col.iloc[:i]
            last_ok = prior[prior].index
            if len(last_ok):
                ducked.append(int(i - close.index.get_loc(last_ok[-1])))

    return {
        "n": len(manifest),
        "ever_tradable": ever,
        "still_tradable_day_before_death": still_tradable,
        "median_sessions_screened_out_first":
            float(np.median(ducked)) if ducked else float("nan"),
    }


def main() -> int:
    p = argparse.ArgumentParser(
        description="Bound the survivorship exposure with synthetic delistings")
    p.add_argument("--rates", type=float, nargs="+",
                   default=[0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12])
    p.add_argument("--seed", type=int, default=20260920)
    p.add_argument("--seeds", type=int, default=3,
                   help="independent draws per rate, since the injection is "
                        "random and one draw is one sample")
    p.add_argument("--start", default="2010-01-01")
    p.add_argument("--top-n", type=int, default=8)
    p.add_argument("--hold", type=int, default=42)
    p.add_argument("--min-adv", type=float, default=10e6)
    p.add_argument("--min-price", type=float, default=5.0)
    p.add_argument("--account", type=float, default=100_000.0)
    p.add_argument("--no-cache", action="store_true")
    args = p.parse_args()

    cfg = replace(production_config(), top_n=args.top_n, hold_days=args.hold)
    cfg_plain = replace(cfg, vix=None)
    rcfg = ReversalConfig()
    lcfg = LiquidityConfig(account_notional=args.account)
    defensive = set(defensive_symbols())

    print("=" * 108)
    print("DELISTING BOUND - how much survivorship would erase the Track J edge?")
    print(f"Started {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"top_n {cfg.top_n}   hold {cfg.hold_days}d   "
          f"screens: ADV >= ${args.min_adv/1e6:.0f}M, price >= ${args.min_price:.0f}")
    print(f"rates {args.rates}   {args.seeds} draw(s) per rate")
    print("=" * 108)

    raw = sorted(set(pd.read_csv(REPO_ROOT / POOL_FILE)["symbol"].dropna())
                 - NON_EQUITY)
    base = load_pool_panel(raw + sorted(defensive), start=args.start,
                           use_cache=not args.no_cache)
    allowed = set(pool_symbols(POOL_FILE)) | defensive
    keep0 = [c for c in base.close.columns if c in allowed]
    base = PriceData(close=base.close[keep0], open_=base.open_[keep0],
                     spy=base.spy, vix=base.vix)
    vol0 = load_volume(raw + sorted(defensive), start=args.start,
                       use_cache=not args.no_cache)
    vol0 = vol0.reindex(index=base.close.index, columns=base.close.columns)

    dcfg0 = DelistingConfig()
    print(f"\nBranch mixture (assumption, not measurement):")
    for b in dcfg0.branches:
        print(f"    {b.name:<16}{b.weight:>6.0%}  terminal {b.terminal_return:>+7.0%}"
              f"  decline {b.decline:>+6.0%} over {b.decline_days:>4d}d")
    print(f"    {'mean terminal':<16}{'':>6}{dcfg0.mean_terminal:>+17.1%}")

    rows, held_rows = [], []

    for rate in args.rates:
        draws = 1 if rate == 0 else args.seeds
        for d in range(draws):
            dcfg = replace(dcfg0, annual_rate=rate)
            c, o, v, manifest = build_synthetic_panel(
                base.close, base.open_, vol0, dcfg, seed=args.seed + 1000 * d)

            if rate > 0:
                a = audit(manifest, c, dcfg)
                if a["count"] == 0:
                    raise ValidationFailure(f"rate {rate} produced no listings")
                if a["alive_before"] != a["count"] or a["dead_after"] != a["count"]:
                    raise ValidationFailure(
                        f"synthetic names are malformed at rate {rate}: "
                        f"{a['alive_before']}/{a['count']} alive before delisting, "
                        f"{a['dead_after']}/{a['count']} dead after")
                if d == 0:
                    print(f"\n  rate {rate:.0%}: injected {a['count']} listings "
                          f"(expected {n_synthetic(len(base.close.columns), len(base.close), dcfg)}), "
                          f"mean terminal {a['mean_terminal']:+.1%}")

            def evaluate(cc, oo, vv):
                """Run the arms on one augmented panel."""
                pr = PriceData(close=cc, open_=oo, spy=base.spy, vix=base.vix)
                tm = tradable_mask(cc, vv, args.min_adv, args.min_price, lcfg)
                rk, bs, _ = compute_scores(pr, cfg)
                tr = build_terms(pr.close, rcfg)
                pb = composite(
                    tr, replace(rcfg, turn_weight=-1.0, strength_weight=0.0,
                                room_weight=1.0), "flip").reindex_like(rk)
                # The screen is applied PER DATE by blanking the score, never by
                # dropping columns.  A whole-sample column cut would be
                # look-ahead, and would also hand the screen credit it has not
                # earned — the question is whether it ducks a dying name in
                # time, which only a per-date test can answer.
                rk = apply_tradable(rk, tm, defensive)
                pb = apply_tradable(pb, tm, defensive)
                sl = slippage_panel(pr.close, vv, lcfg, top_n=cfg.top_n)
                e, _ = equal_weight_arm(
                    pr, cfg, rk, [x for x in rk.columns if x not in defensive], sl)
                nw, nres = run_arm(pb, bs, pr, cfg_plain, sl, return_result=True)
                lv = run_arm(rk, bs, pr, cfg_plain, sl)
                return e, nw, lv, nres, tm

            eq, new, live, new_res, tmask = evaluate(c, o, v)

            # CONTROL: the identical names, donors and count, but they do not
            # die.  Injection does two things at once — it adds delisting risk
            # AND it enlarges the pool, and a bigger pool makes the top-8 cut a
            # more extreme percentile, which raises the measured edge on its
            # own.  Without this arm the curve conflates the two and reads
            # backwards (the edge appeared to IMPROVE with the delisting rate).
            if rate > 0:
                cc, oc, vc, _ = build_synthetic_panel(
                    base.close, base.open_, vol0, dcfg,
                    seed=args.seed + 1000 * d, immortal=True)
                eq_c, new_c, live_c, _, _ = evaluate(cc, oc, vc)
            else:
                eq_c, new_c, live_c = eq, new, live

            rep = screen_report(tmask, manifest, c)
            n_kept = rep.get("still_tradable_day_before_death", 0)

            # Did the book actually hold one, and when?
            touched = 0
            if not manifest.empty:
                syn = set(manifest["Symbol"])
                for names in new_res.holdings_history:
                    if syn.intersection(names["Holdings"]):
                        touched += 1

            rows.append({
                "Rate": rate, "Draw": d,
                "Injected": 0 if manifest.empty else len(manifest),
                "PassedScreen": n_kept,
                "PoolNames": float(tmask.sum(axis=1).mean()),
                "ScreenedOutFirst": rep.get(
                    "median_sessions_screened_out_first", float("nan")),
                "EqualWeight": eq["CAGR"], "Live": live["CAGR"],
                "New": new["CAGR"], "NewSharpe": new["Sharpe"],
                "NewMaxDD": new["MaxDD"],
                "ControlNew": new_c["CAGR"], "ControlEW": eq_c["CAGR"],
                "DelistingCost": new["CAGR"] - new_c["CAGR"],
                "NewMinusEW": new["CAGR"] - eq["CAGR"],
                "ControlNewMinusEW": new_c["CAGR"] - eq_c["CAGR"],
                "NewMinusLive": new["CAGR"] - live["CAGR"],
                "RebalancesTouchingSynthetic": touched,
            })
            if not manifest.empty:
                s = manifest.copy(); s["Rate"] = rate; s["Draw"] = d
                held_rows.append(s)

    out = pd.DataFrame(rows)
    agg = out.groupby("Rate").agg(
        injected=("Injected", "mean"), passed=("PassedScreen", "mean"),
        pool=("PoolNames", "mean"), ew=("EqualWeight", "mean"),
        live=("Live", "mean"), new=("New", "mean"),
        new_sd=("New", "std"), sharpe=("NewSharpe", "mean"),
        dd=("NewMaxDD", "mean"), edge=("NewMinusEW", "mean"),
        ctrl=("ControlNew", "mean"), ctrl_edge=("ControlNewMinusEW", "mean"),
        cost=("DelistingCost", "mean"),
        touched=("RebalancesTouchingSynthetic", "mean"),
        ducked=("ScreenedOutFirst", "mean")).reset_index()

    print("\n" + "=" * 108)
    print("THE CURVE")
    print("=" * 108)
    print("  'control' is the SAME injected names with the same seed, but they")
    print("  never die — so 'cost' isolates delisting from the pool simply")
    print("  getting bigger. Read the cost column; the raw edge is confounded.")
    hdr = (f"    {'rate':>6}{'injected':>10}{'pool':>7}{'equal-wt':>10}"
           f"{'new':>9}{'+/-':>7}{'control':>9}{'cost':>9}{'Sharpe':>8}"
           f"{'MaxDD':>8}{'edge':>9}")
    print(hdr)
    print("    " + "-" * (len(hdr) - 4))
    for _, r in agg.iterrows():
        sd = f"{r['new_sd']:.1%}" if pd.notna(r["new_sd"]) else "-"
        print(f"    {r['Rate']:>6.0%}{r['injected']:>10.0f}"
              f"{r['pool']:>7.0f}{r['ew']:>10.2%}"
              f"{r['new']:>9.2%}{sd:>7}{r['ctrl']:>9.2%}{r['cost']:>+9.2%}"
              f"{r['sharpe']:>8.2f}{r['dd']:>8.1%}{r['edge']:>+9.2%}")

    # --- breakeven ---
    print("\n" + "=" * 108)
    print("BREAKEVEN")
    print("=" * 108)
    print("  Breakeven is read on the CONTROLLED edge — the new score minus "
          "equal weight,\n  with the pool-size effect removed by differencing "
          "against the control arm.")
    agg["controlled_edge"] = agg["edge"] - (agg["ctrl_edge"] - agg.loc[
        agg["Rate"] == 0, "ctrl_edge"].iloc[0]) if (agg["Rate"] == 0).any() \
        else agg["edge"]
    for _, r in agg.iterrows():
        print(f"    rate {r['Rate']:>4.0%}: raw edge {r['edge']:>+7.2%}, "
              f"delisting cost {r['cost']:>+7.2%}, "
              f"controlled edge {r['controlled_edge']:>+7.2%}")

    pos = agg[agg["controlled_edge"] > 0]
    neg = agg[agg["controlled_edge"] <= 0]
    if neg.empty:
        print(f"  The edge over owning the pool survives every rate tested, up "
              f"to {agg['Rate'].max():.0%}/yr.")
        print(f"  No breakeven inside the tested range — widen --rates to find "
              f"one.")
    elif pos.empty:
        print(f"  The edge is gone at the lowest rate tested "
              f"({agg['Rate'].min():.0%}/yr). The wide-pool result is not "
              f"usable without point-in-time data.")
    else:
        lo = pos["Rate"].max()
        hi = neg[neg["Rate"] > lo]["Rate"].min()
        print(f"  The edge over owning the pool survives to {lo:.0%}/yr and is "
              f"gone by {hi:.0%}/yr.")
        print(f"  Compare that against the plausible annual rate for names "
              f"screened at ADV >= ${args.min_adv/1e6:.0f}M and price >= "
              f"${args.min_price:.0f}.")

    print("\n  How much the screens are doing:")
    print("    'still tradable' = cleared the ADV and price floors on the day "
          "BEFORE it died.\n    A working screen ducks most of them well ahead "
          "of the event, not at it.")
    for _, r in agg[agg["Rate"] > 0].iterrows():
        pct = r["passed"] / r["injected"] if r["injected"] else np.nan
        print(f"    rate {r['Rate']:>4.0%}: {r['passed']:.0f} of "
              f"{r['injected']:.0f} were still tradable the day before death "
              f"({pct:.0%});\n              the rest stopped being tradable a "
              f"median {r['ducked']:.0f} sessions earlier; the book held one "
              f"on {r['touched']:.0f} rebalance(s)")

    out.to_csv(REPO_ROOT / OUT, index=False)
    if held_rows:
        pd.concat(held_rows).to_csv(REPO_ROOT / OUT_HELD, index=False)
    print(f"\n  Curve exported to {REPO_ROOT / OUT}")
    if held_rows:
        print(f"  Survivors of the screen exported to {REPO_ROOT / OUT_HELD}")

    print("\n  READ THIS AS A BOUND, NOT A CORRECTION. The rate, the branch "
          "mixture and the\n  shape of the pre-delisting decline are all "
          "assumptions; only their consequences\n  are measured.")
    print(f"\nCompleted {datetime.now():%Y-%m-%d %H:%M:%S}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
