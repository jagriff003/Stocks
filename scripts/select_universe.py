"""
Propose a universe from a screener export, by maximizing independent bets.

Takes a Schwab (or any) screener CSV, and selects a universe from it.  The
selection objective is deliberately NOT strength.

WHY STRENGTH IS NOT USED, IN EITHER DIRECTION
   The screen has already applied its strength filter (5-year total return above
   a floor).  Among names that pass it, nothing in this repo's evidence supports
   preferring the stronger ones, and there is evidence against it:

     - The composite score's information coefficient is indistinguishable from
       zero at every horizon from 5 to 126 days (Track F).
     - The WORST-ranked quintile out-returned the best at six of seven horizons.
       At short horizons this universe mean-reverts.
     - A name up 20% a year for five years is as plausibly a reversion candidate
       as a continuation one.

   So strength is used as a FILTER (by the screener, upstream) and never as a
   RANKING (here).  It is not inverted either - selecting against strength would
   be the same mistake with the sign flipped.  Screen passers are treated as an
   unordered set, and the only thing that orders them is what they contribute to
   diversification.

THE OBJECTIVE
   Maximize effective independent bets: the exponential of the Shannon entropy
   of the correlation matrix's eigenvalue spectrum (`momentum.correlation.
   effective_bets`).  N uncorrelated assets give N effective bets; N assets that
   move together give one.  The live 52-name universe delivers 27.4.

   This is the one selection mechanism in FINDINGS with measured value -
   correlation-aware selection, gated above VIX 25, buys 1.32pp of drawdown for
   no return cost.  Industry is a crude proxy for the same thing; the
   correlation matrix is the real quantity, and an industry cap is retained on
   top of it as a guard against the matrix being confidently wrong.

INCUMBENCY
   A name already in the universe is kept unless it fails the screen outright.
   Universe changes are real trades, and the report flags every case where an
   incumbent was kept despite a better-diversifying candidate being available,
   so the cost of that choice is visible rather than assumed away.

VALIDATION
   The greedy selection recomputes effective bets from the full correlation
   matrix at every step and asserts the score is non-decreasing.  The final
   proposal is scored against the incumbent universe and against random
   same-size draws from the same candidate pool, so "did this actually help?"
   is answered in the output rather than assumed.

Every threshold is a flag.

Run:  python scripts/select_universe.py --screen my_screen.csv
      python scripts/select_universe.py --screen my_screen.csv --size 40
      python scripts/select_universe.py --demo    # uses the live universe
"""

from __future__ import annotations

import argparse
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from momentum.universe import current_symbols, defensive_symbols

OUT = "universe_proposal.csv"

# Column aliases seen in screener exports; extend as needed.
ALIASES = {
    "symbol": ["symbol", "ticker", "sym"],
    "name": ["name", "company", "company name", "security", "description"],
    "industry": ["industry", "sub-industry", "subindustry", "industry name"],
    "sector": ["sector", "sector name"],
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    lower = {str(c).strip().lower(): c for c in df.columns}
    out = {}
    for want, names in ALIASES.items():
        for n in names:
            if n in lower:
                out[want] = lower[n]
                break
    missing = [k for k in ("symbol",) if k not in out]
    if missing:
        raise SystemExit(
            f"Screener CSV has no recognizable {missing} column. "
            f"Columns found: {list(df.columns)}")
    frame = df.rename(columns={v: k for k, v in out.items()})
    keep = [c for c in ("symbol", "name", "sector", "industry")
            if c in frame.columns]
    frame = frame[keep].copy()
    frame["symbol"] = frame["symbol"].astype(str).str.strip().str.upper()
    return frame.drop_duplicates("symbol").reset_index(drop=True)


def effective_bets_from_corr(corr: np.ndarray) -> float:
    """Entropy-based effective bets for a correlation submatrix."""
    if corr.shape[0] < 2:
        return float(corr.shape[0])
    ev = np.clip(np.linalg.eigvalsh(corr), 1e-12, None)
    w = ev / ev.sum()
    return float(np.exp(-(w * np.log(w)).sum()))


def greedy_select(corr: pd.DataFrame, candidates, size, industry, cap,
                  seed_names=(), rng=None):
    """
    Add, one at a time, whichever candidate most raises effective bets.

    Ties are broken randomly rather than by any characteristic of the name, so
    no ordering sneaks in through the back door.
    """
    rng = rng or np.random.default_rng(0)
    syms = [s for s in candidates if s in corr.columns]
    chosen = [s for s in seed_names if s in corr.columns]
    counts = {}
    for s in chosen:
        counts[industry.get(s, "?")] = counts.get(industry.get(s, "?"), 0) + 1

    scores = []
    while len(chosen) < size:
        best, best_score = [], -np.inf
        for s in syms:
            if s in chosen:
                continue
            ind = industry.get(s, "?")
            if cap and counts.get(ind, 0) >= cap:
                continue
            sub = corr.loc[chosen + [s], chosen + [s]].values
            sc = effective_bets_from_corr(sub)
            if sc > best_score + 1e-12:
                best, best_score = [s], sc
            elif abs(sc - best_score) <= 1e-12:
                best.append(s)
        if not best:
            break
        pick = str(rng.choice(best)) if len(best) > 1 else best[0]
        chosen.append(pick)
        counts[industry.get(pick, "?")] = counts.get(industry.get(pick, "?"), 0) + 1
        scores.append(best_score)

    # The objective must never go backwards.
    for i in range(1, len(scores)):
        if scores[i] < scores[i - 1] - 1e-9:
            raise AssertionError(
                f"Effective bets fell at step {i}: {scores[i-1]:.4f} -> "
                f"{scores[i]:.4f}. Greedy selection is not behaving monotonically.")
    return chosen, scores


def main() -> int:
    p = argparse.ArgumentParser(description="Propose a universe from a screen")
    p.add_argument("--screen", help="screener export CSV")
    p.add_argument("--demo", action="store_true",
                   help="run against the live universe as the candidate pool")
    p.add_argument("--size", type=int, default=40)
    p.add_argument("--industry-cap", type=int, default=3,
                   help="maximum names per industry (0 disables)")
    p.add_argument("--min-coverage", type=float, default=0.95,
                   help="fraction of sessions a symbol must have data for")
    p.add_argument("--window", type=int, default=200,
                   help="trading days of history for the correlation estimate")
    p.add_argument("--no-incumbency", action="store_true",
                   help="ignore the current universe entirely")
    p.add_argument("--random-trials", type=int, default=200,
                   help="random same-size draws to score the proposal against")
    p.add_argument("--seed", type=int, default=20260917)
    p.add_argument("--start", default="2010-01-01")
    args = p.parse_args()

    print("=" * 96)
    print("UNIVERSE PROPOSAL")
    print(f"Started {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 96)

    defensive = set(defensive_symbols())
    incumbents = [s for s in current_symbols() if s not in defensive]

    if args.screen:
        raw = pd.read_csv(args.screen)
        cand = normalize_columns(raw)
        print(f"\nScreen: {len(cand)} names from {args.screen}")
    elif args.demo:
        cand = pd.DataFrame({"symbol": incumbents})
        print(f"\nDEMO: using the live universe ({len(cand)} names) as the pool")
    else:
        raise SystemExit("Pass --screen <csv> or --demo")

    if "industry" not in cand.columns:
        pool = REPO_ROOT / "random_pool.csv"
        if pool.exists():
            m = pd.read_csv(pool).set_index("symbol")["sector"].to_dict()
            cand["industry"] = cand["symbol"].map(m)
            print("  no industry column; fell back to sector from random_pool.csv")
        else:
            cand["industry"] = "?"
            print("  WARNING: no industry data; the industry cap is inert")
    industry = dict(zip(cand["symbol"], cand["industry"].fillna("?")))

    import yfinance as yf
    warnings.filterwarnings("ignore")
    syms = sorted(set(cand["symbol"]) | set(incumbents))
    print(f"\nDownloading {len(syms)} symbols "
          f"(~{len(syms)//400 + 1} batched requests)")
    frames = []
    for i in range(0, len(syms), 400):
        d = yf.download(syms[i:i + 400], start=args.start, interval="1d",
                        auto_adjust=True, progress=False, threads=True,
                        group_by="column")
        frames.append(d["Close"] if isinstance(d.columns, pd.MultiIndex) else d)
    close = pd.concat(frames, axis=1)

    # Order matters, and getting it wrong is silent.  Drop sparse COLUMNS
    # first, then incomplete ROWS.  With a few hundred symbols the union index
    # picks up sessions where a single odd ticker traded and nothing else did;
    # every other column is NaN on that row, so dropping incomplete columns
    # first would discard the entire universe.  An earlier version of this did
    # exactly that and reported "1 usable candidate".
    sub = close.iloc[-(args.window + 1):]
    rets = sub.pct_change().iloc[1:]
    n0 = rets.shape[1]

    coverage = rets.notna().mean()
    rets = rets.loc[:, coverage >= args.min_coverage]
    n1 = rets.shape[1]

    rows0 = len(rets)
    rets = rets.dropna(axis=0, how="any")
    print(f"  panel: {n0} symbols -> {n1} with >={args.min_coverage:.0%} "
          f"coverage -> {rets.shape[1]} usable; "
          f"{rows0} sessions -> {len(rets)} complete")
    if len(rets) < 60:
        raise SystemExit(
            f"Only {len(rets)} complete sessions survive. The correlation "
            f"estimate would be meaningless; check the panel.")
    corr = rets.corr()
    usable = [s for s in cand["symbol"] if s in corr.columns]
    if len(usable) < 2:
        raise SystemExit(
            f"Only {len(usable)} candidates have {args.window} clean sessions. "
            f"Nothing can be selected; check the panel rather than trusting a "
            f"proposal built on it.")
    dropped = sorted(set(cand["symbol"]) - set(usable))
    print(f"{len(usable)} candidates with {args.window}d of clean history"
          + (f"; dropped {len(dropped)}: {', '.join(dropped[:8])}"
             + ("..." if len(dropped) > 8 else "") if dropped else ""))

    keep = [] if args.no_incumbency else [s for s in incumbents if s in usable]
    failed = [s for s in incumbents if s not in set(cand["symbol"])]
    if not args.no_incumbency:
        print(f"\nIncumbents kept (still pass the screen): {len(keep)}")
        if failed:
            print(f"Incumbents NO LONGER in the screen — dropped: "
                  f"{', '.join(failed)}")

    rng = np.random.default_rng(args.seed)
    chosen, curve = greedy_select(corr, usable, args.size, industry,
                                  args.industry_cap, seed_names=keep, rng=rng)
    eb = effective_bets_from_corr(corr.loc[chosen, chosen].values)

    inc_ok = [s for s in incumbents if s in corr.columns]
    eb_inc = effective_bets_from_corr(corr.loc[inc_ok, inc_ok].values)

    # Random draws MUST be the same size as the proposal.  Effective bets rises
    # mechanically with the number of names, so scoring a 31-name proposal
    # against 40-name draws is not a fair comparison - it is a rigged one, and
    # it made a working selector look worse than random.
    n_draw = len(chosen)
    draws = []
    for _ in range(args.random_trials):
        pick = list(rng.choice(usable, size=min(n_draw, len(usable)),
                               replace=False))
        draws.append(effective_bets_from_corr(corr.loc[pick, pick].values))
    draws = np.array(draws)

    if len(chosen) < args.size:
        n_ind = len(set(industry.values()))
        print("")
        print(f"  NOTE: selection stopped at {len(chosen)} of {args.size} "
              f"requested.")
        print(f"        The industry cap of {args.industry_cap} binds: "
              f"{n_ind} industries x {args.industry_cap} = "
              f"{n_ind * args.industry_cap} maximum.")
        print("        Raise --industry-cap or supply finer industry labels.")

    print("\n" + "=" * 96)
    print("RESULT")
    print("=" * 96)
    print(f"  Proposed universe          {len(chosen)} names, "
          f"{eb:.2f} effective bets  ({eb/len(chosen):.1%} of ticker count)")
    print(f"  Current universe           {len(inc_ok)} names, "
          f"{eb_inc:.2f} effective bets  ({eb_inc/max(1,len(inc_ok)):.1%})")
    print(f"  Random {n_draw}-name draws       mean {draws.mean():.2f}, "
          f"sd {draws.std():.2f}, best {draws.max():.2f}")
    print(f"  Proposal percentile vs random draws: "
          f"{(draws < eb).mean():.1%}")
    if (draws < eb).mean() < 0.5:
        print("    ^ the proposal is WORSE than a coin flip against random "
              "draws;\n      the greedy objective is not buying anything here.")

    added = [s for s in chosen if s not in incumbents]
    kept = [s for s in chosen if s in incumbents]
    gone = [s for s in inc_ok if s not in chosen]
    print(f"\n  Kept {len(kept)}   Added {len(added)}   Dropped {len(gone)}")
    if added:
        print(f"    added:   {', '.join(added)}")
    if gone:
        print(f"    dropped: {', '.join(gone)}")

    worst = corr.loc[chosen, chosen].where(
        ~np.eye(len(chosen), dtype=bool)).stack().sort_values(ascending=False)
    print(f"\n  Most correlated pairs in the proposal ({args.window}d):")
    for (a, b), v in worst.head(5).items():
        print(f"    {a:<6} {b:<6} {v:>6.2f}")

    ind_counts = pd.Series([industry.get(s, "?") for s in chosen]).value_counts()
    print(f"\n  Industry spread: {len(ind_counts)} industries, "
          f"max {ind_counts.max()} in one")

    pd.DataFrame({"Symbol": chosen,
                  "Industry": [industry.get(s, "?") for s in chosen],
                  "Status": ["kept" if s in incumbents else "added"
                             for s in chosen]}).to_csv(REPO_ROOT / OUT, index=False)
    print(f"\nProposal written to {REPO_ROOT / OUT}")
    print(f"\nCompleted {datetime.now():%Y-%m-%d %H:%M:%S}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
