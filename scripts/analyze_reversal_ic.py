"""
Does a trend-CHANGE signal carry cross-sectional information the production
composite does not?

Track F established that the live ranker's IC is indistinguishable from zero at
every horizon, and negative at 42-63 days: at the live hold, this universe
reverses rather than trends.  `momentum/reversal.py` builds signals that try to
trade that fact deliberately - turn, strength, and position in the 52-week
range - instead of discovering it as a sign error.

This is stage one only.  It measures whether the signals PREDICT, using the
same IC / decile / top-K machinery Track F used, and it does not simulate a
portfolio.  A backtest of a signal that has not cleared this test would mostly
be measuring turnover, and the repo already has five results saying turnover
costs money here.

WHAT IT IS BUILT TO AVOID GETTING WRONG

Everything `analyze_ranker_ic.py` guards - overlapping forward windows, the
wrong eligible pool, the wrong price convention - is inherited by IMPORTING
that script's functions rather than reimplementing them.  There is one IC
implementation in this repo and this uses it.

On top of that, five checks run BEFORE any result is printed, and the script
exits non-zero without reporting if any of them fails:

  1. The vectorized rolling OLS matches `scipy.stats.linregress` fitted
     directly on sampled windows.  An independent implementation, not the same
     algebra twice.
  2. `range_pos` lies in [0, 1] and matches a brute-force rolling min/max.
  3. No look-ahead: every term recomputed on a panel TRUNCATED at date T equals
     its full-panel value at T.  This is the check that matters most for a new
     signal, and the one a plausible-looking result would otherwise hide.
  4. The IC pipeline reproduces Track F's PUBLISHED mean IC table, on Track F's
     own 52-symbol universe, to 5e-4.
  5. The wide-pool loader reproduces production `load_data` exactly on the
     universe both can load.

THE BENCHMARKS THAT MATTER

Not SPY.  Two arms, both required:

  * `composite (live)`   - what the production ranker scores today.
  * `composite negated`  - the free reversal signal.  If this universe reverses
    at 14-63 days, flipping the existing score's sign is a reversal strategy
    that costs nothing to build.  Any new construction that does not beat it
    has rediscovered a minus sign with extra steps.

Run:  python scripts/analyze_reversal_ic.py
      python scripts/analyze_reversal_ic.py --pool universe --floor on
      python scripts/analyze_reversal_ic.py --slope-window 40 --room-weight -1
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from momentum.data import PriceData, load_data
from momentum.experiments import production_config
from momentum.reversal import (ReversalConfig, build_terms, composite,
                               range_position, rolling_ols)
from momentum.strategy import compute_scores
from momentum.universe import current_symbols, defensive_symbols

from scripts.analyze_ranker_ic import (decile_table, eligible_mask,
                                       forward_returns, ic_stats,
                                       print_subperiods, row_spearman,
                                       topk_table)

OUT_IC = "rsi_ma_reversal_ic.csv"
OUT_DECILE = "rsi_ma_reversal_decile.csv"
OUT_TOPK = "rsi_ma_reversal_topk.csv"
OUT_ROBUST = "rsi_ma_reversal_robustness.csv"

POOL_FILE = "random_pool.csv"

#: Track F's published mean IC for the live blended score, FINDINGS.md
#: "The information coefficient: zero at every horizon".  Reproduced here as a
#: reconciliation target, not as a remembered fact.
TRACK_F_IC = {5: 0.0002, 14: 0.0026, 21: -0.0009, 42: -0.0057,
              63: -0.0137, 126: 0.0018}

#: The universe Track F ran on, before the 2026-09-17 update cut it to 46.
TRACK_F_UNIVERSE = "snapshots/universe/universe_2026-09-17_pre-update.csv"

NON_EQUITY = {"SHY", "TLT", "IAU", "SH", "SPY", "^VIX"}


# ==========================================================================
# Panel loading
# ==========================================================================

def pool_symbols(pool_file: str = POOL_FILE, verbose: bool = True) -> list:
    """
    Tradable symbols from a screener pool, with the compliance list applied.

    This is not a detail to handle later.  The 747-name pool is an external
    screener export that nobody curated, so restricted names are expected to
    appear in it — and five do (AMT, CCI, DLR, IRM, WULF).  WULF was among the
    ten most-picked names in the first Track J measurement, which means part of
    that edge was earned on a name the account cannot trade.  Any figure
    computed over an unfiltered pool overstates what is reachable.

    `filter_screen` matches on symbol, industry AND issuer name, so a foreign
    or alternate line of a restricted issuer is caught even under a ticker that
    is not on the list.  Filtering (rather than raising) is correct here for
    the reason that function documents: an uncurated external list containing
    restricted names is expected, not a failure.  The live universe is the
    opposite case and still raises.
    """
    from momentum.restrictions import filter_screen

    frame = pd.read_csv(REPO_ROOT / pool_file)
    kept, dropped = filter_screen(frame, symbol_col="symbol")
    if verbose and len(dropped):
        why = dropped.groupby("restricted_by")["symbol"].apply(
            lambda s: ", ".join(sorted(s))).to_dict()
        print(f"  compliance: dropped {len(dropped)} restricted name(s) from "
              f"{pool_file}")
        for key, syms in why.items():
            print(f"    by {key}: {syms}")
    return sorted(set(kept["symbol"].dropna()) - NON_EQUITY)


def _pool_cache(symbols, start):
    key = f"pool_{len(symbols)}sym_" + hashlib.md5(
        ",".join(sorted(symbols)).encode()).hexdigest()[:12]
    d = REPO_ROOT / ".price_cache"
    d.mkdir(parents=True, exist_ok=True)
    return (d / f"{key}_close_{start}.pkl", d / f"{key}_open_{start}.pkl",
            d / f"{key}_aux_{start}.pkl")


def load_pool_panel(symbols, start="2010-01-01", use_cache=True,
                    verbose=True) -> PriceData:
    """
    A price panel for an arbitrary symbol list, cleaned exactly as `load_data`
    cleans one.

    `load_data` issues a single yfinance request, which is fine for 46 symbols
    and unreliable for 750, so this chunks the download the way
    `analyze_random_baskets.download_panel` does.  The cleaning afterwards is
    deliberately identical to production's - phantom-row drop, then the
    "roughly a year of history somewhere in the record" column filter - because
    check 5 asserts the two loaders agree, and they can only agree if the
    cleaning is the same.
    """
    import yfinance as yf
    warnings.filterwarnings("ignore")

    c_path, o_path, a_path = _pool_cache(symbols, start)
    if use_cache and all(p.exists() for p in (c_path, o_path, a_path)):
        if verbose:
            print(f"  loading pool panel from cache ({c_path.name})")
        close = pd.read_pickle(c_path)
        open_ = pd.read_pickle(o_path)
        aux = pd.read_pickle(a_path)
        return PriceData(close=close, open_=open_,
                         spy=aux["SPY"], vix=aux["VIX"])

    syms = sorted(set(symbols))
    frames_c, frames_o = [], []
    chunk = 400
    for i in range(0, len(syms), chunk):
        part = syms[i:i + chunk]
        d = yf.download(part, start=start, interval="1d", auto_adjust=True,
                        progress=False, threads=True, group_by="column")
        frames_c.append(d["Close"])
        frames_o.append(d["Open"])
        if verbose:
            print(f"    {min(i + chunk, len(syms))}/{len(syms)} symbols")
    close = pd.concat(frames_c, axis=1)
    open_ = pd.concat(frames_o, axis=1)

    aux = yf.download(["SPY", "^VIX"], start=start, interval="1d",
                      auto_adjust=True, progress=False,
                      group_by="column")["Close"]
    spy, vix = aux["SPY"], aux["^VIX"]

    # --- production's cleaning, step for step (momentum/data.py) ---
    phantom = close.isna().all(axis=1)
    if phantom.any():
        close = close.loc[~phantom]
        open_ = open_.reindex(close.index)

    one_year_ago = close.index[-1] - pd.DateOffset(months=12)
    recent_rows = close.loc[one_year_ago:].shape[0]
    close = close.dropna(axis=1, thresh=recent_rows)
    open_ = open_.reindex(columns=close.columns)

    spy = spy.reindex(close.index).ffill()
    vix = vix.reindex(close.index).ffill()

    close.to_pickle(c_path)
    open_.to_pickle(o_path)
    pd.DataFrame({"SPY": spy, "VIX": vix}).to_pickle(a_path)
    return PriceData(close=close, open_=open_, spy=spy, vix=vix)


# ==========================================================================
# Validation gate
# ==========================================================================

class ValidationFailure(RuntimeError):
    pass


def check_rolling_ols(prices: pd.DataFrame, window: int, rng, n_samples=150,
                      tol=1e-8) -> str:
    """
    Check 1: the vectorized rolling OLS against an independent implementation.

    `scipy.stats.linregress` is fitted on the raw window, so this compares
    algebra against a library rather than against a second copy of itself.
    Both slope and standard error are checked - a vectorization can get the
    slope right and the residual variance wrong, and `slope_change_t` divides
    by the latter.
    """
    from scipy.stats import linregress

    slope, se = rolling_ols(prices, window)
    y = np.log(prices.astype(float))

    cols = list(prices.columns)
    worst_slope = worst_se = 0.0
    checked = 0
    guard = 0
    while checked < n_samples and guard < n_samples * 40:
        guard += 1
        col = cols[rng.integers(len(cols))]
        i = int(rng.integers(window, len(prices)))
        if not np.isfinite(slope.iloc[i][col]) or not np.isfinite(se.iloc[i][col]):
            continue
        seg = y[col].iloc[i - window + 1:i + 1]
        if seg.isna().any():
            continue
        fit = linregress(np.arange(window, dtype=float), seg.to_numpy())
        worst_slope = max(worst_slope, abs(fit.slope - slope.iloc[i][col]))
        worst_se = max(worst_se, abs(fit.stderr - se.iloc[i][col]))
        checked += 1

    if checked < 20:
        raise ValidationFailure(
            f"rolling OLS check could only sample {checked} finite windows")
    if worst_slope > tol or worst_se > tol:
        raise ValidationFailure(
            f"rolling OLS disagrees with scipy.linregress: "
            f"slope gap {worst_slope:.3e}, se gap {worst_se:.3e} > {tol:.0e}")
    return (f"slope gap {worst_slope:.2e}, se gap {worst_se:.2e} "
            f"over {checked} sampled windows (tol {tol:.0e})")


def check_range_position(prices: pd.DataFrame, cfg: ReversalConfig, rng,
                         n_samples=150, tol=1e-12) -> str:
    """
    Check 2: `range_pos` is bounded in [0, 1] and matches a brute-force
    rolling min/max on sampled points.
    """
    rp = range_position(prices, cfg)
    finite = rp.to_numpy()[np.isfinite(rp.to_numpy())]
    if finite.size == 0:
        raise ValidationFailure("range_pos is empty")
    lo, hi = float(finite.min()), float(finite.max())
    if lo < -tol or hi > 1 + tol:
        raise ValidationFailure(
            f"range_pos out of bounds: min {lo:.6f}, max {hi:.6f}")

    px = prices.astype(float)
    cols = list(prices.columns)
    w = cfg.room_window
    worst = 0.0
    checked = 0
    guard = 0
    while checked < n_samples and guard < n_samples * 40:
        guard += 1
        col = cols[rng.integers(len(cols))]
        i = int(rng.integers(w, len(prices)))
        if not np.isfinite(rp.iloc[i][col]):
            continue
        seg = px[col].iloc[i - w + 1:i + 1]
        if seg.isna().any():
            continue
        expect = (seg.iloc[-1] - seg.min()) / (seg.max() - seg.min())
        worst = max(worst, abs(expect - rp.iloc[i][col]))
        checked += 1

    if checked < 20:
        raise ValidationFailure(
            f"range_pos check could only sample {checked} finite points")
    if worst > 1e-10:
        raise ValidationFailure(f"range_pos brute-force gap {worst:.3e}")
    return (f"bounds [{lo:.4f}, {hi:.4f}], brute-force gap {worst:.2e} "
            f"over {checked} points")


def check_no_lookahead(prices: pd.DataFrame, cfg: ReversalConfig, rng,
                       n_dates=5, tol=1e-10) -> str:
    """
    Check 3: no term uses data from after the date it is stamped with.

    Recompute every term on a panel truncated at date T and compare against the
    full-panel value at T.  A centred window, an off-by-one shift, or any
    normalization that reaches across the whole history would show up here and
    nowhere else - the resulting scores stay perfectly plausible.
    """
    full = build_terms(prices, cfg)
    n = len(prices)
    worst = {}
    for _ in range(n_dates):
        i = int(rng.integers(int(n * 0.55), n))
        truncated = build_terms(prices.iloc[:i + 1], cfg)
        for name, frame in full.items():
            a = frame.iloc[i]
            b = truncated[name].iloc[-1]
            both = a.notna() & b.notna()
            if not both.any():
                continue
            gap = float((a[both] - b[both]).abs().max())
            worst[name] = max(worst.get(name, 0.0), gap)

    if not worst:
        raise ValidationFailure("look-ahead check compared nothing")
    bad = {k: v for k, v in worst.items() if v > tol}
    if bad:
        raise ValidationFailure(
            "terms change when the panel is truncated (look-ahead): "
            + ", ".join(f"{k} {v:.3e}" for k, v in bad.items()))
    return (", ".join(f"{k} {v:.1e}" for k, v in sorted(worst.items()))
            + f"  (tol {tol:.0e}, {n_dates} truncation dates)")


def check_track_f(start: str, horizons, tol=5e-4) -> str:
    """
    Check 4: this script's IC pipeline reproduces Track F's published table.

    Run on Track F's own 52-symbol universe (the snapshot taken before the
    2026-09-17 update), from the cached panel of that date.  A large
    `cache_max_age_hours` is the point: the check is against a FIXED historical
    panel, and re-downloading would extend it by days and move the numbers for
    reasons that have nothing to do with correctness.

    This is what ties every table below to a published result.  Without it the
    reconciliation would be against my own re-derivation, which is no
    reconciliation at all.
    """
    snap = REPO_ROOT / TRACK_F_UNIVERSE
    if not snap.exists():
        raise ValidationFailure(f"Track F universe snapshot missing: {snap}")

    rows = pd.read_csv(snap)
    syms = sorted(rows[rows["Active"] == "Y"]["Symbol"].tolist())

    prices = load_data(syms, start_date=start, use_cache=True,
                       cache_max_age_hours=1e9, verbose=False)
    cfg = production_config()
    ranking, base, _ = compute_scores(prices, cfg)
    mask = eligible_mask(ranking, base, cfg, set(defensive_symbols()))

    gaps = []
    for h, published in TRACK_F_IC.items():
        if h not in horizons:
            continue
        fwd = forward_returns(prices, h, "open").reindex_like(ranking)
        ic = row_spearman(ranking.where(mask), fwd.where(mask))
        got = ic_stats(ic, h)["Mean IC"]
        gaps.append((h, published, got, abs(got - published)))

    if not gaps:
        raise ValidationFailure("no Track F horizon was checked")
    worst = max(g[3] for g in gaps)
    detail = "  ".join(f"{h}d {pub:+.4f}/{got:+.4f}" for h, pub, got, _ in gaps)
    if worst > tol:
        raise ValidationFailure(
            f"IC pipeline does not reproduce Track F (worst gap {worst:.5f} > "
            f"{tol:.0e}).  Published/reproduced: {detail}.  Refusing to report "
            f"a new signal measured by a pipeline that cannot reproduce the "
            f"old one.")
    return (f"panel {prices.index[0]:%Y-%m-%d}..{prices.index[-1]:%Y-%m-%d}, "
            f"{len(syms)} symbols; worst gap {worst:.5f} (tol {tol:.0e})\n"
            f"                 published/reproduced  {detail}")


def check_pool_loader(start: str, tol=1e-5) -> str:
    """
    Check 5: `load_pool_panel` reproduces production `load_data`.

    The wide-pool results are only comparable to the universe results if both
    panels were built the same way.  This runs the live universe through both
    loaders and requires the close panels to be identical.
    """
    syms = current_symbols()
    prod = load_data(syms, start_date=start, use_cache=True,
                     cache_max_age_hours=1e9, verbose=False)
    mine = load_pool_panel(syms, start=start, use_cache=True, verbose=False)

    shared = [c for c in prod.close.columns if c in mine.close.columns]
    if len(shared) < len(prod.close.columns) * 0.95:
        raise ValidationFailure(
            f"pool loader kept {len(shared)} of {len(prod.close.columns)} "
            f"production columns")
    idx = prod.close.index.intersection(mine.close.index)
    if len(idx) < len(prod.close.index) * 0.95:
        raise ValidationFailure(
            f"pool loader index covers {len(idx)} of {len(prod.close.index)} "
            f"production sessions")

    a = prod.close.loc[idx, shared]
    b = mine.close.loc[idx, shared]
    # RELATIVE, not absolute.  yfinance hands back float32 on one of these two
    # paths, so a $1,000 stock disagrees by ~3e-4 in dollars purely from
    # representation - measured, not assumed: the worst offender is COST at a
    # relative gap of 1.8e-6.  An absolute tolerance tight enough to catch a
    # genuinely misaligned panel would fail on that noise, and one loose enough
    # to pass it would be meaningless on a $15 name.
    diff = ((a - b).abs() / b.abs()).to_numpy()
    compared = int(np.isfinite(diff).sum())

    # np.nanmax, not max: cells NaN in either panel compare to NaN, and a plain
    # max over an array containing NaN returns NaN - which then fails every
    # `gap > tol` test and lets a broken loader through silently.  The count of
    # finite comparisons is asserted for the same reason: a gap of 0.0 over
    # zero compared cells is not a passing check.
    if compared < 0.5 * len(idx) * len(shared):
        raise ValidationFailure(
            f"pool loader check compared only {compared:,} cells of "
            f"{len(idx) * len(shared):,}; the panels barely overlap")
    gap = float(np.nanmax(diff))
    if not np.isfinite(gap) or gap > tol:
        raise ValidationFailure(
            f"pool loader disagrees with load_data by {gap:.3e} relative on "
            f"close prices (tol {tol:.0e})")
    return (f"{len(shared)} shared symbols, {len(idx)} shared sessions, "
            f"{compared:,} cells compared, max relative close gap {gap:.2e} "
            f"(tol {tol:.0e})")


def run_validation(start: str, horizons, seed: int, skip_track_f: bool) -> None:
    """Every check, printed. Raises `ValidationFailure` rather than warning."""
    print("\n" + "=" * 104)
    print("VALIDATION - nothing below is printed unless all of these pass")
    print("=" * 104)

    rng = np.random.default_rng(seed)
    cfg = ReversalConfig()
    prices = load_data(current_symbols(), start_date=start, use_cache=True,
                       cache_max_age_hours=1e9, verbose=False)

    checks = [
        ("1  rolling OLS vs scipy.linregress",
         lambda: check_rolling_ols(prices.close, cfg.slope_window, rng)),
        ("2  range_pos bounds and brute force",
         lambda: check_range_position(prices.close, cfg, rng)),
        ("3  no look-ahead under truncation",
         lambda: check_no_lookahead(prices.close, cfg, rng)),
        ("5  pool loader vs production load_data",
         lambda: check_pool_loader(start)),
    ]
    if not skip_track_f:
        checks.insert(3, ("4  IC pipeline vs Track F published table",
                          lambda: check_track_f(start, horizons)))

    for label, fn in checks:
        detail = fn()
        print(f"  PASS  {label}\n                 {detail}")

    if skip_track_f:
        print("\n  !! CHECK 4 SKIPPED (--skip-track-f).  The IC pipeline is "
              "unreconciled against\n     any published result. Treat every "
              "number below as provisional.")
    print()


# ==========================================================================
# Signals under test
# ==========================================================================

def build_signals(prices: PriceData, cfg: ReversalConfig, model_cfg):
    """
    Every panel to be scored, plus the production benchmarks.

    Returned in a fixed order so the printed matrices stay comparable between
    pools and between runs.
    """
    ranking, base, _ = compute_scores(prices, model_cfg)
    terms = build_terms(prices.close, cfg)

    from dataclasses import replace
    cfg_room_neg = replace(cfg, room_weight=-1.0)
    cfg_room_pos = replace(cfg, room_weight=1.0)

    # The sign-flipped arms are not a second hypothesis - they are the same
    # measurement read in the other direction, and they exist so that the
    # top-K table is computed on the book a trader would actually hold.  An
    # edge table for `flip` says nothing about what ranking by `-flip` buys,
    # because the top 64 of one is not the bottom 64 of the other once other
    # terms are blended in.  Labelled "negated" throughout so no reader mistakes
    # them for independent findings.
    cfg_pullback = replace(cfg, turn_weight=-1.0, strength_weight=0.0,
                           room_weight=1.0)
    cfg_pullback_s = replace(cfg, turn_weight=-1.0, strength_weight=1.0,
                             room_weight=1.0)

    signals = {
        "composite (live)": ranking,
        "composite negated": -ranking,
        "turn_t  (b)": terms["turn_t"],
        "turn_t negated": -terms["turn_t"],
        "flip    (c)": terms["flip"],
        "flip negated": -terms["flip"],
        "strength_t": terms["strength_t"],
        "range_pos": terms["range_pos"],
        "b: turn+strength": composite(terms, cfg, "turn_t"),
        "c: flip+strength": composite(terms, cfg, "flip"),
        "b + room(-1)": composite(terms, cfg_room_neg, "turn_t"),
        "b + room(+1)": composite(terms, cfg_room_pos, "turn_t"),
        "c + room(-1)": composite(terms, cfg_room_neg, "flip"),
        "PULLBACK -flip+room": composite(terms, cfg_pullback, "flip"),
        "PULLBACK +strength": composite(terms, cfg_pullback_s, "flip"),
    }
    signals = {k: v.reindex_like(ranking) for k, v in signals.items()}
    return signals, ranking, base


# ==========================================================================
# Robustness: is the top-K edge real, and what is it made of?
# ==========================================================================

def edge_periods(panel, fwd, mask, h: int, k: int) -> pd.Series:
    """
    Per-period top-K edge over the eligible-pool mean, on NON-OVERLAPPING
    periods.

    The headline top-K table reports a mean with no standard error, which is
    how a large and meaningless number gets mistaken for an edge.  Sampling
    every h-th date gives one observation per disjoint holding period, so the
    t-statistic computed from this is honest in the same sense `ic_stats` is.
    """
    s = panel.where(mask)
    f = fwd.where(mask & fwd.notna())
    rank = s.rank(axis=1, ascending=False, method="first")
    top = f.where(rank <= k).mean(axis=1)
    return (top - f.mean(axis=1)).dropna().iloc[::h]


def _edge_row(e: pd.Series) -> dict:
    n = len(e)
    if n < 6:
        return {}
    thirds = np.array_split(np.arange(n), 3)
    p = [float(e.iloc[x[0]:x[-1] + 1].mean()) for x in thirds]
    return {"Edge": float(e.mean()), "sd": float(e.std()), "N": n,
            "t": float(e.mean() / (e.std() / np.sqrt(n))),
            "P1": p[0], "P2": p[1], "P3": p[2]}


def print_edge_table(rows: list, title: str, note: str = "") -> None:
    print(f"\n  {title}")
    if note:
        print(f"  {note}")
    hdr = (f"    {'variant':<44}{'h':>4}{'K':>4}{'Edge':>10}{'t':>7}"
           f"{'P1':>9}{'P2':>9}{'P3':>9}")
    print(hdr)
    print("    " + "-" * (len(hdr) - 4))
    for r in rows:
        flag = " *" if abs(r["t"]) >= 2.0 else ""
        print(f"    {r['Variant']:<44}{r['Horizon']:>4}{r['K']:>4}"
              f"{r['Edge']:>9.2%}{r['t']:>7.2f}"
              f"{r['P1']:>9.2%}{r['P2']:>9.2%}{r['P3']:>9.2%}{flag}")


def robustness_report(signals, prices, base_mask, fwd, arms, horizons, ks,
                      pool_file) -> pd.DataFrame:
    """
    Three questions the IC and top-K tables cannot answer.

    1. SIGNIFICANCE.  Does the per-period edge clear |t|=2 on non-overlapping
       periods, and is it present in all three subperiods?  FINDINGS' standing
       bar is that a result winning the full sample by winning one segment is a
       fit, not a finding.

    2. SURVIVORSHIP.  The pool was built in 2026 from names liquid TODAY.  A
       signal that buys pullbacks in strong performers is the most exposed
       construction there is: every dip in this sample was followed by a
       recovery, because names whose dips did not recover are not in the pool.
       That cannot be fixed without point-in-time data, but it can be BOUNDED -
       drop the names we know ex-post were the biggest winners and re-measure.
       If the edge collapses, it was memory rather than signal.

    3. SIZE.  Where does the edge live?  An edge concentrated in $1B names is
       not tradable at the 7.5 bps this repo assumes, and is not reachable from
       a universe made of mega-caps.
    """
    all_rows = []
    pool = pd.read_csv(REPO_ROOT / pool_file).set_index("symbol")
    caps = pool["marketCap"]
    close = prices.close
    full_ret = close.ffill().iloc[-1] / close.bfill().iloc[0] - 1

    def measure(arm, variant, mask):
        out = []
        for h in horizons:
            for k in ks:
                st = _edge_row(edge_periods(signals[arm], fwd[h], mask, h, k))
                if st:
                    st.update({"Variant": variant, "Arm": arm,
                               "Horizon": h, "K": k})
                    out.append(st)
        all_rows.extend(out)
        return out

    print("\n" + "=" * 104)
    print("ROBUSTNESS - significance, survivorship bound, and where the edge lives")
    print("=" * 104)

    for arm in arms:
        print_edge_table(
            measure(arm, "all names (as reported above)", base_mask),
            f"{arm}  -  per-period edge, non-overlapping periods",
            "* marks |t| >= 2. Read P1/P2/P3 before the mean.")

        surv = []
        for pct in (0.10, 0.20, 0.30):
            cutoff = full_ret.quantile(1 - pct)
            drop = set(full_ret[full_ret >= cutoff].index)
            m = base_mask.copy()
            m[[c for c in m.columns if c in drop]] = False
            surv += measure(arm, f"ex top {pct:.0%} realized winners "
                                 f"(-{len(drop)} names)", m)
        print_edge_table(surv, f"{arm}  -  SURVIVORSHIP BOUND",
                         "Names we know ex-post were the biggest winners, "
                         "removed. A collapse here means\n  the edge was "
                         "hindsight; survival means it is doing something else.")

        size = []
        for floor_cap, lab in ((2e9, "$2B"), (10e9, "$10B"), (50e9, "$50B")):
            keep = set(caps[caps >= floor_cap].index)
            m = base_mask.copy()
            m[[c for c in m.columns if c not in keep]] = False
            size += measure(arm, f"only names above {lab} cap (today)", m)
        print_edge_table(size, f"{arm}  -  WHERE THE EDGE LIVES, by size",
                         "Cap is TODAY's, so this is a descriptive cut and not "
                         "a tradable filter.")

        picks = picked_names(signals[arm], base_mask, k=max(ks), caps=caps)
        print(f"\n  {arm}  -  what it buys at K={max(ks)}: "
              f"{picks['n_rows']:,} (date, pick) rows over "
              f"{picks['n_names']} distinct names")
        print(f"    median cap of picks ${picks['median']/1e9:,.1f}B "
              f"vs pool median ${caps.median()/1e9:,.1f}B; "
              f"{picks['under_2b']:.1%} of picks under $2B "
              f"vs {(caps < 2e9).mean():.1%} of the pool")
        print(f"    most-picked: {', '.join(picks['top_names'])}")

    return pd.DataFrame(all_rows)


def picked_names(panel, mask, k, caps) -> dict:
    """Composition of the top-K book, for reading what a signal actually buys."""
    rank = panel.where(mask).rank(axis=1, ascending=False, method="first")
    sel = rank.le(k).stack()
    sel = sel[sel].reset_index()
    sel.columns = ["Date", "Ticker", "_"]
    cap = sel["Ticker"].map(caps)
    return {
        "n_rows": len(sel), "n_names": sel["Ticker"].nunique(),
        "median": float(cap.median()),
        "under_2b": float((cap < 2e9).mean()),
        "top_names": list(sel["Ticker"].value_counts().head(10).index),
    }


# ==========================================================================
# Reporting
# ==========================================================================

def print_matrix(table: pd.DataFrame, value: str, horizons, title: str,
                 fmt: str, flag_t: bool = False) -> None:
    print(f"\n  {title}")
    header = f"    {'signal':<20}" + "".join(f"{str(h) + 'd':>10}" for h in horizons)
    print(header)
    print("    " + "-" * (len(header) - 4))
    for sig in table["Signal"].unique():
        sub = table[table["Signal"] == sig].set_index("Horizon")
        cells = ""
        for h in horizons:
            if h in sub.index:
                v = sub.loc[h, value]
                mark = ""
                if flag_t and abs(sub.loc[h, "t"]) >= 2.0:
                    mark = "*"
                cells += f"{format(v, fmt) + mark:>10}"
            else:
                cells += f"{'-':>10}"
        print(f"    {sig:<20}{cells}")


def main() -> int:
    p = argparse.ArgumentParser(
        description="Cross-sectional information in trend-change signals")
    p.add_argument("--horizons", type=int, nargs="+",
                   default=[5, 14, 21, 42, 63, 126])
    p.add_argument("--pool", choices=["universe", "wide", "both"],
                   default="both")
    p.add_argument("--floor", choices=["on", "off", "both"], default="both",
                   help="apply the min_level_threshold eligibility floor")
    p.add_argument("--bins", type=int, default=5)
    p.add_argument("--topk", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--price", choices=["open", "close"], default="open")
    p.add_argument("--start", default="2010-01-01")
    p.add_argument("--seed", type=int, default=20260920)
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--robustness-arms", nargs="+",
                   default=["flip negated", "composite (live)"],
                   help="arms to put through the significance / survivorship / "
                        "size cuts (wide pool only)")
    p.add_argument("--robustness-k", type=int, nargs="+", default=[4, 8])
    p.add_argument("--robustness-horizons", type=int, nargs="+",
                   default=[14, 63])
    p.add_argument("--skip-track-f", action="store_true",
                   help="skip the published-value reconciliation (check 4). "
                        "Prints a warning banner and marks results provisional.")

    # Every ReversalConfig knob, so nothing here is a hard-coded threshold.
    p.add_argument("--slope-window", type=int, default=60)
    p.add_argument("--long-window", type=int, default=252)
    p.add_argument("--long-skip", type=int, default=21)
    p.add_argument("--short-window", type=int, default=63)
    p.add_argument("--room-window", type=int, default=252)
    p.add_argument("--turn-weight", type=float, default=1.0)
    p.add_argument("--strength-weight", type=float, default=1.0)
    p.add_argument("--room-weight", type=float, default=0.0)
    p.add_argument("--normalization", choices=["cross_sectional", "rank"],
                   default="cross_sectional")
    args = p.parse_args()

    cfg = ReversalConfig(
        slope_window=args.slope_window, long_window=args.long_window,
        long_skip=args.long_skip, short_window=args.short_window,
        room_window=args.room_window, turn_weight=args.turn_weight,
        strength_weight=args.strength_weight, room_weight=args.room_weight,
        normalization=args.normalization,
    )

    print("=" * 104)
    print("TREND-CHANGE SIGNALS - CROSS-SECTIONAL INFORMATION")
    print(f"Started {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"horizons {args.horizons}   forward returns "
          f"{args.price}-to-{args.price}   normalization {cfg.normalization}")
    print(f"slope window {cfg.slope_window}   flip {cfg.long_window}/"
          f"{cfg.long_skip} vs {cfg.short_window}   room {cfg.room_window}")
    print("=" * 104)

    try:
        run_validation(args.start, args.horizons, args.seed, args.skip_track_f)
    except ValidationFailure as exc:
        print("\n" + "=" * 104)
        print("VALIDATION FAILED - refusing to report")
        print("=" * 104)
        print(f"  {exc}")
        return 3

    model_cfg = production_config()
    defensive = set(defensive_symbols()) | NON_EQUITY

    pools = []
    if args.pool in ("universe", "both"):
        pools.append(("universe (46)", None))
    if args.pool in ("wide", "both"):
        pools.append(("wide pool (747)", POOL_FILE))

    floors = (["on", "off"] if args.floor == "both" else [args.floor])

    ic_rows, dec_rows, topk_rows, robust_written = [], [], [], []

    for pool_label, pool_file in pools:
        print("\n" + "=" * 104)
        print(f"POOL: {pool_label}")
        print("=" * 104)

        if pool_file is None:
            prices = load_data(current_symbols(), start_date=args.start,
                               use_cache=not args.no_cache, verbose=False,
                               drop_unsettled=True)
        else:
            syms = pool_symbols(pool_file)
            print(f"  {len(syms)} tradable symbols from {pool_file}")
            prices = load_pool_panel(syms, start=args.start,
                                     use_cache=not args.no_cache)

        print(f"  panel: {prices.close.shape[1]} tickers, "
              f"{prices.index[0]:%Y-%m-%d} to {prices.index[-1]:%Y-%m-%d}")

        print("  scoring...", flush=True)
        signals, ranking, base = build_signals(prices, cfg, model_cfg)

        fwd = {h: forward_returns(prices, h, args.price).reindex_like(ranking)
               for h in args.horizons}

        for floor in floors:
            if floor == "on":
                mask = eligible_mask(ranking, base, model_cfg, defensive)
            else:
                # Same history gate, no level floor: the freefall filter is
                # exactly what a turn signal needs removed, since it screens
                # out the names that were recently bad.
                counts = ranking.notna().cumsum()
                mask = ranking.notna() & (counts >= model_cfg.min_data_days)
                for sym in defensive:
                    if sym in mask.columns:
                        mask[sym] = False

            pool_size = mask.sum(axis=1)
            print(f"\n  ELIGIBILITY FLOOR {floor.upper()}  -  "
                  f"{pool_size.mean():.1f} names per date "
                  f"({pool_size.min():.0f} to {pool_size.max():.0f})")

            # The live book is 4 names out of ~41 eligible - the top ~9.8%.
            # On a 700-name pool a flat K=4 is a far sharper cut than the model
            # actually makes, so the proportional cut is carried alongside and
            # is the one the headline edge table reports.
            matched = max(1, int(round(0.098 * pool_size.mean())))
            ks = sorted(set(args.topk) | {matched})
            print(f"  top-K cuts {ks}  (the live book is 4 of ~41 = top 9.8%, "
                  f"so K={matched} is the size-matched cut here)")

            ic_keep = {}
            for name, panel in signals.items():
                for h in args.horizons:
                    ic = row_spearman(panel.where(mask), fwd[h].where(mask))
                    ic_keep[(name, h)] = ic
                    st = ic_stats(ic, h)
                    if not st:
                        continue
                    # Smallest mean IC this test could have detected, at 80%
                    # power and two-sided alpha=0.05, on the non-overlapping
                    # dates.  Reported because a null result is uninterpretable
                    # without it: Track F's IC-decline finding turned entirely
                    # on the same calculation (it could not detect a decline
                    # smaller than eighteen times the effect it was measuring).
                    st["MDE(80%)"] = (2.802 * st["IC sd"] / np.sqrt(st["N indep"])
                                      if st["N indep"] > 1 else np.nan)
                    st.update({"Pool": pool_label, "Floor": floor,
                               "Signal": name})
                    ic_rows.append(st)

                    d = decile_table(panel, fwd[h], mask, h, args.bins)
                    d["Pool"], d["Floor"], d["Signal"] = pool_label, floor, name
                    dec_rows.append(d)

                    t = topk_table(panel, fwd[h], mask, h, ks)
                    t["Pool"], t["Floor"], t["Signal"] = pool_label, floor, name
                    topk_rows.append(t)

            ic = pd.DataFrame([r for r in ic_rows
                               if r["Pool"] == pool_label and r["Floor"] == floor])
            print_matrix(ic, "Mean IC", args.horizons,
                         "Mean information coefficient  "
                         "(* marks |t| >= 2 on non-overlapping dates)",
                         "+.4f", flag_t=True)
            print_matrix(ic, "t", args.horizons,
                         "t-statistic, overlap-corrected", "+.2f")
            print_matrix(ic, "MDE(80%)", args.horizons,
                         "Smallest mean IC detectable at 80% power  "
                         "(compare against the IC above before reading a "
                         "null as evidence of no effect)", ".4f")

            dec = pd.concat([d for d in dec_rows
                             if d["Pool"].iloc[0] == pool_label
                             and d["Floor"].iloc[0] == floor])
            spread = []
            for (sig, h), g in dec.groupby(["Signal", "Horizon"]):
                best = g[g["Bin"] == args.bins]["Mean Fwd Return"]
                worst = g[g["Bin"] == 1]["Mean Fwd Return"]
                if len(best) and len(worst):
                    spread.append({"Signal": sig, "Horizon": h,
                                   "Spread": float(best.iloc[0] - worst.iloc[0])})
            print_matrix(pd.DataFrame(spread), "Spread", args.horizons,
                         f"Top-minus-bottom {args.bins}-bin spread in forward "
                         f"return", "+.2%")

            tk = pd.concat([t for t in topk_rows
                            if t["Pool"].iloc[0] == pool_label
                            and t["Floor"].iloc[0] == floor])
            sub = tk[tk["K"] == matched]
            print_matrix(sub, "Edge", args.horizons,
                         f"Top-{matched} edge over the eligible-pool mean "
                         f"(the quantity a book monetizes)", "+.2%")

            # --- what survived, and what that is worth after multiplicity ---
            n_cells = len(signals) * len(args.horizons)
            # Half the arms are sign flips of another arm, so they carry the
            # same information with the opposite sign and are not new tests.
            n_families = n_cells / 2
            expected = 0.05 * n_families
            flagged = ic[ic["t"].abs() >= 2.0]

            print(f"\n  MULTIPLICITY.  {n_cells} (signal x horizon) cells "
                  f"printed above, about {n_families:.0f} independent of sign; "
                  f"at\n  alpha=0.05 roughly {expected:.1f} would clear |t|>=2 "
                  f"by chance alone.  {len(flagged)} did.")
            if len(flagged):
                print("  Horizons overlap and the arms are correlated, so "
                      "treat these as leads to\n  re-test, not as findings. "
                      "Subperiod stability is the first filter:")
                for _, r in flagged.iterrows():
                    print(f"\n  {r['Signal']} at {r['Horizon']:.0f}d  "
                          f"(IC {r['Mean IC']:+.4f}, t {r['t']:+.2f}, "
                          f"MDE {r['MDE(80%)']:.4f})")
                    print_subperiods(ic_keep[(r["Signal"], int(r["Horizon"]))],
                                     int(r["Horizon"]), 3)

            # The survivorship and size cuts only mean something on the wide
            # pool - a 46-name universe has nothing to drop and no size range.
            if pool_file is not None and floor == "on":
                arms = [a for a in args.robustness_arms if a in signals]
                missing = set(args.robustness_arms) - set(arms)
                if missing:
                    print(f"\n  (robustness arms not in the signal set, "
                          f"skipped: {sorted(missing)})")
                if arms:
                    rob = robustness_report(
                        signals, prices, mask, fwd, arms,
                        args.robustness_horizons, args.robustness_k, pool_file)
                    rob.to_csv(REPO_ROOT / OUT_ROBUST, index=False)
                    robust_written.append(len(rob))

    ic_out = pd.DataFrame(ic_rows)
    dec_out = pd.concat(dec_rows, ignore_index=True)
    tk_out = pd.concat(topk_rows, ignore_index=True)
    ic_out.to_csv(REPO_ROOT / OUT_IC, index=False)
    dec_out.to_csv(REPO_ROOT / OUT_DECILE, index=False)
    tk_out.to_csv(REPO_ROOT / OUT_TOPK, index=False)

    print("\n" + "=" * 104)
    print("EXPORTS")
    print("=" * 104)
    print(f"  IC       {REPO_ROOT / OUT_IC}  ({len(ic_out):,} rows)")
    print(f"  Deciles  {REPO_ROOT / OUT_DECILE}  ({len(dec_out):,} rows)")
    print(f"  Top-K    {REPO_ROOT / OUT_TOPK}  ({len(tk_out):,} rows)")
    if robust_written:
        print(f"  Robust   {REPO_ROOT / OUT_ROBUST}  "
              f"({robust_written[-1]:,} rows)")
    print(f"\nCompleted {datetime.now():%Y-%m-%d %H:%M:%S}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
