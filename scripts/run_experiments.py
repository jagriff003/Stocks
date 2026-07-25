"""
Parameter sweeps.

    python scripts/run_experiments.py execution   how much did the unachievable
                                                  same-close fill flatter us?
    python scripts/run_experiments.py velocity    level/velocity weight sweep
    python scripts/run_experiments.py vix         legacy VIX regime thresholds
    python scripts/run_experiments.py zscore      z-score method and window
    python scripts/run_experiments.py all

Every suite runs over one shared price panel so differences are attributable to
the parameter under test.  Read the deltas, not the levels: absolute figures
carry survivorship bias from applying today's screened universe backwards.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from momentum.config import GraduatedVixConfig, VelocityConfig, VixRegimeConfig
from momentum.data import load_data
from momentum.experiments import (Experiment, export_comparison, legacy_config,
                                  print_comparison, print_subperiods,
                                  production_config, run_experiments, variant)
from momentum.universe import current_symbols


# --------------------------------------------------------------------------
# Suites
# --------------------------------------------------------------------------

def suite_execution():
    """
    What the execution assumption is worth.

    The pre-refactor backtest filled at the close that generated the signal.
    This isolates that assumption from everything else: same scores, same
    selections, only the fill convention and slippage change.

    If the strategy's edge largely evaporates between 'same_close' and
    'next_open', that is the single most important thing to know before tuning
    anything else — because every historical parameter choice was made against
    the flattering version.
    """
    base = legacy_config()   # legacy blend, so ONLY execution varies here
    return [
        variant(base, "same_close_0bps", "pre-refactor assumption",
                execution__execute_at="same_close", execution__slippage_bps=0.0),
        variant(base, "same_close_7.5bps", "same fill, realistic slippage",
                execution__execute_at="same_close", execution__slippage_bps=7.5),
        variant(base, "next_open_0bps", "realistic fill, no slippage",
                execution__execute_at="next_open", execution__slippage_bps=0.0),
        variant(base, "next_open_7.5bps", "realistic fill and slippage",
                execution__execute_at="next_open", execution__slippage_bps=7.5),
        variant(base, "next_close_7.5bps", "full day of action lag",
                execution__execute_at="next_close", execution__slippage_bps=7.5),
    ]


def suite_velocity():
    """
    Level vs velocity weighting, on a corrected scale.

    The previous sweep that selected 0.7/0.3 ran against a normalization
    mismatch — the level was a rolling z-score, the velocity was
    cross-sectional, so the nominal weights did not correspond to actual
    influence.  This re-runs the question with both components on the same
    scale, which makes 'invert the weights' a meaningful test rather than a
    confounded one.

    L0/V100 is included because it *is* Track C's "rank on the rate of change of
    the z-score instead of its level" — no new machinery needed.
    """
    base = production_config()
    experiments = [
        variant(base, "legacy_scale_L70V30", "old confounded scale (reference)",
                velocity__blend_normalization="legacy"),
        variant(base, "level_only", "no velocity component",
                velocity=None),
    ]

    for window in (5, 10, 20):
        for lw in (1.0, 0.7, 0.5, 0.3, 0.0):
            name = f"w{window}_L{int(lw*100)}V{int((1-lw)*100)}"
            experiments.append(
                variant(base, name,
                        "velocity-only ranking (Track C #2)" if lw == 0.0 else "",
                        velocity__velocity_window=window,
                        velocity__level_weight=lw,
                        velocity__velocity_weight=round(1.0 - lw, 4))
            )

    # Rank-based normalization at the current weighting, as an outlier-robustness
    # check on the whole approach.
    experiments.append(
        variant(base, "w10_L70V30_rank", "rank-normalized blend",
                velocity__blend_normalization="rank")
    )
    return experiments


def suite_robustness():
    """
    Is the velocity-window optimum a plateau or a spike?

    The main sweep hands back w5_L70V30 winning CAGR, Sharpe, drawdown and
    Calmar simultaneously, while its immediate neighbour w10_L70V30 gives up
    six points of CAGR.  A parameter surface that sharp is the signature of
    fitting to history, and the strategy's own design philosophy flags exactly
    this: weights calibrated to historical relationships decay as markets
    evolve.

    Two questions this answers:
      1. Sweeping the window finely (3/5/7/10/15) — does performance fall off a
         cliff either side of 5, or degrade gracefully?
      2. Does rank normalization, which is immune to velocity outliers, flatten
         the surface?  If the z-scored version is spiky and the rank version is
         flat at a similar level, the spike was outlier luck and rank is the
         safer choice even where it scores marginally lower.
    """
    base = production_config()
    experiments = []

    for window in (3, 5, 7, 10, 15):
        for norm in ("cross_sectional", "rank"):
            tag = "z" if norm == "cross_sectional" else "rank"
            experiments.append(
                variant(base, f"w{window}_L70V30_{tag}", "",
                        velocity__velocity_window=window,
                        velocity__level_weight=0.7,
                        velocity__velocity_weight=0.3,
                        velocity__blend_normalization=norm)
            )

    # True no-velocity control.  Note this is NOT the same as L100/V0: with
    # velocity_weight=0.0 the term 0.0 * NaN is still NaN, so a L100V0 config
    # silently drops stocks that lack velocity history.  Only velocity=None
    # isolates the level cleanly.
    experiments.append(variant(base, "level_only", "true no-velocity control",
                               velocity=None))
    return experiments


def suite_correlation():
    """
    Does correlation-aware selection pay, and when?

    The ranker scores each stock alone, so nothing stops two of four slots going
    to the same bet in different tickers.  This tests folding the correlation
    matrices — currently exported and read by eye — into selection.

    The central question is not "is diversification good" but WHEN it is worth
    paying for.  Forcing a swap down the ranking costs expected return every
    time it fires; correlation only costs you when correlated names gap
    together, which is a stress phenomenon.  In a calm tape, three correlated
    winners are concentration you are being paid for.

    So the sweep varies the VIX gate (never / 25 / 20 / 15 / always) as the
    primary axis, and threshold style as the secondary one:

      absolute  a fixed cap.  Simple, but never binds in calm markets and can
                reject everything in a selloff, when pairwise correlations all
                converge toward 0.9.
      relative  a percentile of TODAY's correlation distribution.  Always
                excludes the most-redundant tail of what is on offer, whatever
                the absolute level.
    """
    base = production_config()
    exps = [variant(base, "no_corr_filter", "baseline: rank only",
                    correlation__enabled=False)]

    for gate in (25.0, 20.0, 15.0, None):
        label = "always" if gate is None else f"vix{gate:g}"
        exps.append(
            variant(base, f"rel85_{label}",
                    f"relative p85, gated above VIX {gate or 0:g}",
                    correlation__enabled=True,
                    correlation__method="relative",
                    correlation__max_percentile=0.85,
                    correlation__apply_above_vix=gate)
        )

    # Threshold style and strictness, at the gate that the axis above favours.
    for pct in (0.70, 0.95):
        exps.append(
            variant(base, f"rel{int(pct * 100)}_vix20", "",
                    correlation__enabled=True,
                    correlation__method="relative",
                    correlation__max_percentile=pct,
                    correlation__apply_above_vix=20.0)
        )

    for cap in (0.70, 0.85):
        exps.append(
            variant(base, f"abs{int(cap * 100)}_vix20", "",
                    correlation__enabled=True,
                    correlation__method="absolute",
                    correlation__max_correlation=cap,
                    correlation__apply_above_vix=20.0)
        )

    # Correlation window: how much history the estimate uses.
    for window in (20, 200):
        exps.append(
            variant(base, f"rel85_vix20_w{window}", f"{window}d correlation window",
                    correlation__enabled=True,
                    correlation__method="relative",
                    correlation__max_percentile=0.85,
                    correlation__window=window,
                    correlation__apply_above_vix=20.0)
        )

    # Hold fewer, better-diversified positions rather than backfilling.
    exps.append(
        variant(base, "rel85_vix20_holdfewer", "concentrate size, not exposure",
                correlation__enabled=True,
                correlation__method="relative",
                correlation__max_percentile=0.85,
                correlation__apply_above_vix=20.0,
                correlation__on_infeasible="hold_fewer")
    )
    return exps


def suite_ladder():
    """
    Track A — graduated absolute-level VIX ladder vs the legacy z-score regime.

    Two independent changes are being tested, and the sweep separates them:

      1. ABSOLUTE LEVEL instead of z-score.  A z-score measures deviation from a
         recent baseline, so a sustained moderate VIX becomes the new baseline
         and stops registering — 88% of days in the 15-20 band were classified
         'normal'.  VIX has spent 52% of the last three years in that band.

      2. DAILY evaluation instead of rebalance-gated.  Of days the old detector
         flagged as crisis, only 25% were actually holding defensive positions;
         the rest were riding pre-crisis picks waiting for the clock.

    'ladder_X_rebalance_only' isolates (1) by keeping the old timing, so the
    difference against 'ladder_X' is the value of daily evaluation alone.

    Ladder shapes, as momentum slots out of 4 per band [<15, 15-20, 20-25, 25+]:
      gentle    [4,4,3,1]  barely reacts until genuinely stressed
      moderate  [4,3,2,0]  steps down through every band
      steep     [4,3,1,0]  aggressive de-risking
      binary    [4,4,4,0]  a control: no graduation at all, just a cliff at 25
    """
    base = production_config()
    shapes = {
        "gentle":   [4, 4, 3, 1],
        "moderate": [4, 3, 2, 0],
        "steep":    [4, 3, 1, 0],
        "binary":   [4, 4, 4, 0],
    }

    exps = [
        variant(base, "legacy_zscore_vix", "current production regime",
                graduated_vix=None),
        variant(base, "no_regime", "no volatility overlay at all",
                vix=None, graduated_vix=None),
    ]

    for name, slots in shapes.items():
        exps.append(
            variant(base, f"ladder_{name}", f"slots {slots}, daily",
                    vix=None,
                    graduated_vix=GraduatedVixConfig(momentum_slots=slots))
        )

    # Isolate daily evaluation from the absolute-level change.
    exps.append(
        variant(base, "ladder_moderate_reb_only", "same ladder, rebalance-gated",
                vix=None,
                graduated_vix=GraduatedVixConfig(momentum_slots=[4, 3, 2, 0],
                                                 evaluate_daily=False))
    )

    # Hysteresis: how much confirmation before re-risking.
    for step_up in (1, 5, 10):
        exps.append(
            variant(base, f"ladder_mod_up{step_up}d", f"re-risk after {step_up}d",
                    vix=None,
                    graduated_vix=GraduatedVixConfig(momentum_slots=[4, 3, 2, 0],
                                                     step_up_days=step_up))
        )

    # Smoothing VIX as an alternative to hysteresis.
    for window in (5, 10):
        exps.append(
            variant(base, f"ladder_mod_smooth{window}", f"{window}d VIX average",
                    vix=None,
                    graduated_vix=GraduatedVixConfig(momentum_slots=[4, 3, 2, 0],
                                                     smoothing_window=window))
        )

    # Band placement: does the 15 edge matter, given 52% of recent days sit there?
    exps.append(
        variant(base, "ladder_edges_17_22_27", "shifted bands",
                vix=None,
                graduated_vix=GraduatedVixConfig(band_edges=[17.0, 22.0, 27.0],
                                                 momentum_slots=[4, 3, 2, 0]))
    )
    return exps


def suite_ladder_extreme():
    """
    Track A, second pass — does the absolute ladder survive at genuine extremes?

    The first ladder sweep rejected the hypothesis as specified: every graduated
    absolute-level ladder lost to the z-score regime on BOTH return and
    drawdown, and lost to having no overlay at all.

    Two mechanisms explain that, and this suite tests whether either can be
    dodged rather than assuming the whole idea is dead:

    1. The ladder was on far too often.  VIX >= 15 is 65% of days, and the
       15-20 band earns +19.1% annualized.  De-risking there swaps a 19% return
       for a ~2% one.  The band table describes conditional MARKET returns; it
       is not evidence that de-risking in those bands helps.

    2. Absolute VIX lags.  By the time it has ARRIVED above 25, the drawdown has
       largely happened — you sell the bottom and miss the recovery.  A z-score
       fires on unusual vol relative to a recent baseline, closer to the onset.

    If (1) is the whole story, pushing the bands out to 25/30/35 — where the
    band really is dire and infrequent — should recover the loss.  If (2)
    dominates, no absolute threshold will help, because the timing is wrong at
    every level.  These two outcomes are distinguishable, which is the point.
    """
    base = production_config()
    exps = [
        variant(base, "legacy_zscore_vix", "current production regime",
                graduated_vix=None),
        variant(base, "no_regime", "no volatility overlay", vix=None,
                graduated_vix=None),
    ]

    ladders = {
        "e25_30_35_mild":  ([25.0, 30.0, 35.0], [4, 4, 3, 1]),
        "e25_30_35_steep": ([25.0, 30.0, 35.0], [4, 3, 2, 0]),
        "e30_35_40":       ([30.0, 35.0, 40.0], [4, 3, 2, 0]),
        "e28_35_45_cliff": ([28.0, 35.0, 45.0], [4, 4, 2, 0]),
        "only_above_30":   ([30.0, 40.0, 50.0], [4, 0, 0, 0]),
    }
    for name, (edges, slots) in ladders.items():
        exps.append(
            variant(base, f"ladder_{name}", f"edges {edges}, slots {slots}",
                    vix=None,
                    graduated_vix=GraduatedVixConfig(band_edges=edges,
                                                     momentum_slots=slots))
        )

    # Does re-risking faster recover the missed recovery? If mechanism (2) is
    # real, a short step_up should help materially at extreme edges.
    for step_up in (1, 10):
        exps.append(
            variant(base, f"ladder_e25_up{step_up}d", f"re-risk after {step_up}d",
                    vix=None,
                    graduated_vix=GraduatedVixConfig(band_edges=[25.0, 30.0, 35.0],
                                                     momentum_slots=[4, 3, 2, 0],
                                                     step_up_days=step_up))
        )
    return exps


def suite_ladder_zscore():
    """
    Track A, third pass — keep the machinery, change the basis.

    The spec bundled two changes and only one is refuted. Absolute-level banding
    fails structurally (see suite_ladder_extreme). But GRADUATION, DAILY
    EVALUATION and HYSTERESIS were never tested on a basis that works — they
    were only tested inside the broken ladder, where the basis dominated.

    This runs the same ladder machinery on VIX z-score bands, isolating each
    surviving idea against the current production regime:

      graduation      3 slots at z>=1.0 rather than jumping straight to 2
      daily eval      exposure re-decided every day, not only at rebalance.
                      This is the direct fix for "only 25% of crisis-flagged
                      days were actually holding defensive positions".
      hysteresis      confirmation before re-risking, to stop boundary churn

    'zladder_match_legacy' reproduces the production thresholds (elevated z=1.5
    -> 2 slots, crisis z=2.5 -> 0) in ladder form. It should land close to
    legacy_zscore_vix; a large gap would mean the ladder machinery itself is
    doing something unintended, and any apparent win from the variants below
    would be an artifact rather than a result.
    """
    base = production_config()
    exps = [
        variant(base, "legacy_zscore_vix", "current production regime",
                graduated_vix=None),
        variant(base, "no_regime", "no volatility overlay", vix=None,
                graduated_vix=None),
    ]

    def zladder(name, note, **kwargs):
        cfg = dict(band_basis="zscore", zscore_window=60,
                   band_edges=[1.5, 2.5, 3.5], momentum_slots=[4, 2, 0, 0])
        cfg.update(kwargs)
        return variant(base, name, note, vix=None,
                       graduated_vix=GraduatedVixConfig(**cfg))

    # Control: production thresholds expressed as a ladder, rebalance-gated.
    exps.append(zladder("zladder_match_legacy", "legacy thresholds, no daily eval",
                        evaluate_daily=False))
    # Same thresholds, daily evaluation — isolates the timing change alone.
    exps.append(zladder("zladder_daily", "legacy thresholds, DAILY"))

    # Graduation: an intermediate step instead of 4 -> 2.
    exps.append(zladder("zladder_grad", "graduated 4/3/2/0, daily",
                        band_edges=[1.0, 1.75, 2.5],
                        momentum_slots=[4, 3, 2, 0]))
    exps.append(zladder("zladder_grad_gentle", "graduated 4/3/1, later edges",
                        band_edges=[1.5, 2.25, 3.0],
                        momentum_slots=[4, 3, 1, 0]))

    # Hysteresis on the re-risk side, with daily evaluation live.
    for step_up in (1, 5, 10):
        exps.append(zladder(f"zladder_daily_up{step_up}d",
                            f"re-risk after {step_up}d", step_up_days=step_up))

    # Does a slower baseline detect transitions better?
    for window in (30, 120):
        exps.append(zladder(f"zladder_daily_win{window}",
                            f"{window}d z-score baseline", zscore_window=window))
    return exps


def suite_overnight():
    """
    Is daily de-risking a bad signal, or a good signal you cannot reach?

    Daily evaluation lost 2.5pp of CAGR under realistic (next_open) execution.
    Two very different explanations fit that:

      A. The signal is wrong — reacting daily chases noise, and no execution
         improvement would rescue it.
      B. The signal is right but unreachable — a regime decision taken on
         close(T) cannot be acted on until open(T+1), and a large share of
         volatility is realized in the overnight gap. You sell AFTER the gap
         down, banking the loss without avoiding it, then buy back after the
         gap up.

    Running each config under both execution conventions separates them. If
    daily evaluation is competitive under 'same_close' and only loses under
    'next_open', explanation B holds: the idea is sound and the constraint is
    the trading window, not the model.

    'same_close' is not achievable — it fills at the close that generated the
    signal. It is included strictly as a diagnostic upper bound.
    """
    base = production_config()

    def zcfg(**kwargs):
        cfg = dict(band_basis="zscore", zscore_window=60,
                   band_edges=[1.5, 2.5, 3.5], momentum_slots=[4, 2, 0, 0])
        cfg.update(kwargs)
        return GraduatedVixConfig(**cfg)

    exps = []
    for execution in ("same_close", "next_open"):
        tag = "SC" if execution == "same_close" else "NO"
        exps.append(
            variant(base, f"rebalance_gated_{tag}", f"{execution}",
                    vix=None, graduated_vix=zcfg(evaluate_daily=False),
                    execution__execute_at=execution)
        )
        exps.append(
            variant(base, f"daily_{tag}", f"{execution}",
                    vix=None, graduated_vix=zcfg(evaluate_daily=True),
                    execution__execute_at=execution)
        )
        # Zero-cost twins, so the comparison is not confounded by the extra
        # slippage daily evaluation incurs from its higher turnover.
        exps.append(
            variant(base, f"daily_{tag}_0bps", f"{execution}, no slippage",
                    vix=None, graduated_vix=zcfg(evaluate_daily=True),
                    execution__execute_at=execution,
                    execution__slippage_bps=0.0)
        )
        exps.append(
            variant(base, f"rebalance_gated_{tag}_0bps", f"{execution}, no slippage",
                    vix=None, graduated_vix=zcfg(evaluate_daily=False),
                    execution__execute_at=execution,
                    execution__slippage_bps=0.0)
        )
    return exps


def suite_exits():
    """
    Track B — per-stock rank exits, decoupled from the buy clock.

    The frequency evidence replicates: 75% of holdings drop out of the top 4
    before their hold ends, median day 4, only 17.8% recover.

    The return evidence does not support the stated premise. From the moment an
    exit would trigger, positions go on to earn +0.39% on average (median
    +0.00%, 44% negative) over the following ~7 days. Rank is RELATIVE — a stock
    slides from 4th to 6th because others rose, not because it fell — so rank
    decay carries little directional information about the holding.

    That leaves one live question, which this suite answers: an exit does not
    have to avoid a loss to be worth it, it only has to redeploy into something
    better. The replacement must beat +0.39% over ~7 days by more than the ~15
    bps round trip.

    Axes:
      exit_rank    how far a holding must fall (5 = twitchy, 8 = only on
                   genuine collapse)
      consecutive  confirmation days, the anti-flapping guard
      replacement  immediate / defensive / defer — Track A's overnight-premium
                   result predicts the non-invested variants lose badly, and
                   they are included to confirm that rather than assume it
      budget       cap on exits per month, since turnover is the binding cost
    """
    base = production_config()
    exps = [variant(base, "no_rank_exit", "baseline: 14-day clock only",
                    exits__enabled=False)]

    for rank in (5, 6, 8):
        for consecutive in (1, 2, 3):
            exps.append(
                variant(base, f"exit_r{rank}_d{consecutive}", "",
                        exits__enabled=True,
                        exits__exit_rank_threshold=rank,
                        exits__consecutive_days=consecutive,
                        exits__replacement="immediate")
            )

    for mode in ("defensive", "defer"):
        exps.append(
            variant(base, f"exit_r6_d2_{mode}", f"replacement={mode}",
                    exits__enabled=True, exits__exit_rank_threshold=6,
                    exits__consecutive_days=2, exits__replacement=mode)
        )

    for budget in (1, 2):
        exps.append(
            variant(base, f"exit_r6_d2_max{budget}pm", f"max {budget} exits/month",
                    exits__enabled=True, exits__exit_rank_threshold=6,
                    exits__consecutive_days=2, exits__replacement="immediate",
                    exits__max_exits_per_month=budget)
        )

    # Minimum hold, guarding against same-week round trips.
    exps.append(
        variant(base, "exit_r6_d2_minhold7", "min 7 days before exit eligible",
                exits__enabled=True, exits__exit_rank_threshold=6,
                exits__consecutive_days=2, exits__replacement="immediate",
                exits__min_hold_days=7)
    )
    return exps


def suite_swaps():
    """
    Track B, second pass — "sell when there's a better buy available".

    The rank-trigger version was rejected: every variant lost, monotonically
    with turnover, because rank is relative and a holding sliding from 4th to
    6th carries no information about its own prospects (measured forward return
    after trigger: +0.39% mean, +0.00% median).

    This tests the opportunity-triggered version instead. A swap fires only when
    the best available candidate beats the weakest holding by more than
    `min_score_gap` composite z-units — the transaction-cost hurdle expressed in
    signal terms.

    The live risk is timing, not logic: if the gap only opens after the
    challenger has already run, the rule buys the top. The model is documented
    to be late to its own winners (60% of new entrants had already spent 10+ of
    the prior 40 sessions in the top 8). `scripts/analyze_swap_quality.py`
    measures challenger-vs-incumbent forward returns to settle that directly,
    rather than inferring it from the CAGR column.

    A gap of 0.0 is included as the degenerate control: swap whenever ANY
    candidate outranks a holding. That is continuous rebalancing, and it should
    be terrible — if it is not, something is wrong with the cost model.
    """
    base = production_config()
    exps = [variant(base, "no_swap", "baseline: 14-day clock only",
                    exits__enabled=False)]

    for gap in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5):
        exps.append(
            variant(base, f"swap_gap{gap:g}", f"challenger must beat by {gap:g}z",
                    exits__enabled=True, exits__mode="score_gap",
                    exits__min_score_gap=gap, exits__consecutive_days=2)
        )

    # Confirmation days at the gap most likely to be viable.
    for days in (1, 3, 5):
        exps.append(
            variant(base, f"swap_gap1.0_d{days}", f"{days}d confirmation",
                    exits__enabled=True, exits__mode="score_gap",
                    exits__min_score_gap=1.0, exits__consecutive_days=days)
        )

    # Turnover budget, the binding constraint on everything in this family.
    for budget in (1, 2):
        exps.append(
            variant(base, f"swap_gap1.0_max{budget}pm", f"max {budget}/month",
                    exits__enabled=True, exits__mode="score_gap",
                    exits__min_score_gap=1.0, exits__consecutive_days=2,
                    exits__max_exits_per_month=budget)
        )
    return exps


def suite_vix():
    """Legacy VIX regime thresholds. Superseded by the Track A ladder."""
    base = production_config()
    return [
        variant(base, "no_vix_filter", "baseline", vix=None),
        variant(base, "vix_z1.0_2.0", "",
                vix=VixRegimeConfig(elevated_zscore=1.0, crisis_zscore=2.0)),
        variant(base, "vix_z0.75_1.5", "reacts sooner",
                vix=VixRegimeConfig(elevated_zscore=0.75, crisis_zscore=1.5)),
        variant(base, "vix_z1.5_2.5", "current production",
                vix=VixRegimeConfig(elevated_zscore=1.5, crisis_zscore=2.5)),
        variant(base, "vix_z1.0_2.0_win30", "faster baseline",
                vix=VixRegimeConfig(elevated_zscore=1.0, crisis_zscore=2.0,
                                    zscore_window=30)),
        variant(base, "vix_z1.0_2.0_roc30", "with spike trigger",
                vix=VixRegimeConfig(elevated_zscore=1.0, crisis_zscore=2.0,
                                    use_roc_trigger=True, roc_threshold=0.30)),
    ]


def suite_zscore():
    """Z-score normalization method and window."""
    base = production_config()
    return [
        variant(base, "cross_sectional", "",
                scoring__zscore_method="cross_sectional",
                scoring__zscore_window=252, min_data_days=200),
        variant(base, "rolling_63d", "",
                scoring__zscore_method="rolling",
                scoring__zscore_window=63, min_data_days=200),
        variant(base, "rolling_126d", "current production",
                scoring__zscore_method="rolling",
                scoring__zscore_window=126, min_data_days=200),
        variant(base, "rolling_252d", "",
                scoring__zscore_method="rolling",
                scoring__zscore_window=252, min_data_days=252),
        variant(base, "rolling_504d", "",
                scoring__zscore_method="rolling",
                scoring__zscore_window=504, min_data_days=504),
    ]


SUITES = {
    "execution": (suite_execution, "rsi_ma_execution_comparison.csv",
                  "same_close_0bps"),
    "velocity":  (suite_velocity,  "rsi_ma_velocity_comparison.csv",
                  "legacy_scale_L70V30"),
    "robustness": (suite_robustness, "rsi_ma_velocity_robustness.csv",
                   "level_only"),
    "correlation": (suite_correlation, "rsi_ma_correlation_filter_comparison.csv",
                    "no_corr_filter"),
    "ladder":    (suite_ladder,    "rsi_ma_ladder_comparison.csv", "legacy_zscore_vix"),
    "ladder2":   (suite_ladder_extreme, "rsi_ma_ladder_extreme.csv", "legacy_zscore_vix"),
    "zladder":   (suite_ladder_zscore, "rsi_ma_ladder_zscore.csv", "legacy_zscore_vix"),
    "overnight": (suite_overnight, "rsi_ma_overnight_gap.csv", "rebalance_gated_NO"),
    "exits":     (suite_exits, "rsi_ma_exit_comparison.csv", "no_rank_exit"),
    "swaps":     (suite_swaps, "rsi_ma_swap_comparison.csv", "no_swap"),
    "vix":       (suite_vix,       "rsi_ma_vix_regime_comparison.csv",
                  "no_vix_filter"),
    "zscore":    (suite_zscore,    "rsi_ma_zscore_comparison.csv",
                  "rolling_126d"),
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run parameter sweeps")
    parser.add_argument("suite", choices=list(SUITES) + ["all"])
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    print("=" * 78)
    print("MOMENTUM PARAMETER SWEEPS")
    print(f"Started {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 78)

    prices = load_data(current_symbols(), start_date=args.start,
                       use_cache=not args.no_cache)

    names = list(SUITES) if args.suite == "all" else [args.suite]

    for name in names:
        builder, out_file, baseline = SUITES[name]
        experiments = builder()

        print(f"\n{'-' * 78}")
        print(f"SUITE: {name}  ({len(experiments)} configs)")
        print(f"{'-' * 78}")
        print((builder.__doc__ or "").rstrip())

        results = run_experiments(experiments, prices)
        print_comparison(results, baseline=baseline)
        print_subperiods(results, n_periods=3)
        export_comparison(results, str(REPO_ROOT / out_file))

    print(f"\nCompleted {datetime.now():%Y-%m-%d %H:%M:%S}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
