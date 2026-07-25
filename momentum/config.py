"""
Configuration objects for the momentum strategy.

Every tunable parameter in the strategy lives in one of these dataclasses.  That
is deliberate: `ScoringConfig` + `VelocityConfig` + `VixRegimeConfig` +
`ExecutionConfig` together are a complete, serializable description of a model
version.  `snapshot_config()` writes that description to a dated JSON file so a
future live-vs-backtest reconciliation can ask "what was the model actually
doing on 2026-04-17?" and get an answer instead of a git archaeology session.

See also `momentum.universe`, which does the same thing for the ticker list.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict, is_dataclass
from datetime import date
from pathlib import Path
from typing import List, Optional, Dict, Any


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

# Repo root — everything the strategy reads or writes is resolved relative to
# this, so the package works regardless of the caller's working directory.
REPO_ROOT = Path(__file__).resolve().parent.parent

SNAPSHOT_DIR = REPO_ROOT / "snapshots"
UNIVERSE_DIR = SNAPSHOT_DIR / "universe"
CONFIG_DIR = SNAPSHOT_DIR / "config"
OUTPUT_DIR = REPO_ROOT


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

@dataclass
class ScoringConfig:
    """
    Parameters for building the composite momentum score.

    The composite is the average of two standardized measures:
        (ma_diff_z + ma_deriv_z) / 2

    where ma_diff is (SMA[ma_short] - SMA[ma_long]) and ma_deriv is its
    `derivative_window`-day change.  RSI and relative-strength-vs-SPY are
    computed and exported for diagnostics but are NOT in the composite — RSI
    has been tested twice and carries no signal here.

    zscore_method
        'rolling'         — each stock standardized against its own
                            `zscore_window`-day history (time-series).  Current
                            production setting.
        'cross_sectional' — each day standardized across the universe.

    Note on `ma_short` / `ma_long`: 50/200 encodes months of trailing
    information for a ~14-day action horizon.  Prior testing showed uniformly
    shortening the whole MA basis hurts CAGR; adding a shorter-window signal
    *alongside* the existing one is untested and is a Track C question.
    """
    rsi_window: int = 14
    ma_short: int = 50
    ma_long: int = 200
    derivative_window: int = 5
    zscore_window: int = 126
    rel_strength_window: int = 20
    zscore_method: str = "rolling"       # 'rolling' | 'cross_sectional'


@dataclass
class VelocityConfig:
    """
    Parameters for blending z-score level with z-score velocity (rate of change).

        blended = level_weight * level_component + velocity_weight * velocity_component

    Both components are put on a common scale before blending (see
    `momentum.signals.apply_velocity_blend` and `blend_normalization` below) so
    the weights have an interpretable meaning: 0.5/0.5 really is equal
    influence.

    IMPORTANT — this differs from the pre-refactor behaviour.  The original code
    blended a *rolling* z-score level against a *cross-sectionally* normalized
    velocity.  Those two quantities have different cross-sectional dispersions,
    so the nominal 0.7/0.3 split did not correspond to 70/30 actual influence.
    Set `blend_normalization='legacy'` to reproduce the old (confounded)
    behaviour for parity checks; use 'cross_sectional' or 'rank' for any test
    where the weight values are supposed to mean something.

    blend_normalization
        'cross_sectional' — standardize BOTH components across the universe each
                            day before weighting.  Recommended default.
        'rank'            — convert both components to cross-sectional
                            percentile ranks before weighting.  Outlier-robust;
                            a stock with a wild velocity reading can't dominate.
        'legacy'          — pre-refactor behaviour: raw level + cross-sectional
                            velocity.  For reproducing historical results only.

    min_level_threshold
        Hard floor applied to the *level* score (never the blend): a stock in
        complete freefall is skipped no matter how strong its velocity bounce.
    """
    velocity_window: int = 10
    level_weight: float = 0.7
    velocity_weight: float = 0.3
    min_level_threshold: float = -3.0
    blend_normalization: str = "cross_sectional"   # 'cross_sectional' | 'rank' | 'legacy'

    @property
    def label(self) -> str:
        return (f"v{self.velocity_window}_L{self.level_weight:.0%}"
                f"V{self.velocity_weight:.0%}_{self.blend_normalization[:5]}")


# --------------------------------------------------------------------------
# Execution and costs
# --------------------------------------------------------------------------

@dataclass
class ExecutionConfig:
    """
    How a signal becomes a fill, and what that costs.

    The pre-refactor backtest ranked on the close of day T and booked the return
    from close(T) to close(T+1) — i.e. it assumed you could execute at the very
    close that produced the signal.  You cannot.  In elevated-VIX regimes the
    open-to-close swing is worth several percent, which is orders of magnitude
    larger than spread slippage, so this assumption is not a rounding error.

    execute_at
        'next_open'  — signal from close(T), fill at open(T+1).  Realistic
                       default; matches "check after the close, trade at the
                       open".
        'next_close' — fill at close(T+1).  Models a full-day action lag.
        'same_close' — fill at close(T).  Not achievable live; retained only to
                       quantify how much of the historical CAGR came from this
                       assumption.

    slippage_bps
        One-way cost in basis points, applied to each side of a trade.  Covers
        bid/ask spread and market impact.  Commission is assumed zero
        (commission-free retail brokerage).
        # REPLACE: raise this if your broker charges commission or you trade
        # size large enough to move the less-liquid names in the universe.

    tax_rate_short_term
        Reserved for a future tax-drag layer.  Left at 0.0 and unused for now;
        gross-of-tax results come first, and the tax layer switches on once the
        gross numbers are in.
        # REPLACE: your marginal short-term capital gains rate, if this is a
        # taxable account and you want tax-adjusted comparisons.

    risk_free_rate
        Used for Sharpe.  Roughly the 10-year Treasury yield.
    """
    execute_at: str = "next_open"        # 'next_open' | 'next_close' | 'same_close'
    slippage_bps: float = 7.5            # one-way, basis points
    tax_rate_short_term: float = 0.0     # not yet applied
    risk_free_rate: float = 0.045

    @property
    def slippage_frac(self) -> float:
        return self.slippage_bps / 10_000.0


# --------------------------------------------------------------------------
# VIX regime
# --------------------------------------------------------------------------

@dataclass
class GraduatedVixConfig:
    """
    Track A — absolute-level VIX ladder, evaluated daily.

    Replaces the z-score regime, which had two structural problems:

      1. It measured deviation from a recent baseline, not absolute level.  A
         sustained moderate VIX becomes the new baseline and stops registering:
         88% of days in the 15-20 band were classified 'normal'.  That band's
         realized Sharpe was 0.88 against 3.30 below 15, and VIX has spent 52%
         of the last three years there.

      2. It only acted at a scheduled rebalance.  Of days flagged 'crisis', just
         25% were actually holding defensive positions — the other 75% were
         riding pre-crisis picks because no rebalance had come due.

    The measured staircase this ladder is fitted to (backtest, by absolute band):

        VIX <15    Sharpe  3.30    ann. return  48.6%
        VIX 15-20  Sharpe  0.88    ann. return  19.1%
        VIX 20-25  Sharpe -0.99    ann. return -13.5%
        VIX 25+    Sharpe -1.48    ann. return -35.0%

    Monotonic, not a cliff — so the response is graduated, not binary.

    band_edges / momentum_slots
        `momentum_slots[i]` is how many of `top_n` stay in momentum names while
        VIX sits in band i.  len(momentum_slots) == len(band_edges) + 1.
        Remaining slots go to `defensive_symbols`.

        Ranking survives at every band but the last: the strategy keeps leaning
        into the best available opportunity rather than defaulting to cash, and
        only abandons relative selection when conditions are genuinely dire.
        A ladder with 0 anywhere but the final band contradicts that.

    step_down_days / step_up_days
        Hysteresis, asymmetric by default.  De-risking on the first day in a
        worse band is deliberate — the cost of being late is the whole point of
        Track A.  Re-risking waits for confirmation, because VIX oscillating
        around a boundary would otherwise churn the book at 2x slippage per
        round trip.

    smoothing_window
        Rolling mean applied to VIX before banding.  1 = spot.  Larger values
        stop a single intraday spike moving the ladder, at the cost of reacting
        later.  An alternative to hysteresis, not a complement — sweep both.
    """
    enabled: bool = True

    band_basis: str = "level"       # 'level' | 'zscore'
    #   MEASURED 2026-07-25, and the reason this field exists:
    #
    #   'level' — absolute VIX. Track A's original proposal. REJECTED on
    #   evidence. Every ladder shape lost to the z-score regime on BOTH return
    #   and drawdown, and lost to having no overlay at all. Pushing the bands
    #   out to 30/35/40 recovered the return (17.02% vs 16.96% for no overlay)
    #   but essentially none of the drawdown protection (-28.50% vs -29.02% for
    #   no overlay, against -19.23% for the z-score regime).
    #
    #   The reason is structural, not a tuning problem: absolute VIX crossing a
    #   threshold IS the drawdown. It is coincident-to-lagging, so acting on it
    #   sells into the loss and misses the recovery. No threshold fixes that,
    #   which is why every band placement failed the same way.
    #
    #   The band table that motivated the ladder describes conditional MARKET
    #   returns, not the value of de-risking. VIX 15-20 earns +19.1% annualized;
    #   de-risking across it swaps a 19% return for roughly 2%.
    #
    #   'zscore' — bands on the VIX z-score instead, keeping the graduation,
    #   daily evaluation and hysteresis that the ladder machinery provides while
    #   using the basis that actually detects transitions early.

    zscore_window: int = 60         # used when band_basis='zscore'

    band_edges: List[float] = field(default_factory=lambda: [15.0, 20.0, 25.0])
    momentum_slots: List[int] = field(default_factory=lambda: [4, 3, 2, 0])

    defensive_symbols: List[str] = field(default_factory=lambda: ["SHY", "TLT", "IAU"])

    # --- hysteresis ---
    step_down_days: int = 1     # days in a worse band before de-risking
    step_up_days: int = 3       # days in a better band before re-risking
    smoothing_window: int = 1   # rolling mean on VIX before banding

    evaluate_daily: bool = False
    #   MEASURED 2026-07-25 — daily evaluation was REJECTED on evidence, on a
    #   z-score basis where the regime signal itself works:
    #
    #       rebalance-gated  CAGR 20.05%  Sharpe 0.94  MaxDD -18.80%  130 trd/yr
    #       daily            CAGR 17.52%  Sharpe 0.81  MaxDD -18.94%  165 trd/yr
    #
    #   It costs 2.53pp of CAGR to buy 0.14pp of drawdown. All nine daily
    #   variants tested lost; none recovered it through hysteresis or smoothing.
    #
    #   Why: the 14-day rebalance clock was acting as an unintentional noise
    #   filter. Re-deciding exposure daily reacts to transient z-score spikes —
    #   de-risking into a dip and re-risking higher. Turnover rises from 130 to
    #   165 trades/year, which is ~0.8pp of slippage before any whipsaw cost.
    #
    #   This reframes the observation that only 25% of crisis-flagged days were
    #   actually holding defensive positions. That is not a defect to fix; it is
    #   the strategy staying invested through transient flags, which the data
    #   says was the right call.
    #
    #   Track B's per-stock rank exits are a DIFFERENT question and are not
    #   covered by this result — that is about a single position's own decay,
    #   not a portfolio-wide risk dial.

    def __post_init__(self):
        if len(self.momentum_slots) != len(self.band_edges) + 1:
            raise ValueError(
                f"momentum_slots must have {len(self.band_edges) + 1} entries "
                f"for {len(self.band_edges)} band edges, got "
                f"{len(self.momentum_slots)}"
            )
        if list(self.band_edges) != sorted(self.band_edges):
            raise ValueError("band_edges must be ascending")
        if any(a < b for a, b in zip(self.momentum_slots, self.momentum_slots[1:])):
            raise ValueError(
                "momentum_slots must be non-increasing — a higher VIX band "
                "cannot carry more momentum exposure than a calmer one"
            )

    @property
    def band_labels(self) -> List[str]:
        labels = [f"<{self.band_edges[0]:g}"]
        for lo, hi in zip(self.band_edges, self.band_edges[1:]):
            labels.append(f"{lo:g}-{hi:g}")
        labels.append(f"{self.band_edges[-1]:g}+")
        return labels


@dataclass
class CorrelationConfig:
    """
    Correlation-aware selection and universe screening.

    Two separate jobs, deliberately not conflated:

    1. SELECTION FILTER (medium-term window).  Stops two highly correlated names
       taking two of only four slots.  Applied greedily down the ranked list: the
       top-ranked candidate is always taken, then each next candidate is accepted
       only if it is not too correlated with what has already been accepted.
       Ranking stays the primary mechanism — this only ever says "not this one,
       take the next best", never "hold cash instead".

    2. UNIVERSE SCREEN (long-term window).  A quarterly diagnostic listing
       redundant pairs, so the screener can swap one out for something that adds
       an independent bet.  Advisory only; changes nothing at runtime.

    method — how the cap is set
        'absolute'  reject above `max_correlation`.  Simple and interpretable,
                    but behaves badly across regimes: in calm markets it never
                    binds, and in a selloff, when everything converges toward
                    0.9, it can reject every available candidate.
        'relative'  reject above the `max_percentile` quantile of TODAY's
                    pairwise correlation distribution.  Self-scaling: it always
                    excludes the most-redundant tail of what is currently on
                    offer, whatever the absolute level.  Consistent with ranking
                    against the universe rather than against a fixed bar.

    on_infeasible — when fewer than top_n candidates clear the cap
        'relax'      fill remaining slots with the best rejected candidates, in
                     rank order.  Never ends up under-invested.  Default.
        'hold_fewer' hold a smaller, genuinely diversified book, equal-weighted
                     across fewer names.  Concentrates position size instead of
                     sector exposure — a real trade-off, not a free lunch.

    A caution worth keeping in view: diversification lowers portfolio variance,
    but the constraint forces you down the ranking into lower-scoring names, and
    that costs expected return.  Whether the trade nets out is empirical.  Run
    `scripts/run_experiments.py correlation` rather than assuming it helps.
    """
    enabled: bool = True

    # --- when the filter is live ---
    apply_above_vix: Optional[float] = 25.0
    #   Correlation only costs you when correlation is what hurts.  In a calm
    #   tape, three correlated winners are concentration you are being paid for,
    #   and forcing a swap down the ranking just gives up return for a risk that
    #   is not currently priced.  In stress, the same three names become one
    #   position that gaps together.
    #
    #   So the filter is gated on absolute VIX rather than run continuously.
    #   None = always apply.
    #
    #   Measured (2026-07-25 sweep, 2012-2026, relative p85):
    #       gate      CAGR     Sharpe   MaxDD     Calmar
    #       none    19.85%      0.91   -20.55%     0.97
    #       25.0    19.83%      0.91   -19.23%     1.03   <- default
    #       20.0    19.52%      0.89   -20.18%     0.97
    #       15.0    18.32%      0.83   -25.08%     0.73
    #       always  16.96%      0.75   -26.31%     0.64
    #
    #   Filtering continuously costs 2.9pp of CAGR AND worsens drawdown: you give
    #   up your best-ranked names to insure against a risk that is not priced in
    #   a calm tape.  The crossover sits between 15 and 20.
    #
    #   Caveat on the default: VIX >= 25 is only 13% of days historically and 5%
    #   of the last three years, so the 1.3pp drawdown gain rests on a thin slice
    #   of history.  Read it as approximately-free rather than proven.
    #
    #   Absolute level, not a z-score, for the same reason Track A is moving off
    #   z-scores: a sustained moderate VIX becomes its own baseline and stops
    #   registering as elevated exactly when it matters.

    # --- selection filter ---
    window: int = 50                 # trading days for the medium-term estimate
    method: str = "relative"         # 'absolute' | 'relative'
    max_correlation: float = 0.75    # used when method='absolute'
    max_percentile: float = 0.85     # used when method='relative'
    on_infeasible: str = "relax"     # 'relax' | 'hold_fewer'
    min_positions: int = 2           # floor when on_infeasible='hold_fewer'
    apply_to_defensive: bool = False # defensive sleeve is chosen for low
                                     # correlation already; filtering it just
                                     # blocks the crisis fill

    # --- universe screen ---
    long_term_window: int = 200
    redundant_threshold: float = 0.70
    #   Calibrated to this universe rather than to a textbook number.  The
    #   quarterly screen already diversifies below the sector level, and it
    #   shows: over 1,326 pairs the max is 0.79, only 2 clear 0.70, and none
    #   reach 0.80.  A 0.80 threshold would flag nothing and give false comfort.
    #   Treat this screen as drift monitoring between screens, not as a source
    #   of corrections.


@dataclass
class VixRegimeConfig:
    """
    LEGACY binary/ternary VIX regime based on a rolling z-score.

    Retained so pre-refactor results stay reproducible.  Track A replaces this
    with `GraduatedVixConfig`, which keys off absolute VIX level rather than
    deviation-from-recent-baseline.

    The known failure mode: a z-score measures deviation from a recent
    baseline, so a sustained moderate VIX simply becomes the new baseline.  88%
    of days in the VIX 15-20 band were classified 'normal' by this logic — a
    band whose realized Sharpe was 0.88 versus 3.30 below 15.

    Regime logic (evaluated in order, highest wins):
      vix_z >= crisis_zscore    -> 'crisis'
      vix_z >= elevated_zscore  -> 'elevated'
      vix_roc >= roc_threshold  -> 'elevated'   (optional spike trigger)
      else                      -> 'normal'
    """
    zscore_window: int = 60
    roc_window: int = 5
    elevated_zscore: float = 1.5
    crisis_zscore: float = 2.5
    use_roc_trigger: bool = False
    roc_threshold: float = 0.30
    elevated_top_n: int = 2
    defensive_symbols: List[str] = field(default_factory=lambda: ["SHY", "TLT", "IAU"])
    crisis_symbols: List[str] = field(default_factory=lambda: ["SHY", "TLT"])


# --------------------------------------------------------------------------
# Config snapshotting (Request #1)
# --------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """
    A complete, serializable description of one model version.

    Bundling the pieces means a snapshot is atomic: you can't end up with a
    record of the scoring parameters but not the regime thresholds that were
    live at the same time.
    """
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    velocity: Optional[VelocityConfig] = field(default_factory=VelocityConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    vix: Optional[VixRegimeConfig] = None
    correlation: Optional[CorrelationConfig] = field(default_factory=CorrelationConfig)
    graduated_vix: Optional[GraduatedVixConfig] = None
    #   When set, supersedes `vix` entirely — the two regime systems are
    #   alternatives, not layers. Set `vix=None` alongside it to make that
    #   explicit in snapshots.
    top_n: int = 4
    hold_days: int = 14
    min_data_days: int = 200
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        def convert(obj):
            if is_dataclass(obj) and not isinstance(obj, type):
                return {k: convert(v) for k, v in asdict(obj).items()}
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [convert(v) for v in obj]
            return obj

        return {
            "top_n": self.top_n,
            "hold_days": self.hold_days,
            "min_data_days": self.min_data_days,
            "notes": self.notes,
            "scoring": convert(self.scoring),
            "velocity": convert(self.velocity),
            "execution": convert(self.execution),
            "vix": convert(self.vix),
            "correlation": convert(self.correlation),
            "graduated_vix": convert(self.graduated_vix),
        }


def snapshot_config(config: ModelConfig, as_of: Optional[date] = None,
                    label: str = "") -> Path:
    """
    Write a dated JSON snapshot of `config` to snapshots/config/.

    Overwrites an existing snapshot for the same date and label — re-running the
    live model twice in one day should not produce two competing records.

    Returns the path written.
    """
    as_of = as_of or date.today()
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_{label}" if label else ""
    path = CONFIG_DIR / f"config_{as_of.isoformat()}{suffix}.json"

    payload = config.to_dict()
    payload["_snapshot_date"] = as_of.isoformat()
    payload["_label"] = label

    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def load_config_snapshot(as_of: Optional[date] = None,
                         label: str = "") -> Dict[str, Any]:
    """
    Load the config snapshot in effect on `as_of` (the most recent snapshot on
    or before that date).  Returns the raw dict rather than reconstructing
    dataclasses, so snapshots written by older code versions stay readable even
    if fields have since been added or renamed.
    """
    if not CONFIG_DIR.exists():
        raise FileNotFoundError(f"No config snapshots found in {CONFIG_DIR}")

    suffix = f"_{label}" if label else ""
    candidates = sorted(CONFIG_DIR.glob(f"config_*{suffix}.json"))
    if not candidates:
        raise FileNotFoundError(f"No config snapshots matching label={label!r}")

    if as_of is None:
        return json.loads(candidates[-1].read_text(encoding="utf-8"))

    eligible = [p for p in candidates
                if p.stem.split("_")[1] <= as_of.isoformat()]
    if not eligible:
        raise FileNotFoundError(
            f"No config snapshot on or before {as_of.isoformat()}; "
            f"earliest is {candidates[0].stem}"
        )
    return json.loads(eligible[-1].read_text(encoding="utf-8"))
