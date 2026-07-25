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
    graduated_vix: Optional[Any] = None   # GraduatedVixConfig, set in Phase 2
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
