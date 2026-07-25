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

    velocity_window
        TRADING SESSIONS, not calendar days, and not the trading cadence.  The
        velocity term is `z(today) - z(velocity_window sessions ago)`, so 5 is
        roughly a one-week rate of change.

        Do not confuse it with two nearby parameters:
          ModelConfig.hold_days = 14      calendar days between rebalances —
                                          THIS is the trading frequency
          ScoringConfig.derivative_window = 5   sessions, the change in the MA
                                          gap that feeds the composite itself

        So there are two nested derivatives: a 5-session derivative of the
        moving-average spread inside the composite, and a 5-session derivative
        of the resulting composite z-score here.  Same length, different
        objects.
    """
    velocity_window: int = 5
    #   Was 10.  Changed 2026-07-25 to match the production value, so a bare
    #   VelocityConfig() no longer silently constructs the rejected setting.
    #   On a corrected blend scale, window 10 is beaten by disabling velocity
    #   entirely in every subperiod; window 5 leads on CAGR, Sharpe, drawdown
    #   and Calmar.  See production_config() for the full result and caveats —
    #   this is an in-sample selection on a sharp surface, not a proven optimum.
    level_weight: float = 0.7
    velocity_weight: float = 0.3
    min_level_threshold: float = -3.0
    blend_normalization: str = "cross_sectional"   # 'cross_sectional' | 'rank' | 'legacy'

    # --- Track C: additional derived components ---
    #
    # These extend the same standardize-weight-rank architecture rather than
    # replacing it with a threshold rule.  The candidate entry rule tested
    # previously required four conditions to clear independently (z > 1, best
    # trailing rank <= 4, positive 1st derivative, positive 2nd derivative).  A
    # hard AND is brittle: it discards a candidate that misses on one condition
    # by any margin, and the four conditions are correlated enough that stacking
    # them bought little discriminative power (53.3% precision against a 49.5%
    # naive baseline).  Expressing them as weighted, standardized components
    # lets a strong reading on one offset a marginal miss on another.
    #
    # All weights are on the same cross-sectional scale, so they are directly
    # comparable.  Set any to 0.0 to drop that component.

    acceleration_weight: float = 0.0
    acceleration_window: int = 5
    #   Second derivative of the composite z-score — momentum of momentum.
    #   Targets the distinction the level cannot make: a stock decelerating from
    #   a high level versus one accelerating from a lower one.
    #
    #   MEASURED 2026-07-25 — REJECTED. Left at 0.0.
    #       weight 0.1  CAGR 15.25%  Sharpe 0.61  MaxDD -22.29%
    #       weight 0.3  CAGR 15.42%  Sharpe 0.64  MaxDD -26.07%
    #       weight 0.5  CAGR 12.85%  Sharpe 0.48  MaxDD -28.98%
    #       (baseline   CAGR 19.84%  Sharpe 0.91  MaxDD -19.23%)
    #
    #   Derivative window sensitivity, at weight 0.3:
    #       3-session   CAGR 12.65%  MaxDD -41.95%   <- more than double the
    #      10-session   CAGR 14.47%  MaxDD -24.93%      baseline drawdown
    #      15-session   CAGR 14.02%  MaxDD -23.74%
    #
    #   The window test is the informative one: differencing twice amplifies
    #   noise, and a short window amplifies it catastrophically. The
    #   acceleration term carries noise, not timing information. Even a 10%
    #   weight costs 4.6pp of CAGR.
    #
    #   This also disposes of the AND-of-four candidate entry rule. Expressed as
    #   a weighted composite (level 0.4 / velocity 0.3 / accel 0.2 / rank 0.1)
    #   it scores 14.62% against the baseline's 19.84%. The rule looked
    #   selective — firing on 3.2% of stock-days — because it was firing on
    #   noise that happened to be rare, not because it was discriminating.

    best_rank_weight: float = 0.0
    best_rank_window: int = 20
    #   Best rank achieved in the trailing window, inverted so higher is better.
    #   Encodes "has been near the top recently" without requiring it to be
    #   there now.
    #
    #   MEASURED — the only Track C component that is not actively harmful, but
    #   still not an improvement:
    #       weight 0.1  CAGR 19.01%  Sharpe 0.85  MaxDD -18.80%  126 trd/yr
    #       weight 0.3  CAGR 17.57%  Sharpe 0.77  MaxDD -23.47%  122 trd/yr
    #   At 0.1 it costs 0.83pp of CAGR for 0.43pp of drawdown and 2 fewer trades
    #   a year. Left at 0.0; revisit only if drawdown becomes the binding
    #   constraint.

    @property
    def label(self) -> str:
        parts = [f"v{self.velocity_window}",
                 f"L{self.level_weight:.0%}V{self.velocity_weight:.0%}"]
        if self.acceleration_weight:
            parts.append(f"A{self.acceleration_weight:.0%}")
        if self.best_rank_weight:
            parts.append(f"R{self.best_rank_weight:.0%}")
        parts.append(self.blend_normalization[:5])
        return "_".join(parts)


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
class ExitConfig:
    """
    Track B — per-stock rank exits, decoupled from the buy clock.

    Buy and sell currently run on the same 14-day timer for no reason other
    than that they always have.  The evidence for splitting them:

      - 68% of holdings drop out of the top 4 before their hold ends, at a
        median of day 3 out of ~10 trading days.
      - That leaves ~6.5 trading days per instance holding a stock the model no
        longer ranks.
      - Only 17% recover their rank, so this is persistent decay rather than
        boundary noise.

    This is NOT the same question Track A answered.  Track A tested a
    portfolio-wide risk dial and found that de-risking forfeits the overnight
    premium — 63% of the return stream for 36% of the variance.  A rank exit
    does not de-risk: it rotates from a decayed name into a better-ranked one,
    staying fully invested.  The overnight-premium result therefore argues for
    `replacement='immediate'` and against the defensive variants, but says
    nothing about whether the exit itself is right.

    exit_rank_threshold
        Rank a holding must fall TO before it is a candidate for exit.  Setting
        this above top_n creates the buffer: with top_n=4 and a threshold of 6,
        a stock sitting at rank 5 is left alone.  Rank 5 today is often rank 4
        tomorrow, and paying 2x slippage to round-trip that is pure loss.

    consecutive_days
        Days it must stay at or beyond that rank before exiting.  The second
        half of the anti-flapping guard.

    replacement — what fills the vacated slot
        'immediate'  best-ranked stock not already held.  Stays fully invested.
        'defensive'  a defensive symbol.  Expected to underperform on the
                     overnight-premium result; included to confirm that.
        'defer'      hold fewer positions until the next scheduled rebalance.
                     Equal-weights across a smaller book, so it concentrates
                     rather than de-risks.

    max_exits_per_month
        Turnover budget.  0 = unlimited.  The strategy already runs ~130 trade
        legs and 16.9x portfolio turnover a year, where 7.5 bps one-way costs
        2.9pp of CAGR.  Any exit rule must clear roughly 0.17pp of CAGR per
        extra 1x of annual turnover just to break even.
    """
    enabled: bool = False

    mode: str = "rank"                 # 'rank' | 'score_gap'
    #   'rank'      exit when the HOLDING decays past a rank threshold. Tests
    #               the premise "sells too late". REJECTED on evidence — every
    #               variant lost, monotonically with turnover, from -1.2pp
    #               (budgeted) to -10.5pp (twitchy). Rank is relative, so a
    #               holding falling from 4th to 6th usually means others rose,
    #               not that it fell: measured return after trigger is +0.39%
    #               mean, +0.00% median.
    #
    #   'score_gap' exit when a BETTER BUY exists — trigger on the opportunity
    #               rather than the decay. Swap only when the best available
    #               candidate beats the worst holding by more than
    #               `min_score_gap`. This is the economically coherent version:
    #               it compares the size of the opportunity against the cost of
    #               taking it, instead of selling on a relative-rank move that
    #               carries no directional information.
    #
    #               Worth ~8pp of CAGR over the rank trigger, and the best
    #               setting (gap 1.0z, max 2/month) is break-even against not
    #               trading at all: 19.92% vs 19.84%, Sharpe 0.92 vs 0.91.
    #               Performance rises monotonically as the gap widens — the
    #               better the rule, the less it trades — which is the signature
    #               of no edge rather than a tuning opportunity.
    #
    #               Measured directly (analyze_swap_quality.py, 216 swaps):
    #                   horizon   incoming  outgoing    edge   win rate  t
    #                       5d      0.46%     0.63%   -0.17%     45.6%  -0.43
    #                      10d      0.74%     0.70%   +0.04%     40.6%   0.08
    #                      21d      1.65%     1.34%   +0.31%     52.8%   0.36
    #
    #               The challenger is statistically indistinguishable from the
    #               name it displaces. The signal arrives too late to carry
    #               information, so the swap pays ~15 bps to exchange a position
    #               for an equivalent one. Left disabled.

    min_score_gap: float = 0.5
    #   Composite z-score units the challenger must beat the incumbent by,
    #   used when mode='score_gap'. This is the transaction-cost hurdle
    #   expressed in signal terms: a round trip costs ~15 bps, so a gap too
    #   small to be worth 15 bps should not trigger a trade. Set it too low and
    #   you churn on noise; too high and it never fires.

    exit_rank_threshold: int = 6
    consecutive_days: int = 2
    replacement: str = "immediate"     # 'immediate' | 'defensive' | 'defer'
    max_exits_per_month: int = 0       # 0 = unlimited

    min_hold_days: int = 0
    #   Optional floor on how soon after entry a position can be exited.  Guards
    #   against buying and selling a name within days on a marginal rank wobble.


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
    exits: Optional[ExitConfig] = field(default_factory=ExitConfig)
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
            "exits": convert(self.exits),
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
