"""
Model health monitor — is the live model still doing what it did in backtest?

This is not an experiment and it does not select anything.  It answers one
question on a schedule: has the model's performance relative to the market
fallen far enough, for long enough, that something has probably broken?

WHAT IS MEASURED

  6-month (126 trading day) log excess return against SPY — both raw and
  beta-adjusted — scaled by the historical spread of that same quantity.  Each
  piece earns its place:

  Log excess, not a ratio of wealth indices.  `ln(1+r_model) - ln(1+r_spy)`
  summed over the window is the same "indexed over SPY" idea, but additive
  across time and symmetric in sign — a 20% shortfall and a 20% surplus get
  equal weight, which a wealth ratio does not give you.

  Beta-adjusted AND raw, because which one belongs in the alarm is an empirical
  question and the measurement answered it against expectation.  A four-name
  concentrated book was assumed to run beta above 1, in which case adjusting
  would stop the monitor firing in every selloff on nothing but market
  exposure.  Measured, this book's beta is about 0.48 (full-sample OLS 0.427;
  correlation to SPY 0.42 at near-identical volatility, 16.8% vs 16.6% —
  four idiosyncratic names plus defensive holdings in stressed regimes are
  simply not very index-like).

  At beta 0.48 the adjustment does the opposite of what was intended: it
  forgives lagging in RISING markets, because CAPM only asks the model to beat
  half of SPY's move.  On 2026-09-02 the model returned 6.65% against SPY's
  12.27% over six months — trailing by 5.6 points — and scored an adjusted z of
  +0.04.  Over the whole record the adjusted series never reaches z <= -2 at
  all.  As an alarm it is close to inert.

  So the raw excess is the alarm, and the beta-adjusted series runs beside it as
  the diagnosis: raw bad and adjusted fine means the market ran and a 0.5-beta
  book did not keep up, which is structural; both bad means the picks
  themselves stopped working, which is not.

  Centered on ZERO, not on the historical mean.  The backtest applies today's
  screened universe back to 2010, so its historical excess return is inflated
  by survivorship; z-scoring against that mean would report degradation the
  moment live trading merely failed to reproduce a biased number.  Zero is the
  honest bar and it does not drift: z = 0 means "matched the benchmark",
  negative means "lagged it".  The historical spread is used only to set the
  SCALE — how big a shortfall is unusual — which survivorship inflates far less
  than it inflates the level.

WHAT THE Z-SCORE DOES NOT MEAN

  Consecutive 126-day windows computed daily share 125 of their 126 days.  The
  series is therefore massively autocorrelated: over the scored record there
  are only ~30 independent windows, so `z < -2` is nowhere near a 2.5% event,
  and breaches arrive in long clusters rather than as isolated days.

  Two consequences, both structural rather than cosmetic:

    - Thresholds must be calibrated against the historical episode record, not
      read off a normal table.  `calibration_table` exists for that.
    - Counting alarm DAYS is meaningless.  Count EPISODES.

  This is also why the duration requirement lives in `confirm_days` rather than
  in a second moving average.  A 126-day rolling return is already a 126-day
  moving average; smoothing it again buys little noise reduction and costs lag,
  which is the one thing a warning signal cannot afford.  "Below the line for N
  consecutive sessions" states the same intent directly and is legible when it
  fires.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass
class HealthConfig:
    """
    Monitor parameters.

    window
        Trading days in the performance window.  126 ~ 6 months.

    beta_window / min_beta_obs
        Trailing window for the beta estimate, and the minimum observations
        before one is produced.  Estimated point-in-time and lagged a day, so
        the beta applied on date T uses only returns through T-1.  Two years is
        long enough to be stable and short enough to track a real change in how
        concentrated the book is.

    burn_in
        Days of history required before any z is emitted.  The scale estimate
        is the whole basis of the alarm; building it on a handful of
        overlapping windows would make the early record noise.

    scale
        'expanding'    SD from all history to date — live-computable, no
                       look-ahead, and the honest default.
        'full_sample'  one SD over the whole record.  Uses future data, so it
                       is a diagnostic comparison rather than a live signal;
                       run both to see whether the choice moves anything.

    risk_free_rate
        Annual, matching ExecutionConfig.  Enters through the CAPM residual:
        at beta ~0.5 the (1 - beta) * rf term is worth a few tenths of a point
        over six months, which is small but free to get right.

    measure
        'raw'       excess against SPY itself.  The alarm runs on this.
        'adjusted'  excess against a beta-equivalent SPY position.

        Measured beta is ~0.48, so 'adjusted' asks the model to beat only half
        of SPY's move and is close to inert as an alarm: it never reaches z -2
        over the whole record, and reads +0.04 on a day the model trails SPY by
        5.6 points.  Both series are always computed; this only picks which one
        arms the alarm.

    alarm_z / alarm_points / confirm_days
        The alarm condition: z at or below `alarm_z` AND the excess at or below
        `alarm_points`, held for `confirm_days` consecutive sessions.  See
        `breach_flags` for why both are required.

        Chosen from the historical episode record, not from a normal table.  On
        the default record the rule fires twice in eleven years (2019-07 and
        2021-08), median 28 days, 2.0% of the record.  That RARITY is the part
        of the calibration that holds up: across every record length and
        threshold tried, the shipped settings fire between two and five times,
        which is the right order for something meant to prompt a review.

        Note which condition is actually binding.  At today's scale z -1.25 is
        worth about -10.9%, so the -12.5% floor is stricter and does the work;
        z -1.00 through -1.50 produce identical episode lists.  The z only takes
        over if the spread widens enough to make -12.5% an ordinary reading.
        That is the intended division of labour, but it does mean tuning
        `alarm_z` alone will look inert until `alarm_points` moves with it.

        What did NOT calibrate: whether firing predicts anything.  The median
        forward-six-month excess after a trigger swings from -2.6% to +4.8%
        depending on record start, floor, and threshold — on two to five
        observations.  It is not that the answer came out negative; it is that
        the answer is not estimable at this sample size.  So this is a REVIEW
        TRIGGER — "you are in the worst tail of your own history, go look" — and
        must not be described, or acted on, as a predictive warning.

    warn_z / warn_points
        The softer pair behind WATCH.
    """
    window: int = 126
    beta_window: int = 504
    min_beta_obs: int = 252
    burn_in: int = 756
    scale: str = "expanding"
    risk_free_rate: float = 0.045

    measure: str = "raw"
    alarm_z: float = -1.25
    alarm_points: float = -0.125
    confirm_days: int = 20
    warn_z: float = -1.0
    warn_points: float = -0.08

    def __post_init__(self):
        if self.scale not in ("expanding", "full_sample"):
            raise ValueError(f"Unknown scale: {self.scale!r}")
        if self.measure not in ("raw", "adjusted"):
            raise ValueError(f"Unknown measure: {self.measure!r}")
        if self.window < 20:
            raise ValueError("window is too short to be a performance measure")

    @property
    def column(self) -> str:
        """The z column the alarm reads."""
        return "z_raw" if self.measure == "raw" else "z_adj"

    @property
    def scale_column(self) -> str:
        return "scale_raw" if self.measure == "raw" else "scale_adj"


# --------------------------------------------------------------------------
# Core computation
# --------------------------------------------------------------------------

def _as_returns(series: pd.Series, index: pd.Index) -> pd.Series:
    """Daily simple returns of a price series, aligned to `index`."""
    prices = pd.Series(series).astype(float)
    return prices.pct_change().reindex(index)


def _trailing_beta(model_ex: pd.Series, market_ex: pd.Series,
                   config: HealthConfig) -> pd.Series:
    """
    Point-in-time beta of the model against the market.

    Rolling over `beta_window`, falling back to an expanding estimate before
    that window is full so the early record is usable rather than blank.  The
    result is lagged one day: the beta applied on date T is estimated only from
    returns through T-1, which keeps the monitor honest as a live signal.
    """
    roll_cov = model_ex.rolling(config.beta_window,
                                min_periods=config.min_beta_obs).cov(market_ex)
    roll_var = market_ex.rolling(config.beta_window,
                                 min_periods=config.min_beta_obs).var()

    exp_cov = model_ex.expanding(min_periods=config.min_beta_obs).cov(market_ex)
    exp_var = market_ex.expanding(min_periods=config.min_beta_obs).var()

    beta = (roll_cov / roll_var).fillna(exp_cov / exp_var)
    return beta.shift(1)


def compute_health(model_returns: pd.Series,
                   spy: pd.Series,
                   config: Optional[HealthConfig] = None) -> pd.DataFrame:
    """
    The full monitor series, one row per trading day.

    Parameters
    ----------
    model_returns : daily NET returns of the strategy (what you would earn)
    spy           : SPY price series, any index covering `model_returns`
    config        : HealthConfig

    Columns
    -------
    model_6m, spy_6m      simple returns over the window, for reading
    beta                  trailing beta applied on that date
    excess_6m_raw         log excess vs SPY over the window
    excess_6m_adj         log excess vs a beta-equivalent SPY position (CAPM)
    scale_raw, scale_adj  SD used for the z on that date
    z_raw, z_adj          zero-centered z-scores
    """
    config = config or HealthConfig()

    model_returns = pd.Series(model_returns).astype(float).dropna()
    index = model_returns.index
    market_returns = _as_returns(spy, index)

    frame = pd.DataFrame({"model": model_returns, "market": market_returns}).dropna()
    if len(frame) < config.window + 1:
        raise ValueError(
            f"Need more than {config.window} aligned observations to measure a "
            f"{config.window}-day window; got {len(frame)}"
        )

    log_model = np.log1p(frame["model"])
    log_market = np.log1p(frame["market"])
    log_rf = np.log1p(config.risk_free_rate) / TRADING_DAYS

    model_ex = log_model - log_rf
    market_ex = log_market - log_rf

    beta = _trailing_beta(model_ex, market_ex, config)

    # Daily excess, two ways.  The adjusted one is the CAPM residual: what the
    # model earned beyond a beta-scaled market position.
    daily_raw = log_model - log_market
    daily_adj = model_ex - beta * market_ex

    w = config.window
    out = pd.DataFrame(index=frame.index)
    out["model_6m"] = np.expm1(log_model.rolling(w).sum())
    out["spy_6m"] = np.expm1(log_market.rolling(w).sum())
    out["beta"] = beta
    out["excess_6m_raw"] = daily_raw.rolling(w).sum()
    out["excess_6m_adj"] = daily_adj.rolling(w, min_periods=w).sum()

    for name in ("raw", "adj"):
        excess = out[f"excess_6m_{name}"]
        if config.scale == "expanding":
            # Shifted so a day is never part of the scale it is judged against.
            scale = excess.expanding(min_periods=config.burn_in).std().shift(1)
        else:
            scale = pd.Series(excess.std(), index=out.index)
            scale[excess.isna()] = np.nan

        out[f"scale_{name}"] = scale
        out[f"z_{name}"] = excess / scale.replace(0.0, np.nan)

    # Nothing is emitted before there is enough history to scale against.
    if config.scale == "expanding":
        out.loc[out.index[:config.burn_in], ["z_raw", "z_adj"]] = np.nan

    return out


# --------------------------------------------------------------------------
# Episodes — the unit that actually means something
# --------------------------------------------------------------------------

def excess_column(column: str) -> str:
    """The excess series a z-column was built from."""
    return "excess_6m_raw" if column == "z_raw" else "excess_6m_adj"


def breach_flags(health: pd.DataFrame, column: str, threshold_z: float,
                 threshold_points: Optional[float] = None) -> pd.Series:
    """
    Days meeting the alarm condition, before the duration requirement.

    Both conditions must hold when `threshold_points` is given: the shortfall
    has to be statistically unusual AND large enough in points to care about.
    Each guards a different failure of the other.

    A z alone drifts.  The scale is an expanding estimate over ~30 independent
    windows, so it carries real estimation error and it moves as history
    accumulates — extending the record from 2010 to 2005 moved the worst raw
    reading from z -3.79 to -2.23 without a single return changing.  A rule that
    can be loosened by adding old data is not a rule.

    Points alone ignore that the spread's natural size can legitimately change.
    A 13% shortfall means something different for a book whose 6-month excess
    normally swings 5 points than for one that swings 15.
    """
    z = health[column]
    flags = (z <= threshold_z) & z.notna()
    if threshold_points is not None:
        excess = health[excess_column(column)]
        flags = flags & (excess <= threshold_points) & excess.notna()
    return flags


def _runs(flags: pd.Series) -> List[Tuple]:
    """Maximal runs of consecutive True in a boolean series."""
    runs, start, prev = [], None, None
    for date, flag in flags.items():
        if flag and start is None:
            start = date
        elif not flag and start is not None:
            runs.append((start, prev))
            start = None
        prev = date
    if start is not None:
        runs.append((start, prev))
    return runs


def episodes(health: pd.DataFrame,
             threshold: float = -1.25,
             confirm_days: int = 20,
             column: str = "z_raw",
             threshold_points: Optional[float] = None,
             forward_window: int = 126) -> pd.DataFrame:
    """
    Every historical episode the monitor would have flagged.

    An episode is a maximal run of days meeting the breach condition lasting at
    least `confirm_days`; the run's `confirm_days`-th day is the trigger, which
    is the first date the alarm could actually have been raised.

    The forward columns are the point of the whole exercise.  A warning that is
    reliably followed by recovery is not a warning, it is a contrarian buy
    signal wearing the wrong label, and the only way to tell the difference is
    to look at what happened next.

    Columns
    -------
    start, trigger, end, days      the run and when it would have fired
    trough_z, trough_date          the worst reading in the episode
    excess_at_trigger              the excess (matching `column`) when it fired
    fwd_excess_after_trigger       excess over the next `forward_window` days
    fwd_excess_after_end           the same, measured from the episode's end
    recovered                      did the measure climb back above 0 afterwards
    """
    series = health[column]
    excess = health[excess_column(column)]
    flags = breach_flags(health, column, threshold, threshold_points)

    rows = []
    for start, end in _runs(flags):
        window = series.loc[start:end]
        if len(window) < confirm_days:
            continue

        trigger = window.index[confirm_days - 1]

        def forward(anchor):
            """Excess over the `forward_window` days after `anchor`."""
            after = excess.loc[excess.index > anchor]
            if len(after) < forward_window:
                return np.nan
            return float(after.iloc[forward_window - 1])

        later = series.loc[series.index > end]
        rows.append({
            "start": start.date(),
            "trigger": trigger.date(),
            "end": end.date(),
            "days": len(window),
            "trough_z": float(window.min()),
            "trough_date": window.idxmin().date(),
            "excess_at_trigger": float(excess.loc[trigger]),
            "fwd_excess_after_trigger": forward(trigger),
            "fwd_excess_after_end": forward(end),
            "recovered": bool((later > 0).any()) if len(later) else np.nan,
        })

    return pd.DataFrame(rows)


def calibration_table(health: pd.DataFrame,
                      thresholds: Sequence[float] = (-1.0, -1.25, -1.5, -1.75,
                                                     -2.0, -2.5),
                      confirm_days: Sequence[int] = (1, 10, 20, 40),
                      column: str = "z_raw",
                      threshold_points: Optional[float] = None,
                      forward_window: int = 126) -> pd.DataFrame:
    """
    What each candidate threshold would have done over the whole record.

    `median_fwd_excess` against `unconditional_fwd` is the test that matters: if
    firing does not precede materially worse forward performance than an average
    day, the rule is decorating noise regardless of how sensible the threshold
    looks.
    """
    excess = health[excess_column(column)]
    valid = health[column].notna()
    unconditional = float(excess.loc[valid].shift(-forward_window).median())

    rows = []
    for threshold in thresholds:
        for days in confirm_days:
            eps = episodes(health, threshold, days, column,
                           threshold_points, forward_window)
            fired_days = int(eps["days"].sum()) if not eps.empty else 0

            rows.append({
                "threshold": threshold,
                "threshold_points": threshold_points,
                "confirm_days": days,
                "episodes": len(eps),
                "alarm_days": fired_days,
                "pct_of_record": (fired_days / int(valid.sum())
                                  if valid.any() else np.nan),
                "median_days": float(eps["days"].median()) if not eps.empty else np.nan,
                "worst_z": float(eps["trough_z"].min()) if not eps.empty else np.nan,
                "median_fwd_excess": (float(eps["fwd_excess_after_trigger"].median())
                                      if not eps.empty else np.nan),
                "unconditional_fwd": unconditional,
                "recovered_all": (bool(eps["recovered"].fillna(False).all())
                                  if not eps.empty else np.nan),
            })

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Reading the current state
# --------------------------------------------------------------------------

def independent_windows(health: pd.DataFrame, config: HealthConfig) -> float:
    """
    Roughly how many non-overlapping windows the record contains.

    Printed next to any exceedance count, because it is the number that governs
    how much the record can support.  ~30 independent observations is a small
    sample however many rows the CSV has.
    """
    n = int(health[config.column].notna().sum())
    return n / config.window if config.window else np.nan


def current_state(health: pd.DataFrame,
                  config: Optional[HealthConfig] = None) -> Dict:
    """
    Today's reading and the run length behind it.

    Three states, and the distance between them is deliberate:

      OK     nothing to do.
      WATCH  the breach condition is met today but has not persisted long
             enough to mean anything, or the softer warn levels have persisted.
      ALARM  the breach condition has held for `confirm_days` consecutive
             sessions.

    ALARM is a prompt to investigate, not to trade.  Every historical episode of
    this kind was followed by recovery, and the repo's walk-forward work already
    established that re-tuning on recent weakness makes results worse — so the
    productive response is to find out WHERE the shortfall came from (universe,
    regime positioning, a couple of bad names), not to change parameters.
    """
    config = config or HealthConfig()
    column = config.column
    live = health.dropna(subset=[column])
    if live.empty:
        return {"state": "UNKNOWN", "reason": "not enough history to score yet"}

    row = live.iloc[-1]

    def run_length(threshold_z: float, threshold_points: Optional[float]) -> int:
        flags = breach_flags(live, column, threshold_z, threshold_points)
        if not bool(flags.iloc[-1]):
            return 0
        count = 0
        for flag in flags.values[::-1]:
            if not flag:
                break
            count += 1
        return count

    alarm_run = run_length(config.alarm_z, config.alarm_points)
    warn_run = run_length(config.warn_z, config.warn_points)

    if alarm_run >= config.confirm_days:
        state = "ALARM"
    elif alarm_run > 0 or warn_run >= config.confirm_days:
        state = "WATCH"
    else:
        state = "OK"

    return {
        "date": live.index[-1],
        "state": state,
        "measure": config.measure,
        "z": float(row[column]),
        "excess": float(row[excess_column(column)]),
        "z_adj": float(row["z_adj"]),
        "z_raw": float(row["z_raw"]),
        "excess_6m_adj": float(row["excess_6m_adj"]),
        "excess_6m_raw": float(row["excess_6m_raw"]),
        "model_6m": float(row["model_6m"]),
        "spy_6m": float(row["spy_6m"]),
        "beta": float(row["beta"]),
        "scale": float(row[config.scale_column]),
        "alarm_run": alarm_run,
        "warn_run": warn_run,
    }


def status_line(state: Dict, config: Optional[HealthConfig] = None) -> str:
    """One line for the tail of a live run."""
    config = config or HealthConfig()
    if state.get("state") == "UNKNOWN":
        return f"MODEL HEALTH: UNKNOWN — {state['reason']}"

    line = (f"MODEL HEALTH: {state['state']}  "
            f"{config.window}d model {state['model_6m']:+.1%} vs SPY "
            f"{state['spy_6m']:+.1%}, excess {state['excess']:+.1%} "
            f"(z {state['z']:+.2f})")
    if state["alarm_run"]:
        line += (f"\n  breaching {state['alarm_run']}/{config.confirm_days} "
                 f"sessions (z <= {config.alarm_z:g} and "
                 f"{config.alarm_points:.0%})")
    return line
