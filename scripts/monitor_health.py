"""
Model health monitor — calibration report and current reading.

    python scripts/monitor_health.py               full report
    python scripts/monitor_health.py --window 63   3-month window instead
    python scripts/monitor_health.py --scale full_sample

The report has three parts, in the order you need to read them:

  1. WHAT THE SPREAD LOOKS LIKE — the scale the z-score is built on, in points,
     so a threshold means something concrete before you pick one.
  2. HISTORICAL EPISODES — every stretch the monitor would have flagged, and
     crucially what happened in the six months after it fired.
  3. CALIBRATION GRID — threshold x duration, with the false-alarm profile.

The point of (3) is that a threshold cannot be chosen from the normal table.
Overlapping windows make the z-score's tail probabilities meaningless; the only
usable evidence is what the rule would have done over the actual record.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from momentum.data import load_data
from momentum.experiments import production_config
from momentum.health import (HealthConfig, calibration_table, compute_health,
                             current_state, episodes, excess_column,
                             independent_windows)
from momentum.strategy import run_strategy
from momentum.universe import current_symbols


def print_state(state: dict, config: HealthConfig) -> None:
    """The current reading, in the units the decision is made in."""
    print("\n" + "=" * 78)
    print("CURRENT READING")
    print("=" * 78)

    if state["state"] == "UNKNOWN":
        print(f"  {state['reason']}")
        return

    print(f"  as of {state['date']:%Y-%m-%d}          STATE: {state['state']}"
          f"   (alarm on {state['measure']} excess)")
    print(f"  {config.window}-day model return   {state['model_6m']:>8.2%}")
    print(f"  {config.window}-day SPY return     {state['spy_6m']:>8.2%}")
    print(f"  beta (trailing {config.beta_window}d) {state['beta']:>8.2f}")
    print(f"  excess vs SPY, raw     {state['excess_6m_raw']:>8.2%}  (log)")
    print(f"  excess vs beta*SPY     {state['excess_6m_adj']:>8.2%}  (log, CAPM residual)")
    arrow = "   <- the alarm reads this"
    print(f"  z (raw)                {state['z_raw']:>8.2f}"
          + (arrow if config.measure == "raw" else ""))
    print(f"  z (adjusted)           {state['z_adj']:>8.2f}"
          + (arrow if config.measure == "adjusted" else ""))
    if state["alarm_run"]:
        print(f"  breach run             {state['alarm_run']:>8}"
              f" of {config.confirm_days} sessions required")
    elif state["warn_run"]:
        print(f"  warn run               {state['warn_run']:>8}"
              f" of {config.confirm_days} sessions required")


def print_scale(health: pd.DataFrame, config: HealthConfig) -> None:
    """Translate the z-axis into points, before any threshold is discussed."""
    live = health.dropna(subset=[config.column])
    scale = float(live[config.scale_column].iloc[-1])
    excess = live[excess_column(config.column)]

    print("\n" + "=" * 78)
    print(f"THE SPREAD BEING MEASURED  ({config.window}-day windows, "
          f"{live.index[0]:%Y-%m-%d} to {live.index[-1]:%Y-%m-%d})")
    print("=" * 78)
    print(f"  scored days                 {len(live):>8,}")
    print(f"  independent windows         {independent_windows(health, config):>8.0f}"
          f"   <- the sample size that governs everything below")
    print(f"  {'mean ' + config.measure + ' excess':<28}{excess.mean():>8.2%}"
          f"   (survivorship-inflated; NOT the z's center)")
    print(f"  {'SD of ' + config.measure + ' excess':<28}{scale:>8.2%}"
          f"   <- 1.00 z is worth this much TODAY")
    print(f"  share of windows below zero {(excess < 0).mean():>8.1%}")

    # The z on a given date used the scale in force on that date, which under an
    # expanding estimate is smaller earlier on.  Reporting min(excess)/scale_today
    # would quietly rescale history to today's yardstick and understate how bad
    # the worst reading actually scored when it happened.
    print(f"\n  {'':<26}{'raw vs SPY':>14}{'beta-adjusted':>16}")
    print("  " + "-" * 56)
    worst_raw_pts, worst_adj_pts = live["excess_6m_raw"].min(), live["excess_6m_adj"].min()
    worst_raw_z, worst_adj_z = live["z_raw"].min(), live["z_adj"].min()
    print(f"  {'worst window, points':<26}{worst_raw_pts:>13.2%}{worst_adj_pts:>16.2%}")
    print(f"  {'worst window, z as scored':<26}{worst_raw_z:>13.2f}{worst_adj_z:>16.2f}")
    print(f"  {'days at or below z -2':<26}{int((live.z_raw <= -2).sum()):>13,}"
          f"{int((live.z_adj <= -2).sum()):>16,}")
    print(f"  {'median beta applied':<26}{'':>13}{live['beta'].median():>16.2f}")

    print(f"\n  A z of -2 on today's scale is a shortfall of about "
          f"{2 * float(live['scale_raw'].iloc[-1]):.1%} (raw) / {2 * scale:.1%} "
          f"(adjusted)\n  over {config.window} trading days.")


def print_episodes(health: pd.DataFrame, threshold: float,
                   confirm_days: int, config: HealthConfig,
                   column: str = "z_adj") -> pd.DataFrame:
    """Every episode, with what followed it."""
    eps = episodes(health, threshold=threshold, confirm_days=confirm_days,
                   column=column, threshold_points=config.alarm_points,
                   forward_window=config.window)

    print("\n" + "=" * 78)
    measure = "raw" if column == "z_raw" else "beta-adjusted"
    print(f"HISTORICAL EPISODES  ({measure} z <= {threshold} AND excess <= "
          f"{config.alarm_points:.1%})")
    print(" " * 21 + f"for >= {confirm_days} consecutive sessions")
    print("=" * 78)

    if eps.empty:
        print("  None. The rule never fired over this record.")
        return eps

    header = (f"{'trigger':<12}{'start':<12}{'end':<12}{'days':>6}"
              f"{'trough z':>10}{'excess':>10}{'fwd 6m':>10}")
    print(header)
    print("-" * len(header))
    for _, r in eps.iterrows():
        fwd = (f"{r['fwd_excess_after_trigger']:>9.2%}"
               if pd.notna(r["fwd_excess_after_trigger"]) else "        -")
        print(f"{str(r['trigger']):<12}{str(r['start']):<12}{str(r['end']):<12}"
              f"{r['days']:>6}{r['trough_z']:>10.2f}"
              f"{r['excess_at_trigger']:>9.2%}{fwd:>10}")
    print("-" * len(header))
    print(f"  'fwd 6m' is the {measure} excess over the {config.window} days "
          "AFTER the alarm fired.")
    print("  Positive there means the alarm was followed by recovery — a "
          "contrarian\n  signal, not a warning.")
    return eps


def print_calibration(health: pd.DataFrame, config: HealthConfig,
                      column: str = "z_adj") -> pd.DataFrame:
    """Threshold x duration, judged on what each would have done."""
    # The floor is part of the rule, so the grid has to carry it too — sweeping
    # z alone would report episode counts the configured monitor never produces.
    table = calibration_table(health, column=column,
                              threshold_points=config.alarm_points,
                              forward_window=config.window)
    table.insert(0, "measure", column)

    print("\n" + "=" * 78)
    label = ("raw excess vs SPY" if column == "z_raw"
             else "beta-adjusted excess (CAPM residual)")
    print(f"CALIBRATION GRID — {label}, floor {config.alarm_points:.1%}")
    print("=" * 78)
    print("  A usable rule fires a handful of times over 14 years, stays fired "
          "long enough\n  to act on, and is followed by WORSE performance than "
          "an average day.")

    unconditional = float(table["unconditional_fwd"].iloc[0])
    print(f"\n  Unconditional median forward {config.window}-day excess: "
          f"{unconditional:>7.2%}")
    print("  (the bar to beat downward — an alarm should precede something "
          "worse than this)\n")

    scale_col = "scale_raw" if column == "z_raw" else "scale_adj"
    scale_now = float(health[scale_col].dropna().iloc[-1])

    header = (f"{'z <=':>7}{'= pts':>8}{'for':>6}{'episodes':>10}{'days':>7}"
              f"{'% record':>10}{'med len':>9}{'med fwd 6m':>12}{'verdict':>12}")
    print(header)
    print("-" * len(header))

    for _, r in table.iterrows():
        if r["episodes"] == 0:
            verdict, fwd, med = "never fires", "        -", "        -"
        else:
            fwd = f"{r['median_fwd_excess']:>11.2%}" if pd.notna(
                r["median_fwd_excess"]) else "          -"
            med = f"{r['median_days']:>8.0f}"
            if pd.isna(r["median_fwd_excess"]):
                verdict = "no forward"
            elif r["median_fwd_excess"] > unconditional:
                verdict = "contrarian"
            elif r["episodes"] > 8:
                verdict = "too noisy"
            else:
                verdict = "candidate"
        print(f"{r['threshold']:>7.2f}{r['threshold'] * scale_now:>8.1%}"
              f"{r['confirm_days']:>6}{r['episodes']:>10}"
              f"{r['alarm_days']:>7}{r['pct_of_record']:>9.1%}{med:>9}"
              f"{fwd:>12}{verdict:>12}")

    print("-" * len(header))
    print("  'contrarian' = firing was followed by BETTER than average "
          "performance.")
    print("  Read that column with the episode count beside it: at two to five episodes")
    print("  it flips sign with record length and floor, so the grid establishes that")
    print("  firing is RARE, not that firing PREDICTS. An alarm is a prompt to look.")
    return table


def main() -> int:
    parser = argparse.ArgumentParser(description="Model health calibration report")
    # Matches run_live's default deliberately.  The z depends on the expanding
    # scale, so the scale depends on how much history was loaded: the same day
    # scores -0.47 from a 2005 start and -0.59 from 2010.  A monitor whose
    # reading changes with an unrelated CLI flag is a trap, so the default is
    # pinned to what the live run uses.  `--start 2005-01-01` buys ~6 more
    # independent windows and is the right call for a calibration pass — just
    # read the z it prints as belonging to that record, not to the live one.
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--window", type=int, default=126,
                        help="trading days in the performance window")
    parser.add_argument("--scale", default="expanding",
                        choices=["expanding", "full_sample"])
    parser.add_argument("--threshold", type=float, default=-1.25,
                        help="z threshold for the alarm")
    parser.add_argument("--points", type=float, default=-0.125,
                        help="points floor for the alarm, e.g. -0.125 for -12.5%%")
    parser.add_argument("--measure", default="raw", choices=["raw", "adjusted"],
                        help="which excess series arms the alarm")
    parser.add_argument("--warn-z", type=float, default=-1.0)
    parser.add_argument("--warn-points", type=float, default=-0.08)
    parser.add_argument("--beta-window", type=int, default=504)
    parser.add_argument("--burn-in", type=int, default=756)
    parser.add_argument("--confirm-days", type=int, default=20,
                        help="consecutive sessions required to fire")
    args = parser.parse_args()

    config = production_config()
    health_config = HealthConfig(
        window=args.window,
        beta_window=args.beta_window,
        burn_in=args.burn_in,
        scale=args.scale,
        risk_free_rate=config.execution.risk_free_rate,
        measure=args.measure,
        alarm_z=args.threshold,
        alarm_points=args.points,
        confirm_days=args.confirm_days,
        warn_z=args.warn_z,
        warn_points=args.warn_points,
    )

    print("=" * 78)
    print("MODEL HEALTH — CALIBRATION REPORT")
    print("=" * 78)

    prices = load_data(current_symbols(), start_date=args.start,
                       use_cache=not args.no_cache)
    result = run_strategy(prices, config, verbose=False)

    health = compute_health(result.returns, prices.spy, health_config)

    print_scale(health, health_config)
    print_state(current_state(health, health_config), health_config)
    eps = print_episodes(health, args.threshold, args.confirm_days,
                         health_config, column=health_config.column)
    table = pd.concat([print_calibration(health, health_config, "z_raw"),
                       print_calibration(health, health_config, "z_adj")],
                      ignore_index=True)

    health.to_csv(REPO_ROOT / "rsi_ma_model_health.csv")
    table.to_csv(REPO_ROOT / "rsi_ma_health_calibration.csv", index=False)
    if not eps.empty:
        eps.to_csv(REPO_ROOT / "rsi_ma_health_episodes.csv", index=False)

    print("\nExported: rsi_ma_model_health.csv, rsi_ma_health_calibration.csv"
          + (", rsi_ma_health_episodes.csv" if not eps.empty else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
