"""
Context signals, with what they have historically meant.

These series never enter the scoring panel.  Testing put every one of them at
the noise floor as a monitor - individually +0.2 to +0.8pp, but flipping sign
across subperiods, and materially NEGATIVE in combination (all monitors
together: +0.02pp of CAGR and drawdown out to -20.84% from -18.80%).  They are
displayed and nothing more.

A LEVEL IS NOT A SIGNAL
   "RSP/SPY is 77.6" tells a reader nothing.  Every series here is therefore
   reported three ways: its current STATE, what that state has historically been
   followed by, and how that compares to the base rate.  A context panel that
   cannot say what a reading meant last time is decoration, and decoration on a
   trading screen is worse than a blank space because it invites invention.

   The bar is the one TODO item 5 sets for any new indicator: what does it fire
   on historically, how often, and what happened next.

OVERLAPPING WINDOWS, AGAIN
   Forward 14-day returns sampled daily share 13 of 14 days.  A naive t-statistic
   over every date is inflated by roughly sqrt(14), which is the same trap that
   made a 63-day IC read -4.11 when the honest figure was -0.53.  The
   significance test here samples every `horizon`-th date so no two observations
   share a forward window, and the effective sample size is printed beside the
   result so a reader can see how thin it is.

WHAT THESE ARE NOT
   Not a forecast, and not a trigger.  The strategy's own record is that every
   timing idea tested failed, and nothing here is exempt from that.  These
   answer "what kind of market is this" for a human deciding whether to look
   harder, not "what should the model do".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass
class ContextSignal:
    """A displayed series, its plain-language meaning, and how to read it."""
    name: str
    label: str
    rising_means: str
    falling_means: str
    why_it_matters: str
    risk_on: Optional[str] = "above"
    green_means: str = ""

    @property
    def colour_note(self) -> str:
        """What the shading means for THIS signal, spelled out."""
        if self.risk_on is None:
            return "grey = above trend (direction is ambiguous for equities)"
        return f"green = {self.green_means}"


SIGNALS: Dict[str, ContextSignal] = {
    "RSP_SPY": ContextSignal(
        "RSP_SPY", "Breadth: equal-weight vs cap-weight S&P",
        "the average stock is keeping up - broad participation",
        "a handful of mega-caps are carrying the index - narrow leadership",
        "a 4-name book lives or dies on whether leadership is broad enough "
        "to have picked from",
        risk_on="above",
        green_means="broad participation - more names to pick from"),
    "HYG_LQD": ContextSignal(
        "HYG_LQD", "Credit appetite: high-yield vs investment-grade",
        "investors are paid to take credit risk and are taking it",
        "credit is being repriced - stress that equities often lag",
        "duration cancels out, so this is risk appetite rather than rates",
        risk_on="above",
        green_means="credit is calm - stress usually shows here before equities"),
    "IWM_SPY": ContextSignal(
        "IWM_SPY", "Risk appetite: small-cap vs large-cap",
        "money is moving down the size curve - risk-seeking",
        "money is hiding in size - defensive rotation",
        "the universe is mid/large cap, so this says whether that tilt is "
        "where the market wants to be",
        risk_on="above",
        green_means="risk-seeking - money moving down the size curve"),
    "AD_LINE": ContextSignal(
        "AD_LINE", "Breadth: share of the pool advancing (20d avg)",
        "more than half the pool is advancing - participation is real",
        "the index is being held up by fewer names than it appears",
        "computed from the pool, not bought: Yahoo serves no breadth index",
        risk_on="above",
        green_means="more than half the pool advancing - a real rally, not a few names"),
    "UUP": ContextSignal(
        "UUP", "US dollar index",
        "a headwind for multinational earnings and for commodities",
        "a tailwind for multinationals, commodities and emerging markets",
        "the universe is 48 US equities and cannot see the dollar at all",
        risk_on="below",
        green_means="a WEAK dollar - note the inversion: dollar strength is the "
                    "headwind, so green here is the LOWER line"),
    "DBC": ContextSignal(
        "DBC", "Broad commodities",
        "inflationary pressure and real-asset strength",
        "disinflation, or demand weakness",
        "held in the model once and dropped: too volatile on a two-week hold. "
        "Direction here is a PRIOR, not a measured effect - see below",
        risk_on="below",
        green_means="cheap commodities. Rising commodities read as a cost and "
                    "rate headwind: SPY returned +0.59% over the next 14d with "
                    "DBC above its 200d average against +1.07% below, and the "
                    "sign held in 14 of 15 tests across 3 state definitions and "
                    "5 horizons - but max |t| was 1.31, the tests overlap "
                    "heavily, and none is individually significant"),
    "SH": ContextSignal(
        "SH", "Short S&P 500 (monitor only, never held)",
        "the counter-trade is ranking well - breadth is deteriorating",
        "the market trend is intact",
        "bought on 14.6% of book-days before being made a monitor, and it was "
        "wrong: SPY rose over the next 20 sessions on 69% of those entries"),
}


def trend_state(series: pd.Series, window: int = 200) -> pd.Series:
    """Above or below its own moving average - the simplest honest state."""
    ma = series.rolling(window, min_periods=window // 2).mean()
    return pd.Series(np.where(series > ma, "above", "below"),
                     index=series.index).where(ma.notna())


def forward_return(series: pd.Series, horizon: int) -> pd.Series:
    """Return over the next `horizon` sessions."""
    return series.shift(-horizon) / series - 1


def conditional_outcomes(state: pd.Series, outcome: pd.Series,
                         horizon: int) -> pd.DataFrame:
    """
    What followed each state, with an overlap-aware significance test.

    `outcome` is a forward return sampled daily, so consecutive observations
    overlap.  The reported t-statistic uses every `horizon`-th date within each
    state; `N indep` is how many genuinely independent observations that leaves,
    and it is usually small enough to matter.
    """
    df = pd.concat([state.rename("state"), outcome.rename("fwd")],
                   axis=1).dropna()
    rows = []
    for st, grp in df.groupby("state"):
        indep = grp["fwd"].iloc[::horizon]
        n = len(indep)
        t = (indep.mean() / (indep.std() / np.sqrt(n))
             if n > 2 and indep.std() > 0 else np.nan)
        rows.append({
            "State": st,
            "Days": len(grp),
            "Share of history": len(grp) / len(df),
            "Mean fwd": float(grp["fwd"].mean()),
            "Median fwd": float(grp["fwd"].median()),
            "Hit rate": float((grp["fwd"] > 0).mean()),
            "N indep": n,
            "t": float(t) if pd.notna(t) else np.nan,
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        base = df["fwd"].mean()
        out["vs base rate"] = out["Mean fwd"] - base
    return out


def state_difference(state: pd.Series, outcome: pd.Series,
                     horizon: int) -> Optional[Dict[str, float]]:
    """
    Test whether the two states actually differ - the only question that matters.

    A per-state t-statistic against zero is nearly useless here: SPY drifts up,
    so EVERY state shows a significant positive mean and a reader sees t = 3.55
    and concludes the signal works.  It does not test that.  This tests
    above-minus-below on non-overlapping samples, which is the comparison a
    context panel is implicitly making.
    """
    df = pd.concat([state.rename("state"), outcome.rename("fwd")],
                   axis=1).dropna()
    groups = [g["fwd"].iloc[::horizon] for _, g in df.groupby("state")]
    labels = [k for k, _ in df.groupby("state")]
    if len(groups) != 2 or min(len(g) for g in groups) < 3:
        return None
    a, b = groups
    diff = a.mean() - b.mean()
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return {"a": labels[0], "b": labels[1], "diff": float(diff),
            "t": float(diff / se) if se > 0 else np.nan,
            "na": len(a), "nb": len(b)}


def report(close: pd.DataFrame, spy: pd.Series,
           names: Sequence[str],
           model_returns: Optional[pd.Series] = None,
           horizon: int = 14, window: int = 200) -> str:
    """
    Render the context panel: state now, and what that state has been worth.

    `model_returns` is optional; when given, the panel also reports what the
    STRATEGY did over the following `horizon` days in each state, which is the
    more directly useful number than what SPY did.
    """
    lines: List[str] = []
    spy_fwd = forward_return(spy.reindex(close.index).ffill(), horizon)

    model_fwd = None
    if model_returns is not None and len(model_returns):
        wealth = (1 + model_returns).cumprod()
        model_fwd = forward_return(wealth, horizon)

    lines.append("=" * 84)
    lines.append(f"CONTEXT  — state now, and what it has been followed by "
                 f"({horizon} sessions)")
    lines.append("=" * 84)
    lines.append("  Displayed only. None of these is in the scoring panel; all "
                 "tested at the")
    lines.append("  noise floor as monitors and negative in combination.")

    for nm in names:
        if nm not in close.columns:
            continue
        sig = SIGNALS.get(nm)
        s = close[nm].dropna()
        if len(s) < window + horizon:
            continue
        st = trend_state(s, window)
        now = st.dropna().iloc[-1] if st.notna().any() else None
        if now is None:
            continue

        label = sig.label if sig else nm
        meaning = (sig.rising_means if now == "above" else sig.falling_means) \
            if sig else ""
        chg = s.iloc[-1] / s.iloc[-horizon - 1] - 1 if len(s) > horizon else np.nan

        lines.append("")
        lines.append(f"  {label}")
        lines.append(f"    now: {now.upper()} its {window}d average"
                     f"   ({horizon}d change {chg:+.1%})")
        if meaning:
            lines.append(f"    reads as: {meaning}")
        if sig and sig.why_it_matters:
            lines.append(f"    why: {sig.why_it_matters}")

        tbl = conditional_outcomes(st, spy_fwd, horizon)
        if tbl.empty:
            continue
        lines.append(f"    {'state':<8}{'days':>7}"
                     f"{'SPY next ' + str(horizon) + 'd':>17}{'hit':>7}")
        for _, r in tbl.iterrows():
            mark = "  <-- now" if r["State"] == now else ""
            lines.append(f"    {r['State']:<8}{r['Days']:>7,}"
                         f"{r['Mean fwd']:>16.2%} {r['Hit rate']:>6.0%}{mark}")

        d = state_difference(st, spy_fwd, horizon)
        if d:
            verdict = ("DISTINGUISHABLE" if abs(d["t"]) >= 2
                       else "not distinguishable from noise")
            lines.append(f"    SPY, {d['a']} minus {d['b']}: {d['diff']:>+.2%}"
                         f"   t {d['t']:>5.2f}  (n {d['na']}/{d['nb']})"
                         f"   -> {verdict}")

        if model_fwd is not None:
            mt = conditional_outcomes(st, model_fwd, horizon)
            if not mt.empty:
                bits = [f"{r['State']} {r['Mean fwd']:+.2%}"
                        for _, r in mt.iterrows()]
                lines.append(f"    model next {horizon}d:  " + "   ".join(bits))
            dm = state_difference(st, model_fwd, horizon)
            if dm:
                v = ("DISTINGUISHABLE" if abs(dm["t"]) >= 2
                     else "not distinguishable from noise")
                lines.append(f"    model, {dm['a']} minus {dm['b']}: "
                             f"{dm['diff']:>+.2%}   t {dm['t']:>5.2f}   -> {v}")

    lines.append("")
    lines.append("  The difference line is the one that matters. A per-state "
                 "t against zero would")
    lines.append("  read significant in every state simply because the market "
                 "drifts up; the test")
    lines.append("  here is above-minus-below on non-overlapping windows. "
                 "Most of these are noise.")
    return "\n".join(lines)
