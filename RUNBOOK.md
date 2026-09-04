# Runbook — what to do when the monitor says something

Written 2026-09-03, while nothing is wrong. That is the point: a decision made
ten months into a shortfall is a worse decision than the same one made now, and
the documented failure mode of this strategy is re-tuning under discomfort.

Every number below is reproducible from `python scripts/monitor_health.py`.

---

## First, the base rates

Do not treat these as problems. They are the normal operating range.

| | |
|---|---|
| 6-month windows trailing SPY | **42%** |
| 5th percentile 6-month shortfall | -10.9% |
| 1st percentile | -17.5% |
| Longest continuous stretch trailing SPY | **356 days** (2019-01 → 2020-01) |
| Next longest | 268 days, 170 days |
| Max drawdown on record | -19.23% |

A -20% drawdown and a -18% relative shortfall are both normal, but they are
different weather. A drawdown is capital leaving the account. A shortfall is the
index compounding without you — in both historical alarm episodes the model was
roughly flat (-1.8%, -2.6%) while SPY ran +17%. Do not read one as a proxy for
the other.

**Nearly a year of trailing the index is inside historical experience for a
strategy compounding at 19.8%.** Discomfort is not evidence.

---

## The states

Printed on every `run_live.py`, and in full by `monitor_health.py`.

| State | Condition | Meaning |
|---|---|---|
| **OK** | neither breach | nothing |
| **WATCH** | breach today, not yet sustained; or the warn pair sustained | note it, no action |
| **ALARM** | raw z ≤ -1.25 **and** excess ≤ -12.5%, for 20 consecutive sessions | work the ladder below |

ALARM fired twice in eleven years (2019-07, 2021-08). It is rare by
construction, and its rarity is the only property of it that calibrated — see
`momentum/health.py` on why it must not be read as predictive.

---

## The ladder

Work it in order. Each step is cheaper and more likely to be the answer than the
one after it. **Do not skip to step 4.**

### 1. Reconcile before diagnosing

The monitor scores the *simulated* return stream — next-open fills, 7.5 bps —
not your account. Before concluding anything about the model, confirm the two
still describe the same thing:

- Did you trade the aligned rotation set, on the rotation date?
- Were fills near the open, or did you trade midday / a day late?
- Does your actual 6-month return resemble the `126d model` figure printed?

If your P&L and the model's diverge, **the model is not what degraded** and
nothing below applies. This is the most likely cause and the easiest to miss.

### 2. Decompose: overlay drag, or bad picks?

The report prints both. They point in different directions:

| Raw | Beta-adjusted | Reading |
|---|---|---|
| bad | fine | A half-beta book failed to keep up with a rally. Structural, not broken. Check the defensive posture block. |
| bad | bad | The picks themselves stopped working. Continue to step 3. |

Both historical episodes were the first kind: the model held a defensive name on
~35% of days while SPY ran +17%. Check `DEFENSIVE POSTURE` — if the recent share
is well above the 39% long-run figure, the overlay is the story.

### 3. Screen the universe

`python scripts/screen_universe.py`

FINDINGS' standing conclusion is that every timing idea tested failed, and what
remaining upside exists is in the candidate set rather than in the timing of
trades among current candidates. A sustained shortfall with a fixed ranking
method is most consistent with a stale universe.

You already re-screen roughly quarterly. An ALARM is a reason to do it now and
to look harder, not to invent a new process.

### 4. Parameters — reluctantly, and with a prior against

Walk-forward validation found that re-tuning on a trailing window **beat the
median fixed configuration only 13% of the time and cost 3pp of CAGR**. The
instinct to tune when it hurts is the documented way to lose money here.

If you get this far, the bar is: name the specific parameter, state in advance
what result would change your mind, and run it as a suite with the subperiod
stability check — the same standard Tracks A-D were held to. A change that wins
the full sample by winning one segment is a fit, not a finding.

---

## What ALARM never means

- **Do not de-risk into it.** Track A found that cutting exposure forfeits the
  overnight premium — 63% of the return stream for 36% of the variance.
- **Do not trade the CURRENT SET off-cycle** because it looks better than what
  you hold. The rank-exit and score-swap suites both tested versions of that and
  both were rejected.
- **Do not treat it as predictive.** Forward six-month excess after a trigger
  swings from -2.6% to +4.8% depending on settings, on two to five observations.
  It is a prompt to look, and nothing more.
