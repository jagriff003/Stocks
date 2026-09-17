# TODO — open work, roughly in order of expected value

Opened 2026-09-03. Each item states what to measure and what would count as an
answer, so it can be picked up cold. Nothing here is urgent; the model is
operating normally.

---

## 1. Defensive posture — is it held too long, and what does it cost?

**Raised 2026-09-03.** The regime overlay puts SHY/TLT/IAU in the book, and
momentum picks SH outright in a downturn. Across the record the book holds at
least one low-beta name on **39% of days**, and that composition is what drags
the blended beta to 0.48 from the 0.99 the momentum picks alone produce.

The concern is that the model can sit defensive through a rally, trailing the
index for reasons unrelated to whether the ranker works. Both historical health
alarms were exactly this shape: model roughly flat, SPY +17%.

What the new posture diagnostic already answers: the longest unbroken defensive
stretch on record is **68 sessions (~3 months)**, not years. So the posture does
release. The open question is what it costs while held.

To measure:

- Segment the return stream by defensive weight (0, 25%, 50%+ of book) and
  compare returns, and returns *relative to SPY*, in each bucket.
- For each defensive episode: what did the book earn, what did SPY earn, and
  what would the same-dated pure-momentum selection have earned? That last one
  is the counterfactual that matters — the overlay's cost is the gap between
  what it held and what it would have held.
- Separate the two sources. SHY/TLT/IAU arrive by regime rule; SH arrives by
  ranking. They are different mechanisms and may have opposite records.
- Condition on VIX band, so "the overlay was right in crises but wrong in
  ordinary elevated readings" is distinguishable from a flat verdict.

What would count as an answer: a per-episode table of overlay cost versus the
pure-momentum counterfactual, with the subperiod split. Track A already found
that de-risking forfeits the overnight premium, so the prior is that the overlay
costs return; this would quantify it and say whether the crisis protection is
worth the drag.

Note the standing caution before acting on any result: Tracks A-D were all
rejected, and the overlay is load-bearing for drawdown (-19.23% max). Measuring
its cost is not the same as removing it.

---

## 1b. Does equal-weight drift understate the tail? (raised 2026-09-17)

The outsized-event analysis concluded that extreme single-stock moves supply
only 8-15% of net P&L. That rests on the simulator's equal-weight-reset
convention: a name that gaps +20% is rebalanced back to 1/N the next day, so it
never compounds at its enlarged weight.

A real book does not do that. Within a 14-day hold, a name up 20% carries ~28%
of a four-name book rather than 25% for the remainder of the period, so the true
tail contribution is somewhat **larger** than measured. The bias is bounded by
the hold period, not by the life of the position, so the effect should be small
— but "should be" is not a measurement.

To measure: re-run `scripts/analyze_outsized.py` against a drift-aware
simulation (weights float within a hold, reset at rotation) and compare the net
share of tagged P&L. What would count as an answer: the σ-sweep table re-run on
drifting weights. If the 8-15% band moves past ~25% the conclusion weakens.

Note this cannot become the default simulator — `_segment_return`'s convention
is load-bearing for comparability with every historical result in FINDINGS. It
would be a parallel path used only for this diagnostic.

---

## 2. Live-vs-simulated reconciliation

**The health monitor's blind spot.** It scores the simulated return stream —
next-open fills, 7.5 bps slippage — not the actual account. If real execution
drifts (trading a day late, filling midday, odd lots, partial rebalances), the
monitor reports perfect health while the account underperforms the thing being
monitored. Known to happen already: 2026-09-03's trade went in a day after the
rotation.

What exists: dated universe and config snapshots, accumulating from 2026-07-25.

What is missing: a record of actual trades and realized P&L.

To do: a simple dated trade log — symbol, date, side, fill price — and a
comparison script that reports the gap between realized and simulated returns
over the same window, attributed to timing versus price. Even a hand-maintained
CSV would do; the value is in having the series at all.

What would count as an answer: the ability to say "the model degraded" or "my
execution degraded" without guessing. These call for opposite responses.

---

## 3. Recalibrate the health monitor without survivorship bias

The monitor's scale is estimated over a backtest that applies today's screened
universe back to 2010, so the historical excess distribution is optimistic and
live shortfalls will be more common than the calibration implies.

Point-in-time universe snapshots accumulate from 2026-07-25. Once there are
enough years of them, rebuild the backtest from the snapshots rather than from
today's universe, and re-derive the scale and thresholds from that.

Not actionable for several years. Recorded so the reason the current thresholds
are provisional does not get lost.

---

## 4. Revisit the health thresholds when there are more episodes

Two episodes support a rarity claim and nothing else. Forward six-month excess
after a trigger swings from -2.6% to +4.8% depending on record start, points
floor, and threshold.

Revisit after any new ALARM: add it to the record, re-run the calibration grid,
and check whether the forward-outcome column has stabilized. If it has not after
four or five episodes, accept permanently that this is a review trigger and stop
testing it for predictive power.

Also note: at today's scale the -12.5% points floor is stricter than z -1.25
(-10.9%), so the floor is the binding condition and tuning `alarm_z` alone will
look inert. Move both or neither.

---

## 5. Discipline: do not add more monitors

Recorded as a decision, not a task.

The health monitor barely calibrated — rarity yes, predictiveness no. A second
and third indicator, each resting on two or three observations, produces a
dashboard with something permanently amber, and that is the condition under
which people start tuning on noise. The walk-forward result says that is the
main way to lose money with this model.

Before adding any new signal, it must clear the same bar this one was held to:
what does it fire on historically, how often, and what happened next?
