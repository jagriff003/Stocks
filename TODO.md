# TODO — open work, roughly in order of expected value

Opened 2026-09-03. Each item states what to measure and what would count as an
answer, so it can be picked up cold. Nothing here is urgent; the model is
operating normally.

Items 0, 0b, 6, 7 and 8 were added 2026-09-17 and differ in kind from the rest:
they are not extensions of tested ideas but gaps in the research program itself.
Item 0 was run the same day and is answered — see FINDINGS Track F. Its result
promotes **item 6 to the top of this list**: the ranker turns out not to produce
the return, the universe does, and the universe number is the one survivorship
bias contaminates most.

---

## 0. What is the ranker actually worth? — ANSWERED 2026-09-17

**Raised and answered 2026-09-17.** Kept in full below, because the design is
reusable and because what it asked for is the reason the answer is trustworthy.

**The answer: the ranking is worth +0.84pp gross and sits at the 61st percentile
of random draws from its own universe. The IC is indistinguishable from zero at
every horizon from 5 to 126 days. Equal-weighting the whole universe and never
trading earns 19.99% against the model's 19.58%.** The model's contribution is
drawdown (-19.23% against -35.22%), not return. Full write-up in FINDINGS,
Track F.

Two follow-ons, neither of them a model change:

- The ranker's parameters are closed to further tuning. Two independent
  measurements say there is no information there to optimize.
- **Item 6 (bound the survivorship bias) is now the most valuable open item in
  this file** and should be read as item 0's successor. The universe is what
  carries the return, and the universe figure is the one most contaminated by
  applying today's screen backwards.

What follows is the original specification.

Every result in FINDINGS is benchmarked against SPY. SPY is the wrong control.
It answers "did we beat the market", which is not the question the record
actually leaves open. The universe is 52 hand-picked names — GOOGL, NVDA, META,
LLY, TSLA — selected *today* on a momentum screen and applied backwards to 2010.
Against that universe, a large part of 19.58% may be available without any
ranking at all.

**Nothing in the repo has ever tested the ranker against the universe it picks
from.** Grep confirms it: no random-portfolio test, no equal-weight-universe
benchmark, no information coefficient, no decile spread. The composite score has
been tuned, blended, offset, gated and swept — never once shown to carry
cross-sectional information.

### What to measure

**A. The null portfolio.** Draw N names at random from the same universe on the
same 14-day clock, through the same `simulate_portfolio` with the same fills and
slippage. Repeat across many seeds (1,000 is cheap here) and report the full
distribution of CAGR, Sharpe and MaxDD. Then state where the live model sits in
it as a percentile.

Run four arms, because two different things are being credited to the ranker
right now and they need separating:

| Arm | Selection | Overlay | Isolates |
|---|---|---|---|
| 1 | ranked (live) | on | the model as it stands |
| 2 | random | on | what the overlay alone earns |
| 3 | ranked | off | the ranker alone |
| 4 | random | off | the universe alone |

Arm 4 is the real baseline. Arm 1 minus arm 2 is what the ranking is worth.

**B. Equal-weight the whole universe**, rebalanced on the same clock. A single
number, and the most intuitive statement of "what did picking get you".

**C. The mechanism half — information coefficient.** Cross-sectional Spearman
correlation between the composite score on day T and forward returns over the
hold horizon, per date, across the record. Report the mean IC, its t-statistic,
and the mean forward return by score decile. Also run it at several horizons
(5, 10, 14, 21, 42 days) so "the signal works but not at 14 days" is
distinguishable from "the signal does not work".

### What would count as an answer

A percentile for arm 1 against arm 4's distribution, and a mean IC with a
t-stat. Thresholds configurable; suggested reading of the result:

- **Above the 90th percentile, IC clearly positive** — the ranker earns its
  keep, and every prior finding stands as written.
- **50th to 90th, IC near zero** — the return is the universe and the overlay.
  The ranking machinery is elaborate noise, `velocity_window=5`'s +6.22pp is a
  selection artifact, and the research program should move to the universe.
- **Below the 50th** — picking is actively hurting, and the cheapest available
  improvement is to stop.

### Why this is worth doing even though it might be unwelcome

It is the one test that can *reduce* confidence in the model, which is exactly
why it has not been run and exactly why it should be. It also offers a single
coherent explanation for the standing puzzle in FINDINGS — that Tracks A, B, C
and D all failed, from four unrelated directions. "The composite does not
produce *timely* information" is already the recorded conclusion. This tests the
stronger and simpler version: whether it produces information at all.

Note the result is not fatal even in the bad case. A 19.58% CAGR at -19.23%
drawdown is worth having however it arises; what changes is where the next
decade of effort goes, and whether the ranker's parameters deserve any further
attention.

---

## 0b. Position sizing — the one dimension never tested

**Raised 2026-09-17.** Every experiment in the record varies *what* to hold
(universe, ranking, offset) or *when* to hold it (hold days, exits, regime
ladder, entry timing). Track E added *how many*. **Nothing has ever varied how
much.** The book is equal-weighted, always, and that is a strategy choice that
has been treated as a simulation convention.

This is the most promising untested lever specifically because it does not
contradict the unifying result. Every rejected track lost by trading more.
Sizing rules change the weights, not the rotation dates — turnover moves
marginally or not at all, so the 2.90pp turnover tax that killed Tracks A-D does
not apply.

It is also pointed at the constraint that actually binds. Track E showed the
thing separating book sizes is drawdown, not CAGR, and the whole RUNBOOK ladder
is about tolerating shortfall. Sizing is the standard lever on exactly that.

### What to measure

Same universe, same ranking, same clock, same fills. Vary only the weights:

- **equal** (the incumbent baseline)
- **inverse volatility** — weight by 1/σ on a trailing window (window a flag)
- **volatility target** — scale gross exposure so the book's ex-ante vol hits a
  target, cash or SHY taking the remainder (target and window both flags)
- **score-proportional** — weight by composite score rank or z, so conviction
  maps to size
- **capped variants** of each, since an uncapped inverse-vol book in this
  universe will concentrate hard into whatever is quietest

Report CAGR, Sharpe, MaxDD, Calmar *and* turnover side by side, subperiod
stability, and the realized share of the book in the largest position.

### What would count as an answer

A comparison table on the existing suite conventions, plus an explicit turnover
column proving the variant did not smuggle in extra trading. The bar: beat the
median configuration in all three subperiods, per the standing convention.

Two cautions to carry in. First, vol-targeting is a cousin of the regime overlay
that Track A already rejected — if it de-risks into calm-then-violent markets it
will forfeit the overnight premium the same way, and the honest prior is that
the *de-risking* variants lose and the *re-weighting* variants are the live
possibility. Second, `_segment_return` resets to equal weight daily, so this
work requires a weight-aware simulation path; see item 1b, which needs the same
machinery. Build it once for both.

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

## 1b. Does equal-weight drift understate the tail? — ANSWERED, CLOSED 2026-09-17

**Answered by Track G.** `momentum/drift.py` was built for the wide-book work
and prices this directly: the legacy convention's free daily rebalance is worth
**0.07pp gross**, and rebalancing policy (`never` / `on_rotation` / `periodic` /
`band`) moves nothing by more than 0.2pp at any book size or hold length. The
approximation is real and negligible. No re-run of the outsized-event analysis
is warranted. Original text follows.

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

---

## 6. Bound the survivorship bias now, instead of waiting years

**THE DECISIVE EXPERIMENT FOR THE WHOLE PROGRAM**, promoted 2026-09-17 by
Track F and made decisive the same day by Track H. It sits at 6 only because
renumbering the file would break every reference to it.

Track H ran the model on 100 random baskets. It beat its own basket in 2 of
them, giving up 8.8pp of CAGR on average, with Sharpe collapsing from ~0.65 to
~0.17. On the live universe it gives up 1.44pp and cuts drawdown by 16pp. The
live universe sits at the 96th percentile of the random distribution, z +1.92.

That leaves exactly two readings - the curation is hindsight, or the curation is
a repeatable rule - and no test can separate them using a universe that was
assembled with the answer visible. This item is that test: apply the selection
rules using only information available in 2010 and see whether the model still
works. Until it is run, the model's mechanism is unestablished.

Track F established that the universe, not the ranker, produces the return:
equal-weighting all 49 momentum names and never trading earns 19.99%. That
number is also the single most survivorship-contaminated figure in the repo,
because it is exactly "what did the names I picked in 2026 do since 2010" with
no selection skill in between to muddy it. Every headline in FINDINGS rests on
it. Until it is bounded, the honest statement about this strategy's absolute
return is that it is unknown.

**Raised 2026-09-17.** Item 3 parks the survivorship problem until enough
point-in-time snapshots accumulate. At one snapshot per rotation from
2026-07-25, a usable record is most of a decade away, and in the meantime every
absolute number in FINDINGS carries an unknown and unstated inflation.

It does not have to stay unknown. The bias can be *bounded* now, with data
already in hand, by running the existing model against universes chosen without
hindsight:

- **A frozen 2010 universe.** Take a plausible 2010-vintage list — S&P 100
  constituents as of 2010-01-01, or the sector-balanced 50 largest US names by
  market cap then — and run the unchanged model on it from 2010. Everything that
  subsequently died or stagnated stays in, which is the point.
- **A rolling-vintage universe.** Re-screen every N years on the rules in
  `screen_universe.py` using only data available at that date, and splice the
  segments. Closer to real practice; more work, and the screen's own rules are
  partly hindsight.
- **Delisted-name recovery.** Establish whether the price source returns data
  for names that no longer trade. If it does not, that is itself a finding, and
  it caps what any of the above can achieve.

### What would count as an answer

A single number: the CAGR gap between the live universe and the frozen-2010
universe over the same window. That number is the survivorship premium, and once
it exists it can be subtracted — from the headline, from the health monitor's
scale (item 3), and from the outsized-event analysis, whose tail estimate is
biased in the reassuring direction for the same reason.

Even a rough figure is a large improvement on the current position, which is
that the bias is acknowledged in prose and quantified nowhere.

---

## 7. Account for how many hypotheses have been tested

**Raised 2026-09-17.** FINDINGS records five tracks, roughly a dozen suites and
well over a hundred configurations, all evaluated on one 14-year price history.
The reported 0.90 Sharpe is the maximum over that search, not a draw from it,
and no correction anywhere in the repo reflects that.

The walk-forward result is the closest thing and it answers a different
question — it shows that *re-tuning on a schedule* is harmful, not how much of
the selected configuration's edge is selection.

### What to measure

- Count the configurations actually evaluated, by suite. FINDINGS and the
  exported comparison CSVs already hold this; it needs tallying, not new runs.
- Compute a deflated Sharpe ratio, or the equivalent multiple-testing haircut,
  using that count and the correlation among the tested variants (they are far
  from independent, which cuts the effective count well below the raw one).
- Report the haircut next to the headline in FINDINGS, permanently.

### What would count as an answer

A deflated Sharpe and the number of effective independent trials behind it. If
the deflated figure stays comfortably positive, that is real reassurance and it
is currently unclaimed. If it does not, it belongs next to the headline anyway.

This pairs naturally with item 0: the null-portfolio distribution is an
empirical version of the same correction, and if both are built the two should
agree. Disagreement between them would itself be informative.

---

## 8. Factor attribution — is this momentum, or low-beta, or something of its own?

**Raised 2026-09-17.** Lower priority than 0, 0b, 6 and 7, and recorded mainly
so it is not mistaken for unexplored ground later.

`health.py` establishes the model runs at roughly half of SPY's beta with 0.42
correlation, which is a useful fact and not an attribution. What is unknown is
how much of the return survives controlling for published factor returns —
market, size, value, momentum, quality, low-volatility. A strategy that is
cross-sectional momentum plus a defensive overlay should load heavily on
momentum and low-beta, and the interesting quantity is the alpha left over.

Requires external data the repo does not have (the Ken French library, or an
equivalent), which is the reason for the lower ranking rather than any doubt
about its usefulness.

### What would count as an answer

A regression of monthly model returns on the factor set, with loadings,
t-statistics and the residual alpha. If alpha is indistinguishable from zero and
the loadings are all on momentum and low-beta, the model is a well-executed
factor portfolio — which is worth knowing plainly, and is a perfectly good thing
to own, but it would reframe the universe work in item 0 as the only place
genuine edge could come from.
