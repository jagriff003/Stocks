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

## 0d. Delisting simulation — bound the survivorship exposure on the wide pool

**Added 2026-09-20.** The Track J backtest earns 24.35% CAGR at 0.65 Sharpe on
635 liquid names against 14.81% for equal weight. The single largest reason not
to believe it is that the pool was built in 2026 from names that still exist,
and `pullback` buys drawdowns — so every dip in the sample was followed by a
recovery, because the names whose dips were terminal are absent.

This cannot be fixed without point-in-time data. It can be **bounded**, and the
bound is the deliverable.

### What has to be settled first (agreed 2026-09-20)

The simulation is only as good as its model of what delisting does to a
position, which is a factual question, not a modelling choice:

- **Delisting is not one event.** Mergers and take-privates pay cash or
  acquirer stock and are the most common exit for a healthy name — those are
  neutral-to-positive and would *help* a momentum signal, since takeovers come
  at a premium. Compliance delistings move to OTC and keep trading at a wide
  spread. Chapter 11 usually cancels or massively dilutes equity. Chapter 7 is
  a total loss.
- So the simulation needs a **mixture**, not a single -100%. Getting the
  mixture weights roughly right matters more than getting any one branch exact,
  because the merger branch and the bankruptcy branch pull in opposite
  directions.
- **The loss is mostly before the event, not at it.** A name takes months to
  travel from deficiency notice to delisting, and a rebalancing strategy with a
  liquidity screen has several chances to exit. The exposure is holding through
  the decline, which is the same exposure the backtest already prices for
  surviving names that fell hard.

### What to measure

1. Historical delisting rate by size/liquidity decile, and the split between
   merger, compliance and bankruptcy exits.
2. Inject synthetic names at those rates into the pool, each carrying a
   terminal path drawn from the matching branch. Let the strategy see them and
   act under its normal rules, **including the liquidity screen** — the point
   is to test whether the screen catches them, not to assume it cannot.
3. Report edge decay as a function of assumed delisting rate, in the same form
   as the breakeven-bps figure: *what rate would it take to erase this?*

### What would count as an answer

A delisting rate high enough to erase the edge that is clearly above the
plausible historical rate for $10M+ ADV names. If the edge dies at a plausible
rate, the wide-pool result is not usable and item 6 becomes mandatory before
anything else.

### Where it landed, 2026-09-20 — RUN, INCONCLUSIVE, ACCEPTED AS A KNOWN RISK

Run at rates 0/4/10% with a survivor control arm. Results:

- The edge over owning the pool is **flat across rates**: +7.98%, +7.74%,
  +8.27%. Absolute CAGR degrades only modestly, 21.5% -> 19.3%.
- **The screen earns its keep.** With realistic starting prices, 37% of
  cause-based delistings stop being tradable a median **95-100 sessions** ahead
  of the event. That is the mitigation actually in place.
- **The cost decomposition is not usable.** It comes out positive — delisting
  appearing to *help* — and the control arm shows why: injecting 497 immortal
  clones alone cost 6pp, while the merger branch added more back. Both artifacts
  exceed the effect being measured.

Root cause of the remaining unreliability: synthetic names are circularly
shifted clones of survivors, which gives realistic volatility but destroys
co-movement with the market — a rolled name does not crash in 2020 with
everyone else, so it does not behave like a real pool member.

**Decision (James, 2026-09-20): stop here.** There is not enough evidence to
action anything beyond the screens already programmed in. The model goes
forward carrying this as a known, unquantified risk rather than a blocker.

### What actually closes it, and the one cheap thing to start now

Only point-in-time data with delisted securities settles this (CRSP, Norgate,
Sharadar). Short of buying it, the free option is a forward record — and it has
to be started deliberately, because it is worthless retroactively:

> **OPERATING REMINDER: run `run_live_trackj.py` WITHOUT `--no-snapshot`.**
> Every example invocation written during development used that flag to avoid
> polluting the record while testing, which makes it easy to copy by accident.
> The runner now prints a loud banner when the flag is used. A rotation that
> goes unrecorded is a permanent hole -- there is no way to reconstruct which
> names were tradable on a past date once the provider has moved on.

**Snapshot the eligible POOL at every rebalance,** the way
`snapshots/universe/` already snapshots the universe. Names that later vanish
from the snapshots are precisely the delistings this study could not see, and
after a few years the snapshots ARE the point-in-time dataset. Costs one CSV
per rebalance and nothing else. Pair it with item 2 (live-vs-simulated
reconciliation) so both records accumulate together from day one.

---

## 0h. Pre-production bug hypotheses for the Track J model — triage list

**Added 2026-09-20.** Not performance risks. These are ways the model could say
something that **does not happen in practice** — divergences between what the
backtest computes and what a live run would face. Ordered by how badly each
would mislead, not by likelihood. Most will be noise; the point is that they are
written down before go-live rather than diagnosed after.

### High — would change the book, silently

1. **Rotation phase is worth MORE at `hold=42`, not less.** FINDINGS records
   that phase is worth ~3.2pp of CAGR at `hold=14`. Going to 42 means six
   rebalances a year instead of eighteen, so *which* 42-day cycle you happen to
   start on matters more, not less. The backtest reports one phase. **Re-run the
   phase sensitivity at `hold=42` before go-live** — this is the one most likely
   to make live results diverge from backtest for a reason nobody suspects.

2. **The eligibility floor still reads the OLD composite.** `min_level_threshold`
   is applied to `base_scores`, which is the production composite. That was
   correct for the comparison — both arms had to face an identical pool — but
   shipping `flip_neg` while keeping a floor derived from a score we have just
   shown carries negative ranking skill is incoherent. Decide: drop the floor,
   or re-derive it from the new score. Note stage one measured the floor as
   near-inert (650.5 vs 652.2 eligible names), so dropping it is probably free.

3. **Tie-breaking is arbitrary and `flip` is built from ranks.** `flip` is a
   difference of two cross-sectional rank panels, so exact ties are genuinely
   possible rather than measure-zero. `rank(method="first")` then breaks them by
   column order, i.e. effectively by ticker alphabetically. Quantify how often
   ties reach the top 8, and break them on something defensible.

4. ~~**The correlation filter is untested at this book size and pool size.**~~
   **ANSWERED 2026-09-20**, `scripts/analyze_book_correlation.py`. Keep the
   inherited `apply_above_vix=25.0` gating: off is worse on everything
   (18.84% / 0.54 against 20.93% / 0.64), and always-on costs 1.5pp of CAGR to
   buy 1pp of drawdown with Sharpe unchanged (0.64 -> 0.65). The filter
   transfers from the 46-name universe without modification.

   Concentration is real but modest: held-book pairwise correlation 0.319
   against 0.278 for random books from the same pool, an excess of **+0.042**,
   exceeding random in 60% of rebalances. Largest single-sector share averages
   40%; the book is >=50% one sector in 28% of rebalances and >=75% in 5%.

   **The open gap is that the filter constrains CORRELATION, not SECTOR.** The
   worst books are miners, not semiconductors -- 2016-11-08 was 100% Basic
   Materials (CDE HL CLF MUX BVN NEM AEM PAAS) and happened *with* the filter
   configured, because that date's VIX was under 25. Precious-metals miners pull
   back together, so a dip-buying score picks them as a group. A sector cap is a
   different instrument and is untested; it belongs with 0e rather than here,
   since the >=75% case is 5% of rebalances.

### Medium — real but bounded

5. **Adjusted closes are restated.** The panel is fully adjusted, so a dividend
   or split tomorrow rewrites yesterday's prices. `range_pos` reads a 52-week
   high off that restated series, which is not the high that was observable at
   the time. Small, but it is exactly the "backtest saw something live cannot"
   class. Measure by comparing `range_pos` computed on adjusted vs unadjusted
   closes.

6. **Data reliability scales with the panel.** At 46 names a missing series is
   obvious; at 635 the chance of at least one bad or stale series on any given
   day is far higher, and the model would rank on it without complaint. The
   existing unsettled-tail guard checks the panel's last date, not per-name
   staleness. Add a per-name freshness check.

7. **The live screen must match the backtest screen exactly.** Trailing median
   dollar volume over `adv_window`, plus the price floor. A live implementation
   that uses a vendor's "average volume" field instead would be a different
   filter, and the divergence would be invisible.

8. **`slippage_panel` takes `top_n` to size the order.** If the live book size
   ever differs from the one the cost panel was built with, the modelled costs
   are wrong in a direction nobody notices.

### Low — worth a look, unlikely to matter

9. Corporate actions landing mid-hold, where an adjustment lag could produce a
   spurious `flip` reading for a day.
10. Defensive sleeve and VIX overlay are currently *off* in the recommended
    configuration; `run_live.py` prints regime and defensive-posture blocks that
    would become misleading rather than wrong. Cosmetic, but it is the kind of
    thing that erodes trust in the output.

### How to work this list

Each item wants the same treatment the repo already uses: an assertion that
fails loudly, or a measurement with a stated tolerance. Prefer a test in
`tests/` over a note here. Items 1 and 2 should be closed before any live
trade; the rest can be triaged after.

---

## 0i. Two ways to stop betting on a single rotation phase

**Added 2026-09-20.** The phase test found the new score's CAGR spans 13.16% to
23.35% across rebalance offsets — a **10.19pp** spread at `hold=42`, against
~3.2pp at `hold=14`. The equal-weight arm's spread is 0.71pp, so this is purely
a rotation artifact, not anything about the data. In live trading you get one
phase and cannot know in advance whether it is a good one.

Two independent ways to stop taking that bet, raised by James:

### 1. Staggered tranches

Split capital into k sleeves started on different offsets, each rotating on its
own `hold`-day clock. The book becomes the union, and realized CAGR approaches
the phase *median* instead of a single draw.

Attractive because it **makes no signal claim at all** — it is pure variance
reduction, and nothing in FINDINGS argues against it. Two things to check
rather than assume:

- **Turnover per dollar should be unchanged**, since each sleeve trades 1/k of
  the capital k times as often. Confirm it, because if true this is nearly free.
- **Slippage may actually improve**: `slippage_panel` scales impact with order
  size, and three smaller orders cost less than one large order in the same name.
  So tranching could be slightly *accretive* on cost while cutting phase risk.
- Position count goes from 8 to 8k unless `top_n` is cut per sleeve. Decide
  which, and note the sleeves will overlap in names.

### 2. Exit rules that break the clock

A trailing stop (or any exit condition) takes a position out mid-hold and frees
the slot for the next-best name — so positions naturally desynchronize and the
book stops rotating in lockstep. Phase diversification arrives as a side effect
of the exit rule rather than as its purpose.

**This one carries a prior against it.** Track B tested rank-triggered exits and
score-gap swaps and rejected both: losses scaled monotonically with turnover,
-1.2pp to -10.5pp. "Exit and replace with next-best" is structurally close to
what was rejected there. Two things differ — Track B ran on the 46-name
universe with the old composite at `hold=14`, and the motivation here is
variance reduction rather than signal improvement — but the burden is the same,
and it should be tested as [[TODO#0g. A sell-side framework — volatility-guided trailing stop|0g]]
with the phase benefit measured separately from the return effect.

Sequence: do (1) first. It is cheaper, it has no prior against it, and if it
resolves the phase problem on its own then (2) only has to justify itself on
drawdown.

---

## 0j. Anchor the rotation to a weekday, not a session count — DONE 2026-09-20

**Added 2026-09-20.** The production model rotates on a Tuesday 28 times out of
its last 30, which fits the operating pattern exactly: run after Tuesday's
close, trade Wednesday's open. Track J at `hold=40` does not — its last six
rotations were Tue, Mon, Mon, Tue, Mon, Mon.

The cause is that **40 sessions is 8 calendar weeks only when no holiday falls
inside the cycle.** Each holiday stretches the cycle by a weekday, so the
rotation walks forward and then oscillates. Choosing a whole-week hold pins the
weekday only in a market with no holidays.

Only two rotation dates coincided between the two models in all of 2026.

### Why it matters more than it sounds

Trading a Monday rotation in a Wednesday window is two sessions late. The
RUNBOOK prices one stale session at about -10 bps and two at about -18 bps, but
the dispersion is the real cost: a one-session-stale panel picks a different
name **56% of the time**. Two sessions is worse. This is not a rounding error on
a 4-to-8 name book.

### What to build

Rotate when *both* conditions hold: at least `hold_days` sessions have elapsed
AND today is the target weekday. That makes the schedule a calendar, publishable
in advance, and immune to holiday drift — at the cost of a hold that varies
between 40 and 44 sessions.

Check before adopting:
- what the varying hold costs against the fixed-40 baseline (the `top_n=8` row
  of the sweep was flat from 5 to 63 sessions, so expect little);
- that it still lands on the same weekday when a holiday falls ON the target day
  — the rule needs a documented fallback, presumably the next session;
- the interaction with tranching: at k=4 each sleeve rotates every 10 sessions,
  so all four want the same weekday, two weeks apart.

**DONE.** `momentum/schedule.py` defines rotations on the calendar — every N
weeks on a named weekday, mapped forward onto real sessions, anchored to a known
production rotation date so both models land on the same Tuesdays. Ten tests
cover it, including that the weekday holds across holidays and that a fallback
moves later rather than earlier.

---

## 0k. The live runner does not implement tranching — DONE 2026-09-20

**Added 2026-09-20.** `run_live_trackj.py` produces a single book. The
recommended configuration is **k=4 staggered sleeves**, which is what the
phase-spread result (11.62pp -> 2.53pp) and the conviction-weighting result are
measured on. The runner and the recommendation currently disagree.

What it needs: four sleeve states, each with its own rotation date, a union book
with per-name sleeve counts (that count IS the position weight, and multi-sleeve
names annualised 31-35% against 14% for single-sleeve), and a clear statement of
which sleeve rotates next.

**DONE.** The runner builds four sleeves from the calendar, prints each
sleeve's book and selection date, the combined book weighted by how many sleeves
hold each name, and which sleeve rotates next. The backtest block simulates the
sleeves on the same calendar rather than the old session clock, so the metrics
describe what is being traded: 20.81% CAGR, 0.73 Sharpe, -42.99% drawdown, which
sits inside the tranching study's k=4 range of 20.66-23.19% as one realisation
rather than a median.

Two effects fell out of it. The health monitor went from **ALARM** (126d excess
-19.3%) to **OK** (-2.0%), and the combined book's sector mix is Technology
7/29, Energy 7/29, Basic Materials 7/29 with no majority — against a
majority-Technology single book. Mean pairwise correlation across the combined
book is 0.16 against 0.34 for the single book.

---

## 0l. Track K — hedge mode: rotate part of the book into real assets when stocks stop working

**Added 2026-09-23 at James's request.** The pool holds plenty of
inflation-*sensitive* equity (13 gold miners, 54 energy, 43 REITs, 13 base
metals) but exactly one direct inflation instrument, IAU. In 2021-22, the one
inflation episode in the panel, the miners lost 3-6x what gold did (2.2x gold
beta plus equity beta) and REITs fell 24% in 2022 on the rate shock; energy
(+97% Jan21-Jun22), ag inputs and base metals were the hedge. Separately, the
book is not the pool: the 0.70 correlation cap admits at most one gold name
per selection, and a 12-month score reaches an inflation regime months late.

A cousin of the VIX overlay, on different terms: not "is volatility high" but
"has the environment turned against owning stocks, and what is working instead."

### Design, agreed 2026-09-23

| question | decision |
|---|---|
| what it protects against | (1) inflation / debasement, (2) deflationary crash. A momentum crash (2009) is a separate problem and out of scope. |
| architecture | **continuous competition, not a regime classifier** ("dual momentum"). Each hedge asset carries its own trend; it takes a slot from the stock book only when it beats the stock book AND beats cash. No inflation forecast; TLT drops out of an inflationary regime on its own. |
| signals | price-only, plus the **stock-bond correlation** — positive in 1946-48, the 1970s and 2022 — as the gate on duration |
| instruments | gold (IAU), silver, broad commodities (PDBC, no K-1), energy basket from the pool, TIPS (VTIP / SCHP), SHY / IEF / TLT, short-lease REIT basket, dollar (UUP). Bitcoin deferred. **The account is tax-advantaged**: collectibles tax and K-1s do not bind. **Compliance**: REIT funds are fine unless data-center focused. |
| size | whole book, not the rotating sleeve; stock sleeves shrink pro-rata; **hedge share capped at 75%** |
| cadence and exits | weekly evaluation with hysteresis (sweep 1/5/10/20 days — daily cost 2.53pp in Track A); each hedge asset exits on its own faster trend break; regime-off returns capital to stocks at the next rotation |
| acceptance bar | **up to 5pp CAGR cost over 2011-2026 is acceptable** if the long-history tests show real protection in 1946-48 and 1973-81. Negotiable after the analysis. Sharpe and the rest reported, no bar set going in. |
| host model | **Track J** (graduating to production over the next several weeks); built as a model-agnostic layer |

### How it is tested — three tiers, each weaker but longer

1. **Tier 3, index level, 1926-2026 (monthly).** Ken French momentum decile as
   the stock book; French industries (Oil, Mines, RlEst), World Bank Pink Sheet
   gold / silver / commodity index (from 1960), Treasuries rebuilt from FRED
   yields, CPI. Covers 1946-48 (yield peg, ~20% CPI, deeply negative real
   rates — the closest US analogue to debt-driven debasement) and 1973-81.
   Calibrates the trigger; cannot validate stock selection.
2. **Tier 2, ETFs, ~2006-2026.** The switch on real instruments; adds 2008,
   when commodities spiked and crashed within months — the exit-speed test.
3. **Tier 1, Track J, 2011-2026.** The real model plus the layer. One
   inflation episode (2021-22), one deflationary crash (2020), and 2015-16 as a
   false-positive check.

No US data contains hyperinflation. USD debasement looks like a more extreme
1970s, and that is the limit of what any tier can say.

### Tier 3 RUN 2026-09-23 — see FINDINGS, Track K Tier 3

Real assets ran hard in every inflationary episode (1973-74: commodities +154%,
stocks -39%); REITs did not. Three design lessons carry into Tiers 2 and 1:

1. **The bare competition is a permanent allocation** (74% of months hedged,
   -8.2pp in 2011-19). It needs a filter.
2. **The filter is decisive hedge strength (`enter_margin`), not stock
   weakness.** A danger gate on the stock book enters 2021-22 five months late;
   an entry margin of +10% over 3 months costs -1.2pp over 2011-26 and keeps
   most of the episode payoff.
3. **Harvest-at-target exits clip the fat tail that makes the episode**
   (1977-81: +50% -> -29% at a +50% target). The relative-strength exit already
   returns the slot to stocks near the stock bottom. Keep `harvest_*` in the
   code as switched-off options; do not carry them forward as defaults.

Next: Tier 2 (ETFs, weekly, 2006-2026) with the entry-margin family as the
candidate, then Tier 1 on Track J's own simulated returns — the question there
is how much of this Track J's score already does by rotating into energy.

### Tier 2 RUN 2026-09-23 — see FINDINGS, Track K Tier 2

The candidate (63 sessions, +10% entry margin, weekly, next-close execution)
halves drawdown for -1.4pp over 2006-26 but costs **-4.8pp over 2011-26**,
inside the bar with little room. The cost is V-shaped rebounds (2020: -46pp).
A fast hand-back to stocks and a trailing-stop harvest were both tried; both
trade crash and inflation protection for rebound participation nearly one for
one, and neither forms a plateau. Over SPY instead of a momentum book the same
layer ADDS +1.3pp and +0.20 Sharpe — so the book decides, and Tier 1 is next.

### Data: what the live decision reads, and how it stays current (built 2026-09-23)

`python scripts/update_market_data.py` every session after 16:15 ET (RUNBOOK,
"Market data store"). Daily store: append-only total returns, overlap-checked,
tracked in git. Long history: dated raw vintages, research only, `--long`
monthly. Nothing in the live decision depends on the long-history sources.

### Tier 1 RUN 2026-09-23 — FAILS THE BAR (see FINDINGS, Track K Tier 1)

On Track J the candidate costs -5.5pp CAGR (phase mean) with Sharpe down and no
drawdown bought; every variant costs 3.8-10.3pp. Track J already rotated into
energy in 2021-22 (+39% while the momentum decile lost 24%), and its returns
mean-revert at 63 sessions (corr -0.22, monotone quintiles), so the layer hedges
just before Track J's recoveries. Judging on SPY instead still costs 5-6pp.

**Decision pending (James).** Options on the table:
- (a) Shelve the layer as a traded mechanism. Keep the store, the layer and the
  three tiers; optionally print the layer's reading in `run_live_trackj.py` as
  information, not a trade.
- (b) Put direct real-asset ETFs (PDBC/DBC, SLV, alongside the IAU already
  there) INTO the Track J pool, so its own score can select them when they
  trend. Tier 3: in 1973-74 commodities and bullion tripled while energy
  equities fell 30% — the direct instrument is what a producer-heavy book lacks.
  Cheap to test: same backtest, pool plus a handful of ETFs.
- (c) A small static real-asset allocation as priced insurance (roughly
  weight x (21% - 7%) a year, e.g. ~1.4pp at 10%). Rejected as a "sleeve" on
  2026-09-23 before the evidence; re-offered because timing has now failed.

### Remaining open items

2. **Does the layer shrink the survivorship exposure? (James, 2026-09-23.)**
   Mechanism: delistings cluster in stress regimes; if the layer is hedged then,
   the book holds fewer stocks exactly when they die. Measurable in principle as
   the book's stock weight on delisting dates, hedged vs not. Two notes:
   - The 0d simulation cannot answer it: its synthetic names are rolled clones
     that do not co-move with the market, so their "deaths" are not timed to
     stress. A variant that draws delisting dates with probability rising in
     market drawdown would test the mechanism directly, with the layer on/off.
   - Survivorship biases the measured hedge value in the CONSERVATIVE direction
     either way: the unhedged backtest understates stress-period losses (the
     dead are missing), so measured protection is understated; and it inflates
     calm-period stock returns, so the measured cost of hedging is overstated.
   - It only helps for slow stresses. The layer was 10% hedged through the
     2020 crash, and it does nothing for idiosyncratic failures in calm markets
     — which is the case a pullback-buying score is most exposed to.

---

## 0g. A sell-side framework — volatility-guided trailing stop

**Added 2026-09-20 at James's request.** Everything in this repo is entry-side:
the model ranks, buys the top N, and holds until the clock says rotate. There
is no exit rule that responds to what a position is doing. At `top_n=8,
hold=42` a name is held six weeks regardless of how it behaves in week two.

Sequenced deliberately **after** the Track J score goes live, not before. It is
a second change, and stacking it on an unproven selection change would make
neither attributable.

### What to build

A trailing stop whose distance scales with the name's own volatility (e.g. k x
ATR or k x rolling sigma), so a 35%-vol name is not stopped by normal noise
while a 15%-vol name is not given 40% of room. Fixed-percentage stops are the
obvious alternative and are known to do badly across a universe with this much
dispersion in volatility — `flip_neg` picks range from utilities to
semiconductors.

### FIRST EVIDENCE, 2026-09-20 — no stop level helps, and the data cannot say otherwise

`scripts/analyze_stop_levels.py`. For every position-day the model held, the
excursion below its peak since entry measured in the name's own daily
volatility, against the return from that point to the end of the hold:

| vols below peak | position-days | return from here | hit rate |
|---|---|---|---|
| -6 to -5 | 1,079 | **+2.38%** | 56% |
| -5 to -4 | 1,514 | **+2.49%** | 58% |
| -4 to -3 | 2,180 | **+2.21%** | 57% |
| -3 to -2.5 | 1,457 | +1.79% | 56% |
| -0.5 to 0 | 10,233 | +1.90% | 57% |

**CORRECTED 2026-09-20, same day.** The table above measures the wrong
quantity for the decision. "Hit rate" was P(positive return from here), which is
not recovery — a position can drift up slightly and never regain its high. On
James's framing the question is *probability of recovery*, and measured properly
there is a strong gradient:

| vols below peak | off peak | P(regains peak) | P(regains entry) | to hold end |
|---|---|---|---|---|
| -0.5 to 0 | -0.2% | **86%** | 97% | +1.90% |
| -1.5 to -1 | -2.9% | 65% | 87% | +1.94% |
| -2.5 to -2 | -5.2% | 46% | 73% | +1.73% |
| -4 to -3 | -7.9% | 28% | 53% | +2.21% |
| -6 to -5 | -12.4% | 14% | 24% | +2.38% |
| < -6 | -16.7% | **5%** | 10% | +2.01% |

**Timing matters as much as depth.** At -3 to -4 vols: 51% recovery early in the
hold, 31% in the middle, **11%** in the last third. At -5 to -6 vols: 42% / 16%
/ **3%**.

**Both a vol-scaled and an absolute threshold carry information, with vols
dominant.** Holding raw percentage fixed, the vol measure still separates 3x (at
-12 to -7% off peak: 13% recovery beyond -4 vols against 43% at -2 to -1 vols).
Holding vols fixed, raw percentage adds a milder gradient. The worst cell is
beyond -4 vols AND more than 20% off peak: **7% recovery** over 662 observations.

**The awkward shape for a stop rule.** A deep position still has a positive
expected return to hold end — it mean-reverts partially without recovering, which
is "losing less" rather than "getting well". So the two readings are both true:
recovery is gone, and selling still forgoes a gain. Worse, the clean stop case is
deep AND late (3-11% recovery, ~+1% remaining), but late in the hold the rotation
is days away anyway, so a stop there saves little by construction. Where a stop
would matter most — deep and early — the forward return is HIGHEST (+4% to +8%).

The original observation below still stands on its own terms:

**Every depth is followed by a positive return, and the deep ones are BETTER
than sitting at the peak.** A stop at -3 vols would have sold 22% of all
position-days and forgone +2.26% each time. Pooled at every threshold from -1.5
to -6 the verdict is "hold".

That is the reversion premise working as designed: for a dip-buying score a deep
excursion is the entry signal arriving late, not a thesis break.

**The caveat is bigger than the result.** This pool contains no names that
failed to recover — it was built in 2026 from survivors, and TODO 0d established
there are no terminal declines in it. A stop hedges catastrophic non-recovery,
and catastrophic non-recovery is the one thing this data structurally cannot
contain. So the finding is narrower than "no stop": *within the recoverable
universe*, no vol multiple identifies a non-recovery point, because there are
none to identify.

Two further limits: position-days overlap heavily so the t-statistics are
inflated (the signs and the monotonicity are what carry), and this conditions on
a state without modelling what the freed slot would buy — a stop could still pay
if the replacement beats +2.3%.

**What would change the answer:** point-in-time data with delisted securities,
which is TODO 0d's real fix. Until then a stop is a judgement about tail risk
this repo cannot measure, not a parameter it can optimise.

### THE DECISION VERSION, 2026-09-20 — holding wins in every state measured

`scripts/analyze_hold_vs_swap.py`. James reframed the question correctly: a
position does not have to recover to be worth keeping, it only has to beat what
the slot would otherwise hold. Conditioned on depth AND days remaining:

**P(holding beats swapping into the best available non-held name)** — never
below 50% in any of 19 cells, ranging 50% to 63%.

**Mean return, holding minus swapping** — positive in every cell, +0.17% to
+4.82%, and LARGEST for deep positions with time left (beyond -4 vols with
16-25 days: +2.99%; -4 to -3 vols with 26+ days: +4.82%).

The deeply underwater positions are the ones most worth keeping.

### Why this contradicts the recovery gradient, and which one to believe

P(regains peak) collapses 86% -> 5% with depth. P(ends above the CURRENT price)
is 53-64% in every cell, essentially flat with depth.

These are consistent, and the first is **arithmetic rather than information**: a
name 6 vols below its peak needs a 6-vol rally to recover, one 1 vol below needs
1 vol. That gradient measures distance to a sunk reference point, not the name's
prospects. The forward distribution from here is roughly independent of how it
got here — which is what a near-random-walk looks like.

**The peak is irrelevant to the decision.** Only the forward comparison is, and
it says hold.

That closes the stop question as originally posed. Any remaining case for an
exit rule has to rest on drawdown reduction being worth paying for, not on
avoiding losses — the data says an exit forgoes gains.

Caveats: the modelled replacement ignores the correlation filter, so it is
BETTER than the real alternative and holding still wins (conservative direction);
position-days overlap heavily so the cells are descriptive; and the pool has no
terminal declines, so the true tail is worse than this can show.

### A re-entry blackout is part of the rule, not a refinement

**Raised by James 2026-09-20.** A stop that sells a name must hold it out for N
periods afterwards. Without that, the score re-ranks the same name the next
rebalance and — because the stop fired on a price *drop*, which is exactly what
`−flip` rewards — the model buys straight back into what just burned it. The
stop and the signal would be working against each other by construction:
`pullback` is a dip-buying score, so a stopped-out name becomes *more*
attractive the moment it is sold.

This makes the blackout load-bearing rather than hygiene. Test N over a short
grid (say 1, 2, 3 rotations) and report:

- how often a stopped name would have been re-bought at the next rebalance
  without the blackout, and what that round trip cost;
- whether the blackout's benefit is the avoided re-entry or just reduced
  turnover — those are different mechanisms and only the first justifies the
  rule;
- N too long starts excluding good names for stale reasons, so expect an
  interior optimum and be suspicious if the best N is the largest tested.

### What to measure, and the standing prior against

Track B tested rank-triggered exits and score-gap swaps and **rejected both**:
losses scaled monotonically with turnover, -1.2pp to -10.5pp. A trailing stop
is a different mechanism — it responds to price rather than to rank — so it is
not the same test, but the burden is the same. Report:

1. CAGR, Sharpe and drawdown against the no-stop baseline at `top_n=8,
   hold=42`.
2. Turnover added, against the 0.17pp-of-CAGR-per-1x bar.
3. How often the stop fires and what the stopped name did afterwards — a stop
   that mostly sells bottoms is worse than no stop.
4. Subperiod stability, as with everything else.

The honest framing: this is aimed at drawdown, and drawdown is not the
selection criterion. It buys the ability to hold the model, which is worth
paying some CAGR for, but the amount should be explicit rather than discovered.

---

## 0e. Drawdown-aware parameterization for the Track J score

**Added 2026-09-20 at James's request.** Drawdown is not his selection
criterion — it is how he judges the fortitude required to hold the model — but
-56% is deeper than anything in this repo's history and is worth attacking
directly rather than accepting as a by-product.

The sweep already shows the lever: drawdown improves monotonically with book
size (`top_n` 2/4/8/16 -> -81%/-70%/-57%/-47% at hold=5) at little CAGR cost
above `top_n=8`. Untested: volatility targeting, a drawdown-triggered de-risk,
or capping the weight of the highest-vol picks. Note Track A's finding that
cutting exposure forfeits the overnight premium, which is the standing prior
against any de-risking rule.

---

## 0f. Would switching between the two scores add anything?

**Added 2026-09-20.** `pullback` and `flip_neg` win in different subperiods,
which invites a rule that picks between them at each rebalance.

**Test the premise before building anything.** Does trailing relative
performance between the two scores predict next-period relative performance? If
that autocorrelation is indistinguishable from zero, switching cannot work and
the item closes for the cost of one short script. The prior against is strong:
Track A rejected VIX switching, daily regime evaluation cost 2.53pp, and
walk-forward found trailing-window selection beat a fixed config 13% of the
time.

Note also that `pullback` is `flip_neg` plus a `range_pos` term, so the two are
correlated by construction and there may be little room between them. A fixed
blend weight is a third model requiring no timing, and the sweep harness
already handles it — try that before any switching machinery.

---

## 0c. Does the Track J signal survive outside the mega-caps it is absent from?

**Added 2026-09-20**, out of FINDINGS Track J. The only ranker this repo has
measured that clears |t| = 2 is `rank(12-1 month) - rank(3-month)` — buy strong
twelve-month momentum that has recently pulled back. It earns **+0.99% per
14-day period at t = 2.66** over a 747-name pool, and **+0.13% at t = 0.43**
restricted to names above $50B, which is the range the live universe occupies.

The edge is real in the measurement and unreachable from the current universe.
That is the whole item.

### What to measure

1. **Fills.** The median pick is a $9.1B name and 10.7% are under $2B. Re-run
   the edge with slippage scaled to spread and ADV instead of a flat 7.5 bps.
   The repo's turnover bar is 0.17pp of CAGR per extra 1x of annual turnover;
   a 4-name book drawn from 650 rotates near 100% per period.
2. **Delisting.** The survivorship bound in Track J removes ex-post *winners*
   and the edge survives. It cannot remove ex-post *losers*, because they are
   not in the pool. This is item 6 again, and it is the binding constraint —
   a dip-buying signal is exactly what a delisting-free pool flatters.
3. **Subperiod.** P1 (2012-2016) is +0.03% at the live hold against +2.49% and
   +1.10% after. Establish whether that is regime or construction before
   anything is built on it.
4. **Portfolio, last.** Only if 1-3 survive. Stage one deliberately stopped
   short of simulating, because a backtest of an unvalidated signal at this
   turnover mostly measures costs.

### What would count as an answer

An edge that holds above $10B with realistic fills, in all three subperiods,
after a delisting-aware pool is available. Anything less is a reason to leave
the universe alone — which is the current state.

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

## 9. Loose threads opened 2026-09-17 and not pursued

Recorded so they are not rediscovered from scratch.

**Scoring horizon is mismatched to where the signal lives.** The IC table found
the composite predicts returns only at **126 days** (+1.38pp top-minus-bottom,
104bps top-4 edge); at 5-63 days it predicts reversal. The composite is built
from RSI(14), 20d relative strength and 5d velocity, with only MA(50/200)
reaching long. Untested: does a long-horizon-weighted composite improve the
top-K edge? The tension to resolve first is that return information sits at 126
days while the risk information the rotation harvests sits near 14, and those
want opposite hold lengths.

**Risk claims rest on max drawdown, which is barely estimable.** It is a
single-episode statistic - one number set by one stretch of one history - so
comparing it across arms has almost no power. Several FINDINGS results turn on
drawdown differences of 2-3pp and are weaker than they read. Re-express them
using measures that aggregate over episodes: downside deviation, Ulcer index,
mean of the worst k drawdowns, 5th percentile of rolling 6-month returns.

**The screen and the thesis point opposite ways.** The Schwab screen filters FOR
five-year total return above 10%; the stated expectation, supported by the
quintile table, is that such names are reversion candidates. Nothing resolves
this. It decides whether names like TSLA should be dropped when their five-year
return decays or held precisely because it has.

> **Partly resolved 2026-09-20 by Track J.** On 747 names, long-run strength and
> recent weakness are *both* the right side of the trade at once:
> `rank(12-1 month) - rank(3-month)` earns +0.99% per 14-day period (t = 2.66),
> and position in the 52-week range predicts positively at all six horizons.
> So the screen's long-horizon filter and the reversion thesis are not in
> conflict — the signal is a recent pullback *inside* an established uptrend,
> not a recovery from a long decline. What stays unresolved is the TSLA case
> specifically, because that is a decayed five-year return, which is the long
> leg going bad rather than a short-leg pullback.

**A candidate mechanical rule, from the TSLA case.** TSLA contributed +0.401 over
2012-2021 and -0.011 over 2022-2026 across 96 position-days. "Drop a name whose
contribution over the trailing N years is below X" would be mechanical, testable,
and appears to encode what judgment did here. It is the first piece of the
operator's discretion that looks convertible into something a script can run
blind, which matters for item 6.

**TSM/NVDA overlap was accepted by hand and never costed.** Both semiconductors,
knowingly correlated. What the duplication costs in effective bets and in
drawdown is unmeasured.

**Universe changes create forced trades the backtest never models.** The
2026-09-17 update orphaned a held position (BR) that is no longer in the
universe. The backtest assumes today's universe always existed, so it prices
none of this. Belongs with item 2.

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

### PARTLY ANSWERED 2026-09-20 — and it is the biggest number in this file

The account's transaction export (2026-01 to 2026-09) was
reconciled against the modelled book by `scripts/reconcile_trades.py`. The
answer to "model or execution" is **neither** — it is that the account and the
model were holding **different books**.

Monthly overlap between the modelled book and actual holdings: **0%, 0%, 17%,
25%, 12%, 25%, 50%, 12%, 62%.** Three of the first four rotations had ZERO names
in common. **+12.80pp of the model's +24.73pp arithmetic YTD came from names
never held at any point** — PWR +5.08% and LMT +3.96% alone are +9.04pp, and
neither appears in the trade history.

Much of the early divergence is model-version drift rather than error: the 2026
scoring fixes mean today's code does not reproduce the book the spring's code
recommended. The convergence toward 62% by September tracks those fixes landing.

**Corrected 2026-09-20:** the early trades were IN model — they followed the
model running at the time, and predate the config snapshots, so there is no
record of what it said. This is version drift, not indiscipline.

**Which makes the +18.04% YTD backtest circular.** Today's config exists because
the old one did badly over exactly this period. Backtesting it over 2026
measures a configuration chosen with knowledge of the outcome; that return was
never available. The same caution applies to Track J, developed against data
through September 2026.

**What this promotes.** A stable model and a recorded book are worth more than
further signal work. +12.80pp of the modelled year came from names never held,
and the snapshots that would have made the early year auditable did not exist
until 2026-07-25. Keep snapshotting, and treat any backtest over a period whose
results drove the config as in-sample.

**Still missing** for a complete answer: a starting balance and the Roth
account, without which the account's true return cannot be computed. Overlap and
missed contribution are the measurable parts; `reconcile_trades.py` refuses to
invent a headline P&L from a transaction list.

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
