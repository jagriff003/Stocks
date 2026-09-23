# Momentum strategy — findings log

Measured 2026-07-25 over 2010-2026, 52-ticker universe, `next_open` execution
with 7.5 bps one-way slippage unless stated. Every number here is reproducible
from a named command; nothing is from memory.

Operational companions: **RUNBOOK.md** (what to do when the health monitor
fires) and **TODO.md** (open work, with what to measure).

Read **deltas, not levels**. Absolute figures carry survivorship bias from
applying today's screened universe backwards. The difference between two
variants over the same universe does not.

---

## Bottom line

The three tracks proposed in the design document were all tested and all
rejected on evidence, as was Track D (rank offset), added 2026-09-03. The
gains that did materialize came from fixing two measurement defects and one
mis-specified parameter, not from new features.

Track E (book size, 2026-09-17) is the first test to *confirm* a live setting
rather than reject a proposed change: `top_n=4` is best on all four headline
metrics and is subperiod-consistent.

| Change | Status | CAGR effect |
|---|---|---|
| Level/velocity scale mismatch | **fixed** | prerequisite — made weights meaningful |
| Execution assumption (same-close → next-open + slippage) | **fixed** | -3.41pp, but honest |
| `velocity_window` 10 → 5 | **adopted** | +6.22pp |
| Correlation filter, gated above VIX 25 | **adopted** | -0.01pp CAGR, +1.32pp drawdown |
| Track A — graduated VIX ladder | **rejected** | -2.8 to -9.8pp |
| Track B — rank exits / score-gap swaps | **rejected** | -1.2 to -10.5pp; best is break-even |
| Track C — acceleration, earlier entry | **rejected** | -0.8 to -9.3pp |
| Track D — rank offset (skip top 1-5) | **rejected** | -2.0 to -9.8pp |
| Track E — book size `top_n` 1..8 | **confirmed 4** | -1.1 to -7.5pp for any other size |
| Track F — null benchmark / ranker IC | **measured** | ranking worth +0.84pp gross; universe carries the rest |
| Track G — wide book + weight overlay | **viable alternative** | -1.6pp CAGR, +0.12 Sharpe, -3.6pp vol, 2.5x trades |
| Track G's risk claim | **retracted** | no decay trend; drawdown gap not resolvable on one history |
| Track H — 100 random baskets | **does not generalize** | model beat its own basket in 2 of 100; edge -8.8pp |
| Model health monitor | **built** | diagnostic, not a return change |
| Track J — trend change, strength, room to run | **hypothesis rejected** | stage one; turn and headroom both wrong-signed |
| Track J inverted — 12-1 momentum minus recent pullback | **lead, not a finding** | +0.99%/14d at t=2.66 on 747 names, but only below $50B cap |
| Track J stage two — portfolio backtest | **partly superseded** | look-ahead screen; see the correction |
| Track J corrected — per-date screen, phase, out-of-sample | **real but noisy** | beats live at 100% of phases; t=1.34 by window, not significant |
| Correlation cap, absolute 0.70 every rebalance | **adopted** | +2.72pp CAGR, Sharpe 0.64->0.78, and t vs live 1.34->2.61 out of sample |
| Live-vs-account reconciliation | **the gap is model-version drift** | overlap averaged 25%; the +18% YTD backtest is circular — today's config was chosen *because* 2026 went badly |
| Track K Tier 3 — hedge layer, 1927-2026 index level | **shape found, not a choice** | entry margin +10%/3m: 0.0pp full, -1.2pp 2011-26, +81% 1973-74, +36% 2021-22; harvest exits and stock-weakness gating both hurt |
| Track K Tier 2 — hedge layer on ETFs, daily, 2006-2026 | **candidate stands, little room** | -1.4pp CAGR, MaxDD -56% -> -30%, **-4.8pp over 2011-26** (bar 5pp); V-shaped rebounds are the cost (2020: -46pp); fast hand-back and trailing stops redistribute rather than fix |
| Track K Tier 1 — hedge layer on Track J, 2012-2026 | **fails the bar** | -5.5pp CAGR (phase mean), Sharpe down, no drawdown bought; Track J rotated into energy itself (+39% in 2021-22) and mean-reverts at 63 sessions, so the layer sells its recoveries |
| Track K option (b) — real-asset ETFs in the Track J pool | **inert** | -0.4pp, 0.5% average weight; the pullback score ranks them in the bottom half, correlation cap irrelevant |

Live model: **19.84% CAGR, 0.91 Sharpe, -19.23% max drawdown, Calmar 1.03**,
net of realistic fills and costs. The honest like-for-like starting point was
12.80%.

**Four standing deductions from that figure**, recorded 2026-09-17, mutually
independent and compounding: the window sits at the **91st percentile** of
available 15.9-year windows (SPY earned 14.02% in it against 8.02% in the median
one); roughly **3.2pp is rotation-phase luck**; survivorship is unbounded (TODO
6); and multiple testing is unaccounted (TODO 7). The phase-averaged figure for
this configuration is 16.42%, before the other three.

Read the levels in this document as inflated by a large and only partly
quantified amount, and the differences between arms as sound. See "The backtest
window is in the top decile of available history".

**Read that next to Track F.** Owning the same universe equal-weighted, trading
almost never, earns 19.99% at a 0.92 Sharpe — the model's return advantage over
its own universe is negative. What the model delivers is drawdown: -19.23%
against -35.22%, Calmar 1.02 against 0.57. It is a risk-management overlay on a
universe, not a stock picker, and it should be understood and defended as one.

---

## The unifying result

Five independent experiments converged on the same conclusion from different
directions: **this strategy is hurt by trading more, and no tested signal
justifies the extra turnover.**

- Absolute-VIX de-risking: worse than no overlay at all.
- Daily regime evaluation: -2.53pp for +0.14pp of drawdown.
- Rank-triggered exits: losses scale monotonically with turnover, -1.2pp to -10.5pp.
- Score-gap swaps: performance improves as the rule trades *less*, converging on the baseline.
- Acceleration weighting: costs rise monotonically with the weight.

Two mechanisms explain nearly all of it.

**1. The overnight risk premium.** Decomposing the universe's returns:

| Segment | Ann. vol | Ann. drift |
|---|---|---|
| Overnight (close→open) | 18.09% | **12.04%** |
| Intraday (open→close) | 23.90% | 7.13% |
| Total | 29.96% | 19.13% |

Overnight carries **36% of the variance but 63% of the return**. Any rule that
reduces exposure surrenders 63% of the return stream to avoid 36% of the
variance. That trade is bad at every VIX level tested, which is why the ladder
failed at every band placement.

**2. Turnover is expensive here.** At 16.9x annual portfolio turnover, 7.5 bps
one-way costs **2.90pp of CAGR** — 18% of gross return. The break-even bar for
any new rule is roughly **0.17pp of CAGR per extra 1x of annual turnover**.

---

## Track A — graduated VIX ladder (rejected)

`run_experiments.py ladder | ladder2 | zladder | overnight`

| Config | CAGR | Sharpe | MaxDD |
|---|---|---|---|
| legacy z-score regime (current) | **19.84%** | **0.91** | **-19.23%** |
| no overlay at all | 16.96% | 0.68 | -29.02% |
| ladder `[4,4,4,0]` at 15/20/25 | 14.01% | 0.60 | -25.57% |
| ladder `[4,3,2,0]` at 30/35/40 | 17.02% | 0.72 | -28.50% |

Three separable ideas, three separate failures:

**Absolute-level bands.** Structural, not a tuning problem. Absolute VIX
crossing a threshold *is* the drawdown — it is coincident-to-lagging, so acting
on it sells into the loss and misses the recovery. Pushing bands out to 30/35/40
recovers the return (17.02% vs 16.96% for no overlay) but essentially none of
the drawdown protection (-28.50% vs -29.02%), while the z-score regime delivers
-19.23%. No threshold fixes it.

**The band table was misread.** VIX 15-20 has Sharpe 0.88 but **+19.1%
annualized return**, and covers 65% of days above 15. It describes conditional
*market* returns, not the value of de-risking. Going defensive there swaps a
19% return for roughly 2%.

**Daily evaluation.** Tested on a z-score basis where the signal works:
20.05% → 17.52% CAGR for +0.14pp of drawdown, +36 trades/year. The 14-day clock
was acting as an unintentional noise filter.

This reframes the "only 25% of crisis-flagged days held defensive positions"
observation. That is not a defect — it is the strategy staying invested through
transient flags, which the data says was correct.

---

## Track B — sell-signal decoupling (rejected)

`run_experiments.py exits | swaps`, `analyze_rank_decay.py`, `analyze_swap_quality.py`

**The frequency evidence replicates.** 75.4% of holdings drop out of the top 4
before their hold ends (prior: 68%), median day 4 (prior: 3), **17.8% recover
(prior: 17%)**, 911 triggered instances (prior: 999).

**The return evidence does not support the premise.** From the moment an exit
would trigger:

| | |
|---|---|
| Mean return after trigger | **+0.39%** |
| Median | **+0.00%** |
| Share negative | 43.9% |
| p10 / p50 / p90 | -2.97% / 0.00% / +3.95% |

Rank is *relative*. A holding slides from 4th to 6th because others rose, not
because it fell, so rank decay carries no directional information about the
name. "Sells too late, after returns have soured" is not in the price data.

**Reframing to "sell when there's a better buy available" is worth ~8pp** over
the rank trigger, but converges on not trading:

| Config | CAGR | Sharpe | Trades/yr |
|---|---|---|---|
| no swap | 19.84% | 0.91 | 127.9 |
| gap 1.0z, max 2/month | 19.92% | 0.92 | 137.4 |
| gap 0.5z | 17.14% | 0.76 | 163.0 |
| gap 0.0 (control) | 10.77% | 0.38 | 243.2 |

Performance rises monotonically as the gap widens — the better the rule, the
less it trades.

**Why: the signal arrives too late.** Across 216 swaps:

| Horizon | Incoming | Outgoing | Edge | Win rate | t |
|---|---|---|---|---|---|
| 5d | 0.46% | 0.63% | -0.17% | 45.6% | -0.43 |
| 10d | 0.74% | 0.70% | +0.04% | 40.6% | 0.08 |
| 21d | 1.65% | 1.34% | +0.31% | 52.8% | 0.36 |

The challenger is statistically indistinguishable from the name it displaces.

---

## Track C — buy-signal timing (rejected)

`run_experiments.py entryscore`, `analyze_entry_timing.py`

**Earlier entry, with false positives counted.** Signal: rank ≤ 8 for 3
consecutive sessions. Precision 53.4% against a 49.5% naive baseline.

| | 21-session return |
|---|---|
| Early signal → winners only | +3.26% |
| Early signal → stalled (46.6%) | -0.69% |
| **Early signal → all triggers** | **+1.42%** |
| Buying on actual arrival | **+1.63%** |

Earlier entry is **-0.21% worse**. Measuring winners alone would have shown it
doubling returns — exactly the hindsight trap the design document flagged.

**Acceleration carries noise, not timing.** The derivative-window test is the
informative one:

| Config | CAGR | Sharpe | MaxDD |
|---|---|---|---|
| baseline | **19.84%** | **0.91** | **-19.23%** |
| accel 0.3, 3-session | 12.65% | 0.47 | **-41.95%** |
| accel 0.3, 10-session | 14.47% | 0.57 | -24.93% |
| accel 0.1 | 15.25% | 0.61 | -22.29% |
| all_four (weighted AND-rule) | 14.62% | 0.59 | -23.40% |

Differencing twice amplifies noise; a short window amplifies it
catastrophically. The AND-of-four rule looked selective at 3.2% of stock-days
because it was firing on rare noise, not because it discriminated.

Also settled and needing no build: **inverting the level/velocity weights is
wrong** (every L30/V70 config underperforms level-only at 35-47 more trades a
year), and **ranking on pure z-score rate of change** — Track C #2 — scores
10.1-12.8% across all windows.

---

## Track D — rank offset, "skip the peaked leader" (rejected)

`run_experiments.py offset`, `rsi_ma_rank_offset.csv`

The premise: the top-ranked name has already made its move, so a book of ranks
2-5 buys names still on the way up rather than the one about to give it back.
Book size held at `top_n=4` throughout, so nothing here is confounded with a
position-sizing change.

| Held ranks | CAGR | Sharpe | MaxDD | Calmar | Trd/Yr |
|---|---|---|---|---|---|
| **1-4 (production)** | **19.80%** | **0.91** | **-19.23%** | **1.03** | **128.0** |
| 2-5 | 17.77% | 0.80 | -26.88% | 0.66 | 142.3 |
| 3-6 | 11.54% | 0.43 | -27.16% | 0.43 | 150.7 |
| 4-7 | 13.15% | 0.52 | -26.06% | 0.50 | 158.7 |
| 5-8 | 10.25% | 0.34 | -29.02% | 0.35 | 163.1 |
| 6-9 | 10.01% | 0.32 | -26.05% | 0.38 | 165.7 |

Baseline wins every headline metric and leads in all three subperiods (14.71% /
19.68% / 25.25%). Offsets of 2 and above beat the median config in at most one
segment. The gradient is smooth apart from the 3-6 / 4-7 inversion, which is
noise in a 52-name universe.

Three specifics worth keeping, because each rules out a different rescue:

- **The offset does not buy risk reduction.** Drawdown gets *worse* by 6.8-9.8pp
  at every offset. There is no risk-adjusted reading under which skipping the
  leader pays; it is not a return-for-safety trade.
- **It costs more to run.** Trades/year rises monotonically, 128 → 166, and
  annual turnover 1631% → 2110%. Deeper ranks are less persistent, so the
  skipped book churns harder — the offset pays more slippage for less return.
- **Offset 1 is the only near-miss, and it is not close enough.** -2.03% CAGR
  and -7.65pp drawdown. It survives the subperiod screen, which means the top
  rank's edge is concentrated rather than universal, but it loses on every
  metric in every segment.

**Scope made no difference.** Standing the offset down under the VIX overlay
(`rank_offset_scope='normal'`) tracks the all-regimes variant within 0.5pp of
CAGR at every offset — 17.52% vs 17.77% at offset 1, 11.99% vs 11.54% at
offset 2. The elevated/crisis books are too small a share of days to matter, so
the simpler scope (`'all'`) is the one to keep.

**What this measures.** The offset is a claim about the composite's behaviour at
its own top end — that the score turns anti-predictive in its highest band while
staying predictive just below. The monotone decay says the opposite: rank 1
carries the *most* information of any rank, and the score is well-ordered right
through the top. That is the same conclusion Tracks B and C reached from the
other direction — the composite's ranking is sound, and what it lacks is
timeliness, not ordering.

The parameter stays in `ModelConfig` at `rank_offset=0`, which is a no-op, so the
result is reproducible without re-deriving the machinery.

---

## Track E — book size (`top_n` 1..8): the live value is already right

**Run 2026-09-17.** `python scripts/analyze_book_size.py`. First sweep of book
size in this framework — `top_n=4` had been held fixed through Tracks A-D, so
nothing previously recorded was confounded with it, and nothing previously
recorded tested it either.

The volatility overlay was held fixed, not scaled: `vix.elevated_top_n` stays at
2 for every book size, so an elevated-VIX regime always keeps the same *number*
of momentum names and the defensive sleeve absorbs the rest. The defensive
*share* therefore varies by construction, and cannot not: only three defensive
tickers exist (SHY/TLT/IAU), so the fill is hard-capped at three names and a
book of 8 can be at most 38% defensive. The realized exposure is reported
alongside, and it is milder than that framing suggests — the mean defensive
weight sits in a 11-15% band across the whole sweep, while the share of *days*
touching the sleeve rises 11% → 45%. The overlay spreads over more days as the
book grows; it does not get heavier.

| top_n | CAGR | Sharpe | MaxDD | Calmar | Trades/yr | Turnover | Mean book |
|---|---|---|---|---|---|---|---|
| 1 | 12.12% | 0.27 | -60.92% | 0.20 | 43.6 | 2106% | 1.04 |
| 2 | 18.27% | 0.65 | -37.49% | 0.49 | 74.6 | 1867% | 2.00 |
| 3 | 17.35% | 0.70 | -30.96% | 0.56 | 104.9 | 1773% | 2.96 |
| **4** | **19.58%** | **0.90** | **-19.23%** | **1.02** | 127.8 | 1630% | 3.91 |
| 5 | 18.47% | 0.88 | -21.72% | 0.85 | 149.5 | 1538% | 4.85 |
| 6 | 15.89% | 0.75 | -22.35% | 0.71 | 170.7 | 1487% | 5.74 |
| 7 | 15.96% | 0.77 | -20.78% | 0.77 | 186.9 | 1409% | 6.62 |
| 8 | 15.84% | 0.78 | -20.14% | 0.79 | 204.8 | 1363% | 7.51 |

`top_n=4` is best on CAGR, Sharpe, MaxDD and Calmar simultaneously, and is one
of only two sizes (with 3) beating the median in all three subperiods. **No
change is indicated.**

**Read the margin honestly.** The sweep is not monotone — 2 beats 3 by 0.92pp,
7 beats 6 by 0.07pp — which puts the noise band at roughly 1-2pp of CAGR. On
CAGR alone, 3, 4 and 5 are one plateau and the win is not significant. What
separates 4 is not its CAGR but that the *drawdown* result is monotone and
large: -60.9% → -37.5% → -31.0% → -19.2%, then flat. Diversification buys
drawdown up to four names and stops paying after. That is the durable part of
this result; the CAGR ranking within 3-5 is not.

**The cross-check that matters.** Re-run with `--scale-elevated`, which
preserves the live 2-of-4 elevated ratio instead of holding the count fixed,
`top_n=4` still wins (19.58%), 5 is still second (18.47%), and 3 and 4 are still
the only subperiod-consistent sizes. The verdict does not depend on how the
overlay is scaled, which is the confound this sweep could not design away.
CRISIS is invariant to `top_n` by construction (it holds `crisis_symbols`
outright), so no part of the sweep moves the crisis book.

**On minimizing trades.** Fewer names is genuinely cheaper — `top_n=2` trades
74.6 times a year against 127.8, and has the best CAGR-per-trade of any size
above 1. It is still the wrong trade: 1.31pp of CAGR is the small half of the
cost, and the large half is doubling max drawdown to -37.5%. Trade count falls
with book size but *turnover* rises, because a smaller book replaces a larger
fraction of itself on each rotation. Cutting to 2 does not buy a quieter
strategy; it buys a louder one that trades less often.

**`top_n=1` is not a strategy.** -60.9% drawdown, 0.27 Sharpe, and a first
subperiod at -3.68%. Recorded so it does not get proposed again.

Defect found and fixed in the course of this: the ELEVATED branch computed
`n_momentum = min(elevated_top_n, len(valid_stocks))` without clamping to
`top_n`, so any book smaller than `elevated_top_n` would *grow* in an elevated
regime — `top_n=1` held two names in the regime whose purpose is cutting
exposure. A no-op at the live 2-of-4 setting (parity test still passes to
floating point), latent only because the sweep had never gone below 2.

---

## Outsized single-stock events supply 8-15% of the gain, not most of it

**Run 2026-09-17.** `python scripts/analyze_outsized.py`. The question: is the
record a handful of earnings surprises wearing a strategy's clothes?

Answer: no. Two independent readings agree, and neither depends on a threshold
chosen after seeing the answer.

The decomposition is exact. Every position-day's contribution to the portfolio
return is derived and reconciled against `simulate_portfolio`'s own return
series to **2.7e-16** before anything is reported; the script refuses to print
if it does not reconcile. (It caught a real bug doing so:
`PortfolioResult.holdings` is the book *realized* on each return date, which is
the target series lagged a day, and feeding it the wrong one shifted every
contribution by a session.)

**1. The tails are near-symmetric.** Over 15,821 position-days:

| Tail depth | Days | Top-tail P&L | Bottom-tail P&L | Ratio |
|---|---|---|---|---|
| 0.1% | 16 | +0.56 | -0.49 | 1.13 |
| 1.0% | 158 | +2.93 | -2.57 | 1.14 |
| 5.0% | 791 | +8.07 | -7.32 | 1.10 |
| 10.0% | 1,582 | +12.02 | -10.96 | 1.10 |

A lottery-dependent strategy has a top tail much fatter than its bottom tail.
This one runs 1.10-1.16 at every depth. The top 1% of position-days supply 13.2%
of gross gains — against a 1% share of days, concentrated, but nowhere near
load-bearing, and very nearly cancelled by the matching bottom 1%.

**2. The trim curve.** Zero out the K most extreme position-days and recompute
(the slot is still held; only that name's move is removed):

| K | % of days | top only | bottom only | **both tails** |
|---|---|---|---|---|
| 0 | — | 19.58% | 19.58% | 19.58% |
| 10 | 0.06% | 16.61% | 22.59% | 18.54% |
| 50 | 0.32% | 9.39% | 29.37% | 15.73% |
| 100 | 0.63% | 3.41% | 36.25% | 14.90% |
| 250 | 1.58% | -8.44% | 52.69% | **14.49%** |

The first column is the alarming one and it is also the meaningless one: delete
the best days of *any* equity strategy and it dies. The third column is the
answer. Removing the 250 largest moves in **both** directions — 1.58% of all
position-days, the entire fat tail — leaves 14.49% CAGR and a 0.74 Sharpe
against 0.90. **Roughly three quarters of the compounding survives the complete
removal of the tail.** The remaining 5.09pp is what genuine positive skew is
worth here, which is real but is not the strategy.

**3. Event tagging, and it is threshold-robust.** Tagging a position-day when
the stock's move net of trailing beta to SPY exceeds σ trailing residual
standard deviations (beta and σ estimated on a window ending the day *before*
the move, so an event never calibrates its own yardstick):

| σ | Tagged days | % of days | Upside P&L | Downside P&L | **Net share** |
|---|---|---|---|---|---|
| 2.5 | 404 | 2.55% | +2.23 | -1.76 | **14.7%** |
| 3.0 | 217 | 1.37% | +1.55 | -1.09 | **14.4%** |
| 4.0 | 89 | 0.56% | +0.85 | -0.61 | **7.6%** |
| 5.0 | 42 | 0.27% | +0.61 | -0.29 | **10.0%** |

The headline number does not move with the threshold: outsized events are worth
**8-15% of net P&L** wherever the line is drawn, because the upside and downside
surprises largely offset. The other 85-92% is the ordinary grind of 15,000-odd
unremarkable position-days each earning ~0.0002.

**The mechanism intuition was right; the magnitude was not.** The extreme tail
*is* news-driven, and increasingly so the further out you go: overnight gap
accounts for a median 37% of the absolute move on untagged days, 49% at σ≥4, and
**72% at σ≥5**. Gap dominance is the earnings signature (no earnings calendar
exists in the repo — this is a proxy, not a lookup). So the biggest events are
indeed earnings and news. They are simply not carrying the return.

**No single-name dependency either.** 51 tickers were held; the top 8 supply
half the summed contribution and no ticker exceeds 9.3% (TSLA 9.3%, NVDA 9.1%,
STX 6.6%). The largest single contribution in sixteen years is META on
2023-02-02, +23.3% on a +19.8% gap, worth 5.82% of that day's book — which the
trim curve prices at 0.47pp of lifetime CAGR.

**The standing caveat applies and cuts against comfort.** These are today's
screened tickers applied backwards; survivorship bias means the real historical
tail was worse than this, in both directions.

---

## Track F — the missing control: what is the ranker actually worth?

**Run 2026-09-17.** `scripts/analyze_null_benchmark.py` (500 trials per random
arm) and `scripts/analyze_ranker_ic.py`. The first benchmark in this repo's
history against anything other than SPY.

The finding, stated plainly: **the stock picking contributes approximately
nothing to return. What the model does is manage drawdown on a universe that is
carrying all of the return.**

### The reference rows

| | CAGR | Sharpe | MaxDD | Calmar | Vol | Trades/yr |
|---|---|---|---|---|---|---|
| SPY buy and hold | 13.81% | 0.54 | -33.72% | 0.41 | 17.10% | 0 |
| **Equal-weight universe** | **19.99%** | **0.92** | -35.22% | 0.57 | 16.79% | 3.2 |
| **The live model** | 19.58% | 0.90 | **-19.23%** | **1.02** | 16.82% | 127.8 |

Owning all 49 momentum names equal-weighted, rebalanced never, earns **more**
than the model (19.99% vs 19.58%) at a marginally better Sharpe (0.92 vs 0.90)
and near-identical volatility (16.79% vs 16.82%).

What the model buys is the drawdown: **-19.23% against -35.22%**, which nearly
doubles Calmar (1.02 vs 0.57). Note the shape of that — the two have the *same*
dispersion and very different peak-to-trough. The model is not a lower-risk
portfolio in the variance sense; it specifically avoids the deep holes.

The SPY bar is cleared by 5.77pp and the reason to run this rather than index
stands. But essentially all of that margin is **universe selection**, not
ranking.

### The four arms

| Arm | CAGR | Gross | Sharpe | MaxDD | Turnover |
|---|---|---|---|---|---|
| ranked_overlay (live) | 19.58% | 22.53% | 0.90 | -19.23% | 1630% |
| random_overlay | 13.40% ±3.34 | 17.48% | 0.52 | -30.69% | 2359% |
| ranked_plain | 16.73% | 19.50% | 0.67 | -29.02% | 1562% |
| random_plain | 14.46% ±3.67 | 18.66% | 0.52 | -37.78% | 2403% |

Where the live model sits in the null distribution:

| Comparison | CAGR | **Gross CAGR** | Sharpe |
|---|---|---|---|
| ranked_overlay vs random_overlay | 96.0th | 91.8th | 96.8th |
| ranked_plain vs random_plain | 74.2nd | **60.8th** | 80.0th |

**Read the gross column of the second row.** Stripped of the overlay and of the
slippage advantage that comes from persistence, the ranker sits at the **61st
percentile of random draws from its own universe**. That is noise. It is exactly
what the IC predicts.

### What each component is worth

| Component | net | gross |
|---|---|---|
| Ranking, overlay on | +6.18% | +5.06% |
| **Ranking, overlay off** | +2.27% | **+0.84%** |
| Overlay, on ranked picks | +2.84% | **+3.03%** |
| **Overlay, on random picks** | -1.07% | **-1.18%** |

Two things to take from this table.

**The ranking is worth +0.84pp gross on its own.** The net figure of +2.27pp is
real money but most of it is not skill — an iid draw has no persistence and
rotates ~100% every cycle (2403% turnover against 1562%), so the random arm is
being taxed for being random. Gross removes that and little survives.

**The overlay helps ranked picks and HURTS random ones** (+3.03pp against
-1.18pp gross). That interaction is the most informative number in the study,
and it has a mechanism. In an elevated regime the overlay cuts the book to 2
momentum names plus defensive fill. The top-K analysis shows the ranker's edge
is concentrated at exactly K=1-2 and gone by K=8 (+0.16% / +0.07% / +0.05% per
14-day period at K=1 / 2 / 4). So the overlay works by *forcing concentration
into the only part of the ranking that carries information*. Applied to random
picks the same rule just concentrates noise and forfeits the overnight premium,
and it loses. Neither piece is worth much alone; the combination is at the 96th
percentile.

### The information coefficient: zero at every horizon

Cross-sectional Spearman(score, forward return) on the eligible pool only,
open-to-open to match the fill, with a t-statistic computed on non-overlapping
dates:

| Horizon | Mean IC | t | t (naive) |
|---|---|---|---|
| 5d | 0.0002 | 0.17 | 0.06 |
| **14d (live)** | 0.0026 | 0.36 | 0.77 |
| 21d | -0.0009 | -0.21 | -0.26 |
| 42d | -0.0057 | -0.53 | -1.72 |
| 63d | -0.0137 | -0.53 | **-4.11** |
| 126d | 0.0018 | -0.75 | 0.54 |

No horizon clears |t| = 2. The quintile ladder is flat, and at most horizons the
*worst* bin has the highest mean forward return. The level-only score scores the
same as the velocity-blended one, which is worth holding against
`velocity_window=5`'s reported +6.22pp — that gain is not showing up as
prediction.

Note the 63-day row: the naive t is -4.11 and the honest one is -0.53. Without
the overlap correction this table would report a significant negative IC that
does not exist. Same trap as the health monitor's 126-day windows.

**The subperiod IC decline is not real.** P1 +0.0175, P2 +0.0018, P3 -0.0057
looks like decay; the difference is t = 0.77, p = 0.44. More to the point, the
smallest decline this test could detect at 80% power is **0.084**, roughly
eighteen times the full-record mean IC of 0.0045. The test cannot detect a
decline of any plausible size, so "stable" is as unsupported as "declining". The
only defensible statement is that the IC is indistinguishable from zero
throughout — and you cannot decay from zero. Recorded because the apparent trend
will otherwise get re-discovered and mistaken for degradation.

### Why the ranker looks dead on IC but the top-K cut shows an edge

They are consistent. A quintile is the top 20% of ~41 eligible names, about
eight; the model buys four. The edge lives at ranks 1-2 and is averaged away by
the time you pool eight names, and completely invisible in a full cross-sectional
Spearman dominated by the middle of the distribution.

Hit rate at the live hold is **lower** for the picks than for the pool (57.5%
against 57.9%) at every K. The picks win less often and earn more when they win.
Whatever edge exists is skew, not accuracy.

**This explains Track E.** The ranker's edge is concentrated at rank 1 and is
roughly +0.16% per 14-day period there. Harvesting it undiluted is what
`top_n=1` does, and `top_n=1` returns 12.12% at -60.92% drawdown: the edge is far
too small to survive the volatility of concentrating into it. `top_n=4` is the
point where enough top-rank signal survives to matter while the book still
compounds. Two results that looked unrelated are one fact.

### What this changes

**Nothing about the live model today.** 19.58% at -19.23% is worth having
however it arises, and the drawdown advantage over owning the universe outright
is large, real, and exactly what the RUNBOOK is written to protect.

What it changes is where effort goes:

1. **The ranker's parameters deserve no further attention.** Two independent
   measurements say it carries no cross-sectional information. Tuning it further
   is fitting noise, which the walk-forward result already warned about from a
   different direction.
2. **The overlay deserves more credit than FINDINGS gives it.** It is recorded
   above mostly through Track A's *rejection* of a graduated version. On this
   evidence the binary overlay is the single largest deliberate contributor:
   +3.03pp gross and the entire drawdown advantage.
3. **Survivorship moves from footnote to central question.** The universe is now
   known to carry the return, and the universe figure is the one most
   contaminated by applying today's screen backwards. "Own the universe" earning
   19.99% is partly a statement about 2010-2026 and partly an artifact of
   choosing the names in 2026. Until that is bounded (TODO item 6) the honest
   headline is unknown. This is now the most valuable open item in the repo.

---

## Track G — the wide-book redesign: it works, but not for the reason it was tried

**Run 2026-09-17.** `scripts/analyze_wide_book.py`, on the new weight-aware
simulator `momentum/drift.py`. Motivated by Track F: if the universe produces
the return and the ranker does not, hold more of the universe and stop paying
attention to the ranking.

Three results, of which the second is the one that matters and the third is the
one that kills the motivation.

### 1. The slot-based overlay does not survive a wide book — it inverts

| top_n | overlay on | overlay OFF | overlay worth |
|---|---|---|---|
| 4 | 19.52% / -19.26% | 16.75% / -28.93% | **+2.77pp** |
| 10 | 14.99% / -24.42% | 14.92% / -22.95% | +0.07pp |
| 20 | 15.66% / -22.35% | 16.81% / -28.96% | -1.15pp |
| 30 | 15.69% / -19.71% | 18.63% / -28.49% | -2.94pp |
| 40 | 15.24% / -21.45% | 18.74% / -31.90% | **-3.50pp** |

The overlay flips from worth +2.77pp to costing -3.50pp, crossing over around
ten names. The mechanism is worse than "it stops helping". In an elevated regime
the rule keeps `elevated_top_n`=2 momentum names and fills the rest from a
three-ticker sleeve, so the book is capped at **five names total regardless of
`top_n`**. A 30-name book does not get de-risked; it gets liquidated into five
positions. It still controls drawdown (-19.71%), by a mechanism nobody would
choose on purpose.

### 2. Expressed as a WEIGHT rather than a slot count, the overlay scales

`drift.defensive_weight_targets` holds the sleeve at a target share of the book
instead of a count of slots, so three tickers can carry 30% of a 30-name book.

| Variant | CAGR | Sharpe | MaxDD | Calmar | Vol | Trades/yr |
|---|---|---|---|---|---|---|
| **Live model (top_n=4)** | **19.52%** | 0.89 | **-19.26%** | **1.01** | 16.80% | **128** |
| top_n=30, def wt 0% | 18.63% | 0.98 | -28.49% | 0.65 | 14.46% | 316 |
| top_n=30, def wt 15/30% | 18.48% | 1.00 | -26.10% | 0.71 | 13.92% | 316 |
| **top_n=30, def wt 30/60%** | 17.90% | **1.01** | -21.55% | 0.83 | **13.23%** | 316 |
| top_n=30, def wt 50/100% | 17.05% | 0.98 | **-18.52%** | 0.92 | 12.86% | 316 |
| top_n=40, def wt 30/60% | 17.85% | **1.01** | -20.99% | 0.85 | 13.25% | 242 |

A wide book with a 30%/60% defensive weight gives up **1.6pp of CAGR** and buys
**a materially better Sharpe (1.01 against 0.89) at 3.6pp less volatility**, with
drawdown within 2.3pp of the live model. At 50%/100% it beats the live model's
drawdown outright (-18.52%) for 2.5pp of CAGR. This is a real, defensible
alternative and the first thing tested in this repo that improves risk-adjusted
return.

**Any move to a wide book requires this change.** Leaving the slot-based overlay
in place while widening the book is the worst of the options tested.

### 3. It is MORE maintenance, not less — which was the whole point

**Trades per year roughly triples: 128 at `top_n`=4 against 316 at `top_n`=30.**
More names means more of them change on each rotation. The wide book was
motivated by reducing the work of maintaining a book through biweekly selection,
and it does the opposite.

The obvious fix — rotate less often — does not work:

| Hold | CAGR | MaxDD | Sharpe | Trades/yr |
|---|---|---|---|---|
| **14d** | **17.90%** | **-21.55%** | **1.01** | 316 |
| 30d | 16.38% | -28.82% | 0.86 | 194 |
| 63d | 16.85% | -34.89% | 0.82 | 129 |
| 126d | 16.35% | -35.68% | 0.78 | 83 |
| 252d | 16.00% | -30.37% | 0.81 | 39 |

Cutting rotation to quarterly halves the trade count and costs **13pp of
drawdown**. There is a genuine trilemma here: few names + few trades + good
drawdown (the current model), or many names + good Sharpe + many trades. Many
names + few trades + good drawdown is not on the menu.

**Rebalancing, separately, is a non-question.** `on_rotation`, `never`,
`periodic:63d` and `band:25%` differ by less than 0.2pp at every hold length and
every book size. Weight drift within a hold simply does not matter here, and the
legacy convention's free daily rebalance is worth only **0.07pp gross**. TODO
item 1b is answered and can be closed: the equal-weight-reset approximation is
real but negligible.

### 4. Why rotation controls drawdown when the IC is zero

Rotation frequency drives drawdown hard (-21.55% at 14 days against -34.89% at
63) even though Track F found no cross-sectional forward information at any
horizon. Two candidate mechanisms, one tested and rejected, one supported:

**Not the level floor.** Disabling `min_level_threshold` entirely (-3.0 -> -99)
moves the result by 0.24pp at 14 days and not at all at 63. The freefall floor
is not what is doing this.

**It is the ranking, and its information decays fast.** Ranked against random
selection, same book size, same overlay:

| Selection | Hold | CAGR | MaxDD | Sharpe |
|---|---|---|---|---|
| ranked | 14d | 17.90% | **-21.55%** | 1.01 |
| random | 14d | 16.51% | -24.45% | 0.88 |
| ranked | 63d | 16.85% | **-34.89%** | 0.82 |
| random | 63d | 18.33% | -27.35% | 0.99 |

Read the two 63-day rows. **A stale ranking is worse than no ranking at all** —
ranked selection held 63 days gives up 7.5pp of drawdown to random selection
held the same period. Meanwhile random selection *improves* with longer holds
(less turnover cost) exactly as expected, while ranked selection deteriorates.

The composite therefore carries information that is real, short-lived, and about
**risk rather than return**. Refreshed every 14 days it is worth ~3pp of
drawdown and 0.13 of Sharpe against random; left for 63 days it becomes actively
harmful, because the book is then concentrated in what *was* trending under
conditions that have since changed.

**This is a scope correction on Track F, not a contradiction of it.** The IC
test measured cross-sectional *return* prediction and correctly found none. It
was never capable of detecting risk information, and the model's value was
mis-stated as "drawdown control from the overlay" when it is drawdown control
from the overlay **and** from the rotation. That also retires the standing
puzzle about why Tracks A-D all failed: the rotation is not buying alpha to be
improved on, it is refreshing a risk posture that decays. Every track that tried
to trade the signal *more cleverly* was operating on the wrong theory of what
the signal is.

### What this changes

Nothing today. The live model remains the best CAGR-and-drawdown combination
tested, and the wide-book variant is a different risk profile rather than a
strict improvement — better Sharpe and lower volatility, worse Calmar, and three
times the trading.

The decision it sets up is a preference, not an optimization: **19.52% at 16.8%
volatility and 128 trades a year, or 17.90% at 13.2% volatility and 316.** That
choice should be made on capital and temperament, and revisited as the account
grows, since concentration risk in four names scales with the balance while the
wide book's trade count does not.

---

## Rotation phase is worth ~3pp of the headline CAGR

**Found 2026-09-17**, while building the signal-decay test, which the effect
invalidated on its first run.

`hold_days` does not only set how long a book is held.  It sets the entire
schedule of dates on which rebalances happen, and that schedule matters on its
own.  Holding the configuration completely fixed at `hold_days=14` and shifting
only the rotation PHASE - walking `min_data_days` from 200 to 213, which moves
the start of the walk by one session at a time and changes the eligibility gate
by at most 13 days out of 200 - produces:

| hold_days | phases | mean CAGR | sd | min | max | range |
|---|---|---|---|---|---|---|
| 10 | 14 | 12.82% | 1.70% | 11.13% | 15.16% | 4.04pp |
| **14** | 14 | **16.42%** | **2.68%** | 12.84% | **19.78%** | **6.94pp** |
| 18 | 14 | 13.78% | 0.78% | 12.96% | 14.88% | 1.93pp |

**The live configuration sits at 19.64%, against a phase-averaged 16.42% for the
same configuration.** It is within 0.14pp of the best of fourteen phases. Nothing
was tuned to achieve that - the phase is an accident of when the record starts -
but it means roughly **3.2pp of the headline is the particular calendar history
handed us**, not the model.

### What this does and does not invalidate

**Comparisons that share a rotation schedule are unaffected.** Track E (book
size), Track F (null benchmark and IC) and Track G sweeps 1 and 2 all held
`hold_days=14` and `min_data_days=200` across every arm, so every configuration
compared sat on the identical schedule and the differences between them are
clean. The null benchmark's percentiles are likewise unaffected: the random arms
ran on the same phase as the ranked ones.

**Comparisons that vary the schedule are confounded.** Track G sweep 3 swept
`hold_days` 14 to 252 and reported a sharp drawdown penalty for longer holds.
Part of that is real and part is phase. It should be re-read as suggestive only
until re-run phase-averaged.

**The absolute headline is inflated.** Every CAGR quoted anywhere in this file
carries this, on top of the survivorship premium that TODO item 6 exists to
bound. The two are independent and they compound. A forward-looking expectation
for this configuration is closer to **16.4% before survivorship** than to 19.6%,
and lower again after.

### Why this was not caught earlier

Because `hold_days` was never swept in this framework. `run_date_sensitivity`
asks a neighbouring but different question - it holds the fill schedule FIXED
and varies only the information date, deliberately, to isolate data staleness.
Nothing measured what happens when the schedule itself moves.

The general lesson is worth stating plainly, because it generalizes past this
parameter: **a backtest of a periodic strategy has a free parameter nobody
declares - which day the clock starts on.** Any result quoted from a single
phase is one draw from a distribution whose spread here is several points of
CAGR. Sweeps of anything that shifts the schedule must be phase-averaged or they
report alignment luck.

---

## Correction: the "ranking buys drawdown" claim is not established

**Written 2026-09-17**, correcting Track G's headline on the same day it was
recorded. The phase-averaged decay test does not reproduce it.

Track G reported that ranked selection beats random by ~3pp of drawdown at a
14-day hold and loses to it by 7.5pp at 63 days, and concluded that the
composite carries short-lived RISK information. Re-run at the live book size
with every arm averaged over 10 rotation phases and 60 random draws per hold:

| Hold | dCAGR | dMaxDD | dVol |
|---|---|---|---|
| 7 | +3.04% | +6.82% | -0.09% |
| 10 | +1.65% | +7.69% | -0.26% |
| 14 | +4.10% | +5.97% | +0.00% |
| 18 | -0.53% | **-7.85%** | -0.07% |
| 21 | +0.98% | +1.70% | -0.18% |
| 28 | -0.20% | +1.14% | +0.41% |
| 42 | +4.36% | +3.07% | +0.61% |
| 63 | -0.79% | **+4.89%** | -0.21% |
| **mean** | **+1.58%** | **+2.93%** | **+0.03%** |
| t | 2.16 | 1.67 | 0.25 |

### What survives

**The ranking is worth roughly +1.6pp of CAGR** across hold lengths, t = 2.16.
Marginal, and consistent with Track F's +0.84pp gross once the persistence
advantage in trading costs is added back.

### What does not

**There is no decay trend.** Spearman correlation of the gap with hold length is
-0.36 for CAGR and -0.52 for drawdown, on eight points. Significance would need
|rho| > 0.71. The curve bounces - +4.10, -0.53, +0.98, -0.20, +4.36 - with no
shape.

**The drawdown claim does not hold up**, t = 1.67 across holds. And the 63-day
sign *flips*: Track G measured ranked selection losing 7.5pp of drawdown to
random at a 63-day hold; phase-averaged at the live book size it gains 4.89pp.
"A stale ranking is worse than no ranking" was one noisy point and should not be
repeated.

**Volatility shows nothing at all**: dVol = +0.03%, t = 0.25. The ranking does
not make the return stream less volatile by any measurable amount.

### Why the per-hold t-statistics lie, and the lesson under it

The within-hold standard errors look decisive - t of 6.73, 7.37, -8.73, 8.03.
They are wrong, and the proof is in the table itself: if those SEs described the
real uncertainty the curve would be smooth, and it is not. The variation ACROSS
hold lengths (sd 2.06pp for CAGR, 4.95pp for drawdown) dwarfs the within-hold SE.

The reason is that **10 rotation phases are not 10 independent samples.** They
are ten nearly-identical schedules over *the same fourteen years of market
history*, sharing every episode that matters. Averaging them removes alignment
luck, which is what it was built for, and does almost nothing against the
dominant source of variation: there is only one realization of history.

That compounds with a second problem. **Max drawdown is a single-episode
statistic** - one number set by one stretch of one history - so comparing max
drawdowns across arms has almost no statistical power by construction. That
dVol is flat while dMaxDD is large and unstable is the signature: the drawdown
difference is plausibly about *which* episode happened to be worst, not about a
persistent property.

**The general constraint, which bounds this whole research program:** with one
14-year history, effects smaller than a few points of CAGR are not resolvable,
and anything resting on max drawdown is barely estimable at all. Future risk
comparisons should use measures that aggregate over many episodes - downside
deviation, Ulcer index, the mean of the worst k drawdowns, the 5th percentile of
rolling 6-month returns - rather than the single worst point.

### What still stands from Track G

Sweeps 1 and 2 held the schedule fixed across arms and are unaffected: the
slot-based overlay does invert on a wide book, and the weight-based overlay does
scale. Those compare configurations on identical phases and identical history.
The rebalancing-policy result also stands, for the same reason. It is sweep 3
and the ranked-versus-random risk interpretation that this corrects.

---

## The backtest window is in the top decile of available history

**Measured 2026-09-17.** The record starts 2010-01-04 because that is where a
clean panel for today's universe begins, not because anyone chose it. That
choice is not neutral.

SPY total return over overlapping 15.9-year windows since 1993 — the same length
as the backtest:

| | SPY CAGR |
|---|---|
| **Our window, 2010-01-04 onward** | **14.02%** |
| Median of 213 comparable windows | 8.02% |
| p25 / p75 | 6.00% / 9.59% |
| Best window on record | 16.53% |
| Worst window on record | 3.41% |
| **Percentile of our window** | **91st** |

By decade:

| Period | SPY CAGR |
|---|---|
| 1993-2000 | 21.29% |
| **2000-2010** | **-0.91%** |
| 2010-2020 | 13.27% |
| 2020-2026 | 15.00% |
| 1993-2026 | 10.79% |

The market returned 14.02% in our window against 8.02% in the median comparable
one. The decade immediately preceding the record returned **-0.91%** — a model
measured over 2000-2010 would be a different document.

The pass-through to the model is not 1:1 and is not estimated here. The blended
book runs at roughly half of SPY's beta, so a naive subtraction would be wrong
in the other direction. What is established is the direction and that the
magnitude is large.

### Four deductions now stand against the headline

| Bias | Size | Status |
|---|---|---|
| Period generosity | ~6pp on the market; pass-through unknown | measured 2026-09-17 |
| Rotation phase | ~3.2pp | measured 2026-09-17 |
| Survivorship | unbounded | TODO 6, unmeasured |
| Multiple testing | unknown | TODO 7, unmeasured |

They are independent and they compound. **No sentence of the form "this strategy
returns 19.6%" is defensible.** The supportable claim is narrower and still
worth having: measured over a top-decile decade on a survivorship-selected
universe, the strategy roughly matched its own universe on return and
substantially beat it on drawdown.

### What this does not touch, and why that matters

Every one of these biases moves LEVELS. None of them moves the DIFFERENCE
between two arms measured on the same window, the same rotation phase and the
same universe — which is how every comparative result in this file was
measured:

- the ranker at the 61st percentile of random draws (Track F)
- the overlay worth +3.03pp gross on ranked picks and -1.18pp on random (Track F)
- `top_n=4` against other book sizes (Track E)
- the slot-based overlay inverting on a wide book (Track G)
- rebalancing policy being irrelevant (Track G)

Those stand. The research program's comparative conclusions are robust to all
four biases precisely because both arms carry each bias equally. It is only the
absolute numbers that are inflated, and they are inflated by a lot.

This is the single most useful distinction to hold when reading this document:
**trust the differences, discount the levels.**

---

## Track H — the model does not generalize to baskets it was not built on

**Run 2026-09-17.** `scripts/build_random_pool.py` then
`scripts/analyze_random_baskets.py`. 100 random baskets of 40 names across four
families, drawn from 747 US equities that are liquid today and have full
2010-2026 history. The live universe run through the identical code path
reconciles with production to 0.064%.

The question was whether the model performs because of what it does, or because
of what it was handed. The answer is the second.

### The model destroys value on baskets it has not seen

Each row compares the model against **equal-weighting that same basket** — the
comparison that is immune to survivorship, period and phase bias, because both
arms carry all three equally.

| Family | model CAGR | EW CAGR | edge | win rate | model DD | EW DD | DD edge | model Sharpe | EW Sharpe |
|---|---|---|---|---|---|---|---|---|---|
| **LIVE universe** | 19.57% | 21.01% | **-1.44%** | — | **-19.2%** | -35.2% | **+16.0%** | 0.90 | 0.99 |
| stratified/cap-matched | 7.91% | 16.68% | -8.78% | 0% | -36.0% | -39.2% | +3.2% | 0.20 | 0.69 |
| stratified/all-cap | 6.31% | 15.53% | -9.21% | 4% | -48.1% | -42.2% | -5.8% | 0.08 | 0.57 |
| random/cap-matched | 9.36% | 17.35% | -7.98% | 0% | -35.6% | -37.9% | +2.3% | 0.28 | 0.73 |
| random/all-cap | 7.40% | 16.62% | -9.22% | 4% | -50.2% | -41.5% | -8.7% | 0.13 | 0.62 |

**The model beat its own basket in 2 of 100 random baskets.** Across all 100 the
edge averages **-8.80%** (sd 3.84%, range -17.90% to +3.82%). Sharpe collapses
from ~0.65 for owning the basket to ~0.17 for running the model on it. The
drawdown protection that is the model's main documented virtue is worth +3.2pp
at best and is *negative* on all-cap baskets.

The live universe's edge of -1.44% sits at the **96th percentile** of that
distribution, z = **+1.92**.

### Two readings, and this test cannot separate them

**Reading one: the curation is hindsight.** The live universe was assembled over
years with 2010-2026 performance visible. A universe selected that way will of
course sit in the right tail of randomly drawn ones, and z = +1.92 is precisely
the size of effect that selection-with-knowledge produces. On this reading the
model has no transferable mechanism and the entire record is universe choice.

**Reading two: the curation is a repeatable process.** The screen targets
liquid, trending, sector-diversified, large-cap names. That is a rule, not a
list, and rules can generalize even when the specific names were chosen late.

Nothing in this study distinguishes them, because there is exactly one live
universe and it was built with the answer visible. **TODO item 6 — running the
model on a frozen 2010-vintage universe — is now the decisive experiment for the
whole program rather than merely the most valuable one.** It applies the
selection rules using only information available in 2010 and asks whether the
model still works. That is the only available test that can tell hindsight from
process.

### One component that does look structural

Cap-matched families hold up markedly better on drawdown than all-cap ones
(+3.2pp and +2.3pp against -5.8pp and -8.7pp), and their model drawdowns are
-35.6%/-36.0% against -48.1%/-50.2%. **A four-name book is only survivable in
mega-caps.** That is a principled, transferable finding rather than a hindsight
artifact, and it is consistent with Track E's result that drawdown improves
monotonically with book size up to four and then flattens: concentration is
tolerable only when the constituents are individually stable.

It does not rescue the return result. Cap-matched families still give up 8.0-8.8pp
of CAGR to owning their own baskets.

### A hypothesis that was tested and failed

The live universe carries instruments the ranker can use as escape hatches that
no equity basket has: **SH** (short SPY) and **HYG** (high-yield credit) sit in
the *momentum* sleeve, and FINDINGS already records that "momentum picks SH
outright in a downturn". The obvious hypothesis was that the drawdown advantage
comes from having somewhere to hide.

It does not. Adding SH and HYG to 25 random baskets made every measure **worse**:
model CAGR 9.98% -> 8.04%, edge -6.77% -> -8.70%, drawdown -47.1% -> -49.6%. The
ranker does use them — 17.2% of book-days — and using them costs money. The
hatch improved the edge in 1 basket of 25.

Recorded because it is a plausible-sounding explanation that happens to be
false, and would otherwise be proposed again.

### What this changes

The live model is unchanged and the decision to keep it is untouched: whatever
the reason, the configuration that is actually running produced 19.57% at -19.2%
drawdown over this window.

What changes is the standing of every general claim about the *mechanism*. The
overlay, the ranker and the book size were all characterized on one universe.
Track F established the ranker contributes little; this establishes that even
what remains does not survive contact with a different basket. The honest
summary of the model is now: **a configuration that works on this universe, for
reasons not yet distinguishable from having chosen the universe.**

---

## The backtest cannot evaluate universe updates

**Found 2026-09-17**, attempting to adopt a mechanically selected universe.

`scripts/select_universe.py` proposed a 40-name universe from the current Schwab
screen, maximizing effective independent bets under an industry cap of 2. The
proposal is good by every forward-looking measure available: 25.35 effective
bets against 22.76 for random same-size draws (97.5th percentile), 63.4% of
ticker count against the incumbent's 56.3%, 33 industries, incumbency honoured
with only 2 additions and 8 genuine screen failures.

It backtests far worse, and consistently:

| Universe | N | CAGR | Sharpe | MaxDD | Calmar |
|---|---|---|---|---|---|
| current | 52 | 22.30% | 1.03 | -18.80% | 1.19 |
| proposed | 44 | 17.60% | 0.82 | -24.07% | 0.73 |

Subperiod deltas: -2.41, -4.21, -5.17, -7.31. Zero of four.

### Where the gap comes from

Adding each dropped name back to the proposal, one at a time:

| Added back | CAGR | vs proposal | that name's 2010-2026 return |
|---|---|---|---|
| **TSLA** | 20.69% | **+3.09%** | **22,383%** |
| AA | 19.49% | +1.89% | 32% |
| HYG | 18.24% | +0.64% | 131% |
| JNJ | 18.11% | +0.51% | 572% |
| all ten (= current) | 20.86% | +3.26% | (SPY: 792%) |

**TSLA alone is 95% of the gap.** It returned 224x over the window and no longer
passes the screen.

### The conclusion, which is structural rather than about this proposal

The incumbent universe was assembled over years with 2010-2026 performance
visible. The backtest applies today's list backwards, so it *knows* TSLA
compounded at 224x. **Any** universe update that drops TSLA will look
catastrophic in backtest, regardless of its forward merit — and every future
update faces the same asymmetry against whatever the current list's biggest
winners happen to be.

So the backtest is not a valid instrument for this decision. It is biased in
favour of the incumbent by exactly the hindsight Track H identified, and the
bias is large: 3.26pp of CAGR, 95% of it one ticker.

**Universe changes must be decided on forward-looking criteria alone** — does a
name pass the screen, does it add an independent bet, is it liquid enough to
trade — and the backtest consulted for nothing except confirming that the
machinery still runs. This is the operational consequence of Track H, and it
disqualifies the most natural way anyone would want to validate a universe
change.

### An unresolved tension worth naming

The screen filters FOR five-year total return above 10%. The stated expectation,
supported by this repo's own quintile table (the worst-ranked quintile
out-returned the best at six of seven horizons), is that names up 20% a year for
five years are reversion candidates. The screen and the thesis point in opposite
directions, and nothing tested here resolves it. Dropping TSLA *because* its
five-year return decayed is, on the mean-reversion reading, dropping it at
exactly the wrong moment.

---

## What did work

**The velocity window, on a corrected scale.** The original 0.7/0.3 selection
ran against a normalization defect: a rolling z-score level blended against a
cross-sectionally normalized velocity, so nominal weights did not correspond to
actual influence. Re-run correctly, `velocity_window=10` is beaten by turning
velocity off entirely in every subperiod. Window 5 leads on all four metrics
and beats the median candidate in all three subperiods.

Caveats recorded in `production_config()`: this is an in-sample selection, and
the surface is sharp (window 7 gives up ~9pp of first-period CAGR).

**Walk-forward validation settled how to maintain it.** Re-tuning on a trailing
3-year window and testing 6 months out produced 11.54% CAGR / 0.40 Sharpe —
worse than the *median* fixed config. It picked the truly-best config in 13% of
23 windows against ~9% for random guessing. **Do not re-optimize on a schedule.**

**Correlation-aware selection, gated on VIX.** Filtering continuously costs
2.88pp of CAGR *and* worsens drawdown. Gated above VIX 25 it is free on return
and buys 1.32pp of drawdown.

| Gate | CAGR | Sharpe | MaxDD | Calmar |
|---|---|---|---|---|
| none | 19.85% | 0.91 | -20.55% | 0.97 |
| **above VIX 25** | 19.83% | 0.91 | **-19.23%** | **1.03** |
| above VIX 15 | 18.32% | 0.83 | -25.08% | 0.73 |
| always | 16.96% | 0.75 | -26.31% | 0.64 |

Caveat: VIX ≥ 25 is 13% of history and 5% of the last three years, so this rests
on a thin slice. Approximately-free rather than proven.

---

## Defects fixed

**Execution assumption.** The backtest filled at the close that generated the
signal. Unachievable. Correcting it costs 0.52pp of CAGR; adding realistic
slippage costs 2.90pp more. Notably, `next_open` slightly *improves* drawdown
(-23.13% vs -24.32%) while `next_close` — a full day of lag — blows it out to
-28.41%. Acting at the open is worth 3.2pp of drawdown and costs nothing.

**Relative strength was silently empty.** `spy_data` remained a one-column
DataFrame, so `Series / DataFrame` aligned the date index against the column
axis and produced all-NaN. Zero rows in every historical export; now 210,060.

**Phantom session rows.** yfinance emits an all-NaN row for the current session
before the close. Harmless for ranking, but it silently zeroed the universe in
anything using `dropna(axis=1)`.

**Sector labels are a poor concentration proxy.** V and STT are both Financials
and correlate at **0.020** (50d). IAU and NEM are different sectors and
correlate at **0.792**. The live report now leads with correlation.

**Growing the book was free.** `_trade_cost` computed turnover as the share of
the *old* book that left, so adding names charged nothing -- three names to
five, all three kept, scored as zero turnover despite two buys and three trims.
Found while making the cost model weight-aware for the sizing work (Track I).
7 occurrences in the live config, 2.7bp of CAGR. Every result before
2026-09-18 carried the subsidy.

**Weight capping oscillated.** The per-position cap redistributed excess onto
*all* other names including ones already at the cap, so capping C lifted B over
the cap, capping B lifted C back over, and the answer depended on which
iteration the loop happened to stop at -- a 3-name book could return a 41%
position under a 35% cap. Capped names are now frozen and only the free ones
share the remaining budget. Caught by a unit test, not by inspection.

---

## Universe health

52 tickers delivering **27.4 effective independent bets**, largest common factor
18.5% of variance. Over 1,326 pairs the maximum is 0.792, only 2 clear 0.70,
none reach 0.80 — the quarterly screen is already diversifying well below the
sector level.

The one flagged pair: **IAU / NEM at 0.792**, cross-sector by label. Gold ETF and
gold miner are one bet, and since IAU is the crisis fill, a stressed regime
could hold gold twice.

`python scripts/screen_universe.py`

---

## Model health monitor (built, with a stated limit)

`monitor_health.py`, `momentum/health.py`, `rsi_ma_health_calibration.csv`

A degradation warning: 126-day log excess return against SPY, zero-centered,
scaled by its own historical spread, requiring both a z and a points breach
sustained for 20 sessions. Surfaced on every live run.

**The book's beta is 0.48, and that is a composition effect.**

| Book composition | Share of days | Beta |
|---|---|---|
| Pure momentum | 58% | **0.994** |
| Holding SHY/TLT/IAU or SH | 42% | 0.120 |
| Holding SH specifically | 14.6% | -0.026 |
| Blended | 100% | 0.431 |

The momentum picks are a market-beta book. They do **not** rise when the market
falls — on pure-momentum days the model is positive on only 26.2% of SPY-down
days, with down capture 0.90 against up capture 1.09. The low headline beta
comes from the 36.3% of days holding regime-assigned SHY/TLT/IAU and the 14.6%
holding SH, which momentum picks outright in a downturn.

This is why the alarm runs on **raw** excess. Beta-adjusting sounds prudent and
does the wrong thing here: at beta 0.48 CAPM asks the model to beat only half of
SPY's move, so it forgives lagging in *rising* markets. On 2026-09-02 the model
returned 6.65% against SPY's 12.27% — trailing by 5.6 points — and scored an
adjusted z of +0.04. The adjusted series never reaches z -2 anywhere in the
record. The beta-adjusted number is kept beside the raw one as a diagnosis: raw
bad and adjusted fine means a half-beta book failed to keep up with a rally;
both bad means the picks stopped working.

**What calibrated, and what didn't.** The rule fires twice in eleven years
(2019-07, 2021-08), median 28 days, 2.0% of the record. That rarity is stable —
across every record length and threshold tried it fires two to five times, the
right order for a review trigger.

What is *not* estimable is whether firing predicts anything. Median forward
six-month excess after a trigger swings from -2.6% to +4.8% depending on record
start, points floor, and threshold, on two to five observations. The answer did
not come out negative; it came out unmeasurable. So the monitor is documented and
reported as a REVIEW TRIGGER — "you are in the worst tail of your own history, go
look" — never as a predictive warning.

Three measurement notes that shaped the build:

- **Overlapping windows.** Daily 126-day windows share 125 of 126 days; the
  record holds ~22-32 independent windows, not thousands. `z < -2` carries none
  of its textbook rarity, so thresholds come from the episode record and
  episodes are counted, not days.
- **The z is not scale-stable.** The same day scores -0.47 from a 2005 start and
  -0.59 from 2010, purely because the expanding SD differs. The monitor's default
  start is pinned to `run_live`'s for that reason, and the points floor exists to
  stop the bar drifting as history accumulates. At today's scale the floor
  (-12.5%) is stricter than z -1.25 (-10.9%) and is the binding condition.
- **Survivorship cuts the other way here.** The historical record is optimistic,
  so live shortfalls will be more common than this calibration implies. The bar
  should be read as "what would I act on", not "what was historically rare".

---

## Track I -- position sizing: the null holds, and choosing costs money

The prior going in was that weighting a 4-name momentum book would not make a
systematic difference. It does not. Four schemes were run over the identical
books on the identical dates -- selection held fixed, so any difference is
sizing or it is noise.

Out of sample, 3y train / 6m test, 24 windows, selecting on Sharpe:

| scheme | OOS CAGR | OOS Sharpe | OOS MaxDD | OOS Calmar |
|---|---|---|---|---|
| equal | 15.68% | **0.66** | -23.00% | 0.68 |
| score_proportional | 15.90% | 0.65 | -26.77% | 0.59 |
| score_proportional cap35% | 15.26% | 0.62 | -26.08% | 0.59 |
| inverse_vol cap35% | 13.92% | 0.61 | -19.86% | **0.70** |
| inverse_vol (uncapped) | 13.01% | 0.59 | -20.72% | 0.63 |
| vol_target 25% cap35% | 11.55% | 0.51 | -19.70% | 0.59 |
| vol_target 15% cap35% | 7.39% | 0.30 | **-16.40%** | 0.45 |

**Nothing beats equal weight on Sharpe.** The sharper version of the result is
that *choosing* a scheme is worse than not choosing one: retuning every six
months on training data returns 14.53% / 0.60, against 15.68% / 0.66 for equal
weight left alone. The trainer picked the best-in-hindsight scheme in 25% of
windows (14% is chance) and switched among five schemes across 24 windows.
This is the same shape as the walk-forward result on `velocity_window` -- the
training window does not predict the next one, so adapting adds turnover and
noise.

**score_proportional is the informative null.** Paired against equal weight on
the same names and the same days, the daily difference is indistinguishable
from zero (annualized -0.43%, t = -0.31, 3669 observations). This is a direct
test of something Track F could not reach: the ranker's score *orders* names
usefully, but the size of the gap between scores carries no usable information.
Weighting by score is weighting by noise. It is worth noting that the weights
really did differ -- mean absolute deviation from 1/N was 0.11, and the mean
top weight 0.43 against 0.26 for equal weight -- so this is a null about the
score, not a null about the book being too homogeneous to weight.

**Drawdown is the one real separation, and it is not free.** vol_target at 15%
cuts MaxDD from -23.00% to -16.40%, the largest reduction available, but gives
up more CAGR than it saves: Calmar falls to 0.45 from 0.68. `inverse_vol
cap35%` is the only scheme that improves on equal weight on any risk-adjusted
measure -- Calmar 0.70 against 0.68 -- and the margin is well inside noise. If
a future requirement is specifically to cap drawdown rather than to maximize
risk-adjusted return, inverse-vol capped at 35% is where to start. Nothing here
justifies moving off equal weight today.

### Two engine changes this required

**The cost model is now weight-aware.** Reweighting the same names was
previously free, because cost was computed from name changes alone. Left alone,
that would have handed every weighted scheme a subsidy the equal-weight
baseline never received -- the easiest available way to manufacture a fake
improvement in exactly this experiment.

**That exposed a real bug.** The old `_trade_cost` measured turnover as the
share of the *old* book that left: `len(set(old) - set(new)) / len(old)`.
Growing the book therefore charged nothing. Going from three names to five
while keeping all three scored as zero turnover, despite two purchases and
three trims to pay for them. In the live config this fires 7 times across 3669
sessions and costs 2.7bp of CAGR. Small, but it was a subsidy, and every
historical result carried it.

Equal weight is otherwise untouched: gross returns match the pre-weights
arithmetic to 1.7e-18, checked against an independent replay of the old
arithmetic rather than by comparing the refactored engine with itself.

Reproduce: `python scripts/analyze_position_sizing.py`

---

## Track J — trend change, strength and room: the hypothesis is backwards, and the inverse is the first ranker in this repo to clear |t|=2

**Run 2026-09-20.** `scripts/analyze_reversal_ic.py`, on `momentum/reversal.py`.
Stage one only: this measures prediction, not a portfolio. Nothing here is in
the live model.

The proposal was to stop scoring where a stock *is* and start scoring whether it
has **turned** — negative trend to positive — how strong that turn is relative
to the universe, and how much **room** it has inside its 52-week range. Motivated
by Track F's finding that the live composite predicts reversal at 5-63 days: if
this universe reverses, trade the reversal deliberately rather than discover it
as a sign error.

Three terms, each self-normalizing before cross-sectional standardization:

| term | definition |
|---|---|
| `turn_t` | t-statistic of the change in OLS log-price slope between two adjacent 60-day windows |
| `flip` | `rank(3-month return) - rank(12-1 month return)` |
| `strength_t` | t-statistic of the current 60-day slope |
| `range_pos` | position in the trailing 52-week range, 0 at the low, 1 at the high |

`range_pos` was deliberately exported in neutral form rather than as "headroom",
because George & Hwang (2004) find *proximity* to the 52-week high predicts
higher returns — the opposite sign to the room-to-run story. The weight was left
free so the direction came off the measurement.

### Both halves of the hypothesis came out backwards

Measured on a 747-name pool (`random_pool.csv`), eligible pool ~650 per date,
open-to-open forward returns:

| signal | 5d | 14d | 21d | 42d | 63d | 126d |
|---|---|---|---|---|---|---|
| composite (live) | -0.0057 | -0.0066 | -0.0081 | -0.0088 | -0.0113 | -0.0013 |
| `turn_t` | -0.0035 | -0.0061 | -0.0082 | -0.0140 | -0.0115 | -0.0096 |
| `flip` | **-0.0149** | -0.0157 | -0.0151 | -0.0197 | -0.0137 | +0.0008 |
| `strength_t` | +0.0006 | -0.0028 | -0.0040 | -0.0039 | -0.0027 | +0.0107 |
| `range_pos` | +0.0103 | +0.0105 | +0.0097 | +0.0124 | +0.0173 | +0.0241 |

**Turn is negative at every horizon.** A stock whose trend has just changed from
falling to rising goes on to *underperform*. Both definitions agree, and adding
strength does not rescue either.

**Room is positive at every horizon on the wide pool**, rising with horizon to
+0.0241 at 126 days — six of six, under both eligibility arms. Nearness to the
52-week high beats headroom: the literature's sign, not the intuitive one.
Weighting the term as headroom (`b + room(-1)`) makes the combined score
significantly *worse* (-0.0199, t = -2.05 at 42 days).

On the 46-name universe the same term is positive at 5, 14 and 21 days and
negative at 42, 63 and 126 — three of six, none significant. The direction is
established on the wide pool only, and the universe neither supports nor
contradicts it.

### Inverted, `flip` is the strongest ranker this repo has measured

Negating `flip` gives `rank(12-1 month) - rank(3-month)`: **buy strong 12-month
momentum that has recently pulled back.** That is classic 12-1 momentum plus
short-term reversal — two of the most replicated effects in the literature — so
this is a recovery of a known result rather than a discovery, which is the
reassuring direction for a signal found by flipping a sign.

Per-period top-K edge over the eligible-pool mean, on non-overlapping periods:

| arm | h | K | edge/period | t | P1 | P2 | P3 |
|---|---|---|---|---|---|---|---|
| `flip negated` | 14 | 8 | **+0.99%** | **2.66** | -0.05% | +2.04% | +0.97% |
| `flip negated` | 63 | 8 | +6.41% | 3.94 | +2.41% | +7.16% | +9.88% |
| composite (live) | 14 | 8 | +0.16% | 0.55 | +0.50% | -0.15% | +0.13% |
| composite (live) | 63 | 8 | -0.76% | -0.84 | +1.84% | -0.63% | -3.62% |

For scale: Track F measured the live ranker's edge at **+0.16% per 14-day period
at K=1**, its best cut. On the wide pool `flip negated` earns +0.99% at K=8.

**The production composite has no edge on the wide pool at all** — negative IC at
every horizon, and its 63-day edge is *negative* and worsening by subperiod. That
is Track H's non-generalization result arriving again by a different route.

### Four reasons not to trade it yet

**1. The first subperiod is empty.** P1 (2012-2016) is -0.05% at the live hold
and is the weak segment in nearly every cut. The edge is a 2016-2026
phenomenon in this sample. FINDINGS' standing bar is that a result winning the
full sample by winning one segment is a fit; this wins two of three, with the
first flat rather than negative — better than a fit, short of stable.

**2. It dies in large caps.** Edge at the live hold, K=4, by today's market cap:

| subset | edge/period | t |
|---|---|---|
| all 747 | +1.21% | 2.12 |
| above $2B | +1.17% | 2.53 |
| above $10B | +0.98% | 2.23 |
| **above $50B** | **+0.13%** | **0.43** |

The median pick is a $9.1B name against a $20.7B pool median, and the most-picked
names were CELH, AEHR, CRK, BLFS, CYTK, AXON, INSM, ~~WULF~~, APPS, SRPT.
**The edge is not in the range the live universe occupies.**

Harvesting it means a different and much smaller-cap universe, where the 7.5 bps
fill assumption is not credible and where the repo's turnover bar (0.17pp of
CAGR per extra 1x of annual turnover) bites harder.

(WULF is on the restricted list and is struck through above because that first
measurement did not apply it — see "The compliance list costs nothing here".)

**3. Survivorship is bounded but not removed.** The pool was built in 2026 from
names liquid today, and "buy the dip in a strong performer" is the most exposed
construction there is — every dip in the sample was followed by a recovery
because the names whose dips did not recover are absent. Removing the names we
know ex-post were the biggest winners does *not* collapse the edge:

| variant | h=14, K=8 | t |
|---|---|---|
| all names | +0.99% | 2.66 |
| ex top 10% realized winners | +1.12% | 3.00 |
| ex top 20% | +0.68% | 1.93 |
| ex top 30% | +0.66% | 1.93 |

So it is not simply hindsight about which names won. But this bound says nothing
about **delisted** names, which are not in the pool at all and cannot be added
without point-in-time data. That remains TODO item 6, and it is the binding
constraint on believing any magnitude here.

**4. Multiplicity.** 90 (signal x horizon) cells were printed, about 45
independent of sign; roughly 2.2 would clear |t|=2 by chance. Seven did. The
arms are correlated and the horizons overlap, so that comparison is crude in
both directions, but the honest reading is "a lead worth re-testing", not "an
effect established".

### The compliance list costs nothing here

**Checked 2026-09-20**, after the first measurement was run on an unfiltered
pool. `random_pool.csv` is a screener export, so restricted names are expected
in it, and five were: **AMT, CCI, DLR, IRM, WULF**. WULF was among the ten
most-picked names, so this was not a theoretical exposure — part of the first
reported edge was earned on a name the account cannot trade.

Re-measured with `momentum.restrictions.filter_screen` applied by symbol,
industry and issuer name:

| cut | with restricted | compliant only |
|---|---|---|
| 14d, K=4 | 1.206% (t 2.12) | 1.142% (t 2.03) |
| 14d, K=8 | 0.986% (t 2.66) | **1.008% (t 2.79)** |
| 63d, K=8 | 6.411% (t 3.94) | **6.413% (t 4.01)** |

Restricted names were **1.31% of top-8 picks** (WULF 283, IRM 77, DLR 19,
CCI 7). At K=8 the compliant edge is marginally *larger*. The result stands on
a tradable pool, which is a stronger claim than the original.

Both scripts now filter by default and the panel carries a hard `check_symbols`
assertion, so this cannot silently regress.

### Comparing horizons: per-period edges are not comparable

A 6.41% edge per 63-day period and a 0.99% edge per 14-day period are figures
over periods of different length. Normalized two ways — annualized edge, and
the annualized information ratio of the edge itself, which also accounts for
dispersion and is the cleaner comparison:

| h | edge/period | N | ann. edge | **ann. IR** | composite ann. IR |
|---|---|---|---|---|---|
| 5 | 0.418% | 733 | 21.08% | 0.70 | -0.11 |
| 14 | 1.008% | 262 | 18.15% | 0.73 | 0.17 |
| 21 | 0.766% | 174 | 9.20% | 0.39 | 0.17 |
| 42 | 2.696% | 87 | 16.18% | 0.63 | 0.34 |
| 63 | 6.413% | 58 | 25.65% | **1.05** | -0.26 |
| 126 | 4.744% | 29 | 9.49% | 0.37 | 0.19 |

63 days is the best horizon on both measures, and **that should not be acted
on**. The profile is non-monotone — 0.70, 0.73, 0.39, 0.63, 1.05, 0.37 — and a
signal that genuinely improved with horizon would not dip at 21 days between 14
and 42. Independent observations also collapse at the long end (58 at 63d, 29
at 126d), so the peak carries a wide interval. Selecting 63 days because it is
the maximum is the sweep-maximum trap wearing different clothes.

The defensible statement is the flat one: **the edge is present at every horizon
at an annualized IR of roughly 0.4 to 1.0, against roughly zero for the live
composite.** That comparison does not depend on choosing a horizon, which is
what makes it worth having.

Do not read the `ann. edge` column as an achievable return. It is a gross
cross-sectional edge; harvesting the 21% at h=5 would mean ~50 rebalances a
year and the turnover would eat it.

### The 46-name universe agrees in sign and cannot resolve anything

Same signs on the two terms that matter — `flip negated` positive at five of six
horizons, `turn_t` negative at all six — but no cell clears |t| = 1.7 under
either eligibility arm (the largest is 1.66). `range_pos` is the exception: it
splits three-three and agrees with the wide pool only at the short horizons.
The minimum detectable IC at 80% power on 46 names is **0.040 at the live hold**
against a measured 0.0118. The universe test is roughly three times too small to
see an effect of the size the wide pool reports. It is consistent, not
confirmatory, and it should not be cited either way.

Worth noting anyway: top-4 edge at 14 days is +0.214% for `flip negated` against
+0.131% for the live composite, and at 63 days +1.167% against +0.178%.

### The eligibility floor is a non-issue

`min_level_threshold = -3.0` was expected to delete exactly the candidates a turn
signal wants. It does not: 650.5 eligible names per date with the floor on
against 652.2 with it off, and every IC in this study moves by less than 0.0003
between the two arms. The concern was reasonable and is now measured and closed.

### What this changes

**Nothing in the live model.** Stage one measures prediction; no portfolio was
simulated and no configuration changed.

1. **"Has it turned?" is answered and the answer is no.** Both definitions
   predict the wrong way at every horizon on 747 names. This should not be
   re-proposed without a new mechanism — it is now a measured negative, not an
   untested idea.
2. **"Room to run" is answered and the answer is the opposite.** Nearness to the
   52-week high is the direction with signal. If the screen or any future scoring
   uses distance-from-high as a positive, that is now known to be backwards.
3. **The live composite's non-generalization is confirmed twice over.** Zero-to-
   negative IC on 747 names, and a 63-day edge that is negative and decaying by
   subperiod. Track H said the model does not generalize to baskets it was not
   built on; this says the *ranker* does not either.
4. **The open question is now a universe question, not a signal question.** The
   only measured edge lives below $50B, which is outside the universe. That
   collides directly with the standing conclusion that remaining upside is in the
   candidate set rather than in timing — and for once points at a specific,
   testable change to the candidate set rather than at a parameter.

### Reproducing

```
python scripts/analyze_reversal_ic.py                       # both pools, both floor arms
python scripts/analyze_reversal_ic.py --pool wide --floor on   # the table above
```

Five checks run before anything prints, and the script exits 3 without reporting
if any fails: the vectorized rolling OLS against `scipy.stats.linregress`
(gap 2.3e-14); `range_pos` bounds and brute-force rolling min/max (gap 0.0);
**no look-ahead**, by recomputing every term on a truncated panel and requiring
the value at T to match (gap 0.0); the IC pipeline reproducing **Track F's
published mean IC table** on Track F's own 52-symbol universe (worst gap
0.00007 across all six horizons); and the wide-pool loader reproducing
production `load_data` (max relative gap 1.8e-6 over 187,410 cells).

The fourth is the one that matters most — it ties every number above to a
published result rather than to a re-derivation. The fifth caught a real defect
during construction: it first "passed" while returning `nan`, because a masked
difference produced NaN and `nan > tol` is False.

---

## Track J, stage two — the portfolio comparison: the new score wins, and the live composite is worse than random off its own universe

> **Superseded in part — see "Correction to Track J stage two" below.**
> The pool screen used here was look-ahead, and the claim that the live
> composite has negative ranking skill is RETRACTED.

**Run 2026-09-20.** `scripts/analyze_reversal_backtest.py`. An information
coefficient is not a P&L, so this puts both scores through identical portfolio
machinery.

### What makes the comparison fair

One input changes. `run_strategy(ranking_override=...)` swaps the panel used for
ranking while eligibility, the level floor, the regime overlay, the correlation
filter, the hold clock, sizing and execution stay as configured — and the level
floor still reads the real composite, so both arms face an identical eligible
pool. That the override is neutral is asserted, not assumed: passing the real
composite through it reproduces the un-overridden run to 0.0e+00 on both CAGR
and the full return stream.

The bar is not SPY and it is not the live model. Every pool carries Track F's
four arms (ranked/random x overlay on/off) plus an equal-weight hold of the
pool, because Track F established that owning the universe beats the model. The
quantity that means something is **arm minus its own random null on the same
pool** — ranking skill with the universe divided out.

### The headline, on 635 liquid compliance-filtered names, realistic costs

| arm (overlay off) | CAGR | Sharpe | Calmar | MaxDD | Vol | Turnover |
|---|---|---|---|---|---|---|
| **pullback** (`-flip` + `range_pos`) | **24.35%** | **0.65** | 0.43 | -56.48% | 30.50% | 1793% |
| `flip negated` | 22.74% | 0.48 | 0.42 | -54.47% | 37.80% | 1482% |
| equal-weight pool | 14.81% | 0.55 | 0.37 | -40.54% | 18.75% | 0% |
| random_plain | 7.02% +/-4.59 | 0.11 | 0.14 | -51.67% | 23.36% | 2591% |
| **composite (live)** | **3.14%** | **-0.05** | 0.06 | -53.59% | 27.70% | 2310% |

Ranking skill over each arm's own null, gross:

| score | plain | overlay |
|---|---|---|
| composite (live) | **-3.44%** | -5.33% |
| `flip negated` | +16.19% | +9.00% |
| **pullback** | **+16.52%** | +13.45% |

**The production composite has negative ranking skill on this pool.** It does
not merely fail to generalize — it picks worse than a random draw from the same
eligible set, and pays 2310% annual turnover to do it. Track F said the ranker
carries no information; Track H said the model does not generalize to baskets
it was not built on. This is both, arriving together.

### The room term reverses sign of usefulness with the pool

On the 46-name universe, adding `range_pos` was clearly harmful: `pullback`
12.34% against `flip negated` 20.38%. On 635 names it is the best arm, and it
is better on *both* criteria — +1.6pp CAGR and Sharpe 0.48 -> 0.65, at lower
volatility (30.5% against 37.8%).

The mechanism is the obvious one and it is worth stating because it generalizes:
**on a universe curated for momentum, every name sits near its 52-week high, so
`range_pos` has almost no cross-sectional dispersion and contributes noise.**
Across 635 diverse names it has real spread and real information. A term can be
worthless on a narrow universe and valuable on a wide one without anything
about the term changing.

### Subperiods: the P1 hole does not survive into the portfolio

Liquidity-costed, equal thirds:

| arm | P1 | P2 | P3 |
|---|---|---|---|
| equal-weight pool | 12.09% | 19.62% | 12.88% |
| composite (live) | 3.54% | 5.43% | 0.50% |
| **pullback** | **17.51%** | 15.94% | **41.12%** |
| `flip negated` | 11.18% | 32.65% | 25.38% |

Positive in all three, and `pullback` beats equal weight in P1 by 5.4pp. The
"2012-2016 is empty" worry came from the IC measurement on the unfiltered pool
and does not reproduce here. Two measurements disagreeing about which segment is
weak is evidence that the period effect is unstable, not that a period is bad.

### It improves as fills get worse

| uniform bps | live | `flip negated` | delta |
|---|---|---|---|
| 7.5 | 5.14% | 20.21% | +15.08% |
| 15 | 1.55% | 17.32% | +15.77% |
| 30 | -5.27% | 11.73% | +17.00% |
| 60 | -17.62% | 1.31% | +18.93% |

No uniform slippage level erases the advantage, because the new score trades
*less* than the composite. That is the opposite of every rejected track in this
repo, all of which lost on turnover.

### The parameter surface: read the shape

20 cells, `top_n` x `hold`, both arms, liquidity-costed. New score CAGR:

| top_n \ hold | 5 | 14 | 21 | 42 | 63 | spread |
|---|---|---|---|---|---|---|
| 2 | 14.1 | 23.6 | **35.6** | 23.0 | 14.1 | **21.4pp** |
| 4 | 16.2 | 16.6 | 24.9 | 15.0 | 22.0 | 10.0pp |
| **8** | 20.3 | 20.4 | 22.0 | **22.6** | 20.4 | **2.3pp** |
| 16 | 17.8 | 19.1 | 17.9 | 18.2 | 19.1 | 1.4pp |

`top_n=2, hold=21` is the grid maximum at 35.6% and must not be acted on: that
row swings 21pp across holds, which is a spike. `top_n=8` is a plateau — 20-23%
at every hold, Sharpe 0.54-0.62, drawdown -41% to -57%.

Robustness across the grid: the new score beats live on CAGR in **95%** of
cells, clears CAGR-and-Sharpe-together in **80%**, median advantage **+11.4pp**,
worst cell -5.4pp. That is a far stronger claim than any single backtest figure.

**The inherited `top_n=4, hold=14` is one of the worst cells for this score**
(16.6% / 0.35 / -63.5%). Every turnover figure above is an artifact of settings
chosen for a different score on a different universe.

### The chosen configuration

**`top_n=8, hold=42`** — 22.6% CAGR, 0.62 Sharpe, -49.9% drawdown, rebalancing
six times a year instead of eighteen. Selected because the surrounding row is
flat, not because the cell is high, which is the only defensible way to pick off
a grid given walk-forward's finding that trailing-window re-tuning costs 3pp.

Drawdown falls monotonically with book size (-81/-70/-57/-47% at hold=5 for
`top_n` 2/4/8/16) at no CAGR cost above 8, which is what makes 8 rather than 4
the right answer for a signal ranking hundreds of names.

### Costs, and the engine change they required

`simulate_portfolio` now accepts `slippage_by_symbol`, a Series or a
dates x symbols frame read at the fill date. `momentum/liquidity.py` builds the
frame: half-spread plus square-root market impact, from **point-in-time**
trailing dollar volume rather than today's, so a name is charged what it would
have cost on the day. Today's volume applied backwards would undercharge every
name that has since grown — most survivors in a pool built today, and precisely
the names this signal picks.

On the 635-name pool it prices the least-liquid ADV quintile at 18.9 bps and the
most liquid at 4.6, median 9.0 against the flat 7.5 assumption.

A uniform per-name vector must reproduce the flat path exactly, or every
historical number in this document would silently move. That is asserted in
`tests/test_sizing.py` end to end through `simulate_portfolio`, and again inside
the run.

### What this does not establish

**Survivorship, which is now the only thing standing between this and a live
trial.** The pool is names liquid in 2026 and `pullback` buys drawdowns, so
every dip in the sample was followed by a recovery — the names whose dips were
terminal are absent. The ex-winner bound survived (Track J stage one), but
delisted names cannot be added without point-in-time data. TODO 0d.

Also outstanding: volatility is 30.5% against equal weight's 18.75%, so a real
share of the CAGR is leverage-like; drawdown at -56% is deeper than anything in
this repo's history; and the null here is 15 trials, coarse though the gaps are
large relative to the +/-4.6pp spread.

### Reproducing

```
python scripts/analyze_reversal_backtest.py --pool universe
python scripts/analyze_reversal_backtest.py --pool screened --trials 15
python scripts/analyze_reversal_backtest.py --pool screened --trials 3 --sweep
```

Four gates run before any arm prints, and the script exits 3 without reporting
if any fails: the live universe reconciles with production through this script's
panel (gap 0.0000%); uniform per-name slippage equals the flat path (0.0e+00);
`ranking_override` is neutral (0.0e+00, return streams equal over 3,669 days);
and the cost model is monotone in liquidity (ADV quintile medians 18.9 -> 4.6
bps, Spearman -0.97).

---

## Correction to Track J stage two, and what replaced it

**Written 2026-09-20, same day as the results above.** The stage-two section is
left standing because the record of what was believed and why it changed is
worth more than a clean page. Three of its numbers are wrong and one of its
conclusions is retracted.

### The defect: the pool screen was look-ahead

The screened pool was built by taking each name's **full-sample median** trailing
dollar volume and keeping those above $10M. A name therefore earned its place in
the 2010 cross-section because of volume it had in 2020 — and worse, it
reintroduces survivorship through the back door, because the names that stayed
liquid are disproportionately the ones that did well.

Replaced by `momentum.liquidity.tradable_mask`: a **per-date** screen on trailing
dollar volume and price that blanks the score on days a name fails, so a name
becomes unpickable and pickable again as conditions change. A price floor was
added at the same time, as the blunt proxy for the exchange continued-listing
minimum.

### What changed, at `top_n=8, hold=42`

| arm (overlay off) | as reported above | corrected |
|---|---|---|
| `flip_neg` | 22.74% / 0.48 | **26.56% / 0.69** |
| `pullback` | 24.35% / 0.65 | 21.46% / 0.66 |
| composite (live) | **3.14% / -0.05** | **14.36% / 0.46** |
| equal-weight pool | 14.81% / 0.55 | 13.48% / 0.48 |
| random_plain | 7.02% | 11.92% |

**RETRACTED: "the production composite has negative ranking skill on this pool".**
It does not. Corrected, its ranking skill is **+2.49pp gross** — positive, and in
line with Track F's +0.84pp on the live universe. The live model is not broken
off its own universe; it is merely beaten. The original claim was an artifact of
the look-ahead screen and of `top_n=4/hold=14`, and it was wrong in the
direction that flattered the new result, which is the direction to be most
suspicious of.

The two scores also swapped places. Parameters changed at the same time as the
screen, so which of `flip_neg` and `pullback` is better is **not settled**.

### Three defects in this family, all of which printed plausible output

Recorded together because they share one signature — none of them raised, and
each produced numbers a reader would have believed:

1. **The pool-loader check returned `nan`.** A masked difference produced NaN,
   `nan > tol` is False, and the check "passed" while comparing nothing. Fixed
   with `np.nanmax`, a relative tolerance, and an assertion on the number of
   cells actually compared.
2. **The delisting generator normalized every synthetic name to $100.** An 85%
   decline left it at $15, so a $5 price floor could never fire. The resulting
   "96% were still tradable the day before death" was a statement about the
   generator, not about the screen. With donor price levels and deeper
   cause-branch declines it is 63%, the rest screened out a median 95-100
   sessions early.
3. **`pd.Grouper(freq='6MS')` anchors bins to each series' own first timestamp.**
   The arms start on different dates — the new score needs 252+63 sessions
   before it can rank — so no two arms ever shared a window label and every
   walk-forward comparison had zero rows. It reported `nan%`. Fixed by binning
   from a fixed origin.

The lesson is the one already in `validate-before-reporting`: in this repo the
failure mode is never a crash, it is a plausible number. Every one of these was
caught by a check that existed to be sceptical of a result, not by the result
looking wrong.

### What the corrected picture supports

Three tests were run after the fix, and they do not all say the same thing.

**Rotation phase (TODO 0h.1) — passed.** Across 14 sampled offsets the new
score's CAGR spans **13.16% to 23.35%**, a 10.19pp spread against 0.71pp for
equal weight, so the dispersion is a rotation artifact rather than anything about
the data. The advantage survives it: the new score beats live at **100%** of
phases and equal weight at 93%, worst phase +1.00pp. But the level was a
favourable draw — **the honest figure is the median 21.56%, not 26.56%.**

**Out of sample by window — the advantage is not significant against live.**
Over 95 six-month windows pooled across three phases, `flip_neg` beat the live
composite in 62% of them at a median +6.60%, with **t = 1.34** — and that t is
optimistic, because pooled windows overlap across phases. By calendar period it
won **17 of 30**. The full-sample gap comes from a minority of windows winning
big. Worst single window: **-73pp** against live. The two most recent full
windows go heavily the other way — 2025-07 (live +22.0% vs -7.6%) and 2026-01
(live +51.1% vs -0.2%).

Against *owning the pool* it does clear the bar: t = 2.13 and 2.86.

**Score switching (TODO 0f) — closed, it does not work.** Selecting between the
scores on trailing performance returned 17.92% at 0.48 Sharpe against a median
fixed config's 0.54, and picked the truly-best config 29% of the time against
33% for a coin. The same answer walk-forward already gave for parameter
re-tuning, arriving again for score selection.

### The standing position

The phase test and the window test say different things and both are true: the
advantage is **not** an artifact of where the rotation clock started, and it is
**not** reliable window to window. A real but noisy edge whose full-sample
numbers oversell it.

That argues for a partial allocation rather than a switch, and it argues for
quoting the phase median. Survivorship (TODO 0d) remains unquantified and
accepted as a known risk by decision, not by evidence.

---

## The account was not running the model — and that dwarfs every signal result in this file

**Run 2026-09-20.** `scripts/analyze_ytd_gap.py` and `scripts/reconcile_trades.py`,
the latter against the account's transaction export (2026-01 to 2026-09).

The production backtest reports **+18.04% YTD**. The account's realised result was far below it. That divergence is larger than every effect measured anywhere in
this document, and until it was explained, every comparison here was a statement
about a simulation rather than about money.

### It is not the universe

The first hypothesis — that the backtest is inflated by ranking names only added
in the 2026-09-17 universe update — is **wrong, and backwards**:

| universe | YTD | full CAGR |
|---|---|---|
| current 46 names (what the backtest uses) | +18.04% | 17.21% |
| pre-update 52 names (what was actually held until September) | **+22.09%** | 22.34% |

The September update made the backtest **4.06pp worse**. Trading the old
universe should have earned more, not less.

### It is not execution either

It is that **the account and the model were holding different books**, for most
of the year.

| | modelled | actually held | overlap |
|---|---|---|---|
| 2026-01-06 | XOM LLY CAH STT | (private) | 0/4 |
| 2026-01-20 | XOM LMT PM ULTA | (private) | 0/4 |
| 2026-02-17 | GILD LMT XOM CNP | (private) | 0/4 |
| 2026-03-17 | PWR COST STX MPC | (private) | 1/4 |
| 2026-05-12 | STX PWR NVDA CAT | (private) | 0/4 |
| 2026-07-07 | LYV V PGR SPG | (private) | 2/4 |
| 2026-09-01 | MPC MSFT WELL KO | (private) | 3/4 |

Monthly overlap: **0%, 0%, 17%, 25%, 12%, 25%, 50%, 12%, 62%.**

### Where the year went

The backtest's YTD is five names — STX +7.64%, PWR +5.08%, MPC +5.05%,
LMT +3.96%, XOM +2.87% — which together are +24.6% of a +24.73% arithmetic
total. Everything else nets to approximately zero.

**Of that, +12.80% came from names never held at any point.** PWR and LMT alone
are +9.04pp, and neither appears anywhere in the trade history.

So more than half the modelled year came from positions that never existed in
the account.

### What this does and does not mean

**It is not evidence the model fails in practice.** It is evidence that
something other than the model was traded. Those are opposite conclusions and
the distinction was unavailable until the trade log arrived.

**The divergence is model-version drift, and the trades were IN model.**
Corrected 2026-09-20 on James's account of what happened: the early-year trades
followed the model as it ran at the time. They were not discretionary deviations
— they simply predate the config snapshots, so there is no record of what the
model said then, and today's code does not reproduce it. A 0/4 January overlap
compares the account against a retrospective book that never existed.

**Which makes the +18.04% YTD backtest circular.** The current live model exists
*because* the old one performed poorly over this very period: the scoring
defects were found and fixed in response to 2026's results. Backtesting today's
code over 2026 therefore measures a configuration selected with knowledge of the
outcome. The +18.04% was never available to anyone, and the realised result is what running
a model in real time actually produced.

This is the cleanest example in the repo of why "read deltas, not levels" is
written at the top of this document — and a caution that applies to Track J
too, which was developed today against data running through September 2026.

**What it does not license** is the comfortable reading that the gap is
somebody else's fault. Whether the book gets followed is still the variable with
the largest coefficient here: +12.80pp of the modelled year came from names
never held. The difference is that the fix is a stable model and a recorded
book, not more discipline.

### What is still unmeasurable

The transaction export has no starting balance, so the account's true return cannot be computed from it. Overlap and missed
contribution are the measurable parts. Anything claiming a headline account P&L
from this data would be invented, and the script says so rather than guessing.

### Reproducing

```
python scripts/analyze_ytd_gap.py
python scripts/reconcile_trades.py --trades <transactions.csv>
```

`reconcile_trades.py` writes its detail to the scratchpad rather than the repo:
account data does not belong in version control, and this repo's .gitignore
would swallow a stray CSV silently rather than flag it.

---

## Track K, Tier 3 — the hedge layer over a century: real assets do run in inflation, harvesting them early does not help, and a stock-weakness gate arrives late

**Measured 2026-09-23.** TODO 0l. `scripts/analyze_hedge_history.py`, layer in
`momentum/hedge.py`, data in `momentum/longhistory.py`. Monthly, 1927-01 to
2026-07. The stock book is Ken French's top prior-return decile, **not Track
J**: read the deltas between hedged and unhedged, not the levels. Checks (all
pass, printed by the script): every series against its traded instrument; zero
cap reproduces the stock book exactly; no look-ahead under truncation; returns
recomputed by hand.

> [!important] Two data constructions that would otherwise have flattered this
> **Monthly averages.** Pink Sheet prices and pre-1962 FRED yields are monthly
> averages. Crediting avg(t)->avg(t+1) to a position opened at the end of t
> leaks the second half of month t: that naive series correlates **0.55** with
> the ETF's return in the month *before* the position opened. The construction
> used credits avg(t+1)->avg(t+2) instead — leak **-0.06** (gold), matching the
> ETF's own autocorrelation — and splices in IAU/SLV/DBC month-end returns once
> they exist. A midpoint interpolation was tried first and rejected: it smooths
> twice and still leaks.
> **French "RlEst" is not REITs** (SIC 6798 sits in "Fin"): 0.79 with VNQ. The
> FTSE Nareit All Equity REITs series replaces it from 1972 (0.998 with VNQ).

### The hypothesis, before any rule: real assets run hard in inflation, and not in deflation

Buy-and-hold over each episode, against the stock book:

| episode | stocks | gold | silver | commodities | energy eq. | REITs | 10y UST |
|---|---|---|---|---|---|---|---|
| 1946-48 peg (CPI 11.8%) | +15% | fixed | — | +55% (PPI proxy) | +40% | -38% (not REITs) | +2% |
| 1973-74 stagflation | **-39%** | **+145%** | **+140%** | **+154%** | -30% | -32% | +2% |
| 1977-81 surge | +175% | +232% | +109% | +176% | +106% | +123% | -7% |
| 2008 GFC | -51% | +18% | -10% | -35% | -41% | -65% | +18% |
| 2021-22 | **-15%** | -15% | -28% | **+71%** | **+176%** | +5% | -22% |

James's hypothesis holds for **commodities, energy and precious metals in
inflationary episodes**, by wide margins. It does **not** hold for REITs
(1973-74 -32%) and it inverts in deflationary crashes, where everything but
Treasuries falls with stocks. Energy *equities* failed in 1973-74 while
commodities and bullion tripled — the direct instrument mattered exactly when
it was needed.

### What the layer does with it

Pre-registered default (lookback 3m, 25% slots, 75% cap, correlation gate,
cash eligible), and the post-hoc variants that matter. Episode columns are
hedged minus unhedged cumulative return; `2011-26` and `calm` are CAGR deltas.

| variant | dCAGR | dSharpe | 2011-26 | calm 2011-19 | % months hedged | turnover/yr | 1946-48 | 1973-74 | 1977-81 | 2008 | 2021-22 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **default L=3** | -1.0% | +0.02 | -3.7% | -8.2% | 74% | 276% | -2.8% | +89% | +50% | +25% | +53% |
| L=1 | +0.3% | +0.10 | -3.6% | -7.6% | 76% | 457% | -3.9% | +80% | +185% | +29% | +27% |
| L=1 danger 6m | -0.7% | +0.00 | -4.3% | -3.1% | 27% | 197% | -1.1% | +35% | +11% | +13% | +3% |
| L=3 harvest +50% | -1.4% | +0.01 | -3.2% | -8.0% | 73% | 275% | -3.2% | +68% | -29% | +25% | +61% |
| **L=3 enter +10%** | **-0.0%** | +0.03 | **-1.2%** | -3.8% | 48% | 136% | +3.3% | +81% | +38% | +16% | +36% |
| L=6 enter +10% | -0.3% | +0.04 | -2.6% | -3.4% | 58% | 110% | +1.0% | +68% | +119% | +18% | +25% |

**1. The competition alone is a permanent allocation, not a hedge mode.** With
seven candidates, one beats the stock book over three months by chance most of
the time: hedged 74% of months, and calm decades pay for it (2011-19 -8.2pp,
Sharpe 0.83 -> 0.41). The inflation payoffs are real and large; so is the drag.

**2. Gating on stock weakness arrives late — the inflation trade starts while
stocks are still rising.** The danger gate (hedge only while the stock book
trails cash) cuts months hedged to 27% and the calm cost to -3.1pp, but first
hedges 2021-22 in **June 2021** against January for the default, and realises
-2.7% per unit hedged there against +85%. 1977-81 falls from +50% to +11%.
Early entry has to come from hedge *strength*, not stock *weakness*.

**3. Harvesting at a target does not help, and the reason is the shape of the
runs.** A +25% or +50% target, or a z >= 2 "topping expectations" exit, cuts
1977-81 from +50% to +17% / -29% / -5% and 1973-74 from +89% to +60% / +68% /
+21%. The runs that matter were long (21-57 months) and fat-tailed (silver
+784% in 1977-81), so a fixed target sells the tail that makes the episode.
The relative-strength exit already does what the harvest was meant to do:
**the slot returns to stocks when the stock book's trailing return overtakes
the hedge's**, which in 1974, 1980 and 2022 is close to the stock bottom.

**4. Demanding decisive strength is the better filter.** An entry margin
(`enter_margin`, the hedge must beat the stock book by 10% over 3 months)
keeps early entry, halves turnover, cuts months hedged to 48%, and costs
**-1.2pp over 2011-26** — inside the agreed 5pp bar — for a full-period CAGR
delta of zero. The margin variants form a smooth trade-off between calm cost
and episode capture, not a spike, and none of them was tuned.

**5. 1946-48 is a data limit, not a verdict.** With no commodity series before
1960 the layer had energy, operators, pegged bonds and cash, and lost 2.8%.
Backfilling commodities with PPI (+55% over the episode) turns the default to
**+8.9%**. The episode closest to today's debt situation is also the one the
data supports least.

### What this does not establish

- **Anything about Track J.** The stock book here is 10% of the market. Track
  J's own score may already rotate into energy in an inflationary regime (its
  combined book was 7/29 energy on 2026-09-20), which would shrink the layer's
  marginal value. Tier 1 answers that.
- **A choice among variants.** 50 variants against six episodes: the table is a
  shape check. The robust statements are 1-4 above, not any single row.
- **Pre-1975 gold was not holdable by US persons**; silver and commodities
  carried 1973-74 regardless (gold averaged 14% of the book there).
- **Monthly decisions.** The agreed cadence is weekly; Tiers 2 and 1 run it.

---

## Track K, Tier 2 — on the real ETFs the layer halves drawdown for 1.4pp, and V-shaped rebounds are what it costs

**Measured 2026-09-23.** `scripts/analyze_hedge_etfs.py`. Daily, 2006-07 to
2026-07, instruments from the new daily store (`data/market/`), the stock book
still Ken French's top momentum decile (not Track J). Checks pass: French
market vs SPY 0.991 daily with identical calendars, BIL within 0.12%/yr of the
T-bill, zero cap exact, no look-ahead, hand recompute.

Candidate, fixed before the run as Tier 3's shape in sessions: lookback 63,
entry margin +10%, 25% slots, 75% cap, weekly decisions, **trades at the next
session's close**, 10bps.

| | stock book | candidate | delta |
|---|---|---|---|
| CAGR 2006-26 | 13.5% | 12.1% | **-1.4pp** |
| Sharpe (over T-bill) | 0.55 | 0.58 | +0.03 |
| MaxDD | -56.4% | -30.1% | +26.3pp |
| **2011-26 CAGR (the 5pp bar)** | | | **-4.8pp** |
| calm 2011-19 | | | -4.2pp |
| days hedged / turnover | | 59% / 329% a year | |

| episode | delta | what it held |
|---|---|---|
| 2008 H1 commodity spike | +15.7% | commodities, silver, gold |
| 2008 H2 crash | +23.2% | **dollar, TLT, IEF** — out of commodities by 2008-07-11 |
| GFC to the low | +39.1% | |
| 2009 rebound | -17.0% | |
| 2020 COVID crash | +3.3% | only 10% hedged: too fast for a 63-day comparison |
| **2020 rebound** | **-45.8%** | TLT, IEF, gold held while the book rose 77% |
| 2021-22 inflation | +37.5% | energy, commodities, REITs, dollar |
| 2015-16 commodity crash | -7.3% | |
| 2025-26 metals run | -7.0% | gold, silver |

**1. The exit-speed test passes.** 2008 was the case monthly data could not
see: commodities spiked and crashed within six months. The layer rode the
spike and was in the dollar and Treasuries for the crash.

**2. The cost is V-shaped rebounds, not calm markets.** Right after a crash the
book's 63-day return is deeply negative, so hedges keep "beating stocks by
10%" for months into the recovery. 2020's rebound alone cost 46pp.

**3. Handing back fast does not fix it — it redistributes.** A post-hoc rule
returning the slot to stocks once the book beats the hedge over 10-21 days
cuts the 2020 rebound loss to -16..-36pp, but gives back crash and inflation
protection nearly one for one (GFC +39% -> -6..+29%; 2021-22 +38% -> +1..+26%).
A short window cannot tell a V-rebound from a bear-market rally, and 2008 was
full of the latter. No setting forms a plateau; the best-looking row (10d,
21-session blackout) sits between neighbours with worse drawdown than no hedge.

**4. The trailing-stop harvest is inert or fragile.** 15-20% trails almost
never fire at these slot sizes. 10% with a 21-session blackout is the best row
in the table (-0.4pp, Sharpe +0.07, and the 2025-26 metals run turns from -7%
to +7%), but the same trail with a 63-session blackout is -1.2pp with a -41%
drawdown. A spike, not a shape; not adopted.

**5. The stock-bond correlation gate never mattered.** Over 756 sessions the
correlation only turned positive in 2026. At 126 or 252 sessions nothing
changes either: in 2022 TLT dropped out because it lost to cash, which is the
competition doing the gate's job, as designed.

**6. Cadence: weekly to monthly are indistinguishable; daily is worst.** Across
every phase: daily -1.9pp; every 5 sessions -0.8pp (range -1.6..+0.9); 10
sessions -0.5pp; 20 sessions -0.2pp (range -2.6..+3.1). The phase spread is as
large as the cadence effect, so weekly stays.

**7. The book matters more than any knob.** The same candidate over SPY adds
**+1.3pp CAGR and +0.20 Sharpe**, halves drawdown (-55% -> -22%) and costs
-1.9pp over 2011-26. Against a high-return momentum book it costs; against the
market it pays. Track J is a momentum book, which is why Tier 1 decides this.

### Where this leaves the design

The candidate stands, inside the 5pp bar but with little room (-4.8pp over
2011-26). What it buys is drawdown and the inflation and crash episodes; what
it pays is the months after a V-shaped bottom. Entry margins of 10-15% form the
plateau (+15%: -0.3pp full, -3.9pp 2011-26). Nothing tested here solves the
rebound problem without surrendering the protection, and that should be read as
the price of the insurance rather than a defect waiting for a parameter.

---

## Track K, Tier 1 — on Track J the layer fails the bar: Track J already hedges itself, and its recoveries are what the layer sells

**Measured 2026-09-23.** `scripts/analyze_hedge_trackj.py`. The stock book is
Track J's own four-sleeve return stream (`trackj_portfolio_performance.csv`,
asserted to reproduce TODO 0k's 20.81% / 0.73 / -42.99%). Instruments from
the daily store; results 2012-01 to 2026-09-18. Checks pass: calendars agree to
the session, zero cap exact, no look-ahead, hand recompute. Correlation window
252 rather than 756 because Track J's stream starts 2011-10 (Tier 2 measured
the two as equivalent).

| variant | dCAGR | dSharpe | dMaxDD | calm 2012-19 | 2020-26 | % hedged |
|---|---|---|---|---|---|---|
| **candidate (Tier 2)** | **-7.7pp** (all 5 weekly phases: -5.5pp, range -7.7..-4.0) | -0.20 | +1.5pp | -4.2pp | -12.3pp | 49% |
| entry +15% | -5.0pp | -0.11 | -1.5pp | -3.5pp | -7.1pp | 38% |
| L=126, entry +20% | -4.4pp | -0.10 | +7.8pp | -2.1pp | -7.6pp | 38% |
| cap 50% | -5.1pp | -0.09 | +1.0pp | -3.3pp | -7.5pp | 49% |
| hand back 10d | -3.8pp | -0.08 | 0.0pp | -3.5pp | -4.3pp | 24% |
| signal judged on SPY, scaling Track J | -5.0 to -6.3pp | -0.08 to -0.16 | -1 to +5pp | | | 34-68% |

Every variant costs Track J 3.8-10.3pp of CAGR with Sharpe falling, and none
buys meaningful drawdown. The only row inside 5pp is the hand-back setting
Tier 2 found to be a spike, and it protects nothing (dMaxDD 0.0).

**1. Track J already does the inflation part.** In 2021-22 the French momentum
decile lost 24%; Track J **gained 39%**. Controlling for SPY, its rolling beta
to XLE rose to +0.24 in 2022 and to DBC +0.19: the score rotated into energy
and commodity producers on its own, as its 7/29-energy book on 2026-09-20
suggested. The episode that paid for the layer on every proxy book pays
nothing here (2021-22 delta -3.8%).

**2. Track J mean-reverts at the horizon the layer compares on.** Next-63-session
return by quintile of the past 63 sessions, non-overlapping, 2012-2026:

| book | corr(past, next) | after worst quintile | ... | after best quintile |
|---|---|---|---|---|
| Track J | **-0.22** | +7.1% | +6.6% / +5.2% / +3.0% | +1.5% |
| French top decile | -0.07 | +7.8% | +1.8% / +0.6% / +6.7% | +4.1% |
| SPY | -0.15 | +5.1% | +3.4% / +2.3% / +2.5% | +3.6% |

Track J's profile is monotone: its weak stretches precede its strong ones —
the recovery gradient found in the hold-vs-swap work, seen from the book level.
A competition against Track J's own trailing return hedges exactly before those
recoveries: -37pp in 2019, -40pp in the 2020 rebound, -75pp across 2023-24.

**3. Judging on the market instead does not rescue it.** With SPY as the
signal (Tier 2's best book) and Track J scaled, the cost is still -5.0 to
-6.3pp with Sharpe down. The arithmetic is plain: moving 35-50% of a book that
compounds near 21% into assets that compound 6-8% costs about 5pp a year, and
only a real inflation or debasement episode repays it. The one in this window,
Track J handled itself.

### What this does and does not say

It says the hedge layer as designed should not sit on Track J: it fails the
agreed 5pp bar and Sharpe falls, which the objective treats as a real
constraint. It does NOT say Track J is protected against a 1970s-style or
debasement regime — 2012-2026 contains no such episode, and whether Track J's
score would rotate into real assets fast enough in one is untested. Tier 3's
finding stands: in 1973-74 commodities and bullion tripled while energy
EQUITIES fell 30%, so a book that hedges through producers can still miss.
Survivorship inflates Track J's level, which overstates the hedge's measured
cost; the direction is conservative, the size is unknown (TODO 0d).

---

## Track K option (b) — real-asset ETFs inside the Track J pool: inert, because the score never wants them

**Measured 2026-09-23.** `scripts/analyze_trackj_real_assets.py`. Track J run
exactly as the live runner builds it (reproduces its stored stream to 1e-16),
then with seven component ETFs added to the pool: DBO (oil), UNG (natural gas),
DBA (agriculture), DBB (base metals), SLV, IAU (already present) and UUP. One
per component, chosen so they do not crowd each other out of the correlation
cap (pairwise 0.02-0.42; DBC left out at 0.90 with DBO). They bypass the $10M
volume floor as IAU/SHY/TLT already do; the slippage model still charges them
on real volume. A plumbing check — the same augmented panel with the
components unselectable — reproduces Track J exactly.

| arm | dCAGR | dSharpe | real-asset weight |
|---|---|---|---|
| + components | -0.43 / -0.41pp (two phases) | -0.02 | 0.5% |
| + components, exempt from the correlation cap | -0.41 / -0.38pp | -0.02 | 0.5% |

**The score does not want them.** On the pullback score these ETFs sit in the
bottom half of the cross-section — median percentile 54% (gold) to 90%
(natural gas) — and make Track J's top 8 on 0-1.7% of rotation dates. In
2021-22 they held 0.3% of the book. The correlation exemption changes nothing
(`CorrelationConfig.exempt_symbols`, added for this, default empty) because the
cap was never what kept them out: pullback-in-a-12-month-uptrend is a
stock-shaped signal, and low-volatility commodity baskets rarely reach the top
1.3% of 610 names on it.

**Confirmed along the way (James asked):** sleeves select independently and
the correlation cap applies within a sleeve only, so a name that re-qualifies
is held by several sleeves and weighted by the count. The live combined book
on 2026-09-15 is 32 slots over 29 names; GOOGL, MU and FDX sit in two sleeves
each at 6.25%.

### Where Track K lands

Every way of making Track J carry real assets has now been measured: as an
overriding layer (Tier 1: -5.5pp, Sharpe down) and as candidates for its own
score (option b: inert). The layer protected strongly in the index-level
inflation episodes (Tier 3: 1973-74 +89%, 1977-81 +50%) and on SPY (Tier 2:
+1.3pp, +0.20 Sharpe), but the one inflation episode in Track J's record,
Track J handled itself. **Track K is a preparation model for a regime that is
not in the record** — option (a) in TODO 0l.

---

## Open questions

1. **`velocity_window=5` is an in-sample choice.** Walk-forward proved re-tuning
   is harmful, so it should be left alone — but "don't re-tune" is not "this is
   correct". If live results drift from backtest, look here first.
2. **Survivorship bias.** Today's screened universe applied back to 2010 inflates
   every absolute number. Point-in-time snapshots now accumulate from
   2026-07-25 forward; unbiased backtesting becomes possible as they build up.
3. **Equal-weight drift.** Returns assume a costless daily rebalance back to
   equal weight. Track I tested *alternative target weights* under this same
   convention and found nothing better; it did not test relaxing the
   convention itself, which remains open. It slightly understates a runaway winner's contribution. Kept
   for comparability with all historical results, but it is an approximation.
   Note this biases the outsized-event analysis in the *reassuring* direction:
   a real book would let a winner run to more than its 1/N weight, so the true
   tail contribution is somewhat larger than the 8-15% measured. The effect is
   bounded by the hold period (14 days), not by the life of the position.
4. **Nothing found improves the signal.** Every timing idea tested failed
   because the composite does not produce timely information — rank decay is
   directionless, challengers arrive too late, acceleration is noise. If there
   is more return available, it is likely in the universe (better candidates)
   rather than in the timing of trades among current candidates.

5. **The one signal that measured positive is out of reach of the universe.**
   Track J found `rank(12-1 month) - rank(3-month)` earns +0.99% per 14-day
   period at t = 2.66 on a 747-name pool — and +0.13% at t = 0.43 above $50B
   market cap, which is where the live universe lives. Item 4 said remaining
   upside is in the candidate set; this is the first measurement that says so
   with a direction attached, and it points somewhere the current screen does
   not go. Whether that is worth following depends on TODO 6 (survivorship) and
   on whether a smaller-cap book survives realistic fills — neither is answered.
