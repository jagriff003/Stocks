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
| Model health monitor | **built** | diagnostic, not a return change |

Live model: **19.84% CAGR, 0.91 Sharpe, -19.23% max drawdown, Calmar 1.03**,
net of realistic fills and costs. The honest like-for-like starting point was
12.80%.

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

## Open questions

1. **`velocity_window=5` is an in-sample choice.** Walk-forward proved re-tuning
   is harmful, so it should be left alone — but "don't re-tune" is not "this is
   correct". If live results drift from backtest, look here first.
2. **Survivorship bias.** Today's screened universe applied back to 2010 inflates
   every absolute number. Point-in-time snapshots now accumulate from
   2026-07-25 forward; unbiased backtesting becomes possible as they build up.
3. **Equal-weight drift.** Returns assume a costless daily rebalance back to
   equal weight. It slightly understates a runaway winner's contribution. Kept
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
