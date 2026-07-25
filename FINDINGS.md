# Momentum strategy — findings log

Measured 2026-07-25 over 2010-2026, 52-ticker universe, `next_open` execution
with 7.5 bps one-way slippage unless stated. Every number here is reproducible
from a named command; nothing is from memory.

Read **deltas, not levels**. Absolute figures carry survivorship bias from
applying today's screened universe backwards. The difference between two
variants over the same universe does not.

---

## Bottom line

The three tracks proposed in the design document were all tested and all
rejected on evidence. The gains that did materialize came from fixing two
measurement defects and one mis-specified parameter, not from new features.

| Change | Status | CAGR effect |
|---|---|---|
| Level/velocity scale mismatch | **fixed** | prerequisite — made weights meaningful |
| Execution assumption (same-close → next-open + slippage) | **fixed** | -3.41pp, but honest |
| `velocity_window` 10 → 5 | **adopted** | +6.22pp |
| Correlation filter, gated above VIX 25 | **adopted** | -0.01pp CAGR, +1.32pp drawdown |
| Track A — graduated VIX ladder | **rejected** | -2.8 to -9.8pp |
| Track B — rank exits / score-gap swaps | **rejected** | -1.2 to -10.5pp; best is break-even |
| Track C — acceleration, earlier entry | **rejected** | -0.8 to -9.3pp |

Live model: **19.84% CAGR, 0.91 Sharpe, -19.23% max drawdown, Calmar 1.03**,
net of realistic fills and costs. The honest like-for-like starting point was
12.80%.

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
4. **Nothing found improves the signal.** Every timing idea tested failed
   because the composite does not produce timely information — rank decay is
   directionless, challengers arrive too late, acceleration is noise. If there
   is more return available, it is likely in the universe (better candidates)
   rather than in the timing of trades among current candidates.
