# Runbook — operating the models

## TL;DR — every run, in order

Rotation Tuesdays (09-29, 10-13, 10-27, …), after 16:15 ET.
PowerShell:

```powershell
# 0. setup, once per terminal
cd $env:USERPROFILE\OneDrive\Analytics\Stocks
$py = ".\.venv-1\Scripts\python.exe"

# 1. production (the traded model until Track J graduates)
& $py scripts\run_live.py

# 2. Track J: fresh prices, saves charts
& $py scripts\run_live_trackj.py --no-cache --save-charts charts

# 3. Track J + Track K together: allocation, charts, logs the recommendation
& $py scripts\run_live_combined.py
```

4. **Check:** every `Signal session` / last-session date is **today**, and each
   script exited 0. If not, see part 5.
5. **Decide** (part 4 if Track K is FIRING). **Trade at Wednesday's open.**
6. **Record and commit:**

```powershell
& $py scripts\record_decision.py --action "what you did" --note "why"
git add snapshots data/market; git commit -m "Rotation YYYY-MM-DD"
```

Off-cycle days: steps 0, 2, 3 only. Nothing to trade: CURRENT is information.

---

For James, at the keyboard. Parts 1–4 are the routine and the decisions; part 5
is what to do when something looks wrong; part 6 is reference. The evidence
behind every statement here is in FINDINGS.md — this file says what to do, not
why it was concluded.

Reorganised 2026-09-23 when Track K joined. The health-monitor ladder, run-timing
measurements and scheduler notes that used to open this file are unchanged in
part 6.

---

## 1. Where things stand

| model | what it is | status | run with |
|---|---|---|---|
| **Production** | 46 curated names, top 4, VIX overlay | **the model actually traded** until Track J graduates | `scripts/run_live.py` |
| **Track J** | pullback score on ~745 liquid names, 4 sleeves of 8, one sleeve rotating every other Tuesday | baseline, graduating to production over the weeks after 2026-09-23 | `scripts/run_live_trackj.py` |
| **Track K** | preparation model for an inflation / debasement regime: hands slots to real assets when its trigger fires | **discretionary input** — informs, does not override | `scripts/run_live_combined.py` |

- **Calendar.** Track J (and Track K with it) rotates every other **Tuesday**,
  anchored 2026-09-15 — so 09-29, 10-13, 10-27, … Build on Tuesday's close,
  trade at **Wednesday's open**. Only one Track J sleeve (a quarter of the book)
  trades per rotation. Production prints its own next rotation date; it is
  anchored to the same Tuesdays.
- **Account.** Tax-advantaged, so K-1s and the collectibles rate do not bind.
  Compliance: nothing data-center related, enforced by `restricted.csv`.
- **Nothing here trades for you.** Every script prints; you place the orders.

---

## 2. The routine

### Copy-paste setup

Every block below is **PowerShell** (the VS Code terminal) and calls the venv's
python directly, so nothing needs activating. Paste this once per terminal:

```powershell
cd $env:USERPROFILE\OneDrive\Analytics\Stocks
$py = ".\.venv-1\Scripts\python.exe"
```

### Rotation Tuesday — after 16:15 ET

```powershell
# 1. production, while it is the traded model (charts on screen)
& $py scripts\run_live.py

# 2. Track J: fresh prices, charts saved to charts\, writes its books for step 3
& $py scripts\run_live_trackj.py --no-cache --save-charts charts

# 3. both models: refreshes the data store, prints Track J + Track K + the allocation,
#    saves the Track K and allocation charts, and opens them WITH Track J's three
& $py scripts\run_live_combined.py
```

Step 2 saves quietly and step 3 opens all five charts together. To see Track
J's charts at step 2 instead, add `--show` (`--save-charts charts --show`).
Close the chart windows to let each script finish.

Then:

4. **Check the dates before anything else.** Each script prints a `Signal
   session:` (or `last session`) line. It must be **today**. If not, re-run
   with `--no-cache` — never trade a stale book (part 6, "If the panel is short").
5. **Check the exit codes.** `0` clean; `2` printed but something was skipped
   or is stale — read the block at the end before trusting it; anything else is
   a failure. Part 5 has the fixes.
6. **Read the combined report** (part 3) and **decide** (part 4).
7. **Trade at Wednesday's open.** Only the rotating Track J sleeve, plus any
   Track K change you chose to act on.
8. **Record what you did** — no need to re-run anything:
   ```powershell
   & $py scripts\record_decision.py --action "rotated sleeve 0; skipped Track K" --note "oil crowded"
   & $py scripts\record_decision.py --show
   ```
   The combined report already logged its recommendation when it ran; this
   fills in your action against that row. `--show` prints the recent log.
9. **Commit the record** (the pool and config snapshots are source, not output
   — they only have value if they accumulate). The decision log is personal and
   git-ignored; it stays in the folder, which is backed up:
   ```powershell
   git add snapshots data/market; git commit -m "Rotation 2026-09-29"
   ```

### Other days — optional

Running Track J and the combined report off-cycle is safe and is logged:

```powershell
& $py scripts\run_live_trackj.py --no-cache --save-charts charts
& $py scripts\run_live_combined.py
```

To look without adding a row to the decision log, add `--no-log` to the second
line. The report then shows **two** recommendations for each model:

- **AT THE LAST ROTATION** — what you should be holding now.
- **CURRENT** — what each model would say if today were a rotation.
  **Information, not a trade.** Trading it off-cycle is a different strategy
  that was never tested, for either model.

Useful when the news is loud and you want to see whether Track K agrees.

### Monthly

```powershell
& $py scripts\update_market_data.py --verify    # full-history audit of the daily store; writes nothing
& $py scripts\update_market_data.py --long      # refresh the research data (French, World Bank, FRED, Nareit)
git add data; git commit -m "Monthly data refresh"
```

Both exit 2 when something needs a look (part 5). Commit `data/` afterwards.

### Quarterly

`& $py scripts\screen_universe.py` — the production universe re-screen, as
before.

---

## 3. Reading the combined report

Top to bottom:

**Header.** Store updated to *date*; Track J books from *date*. They must match.
A mismatch prints a `***` warning and exits 2 — re-run `run_live_trackj.py`.

**TRACK K.** For the last rotation and (off-cycle) for today:

- `Track K is quiet` — nothing to do. It is quiet most of the time: 3% of months
  over 1927–2026 outside the inflation episodes.
- `Track K is FIRING — N consecutive firing decision(s)` — the trigger holds:
  **at least 2 of gold, silver, commodities, energy beat SPY by 10% over three
  months and beat cash, and the 1-year stock-bond correlation is positive.**
  The count matters (part 4).
- The table lists every Track K asset: live ticker, 3-month return, excess over
  SPY, whether it qualifies (`yes`), and `*` for the four trigger assets.

**TRACK J.** The sleeve bought at the last rotation, the next rotation date and
sleeve, and what that sleeve would buy on today's close.

**CHARTS** (saved to `charts\` as `YYYY-MM-DD_combined_*.png`, opened with
Track J's three):

- `_combined_trigger` — top: each trigger asset's 3-month return over SPY
  against the dashed +10% line; bottom: the stock-bond correlation against 0.
  Shaded spans are when it fired; the dotted line is the last rotation. Read it
  for *how close* the trigger is and *how long* it has been firing.
- `_combined_allocation` — the recommended book(s) as 100% bars, with Track J's
  energy & materials split out so the overlap with Track K is visible.

**RECOMMENDED ALLOCATION.** One table per recommendation: symbol, the Track J
and Track K shares, the combined weight and dollars at $100k. While Track K is
quiet this is simply Track J. When it fires, each qualifying real asset takes
25% (up to 75%) and Track J's names shrink pro-rata. A symbol held by several
Track J sleeves appears once with the summed weight (that stacking is
deliberate). Off-cycle, the last line lists the difference between the two
recommendations.

---

## 4. Deciding — especially when Track K fires

Track J's book is the default. Track K is a second opinion about the
**environment**, built for a regime that is not in Track J's record. Obeying it
mechanically on Track J over 2012–2026 cost 0.3–0.7pp a year; its value, if
any, is in a 1970s-like grind the record does not contain.

When it fires, the questions the evidence says to ask:

1. **First firing, or confirmed?** It flickers — spells of 1–21 sessions since
   2021. Requiring two consecutive firing rotations halves the on/off changes
   and cost nothing historically. The report says when a firing is the first.
2. **Is Track J already there?** Its score drifts into producers on its own: on
   2026-09-23 Track J was 44% energy and materials, and following Track K on top
   would have made ~72%. Taking Track K's slots on top is doubling a bet, not
   hedging one. Reasonable responses: take only the Track K asset Track J does
   not already hold; or, as you put it, stay with Track J and trim the sleeves
   least aligned with the regime.
3. **Is it early or late?** In 1973 and 1977 it fired with 2–21% of the runs
   done and the assets then more than doubled. In 2022 it fired with 66–81% of
   the runs done; gold and silver had peaked in 2020. Since 2012, the basket it
   held has usually trailed SPY over the next quarter. Crowding is visible in
   the news before it is visible in a 63-session return — the model cannot see
   "pundits doubling down"; you can.
4. **Is this a slow regime or a fast crash?** Track K is built for grinds
   (1973–74 took 21 months). It cannot see a 23-session crash like 2020 and was
   designed not to react to one.

Whatever you choose, record it with `record_decision.py`. After a few episodes
the log answers whether discretion helped — which nothing else in this repo can.

**What not to do** (each of these was tested and lost):

- Trade either model's CURRENT recommendation off-cycle because it looks better
  than what you hold.
- Put Track K in charge of Track J continuously: −5.5pp a year, because it
  hedges just before Track J's recoveries (Track J mean-reverts at this horizon).
- Take profits on Track K assets at a target: the runs that mattered were long
  and fat-tailed, and fixed targets sold the tail.

---

## 5. When something looks wrong

### Exit codes (every script)

| code | meaning | do |
|---|---|---|
| 0 | complete | nothing |
| 2 | printed, but something was skipped, stale or flagged; the output ends with the reason | read the reason before trusting the book |
| other | failed outright — including a restricted name reaching a panel, which is deliberate | fix the cause; never suppress the restricted-name failure |

### Common cases

| symptom | cause | fix |
|---|---|---|
| `Signal session` is not today | stale price cache, or a provider failure | re-run with `--no-cache` |
| combined report: `Track J's books are from … but the store ends …` | Track J not re-run today | run `run_live_trackj.py --no-cache --no-plots`, then the report |
| combined report: `live/trackj_book.json missing` | Track J never run on this machine since the export was added | run `run_live_trackj.py` |
| combined report: `rotation calendars disagree` | anchor or calendar changed in one place only | stop and look — both must use the 2026-09-15 anchor |
| `record_decision.py` refuses: `already records …` | that row already has an action | `--overwrite` only if the first entry was a mistake |
| combined report: `Track J charts shown alongside: 0` | Track J ran without `--save-charts charts` | re-run step 2 with it, or ignore — the report is complete without them |
| a script seems hung | a chart window is open and waiting | close the chart windows |

### Market data store statuses (`update_market_data.py`, and the report's header)

| status | meaning | do |
|---|---|---|
| `ok` | appended, overlap agreed | nothing |
| `FILLED` | Yahoo had no bar for a session SPY has; carried flat, move kept the next day | nothing; if Yahoo posts the bar later it appears as a revision |
| `REVISION REFUSED` | a stored return changed at source (late dividend, filled hole) | read the detail, then `--accept-revisions` |
| `STALE` | symbol ends before SPY | usually Yahoo lag; re-run later |
| `GAP` | sessions missing inside the history | investigate before trusting the series |
| `BIG MOVE` | a daily return beyond 20% | confirm against a second instrument (SLV 2026-01-30 −28.5% is real) |

`--add SYM` starts tracking a new symbol with its full history.

---

## 6. Reference

### Where things live

| path | what | in git |
|---|---|---|
| `data/market/daily_returns.csv` | daily total returns for every Track K instrument; what the report reads | yes — revisions show as diffs |
| `data/market/update_log.csv` | every store update and its checks | no — personal run history |
| `data/decisions/decision_log.csv` | each report's recommendation, and what you did | **no — personal**; lives in the backed-up folder |
| `private/` | account-level material (reconciliations, trade exports) | **no — never** |
| `data/longhistory/` | 1926– research panel (built) and raw vintages (`raw/`, never deleted) | panel yes, raw no |
| `snapshots/pool/`, `snapshots/config/` | point-in-time pool and config per Track J run — the only thing that can ever settle survivorship | **yes — commit them** |
| `live/trackj_book.json` | Track J's books for the combined report | no (regenerated) |
| `charts/` | `--save-charts` and combined-report charts, dated | no |
| `FINDINGS.md` / `TODO.md` | what was measured / what is open | yes |

### When to run on a rebalance day

Measured 2026-09-15 by sampling Yahoo's daily bar every five minutes across the
close (`analyze_run_date_sensitivity.py` covers the cost side; the settle timing
was a one-off probe).

**The bar settles within minutes of the close.** Prices ticked continuously
until 16:00 ET, froze at 16:05, and the last consolidation adjustments — under
4 bps on every name sampled, zero on most — landed by 16:15 ET.

| | |
|---|---|
| Safe to run | **16:15 ET onward**, same day |
| Max drift after 16:05 ET | 3.5 bps (BR); SPY/AAPL/MSFT 0.0-0.7 bps |

So there is no need to wait for the following morning. Run any time after the
close on the rebalance date and the panel carries that date's close.

#### The gap that actually matters

Not script-run-to-trade. It is **the close the model ranked on, to the open you
fill at** — and the target is to *hit* the modelled one overnight
(`close(T) -> open(T+1)`), not to shrink it. Shrinking it further is
`same_close`, which is unachievable; the overnight suite prices that difference
at 20.90% vs 20.05% CAGR, so ~85 bps/yr is the cost of the overnight you are
structurally stuck with.

Stretching it costs roughly the same per session from either end:

| sessions | stale data (fixed trade date) | late fill (correct book) |
|---|---|---|
| 1 | -9.9 bps (t -1.26) | -4.0 bps (t -0.69) |
| 2 | -18.4 (t -1.77) | -11.3 (t -1.46) |
| 3 | -27.0 (t -2.33) | -31.4 (t -2.98) |

One session either way is inside the noise. Read the **dispersion**, not the
mean: a one-session-stale panel picks a different name **56% of the time**, with
outcomes from -264 to +214 bps. It is not a slightly worse book, it is a coin
flip on roughly one position in four.

#### If the panel is short

On 2026-09-03 a run at 22:04 ET — six hours after that bar had settled —
produced a panel ending 2026-09-02, and nothing in the output said so. That was
a provider or request failure, not a timing one; waiting longer would not have
fixed it.

`load_data` now warns on both failure modes, and `run_live.py` prints
`Signal session:` next to the run date. **If those two dates are not the session
you mean to trade on, do not trade the book** — re-run with `--no-cache`. The
cost of one stale session is small in expectation and wide in outcome, which is
exactly the combination not worth accepting when re-running is free.

### Running unattended

`run_live.py` can run on a schedule, but not in its default form: it ends at
`plt.show()`, which blocks forever when there is no window to close. A
scheduled task that looks hung is almost always this.

| goal | command |
|---|---|
| interactive, charts on screen | `python scripts/run_live.py` |
| scheduled, charts kept as PNGs | `python scripts/run_live.py --save-charts charts` |
| scheduled, no charts at all | `python scripts/run_live.py --no-plots` |

`--save-charts DIR` writes `YYYY-MM-DD_performance.png`, `_held-book.png` and
`_context.png` into `DIR`, creating it if needed, and forces a non-GUI
matplotlib backend so nothing tries to open a window. The same flags work for
`run_live_trackj.py`. (Until 2026-09-23 a stale chart block made every charted
Track J run exit 2; fixed.)

A daily scheduled combined report is fine — it logs every run, so the decision
log also becomes a daily record of Track K's reading. Scheduled runs must not
open windows:

| goal | command |
|---|---|
| Track J, scheduled | `scripts\run_live_trackj.py --no-cache --save-charts charts` (never `--show`) |
| combined, scheduled | `scripts\run_live_combined.py --no-show` (charts saved, no windows) |
| combined, no charts | `scripts\run_live_combined.py --no-plots` |

The context panel, the health monitor and the charts are each allowed to fail
without stopping the run — the book is still worth having when the context
panel cannot fetch a series. On a schedule nobody reads the log, so **read the
exit code**; a task that ignores it will happily report success on a run that
skipped the health monitor for a month.

#### Windows Scheduler

Point the action at the venv's python directly rather than at a shell, so no
console is needed:

```
Program:   %USERPROFILE%\OneDrive\Analytics\Stocks\.venv-1\Scripts\python.exe
Arguments: scripts\run_live.py --save-charts charts
Start in:  %USERPROFILE%\OneDrive\Analytics\Stocks
```

Schedule it **after the close** — the panel must contain the session you intend
to trade. Check `Signal session:` in the output against the date you expect
before acting on any scheduled run's book; everything above about stale panels
applies exactly as much when a machine ran it.

### The health monitor — what to do when it says something

Written 2026-09-03, while nothing was wrong. That is the point: a decision made
ten months into a shortfall is a worse decision than the same one made now, and
the documented failure mode of this strategy is re-tuning under discomfort.

**Scope:** calibrated on the 46-name production model. `run_live_trackj.py`
displays it for Track J, but it is not validated there (TODO 3). Every number
below is reproducible from `python scripts/monitor_health.py`.

#### First, the base rates

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

#### The states

Printed on every `run_live.py`, and in full by `monitor_health.py`.

| State | Condition | Meaning |
|---|---|---|
| **OK** | neither breach | nothing |
| **WATCH** | breach today, not yet sustained; or the warn pair sustained | note it, no action |
| **ALARM** | raw z ≤ -1.25 **and** excess ≤ -12.5%, for 20 consecutive sessions | work the ladder below |

ALARM fired twice in eleven years (2019-07, 2021-08). It is rare by
construction, and its rarity is the only property of it that calibrated — see
`momentum/health.py` on why it must not be read as predictive.

#### The ladder

Work it in order. Each step is cheaper and more likely to be the answer than the
one after it. **Do not skip to step 4.**

**1. Reconcile before diagnosing.** The monitor scores the *simulated* return
stream — next-open fills, 7.5 bps — not your account. Before concluding
anything about the model, confirm the two still describe the same thing:

- Did you trade the aligned rotation set, on the rotation date?
- Were fills near the open, or did you trade midday / a day late?
- Does your actual 6-month return resemble the `126d model` figure printed?

If your P&L and the model's diverge, **the model is not what degraded** and
nothing below applies. This is the most likely cause and the easiest to miss.

**2. Decompose: overlay drag, or bad picks?** The report prints both. They
point in different directions:

| Raw | Beta-adjusted | Reading |
|---|---|---|
| bad | fine | A half-beta book failed to keep up with a rally. Structural, not broken. Check the defensive posture block. |
| bad | bad | The picks themselves stopped working. Continue to step 3. |

Both historical episodes were the first kind: the model held a defensive name on
~35% of days while SPY ran +17%. Check `DEFENSIVE POSTURE` — if the recent share
is well above the 39% long-run figure, the overlay is the story.

**3. Screen the universe.** `python scripts/screen_universe.py`. FINDINGS'
standing conclusion is that every timing idea tested failed, and what remaining
upside exists is in the candidate set rather than in the timing of trades among
current candidates. A sustained shortfall with a fixed ranking method is most
consistent with a stale universe. You already re-screen roughly quarterly. An
ALARM is a reason to do it now and to look harder, not to invent a new process.

**4. Parameters — reluctantly, and with a prior against.** Walk-forward
validation found that re-tuning on a trailing window **beat the median fixed
configuration only 13% of the time and cost 3pp of CAGR**. The instinct to tune
when it hurts is the documented way to lose money here. If you get this far,
the bar is: name the specific parameter, state in advance what result would
change your mind, and run it as a suite with the subperiod stability check —
the same standard Tracks A-D were held to. A change that wins the full sample
by winning one segment is a fit, not a finding.

#### What ALARM never means

- **Do not de-risk into it.** Track A found that cutting exposure forfeits the
  overnight premium — 63% of the return stream for 36% of the variance.
- **Do not trade the CURRENT SET off-cycle** because it looks better than what
  you hold. The rank-exit and score-swap suites both tested versions of that and
  both were rejected.
- **Do not treat it as predictive.** Forward six-month excess after a trigger
  swings from -2.6% to +4.8% depending on settings, on two to five observations.
  It is a prompt to look, and nothing more.
