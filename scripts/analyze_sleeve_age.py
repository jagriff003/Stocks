"""
Where in its 8-week hold does a Track J sleeve earn its return?

Asked while planning the move from the production model to Track J
(2026-09-24): phasing in buys each sleeve fresh, while a fast transition buys
the existing sleeves mid-hold.  Which is better depends on when a sleeve earns.

Each sleeve is simulated exactly as the live runner does (the same builder,
calendar and costs, via analyze_trackj_real_assets), at both biweekly phases.
Every day of each sleeve's return is tagged with the sleeve's age in sessions
since its selection, and the excess over SPY is averaged by two-week bucket.

Run:  python scripts/analyze_sleeve_age.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
warnings.filterwarnings("ignore")

from momentum.backtest import _PortfolioBuilder, simulate_portfolio  # noqa: E402
from momentum.liquidity import LiquidityConfig, slippage_panel  # noqa: E402
from momentum.schedule import TUESDAY, rotation_dates, sleeve_assignment  # noqa: E402
from scripts.analyze_trackj_real_assets import (ACCOUNT, EVERY_WEEKS, TRANCHES,  # noqa: E402
                                                 load_panels, score)
from scripts.run_live_trackj import build_config  # noqa: E402

lcfg = LiquidityConfig(account_notional=ACCOUNT)
cfg = build_config(8, 40)
base, vol, _, _, defensive = load_panels()
sc = score(base, vol, cfg, defensive, lcfg)
spy = base.spy.pct_change()

rows = []
combined_parts = []
for phase, anchor in (("A", "2026-09-15"), ("B", "2026-09-22")):
    rot = rotation_dates(base.close.index, weekday=TUESDAY, every_weeks=EVERY_WEEKS,
                         anchor=pd.Timestamp(anchor))
    assign = sleeve_assignment(rot, TRANCHES)
    builder = _PortfolioBuilder(
        sc, price_columns=list(base.close.columns), top_n=cfg.top_n,
        min_data_days=cfg.min_data_days, hold_days=cfg.hold_days, vix_data=base.vix,
        vix_config=cfg.vix, base_composite_scores=None, velocity_config=cfg.velocity,
        correlation_config=cfg.correlation, graduated_config=cfg.graduated_vix,
        exit_config=None, close=base.close, rank_offset=cfg.rank_offset,
        rank_offset_scope=cfg.rank_offset_scope, monitor_symbols=cfg.monitor_symbols)
    slip = slippage_panel(base.close, vol, lcfg, top_n=cfg.top_n * TRANCHES)
    idx = base.close.index
    for j in range(TRANCHES):
        picks = {}
        for d in [d for d in rot if assign[d] == j]:
            rec = builder.select_on(d)
            if rec:
                picks[d] = list(rec["Selected_Stocks"])
        tgt = pd.Series({d: picks[d] for d in sorted(picks)}, dtype=object)
        sel_dates = pd.DatetimeIndex(sorted(picks))
        tgt = tgt.reindex(idx).ffill().dropna()
        r = simulate_portfolio(tgt, base.close, base.open_, execution=cfg.execution,
                               sizing=cfg.sizing, scores=sc, slippage_by_symbol=slip).returns
        # age = sessions since this sleeve's latest selection date
        pos = np.searchsorted(sel_dates.values, r.index.values, side="right") - 1
        valid = pos >= 0
        sel_for_day = sel_dates[np.clip(pos, 0, None)]
        age = idx.get_indexer(r.index) - idx.get_indexer(sel_for_day)
        df = pd.DataFrame({"r": r.values, "spy": spy.reindex(r.index).values, "age": age},
                          index=r.index)[valid]
        df["phase"] = phase
        rows.append(df)

d = pd.concat(rows).dropna().sort_index()
d = d[d.index >= "2012-01-01"]
d["ex"] = d["r"] - d["spy"]
d["bucket"] = pd.cut(d["age"], [0, 10, 20, 30, 45], labels=["weeks 1-2", "weeks 3-4", "weeks 5-6", "weeks 7-8"])
g = d.groupby("bucket", observed=True)["ex"]
out = pd.DataFrame({"days": g.size(), "excess vs SPY, annualised": g.mean() * 252,
                    "t": g.mean() / g.std() * np.sqrt(g.size()),
                    "share of total excess": g.sum() / d["ex"].sum()})
pd.set_option("display.width", 200)
print(out.assign(**{"excess vs SPY, annualised": out["excess vs SPY, annualised"].map("{:+.1%}".format),
                    "share of total excess": out["share of total excess"].map("{:.0%}".format),
                    "t": out["t"].round(2)}).to_string())
# by phase, to see whether the shape is stable
print()
for ph in ("A", "B"):
    s = d[d.phase == ph].groupby("bucket", observed=True)["ex"].mean() * 252
    print(f"phase {ph}: " + "  ".join(f"{k} {v:+.1%}" for k, v in s.items()))
# subperiods
for a, b in (("2012", "2018"), ("2019", "2026")):
    s = d.loc[a:b].groupby("bucket", observed=True)["ex"].mean() * 252
    print(f"{a}-{b}: " + "  ".join(f"{k} {v:+.1%}" for k, v in s.items()))
