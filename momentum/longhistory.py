"""
A monthly return panel back to 1926, for testing the Track K hedge layer on
episodes the price panel cannot reach.

WHY THIS EXISTS

The price panel starts in 2010 and contains one inflation episode.  A hedge rule
calibrated on 2021-22 alone can be made to look brilliant by construction.  The
only US inflation episodes with teeth are 1946-48 (yield peg, CPI near 20%,
deeply negative real rates) and 1973-81, and reaching them means index-level
data rather than stocks.

WHAT EACH SERIES IS, AND WHAT IT IS NOT

  STOCKS   Ken French top prior-return decile (12-2), value weighted.  A proxy
           for "a momentum stock book".  It is not Track J: no pullback term,
           no correlation cap, 10% of the market rather than 8 names.
  MARKET   French Mkt-RF + RF.
  CASH     French RF (one-month T-bill).
  UST10    A 10-year par Treasury priced from yields — FRED LTGOVTBD (long
           governments, treated as 20-year) before 1953-04, GS10 until 1962,
           DGS10 month-end after.  Before 1962 the yields are monthly
           averages; see MONTHLY AVERAGES below.
  GOLD     World Bank Pink Sheet, from 1960.  Fixed at $35 until 1968, so it
           cannot qualify as a hedge before then — which is also historically
           right: US persons could not hold bullion from 1933 to 1974.
  SILVER   Pink Sheet, from 1960.
  CMDTY    Pink Sheet "Total Index" (energy-heavy, trade-weighted) plus the
           T-bill, as a fully collateralised position.  Roll yield is NOT
           modelled; `validate_against_etfs` reports what that misses against
           DBC.
           `cmdty_backfill=True` extends it to 1926-1959 with FRED PPIACO (PPI
           all commodities) plus the bill, delayed like the averages.  PPI
           includes processed goods and is not a traded basket; it exists so
           1946-48 has a commodity candidate at all, and is a sensitivity
           variant, never the default.
  ENERGY   French 49-industry "Oil", value weighted.  Stands in for the pool's
           energy basket.
  REALEST  FTSE Nareit All Equity REITs total return from 1972.  Before that,
           French 49-industry "RlEst" — which is NOT REITs (French files SIC
           6798 under "Fin"); it is developers and operators, correlating 0.79
           with VNQ.  REITs barely existed before 1972, so the 1946-48 real
           estate leg is the weakest series in the panel.
  CPI      FRED CPIAUCNS, not seasonally adjusted, as monthly inflation.

MONTHLY AVERAGES — THE ONE CONSTRUCTION THAT CAN SILENTLY FLATTER A RESULT

Pink Sheet prices and FRED's monthly yield series are averages over the month,
not month-end values.  Treating avg(t)->avg(t+1) as the return earned by a
position opened at the end of t credits the strategy with the part of the move
that happened in the second half of month t — already visible to the signal.
Averaged series are positively autocorrelated for exactly this reason (Working,
1960), and a trend rule will look better on them than it can trade.

The obvious repair — estimate month-end as the midpoint of adjacent averages —
was tried first and REJECTED: it smooths twice, leaves lag-1 autocorrelation at
0.58 against ~0 for the traded ETF, and still leaks a quarter-month of
already-visible move into the realised return.

What is used instead is a DELAY.  A position opened at the end of month t is
credited avg(t+1) -> avg(t+2): both averages lie wholly after the signal date,
so nothing visible to the signal is in the return.  The cost is roughly half a
month of execution lag, which is conservative for a trend rule.  Where a traded
instrument exists its true month-end returns are spliced in instead (IAU from
2005, SLV and DBC from 2006; DBC also brings real roll costs), so the delayed
construction only carries the pre-ETF history.  No free month-end gold series
reaches the 1970s — FRED dropped LBMA in 2022, and the Bundesbank fixing is in
DM and ends in 1998.

WHERE THE RAW DATA LIVES, AND WHY IT IS VINTAGED RATHER THAN APPENDED

Every download is kept permanently under `data/longhistory/raw/` as
`<name>__<YYYYMMDD>.<ext>`, and a refresh adds a new vintage only when the
bytes changed.  Appending is the wrong model for these sources: Ken French
re-forms portfolios every year from a revised CRSP, and FRED and Nareit revise
too, so an appended series would stitch revised and unrevised history
together.  Vintages keep every published version, and `save_panel` reports
what changed between one build and the next.  They also make the foundational
data survive its source: FRED dropped LBMA gold in 2022 and the Pink Sheet URL
changes monthly.  A failed download falls back to the latest vintage, loudly.

The ETF month-end returns come from the daily store (`momentum/marketstore.py`),
so there is one source of truth for every traded instrument.
"""

from __future__ import annotations

import io
import os
import time
import urllib.request
import warnings
import zipfile
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "longhistory" / "raw"
PANEL_FILE = REPO_ROOT / "data" / "longhistory" / "panel_monthly.csv"

FRENCH = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp"
FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv?id="
# The Pink Sheet URL embeds a hash that changes with each monthly release, so
# it is discovered from the landing page rather than hard-coded.
PINK_PAGE = "https://www.worldbank.org/en/research/commodity-markets"

HEDGE_ASSETS = ["GOLD", "SILVER", "CMDTY", "ENERGY", "REALEST", "UST10", "CASH"]


# --------------------------------------------------------------------------
# download into dated vintages
# --------------------------------------------------------------------------

def vintages(name: str, raw_dir: Path = RAW_DIR):
    stem, ext = os.path.splitext(name)
    return sorted(Path(raw_dir).glob(f"{stem}__*{ext}"))


def _fetch(url: str, name: str, max_age_days: float, verbose: bool) -> bytes:
    """
    The latest vintage if it is younger than `max_age_days`; otherwise download,
    keep a new vintage only if the bytes changed, and fall back to the latest
    vintage if the download fails.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    have = vintages(name)
    latest = have[-1] if have else None
    if latest is not None and (time.time() - latest.stat().st_mtime) < max_age_days * 86400:
        return latest.read_bytes()
    try:
        if verbose:
            print(f"  downloading {name}")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            data = r.read()
    except Exception as e:
        if latest is None:
            raise
        warnings.warn(f"{name}: download failed ({e}); using vintage {latest.name}")
        return latest.read_bytes()
    if latest is not None and latest.read_bytes() == data:
        os.utime(latest)                      # unchanged: refresh its age, no new vintage
        return data
    stem, ext = os.path.splitext(name)
    path = RAW_DIR / f"{stem}__{time.strftime('%Y%m%d')}{ext}"
    path.write_bytes(data)
    return data


def _pink_url(verbose: bool) -> str:
    import re
    html = _fetch(PINK_PAGE, "pink_landing.html", 7, verbose).decode("utf-8", "ignore")
    m = re.search(r"https://thedocs\.worldbank\.org[^\"']*CMO-Historical-Data-Monthly\.xlsx", html)
    if not m:
        raise RuntimeError("Pink Sheet link not found on the World Bank commodity page")
    return m.group(0)


# --------------------------------------------------------------------------
# parsers
# --------------------------------------------------------------------------

def _french_block(zipname: str, max_age_days: float, verbose: bool) -> pd.DataFrame:
    """
    The FIRST data block of a Ken French CSV (value-weighted for portfolio
    files), as decimal returns.  Monthly files (YYYYMM) get a month-end index,
    daily files (YYYYMMDD) a date index.  Missing (-99.99/-999) becomes NaN.
    """
    raw = _fetch(f"{FRENCH}/{zipname}.zip", f"{zipname}.zip", max_age_days, verbose)
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        text = z.read(z.namelist()[0]).decode("latin-1")
    lines = text.splitlines()
    start = next(i for i, ln in enumerate(lines)
                 if ln.startswith(",") and i + 1 < len(lines)
                 and lines[i + 1].split(",")[0].strip().isdigit())
    width = len(lines[start + 1].split(",")[0].strip())
    rows = []
    for ln in lines[start + 1:]:
        head = ln.split(",")[0].strip()
        if not (len(head) == width and head.isdigit()):
            break
        rows.append(ln)
    df = pd.read_csv(io.StringIO("\n".join([lines[start]] + rows)), index_col=0)
    df.columns = df.columns.str.strip()
    if width == 6:
        df.index = pd.to_datetime(df.index.astype(str), format="%Y%m") + pd.offsets.MonthEnd(0)
    else:
        df.index = pd.to_datetime(df.index.astype(str), format="%Y%m%d")
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.mask(df <= -99.0)
    return df / 100.0


_french_monthly = _french_block


def french_daily(max_age_days: float = 30, verbose: bool = True) -> pd.DataFrame:
    """Daily STOCKS (top prior-return decile, VW), MARKET and CASH (RF)."""
    ff = _french_block("F-F_Research_Data_Factors_daily_CSV", max_age_days, verbose)
    mom = _french_block("10_Portfolios_Prior_12_2_Daily_CSV", max_age_days, verbose)
    return pd.DataFrame({"STOCKS": mom["Hi PRIOR"], "MARKET": ff["Mkt-RF"] + ff["RF"],
                         "CASH": ff["RF"]}).dropna()


def _fred(series: str, max_age_days: float, verbose: bool) -> pd.Series:
    raw = _fetch(FRED + series, f"fred_{series}.csv", max_age_days, verbose)
    df = pd.read_csv(io.BytesIO(raw), index_col=0, parse_dates=True)
    s = pd.to_numeric(df.iloc[:, 0], errors="coerce")
    s.name = series
    return s


def _pink(max_age_days: float, verbose: bool) -> pd.DataFrame:
    """Monthly-AVERAGE prices: GOLD, SILVER, and the Total commodity index."""
    raw = _fetch(_pink_url(verbose), "pink_monthly.xlsx", max_age_days, verbose)
    prices = pd.read_excel(io.BytesIO(raw), "Monthly Prices", header=None)
    names = prices.iloc[4].astype(str).str.strip()
    body = prices.iloc[6:]
    idx = pd.to_datetime(body.iloc[:, 0].astype(str).str.replace("M", "-"),
                         format="%Y-%m") + pd.offsets.MonthEnd(0)
    col = {n: i for i, n in enumerate(names)}
    out = pd.DataFrame({
        "GOLD": pd.to_numeric(body.iloc[:, col["Gold"]], errors="coerce").values,
        "SILVER": pd.to_numeric(body.iloc[:, col["Silver"]], errors="coerce").values,
    }, index=idx)

    ind = pd.read_excel(io.BytesIO(raw), "Monthly Indices", header=None)
    assert str(ind.iloc[5, 1]).strip() == "Total Index", ind.iloc[5, 1]
    ibody = ind.iloc[9:]
    iidx = pd.to_datetime(ibody.iloc[:, 0].astype(str).str.replace("M", "-"),
                          format="%Y-%m") + pd.offsets.MonthEnd(0)
    out["CMDTY"] = pd.Series(pd.to_numeric(ibody.iloc[:, 1], errors="coerce").values,
                             index=iidx)
    return out


NAREIT = "https://www.reit.com/sites/default/files/returns/MonthlyHistoricalReturns.xls"


def _nareit(max_age_days: float, verbose: bool) -> pd.Series:
    """FTSE Nareit All Equity REITs, monthly total return, 1972 onward."""
    raw = _fetch(NAREIT, "nareit_monthly.xls", max_age_days, verbose)
    d = pd.read_excel(io.BytesIO(raw), "Index Data", header=None)
    hdr = [str(v).strip() for v in d.iloc[5].tolist()]
    c = hdr.index("All Equity REITs")
    assert str(d.iloc[6, c]).strip() == "Total" and str(d.iloc[7, c]).strip() == "Return"
    body = d.iloc[9:, [0, c]].dropna(subset=[0])
    s = pd.Series(pd.to_numeric(body.iloc[:, 1], errors="coerce").values / 100.0,
                  index=pd.to_datetime(body.iloc[:, 0]) + pd.offsets.MonthEnd(0),
                  name="NAREIT")
    return s.dropna()


# --------------------------------------------------------------------------
# constructions
# --------------------------------------------------------------------------

def delayed_average_return(avg: pd.Series) -> pd.Series:
    """
    Row t holds avg(t) -> avg(t+1), the return credited to a position opened at
    the end of t-1.  See MONTHLY AVERAGES in the module doc.
    """
    return avg.pct_change(fill_method=None).shift(-1)


# Where a traded instrument exists, its month-end total return replaces the
# construction from this date on.
ETF_SPLICE = {"GOLD": ("IAU", "2005-03-31"), "SILVER": ("SLV", "2006-06-30"),
              "CMDTY": ("DBC", "2006-03-31")}


def etf_monthly(symbols, max_age_days: float = 7, verbose: bool = True) -> pd.DataFrame:
    """Calendar-month total returns, compounded from the daily market store."""
    from .marketstore import load_daily_returns, monthly_returns
    return monthly_returns(load_daily_returns(list(symbols)))


def par_bond_return(y_prev: pd.Series, y_now: pd.Series, maturity: float) -> pd.Series:
    """
    One-month total return on a par bond bought at yield `y_prev` (percent) and
    repriced at `y_now`, semiannual coupons.  Ageing by one month is ignored;
    at 10 years it moves the return by a basis point or two.
    """
    c = y_prev / 100.0
    y = y_now / 100.0
    n = 2 * maturity
    with np.errstate(divide="ignore", invalid="ignore"):
        annuity = np.where(np.abs(y) > 1e-9, (1 - (1 + y / 2) ** (-n)) / (y / 2), n)
    price = (c / 2) * annuity + (1 + y / 2) ** (-n)
    return pd.Series(price - 1 + c / 12, index=y_now.index)


def treasury_10y(max_age_days: float = 30, verbose: bool = True) -> pd.Series:
    """Monthly total return of a 10-year par Treasury, 1925 onward."""
    lt = _fred("LTGOVTBD", max_age_days, verbose)
    gs = _fred("GS10", max_age_days, verbose)
    dg = _fred("DGS10", max_age_days, verbose)
    for s in (lt, gs):
        s.index = s.index + pd.offsets.MonthEnd(0)

    # Averaged yields get the same delay as averaged prices: row t is the
    # return from avg(t) to avg(t+1).  DGS10 is daily, so from 1962 the month-end
    # yield is real and no delay is needed.
    r_lt = par_bond_return(lt, lt.shift(-1), 20.0)
    r_gs = par_bond_return(gs, gs.shift(-1), 10.0)
    dg_end = dg.dropna().resample("ME").last()
    r_dg = par_bond_return(dg_end.shift(1), dg_end, 10.0)

    out = pd.concat([r_lt.loc[:"1953-03-31"], r_gs.loc["1953-04-30":"1962-01-31"],
                     r_dg.loc["1962-02-28":]])
    out.name = "UST10"
    return out.dropna()


def load_long_history(max_age_days: float = 30, verbose: bool = True,
                      splice: bool = True, cmdty_backfill: bool = False) -> pd.DataFrame:
    """
    Monthly decimal returns on a month-end index, one column per series in the
    module docstring plus CPI.  Rows run to the last month every core series
    covers; hedge assets that start late are NaN before they exist.

    `splice=False` keeps the pre-ETF construction all the way through, which is
    only useful for measuring that construction against the ETF it stands in for.
    """
    ff = _french_monthly("F-F_Research_Data_Factors_CSV", max_age_days, verbose)
    mom = _french_monthly("10_Portfolios_Prior_12_2_CSV", max_age_days, verbose)
    ind = _french_monthly("49_Industry_Portfolios_CSV", max_age_days, verbose)
    pink = _pink(max_age_days, verbose)
    cpi = _fred("CPIAUCNS", max_age_days, verbose)
    cpi.index = cpi.index + pd.offsets.MonthEnd(0)

    panel = pd.DataFrame({
        "STOCKS": mom["Hi PRIOR"],
        "MARKET": ff["Mkt-RF"] + ff["RF"],
        "CASH": ff["RF"],
        "ENERGY": ind["Oil"],
        "REALEST": ind["RlEst"],
    })
    panel["UST10"] = treasury_10y(max_age_days, verbose)
    for c in ("GOLD", "SILVER", "CMDTY"):
        panel[c] = delayed_average_return(pink[c])
    if cmdty_backfill:
        ppi = _fred("PPIACO", max_age_days, verbose)
        ppi.index = ppi.index + pd.offsets.MonthEnd(0)
        first = panel["CMDTY"].first_valid_index()
        early = delayed_average_return(ppi).loc[:first - pd.offsets.MonthEnd(1)]
        panel.loc[early.index.intersection(panel.index), "CMDTY"] = early
    # the commodity index is a price; a collateralised position also earns the bill
    panel["CMDTY"] = panel["CMDTY"] + panel["CASH"]

    if splice:
        etf = etf_monthly([s for s, _ in ETF_SPLICE.values()], verbose=verbose)
        for col, (sym, start) in ETF_SPLICE.items():
            panel.loc[start:, col] = etf[sym].reindex(panel.loc[start:].index)
    reit = _nareit(max_age_days, verbose)
    panel.loc["1972-01-31":, "REALEST"] = reit.reindex(panel.loc["1972-01-31":].index)
    panel["CPI"] = cpi.pct_change(fill_method=None)

    core = ["STOCKS", "MARKET", "CASH", "ENERGY", "REALEST", "UST10", "CPI"]
    end = min(panel[c].last_valid_index() for c in core + list(ETF_SPLICE))
    return panel.loc["1926-07-31":end]


# --------------------------------------------------------------------------
# validation against traded instruments
# --------------------------------------------------------------------------

def validate_constructions(max_age_days: float = 30, verbose: bool = True) -> pd.DataFrame:
    """
    Every series against the instrument it stands in for, over the months both
    exist.  Two kinds of row:

      measured      French/Nareit/DGS10 series that are month-end by nature.
                    `corr` and `gap_ann` should be close to 1 and 0.
      construction  the pre-ETF construction (monthly averages, delayed), run
                    forward into the ETF era with the splice switched OFF, so it
                    can be scored against the real thing.  Also scored: the
                    NAIVE undelayed average, to show what the delay removes.

    `leak` is the correlation of the series with the ETF's return in the month
    BEFORE the position was opened.  A clean series has ~0; a series that
    credits already-visible moves does not.  That is the number the delay
    exists to fix, and the one a trend rule is flattered by.
    """
    raw = load_long_history(max_age_days, verbose, splice=False)
    pink = _pink(max_age_days, verbose)
    gs = _fred("GS10", max_age_days, verbose)
    gs.index = gs.index + pd.offsets.MonthEnd(0)
    etf = etf_monthly(["IEF", "IAU", "SLV", "DBC", "SPY", "XLE", "VNQ"],
                      max_age_days=7, verbose=verbose)

    series = {
        ("measured", "MARKET"): (raw["MARKET"], "SPY"),
        ("measured", "ENERGY"): (raw["ENERGY"], "XLE"),
        ("measured", "REALEST"): (raw["REALEST"], "VNQ"),
        ("measured", "UST10"): (raw["UST10"], "IEF"),
        ("construction", "UST10 pre-1962"): (par_bond_return(gs, gs.shift(-1), 10.0), "IEF"),
    }
    for c, sym in (("GOLD", "IAU"), ("SILVER", "SLV"), ("CMDTY", "DBC")):
        series[("construction", c)] = (raw[c], sym)
        naive = pink[c].pct_change(fill_method=None)
        if c == "CMDTY":
            naive = naive + raw["CASH"].reindex(naive.index)
        series[("naive average", c)] = (naive, sym)

    ann = lambda r: (1 + r).prod() ** (12 / len(r)) - 1
    rows = []
    for (kind, name), (ours, sym) in series.items():
        d = pd.concat([ours, etf[sym], etf[sym].shift(1)], axis=1).dropna()
        d.columns = ["ours", "etf", "etf_prev"]
        rows.append({
            "kind": kind, "series": name, "vs": sym, "months": len(d),
            "corr": d["ours"].corr(d["etf"]),
            "leak": d["ours"].corr(d["etf_prev"]),
            "gap_ann": ann(d["ours"]) - ann(d["etf"]),
            "ac1": d["ours"].autocorr(1), "ac1_etf": d["etf"].autocorr(1),
        })
    return pd.DataFrame(rows).set_index(["kind", "series"])


# --------------------------------------------------------------------------
# the built panel, and what changed since the last build
# --------------------------------------------------------------------------

def save_panel(panel: pd.DataFrame, path: Path = PANEL_FILE) -> pd.DataFrame:
    """
    Write the built panel and report, per column, how it differs from the
    previous build: months added, and the largest revision to a month both
    builds contain.  A revision is expected (French re-forms annually); an
    unexplained one in a column that should be fixed history is the thing to
    look at.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if path.exists():
        old = pd.read_csv(path, index_col=0, parse_dates=True)
        for c in panel.columns:
            a = old[c].dropna() if c in old else pd.Series(dtype=float)
            b = panel[c].dropna()
            common = a.index.intersection(b.index)
            d = (a.loc[common] - b.loc[common]).abs()
            rows.append({"series": c, "months": len(b),
                         "added": int((b.index > (a.index.max() if len(a) else pd.Timestamp.min)).sum()),
                         "revised > 1bp": int((d > 1e-4).sum()),
                         "max revision": float(d.max()) if len(d) else np.nan,
                         "first revised": (d[d > 1e-4].index[0].strftime("%Y-%m")
                                           if (d > 1e-4).any() else "")})
    else:
        rows = [{"series": c, "months": int(panel[c].notna().sum()), "added": int(panel[c].notna().sum()),
                 "revised > 1bp": 0, "max revision": np.nan, "first revised": ""}
                for c in panel.columns]
    tmp = path.with_suffix(".tmp")
    panel.to_csv(tmp, float_format="%.10g", index_label="Date")
    os.replace(tmp, path)
    return pd.DataFrame(rows).set_index("series")
