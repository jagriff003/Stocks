"""
The rebalance workbook: one household target, two IRAs, trades in shares.

WHY IT EXISTS

Trade days mean turning a model weight into share orders across two separate
accounts (a Roth and a Traditional IRA) that often hold the same name, usually
from a phone or tablet.  This builds an Excel workbook that does that
arithmetic with plain formulas — no macros, nothing mobile Excel cannot run —
and `fill_target` lets `run_live_combined.py` drop the day's recommendation
into it.  The workbook itself holds account data and lives in `private/`
(git-ignored); only this generator is in the repository.

HOW A NAME IS SPLIT ACROSS THE TWO ACCOUNTS

Every name has one household target: its weight x (Roth + Traditional
investable value).  Where it sits is its Placement:

    Auto   held in one account -> stays there; held in both -> keeps the
           current proportions (Keep); held in neither -> Split
    Roth / Trad   the whole name in that account
    Split  shares the leftover: after every fixed placement, each account has
           some free cash, and Split names are divided in proportion to it —
           which is what leaves both accounts fully invested.  With no Split
           names, the leftover shows up as cash after the trades.

Whole shares only (rounded down), and trades smaller than a threshold are
skipped unless they close a position.

SHEETS

    Guide     what to type where, the rotation-day steps, an example row
    Setup     account cash, cash to hold back, minimum trade; values; checks
    Target    Symbol, Model (J/K), weight, price, Take/My weight for Track K,
              Placement — written by the report or typed by hand
    Holdings  shares held per account (update after each trade day)
    Plan      all the arithmetic, one row per name (target, then anything held
              that is no longer targeted, which is sold)
    Trades    the phone view: Roth sells, Roth buys, Traditional sells,
              Traditional buys, with a Done column
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from openpyxl import Workbook, load_workbook
from openpyxl.comments import Comment
from openpyxl.formatting.rule import CellIsRule, FormulaRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.workbook.properties import CalcProperties
from openpyxl.worksheet.datavalidation import DataValidation

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PATH = REPO_ROOT / "private" / "Rebalance.xlsx"

T0, T1 = 6, 50      # Target data rows
H0, H1 = 6, 50      # Holdings data rows
P0, P1 = 6, 95      # Plan data rows (target + held-but-not-targeted)
BLOCK = 45          # rows per Trades block (>= Target rows)

FONT = "Arial"
INPUT_FILL = PatternFill("solid", fgColor="FFF2CC")      # soft yellow: type here
HEAD_FILL = PatternFill("solid", fgColor="D9E1F2")
TITLE_FILL = PatternFill("solid", fgColor="1F3864")
BLUE, BLACK, GREEN, GREY = "0000FF", "000000", "008000", "808080"
THIN = Side(style="thin", color="BFBFBF")
BOX = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)

USD = '$#,##0;($#,##0);"-"'
USD2 = '$#,##0.00;($#,##0.00);"-"'
PCT = '0.00%;(0.00%);"-"'
SH = '#,##0;(#,##0);"-"'


def _f(size=10, bold=False, color=BLACK, italic=False):
    return Font(name=FONT, size=size, bold=bold, color=color, italic=italic)


def _title(ws, text, width_cols=8):
    ws["A1"] = text
    ws["A1"].font = Font(name=FONT, size=14, bold=True, color="FFFFFF")
    for c in range(1, width_cols + 1):
        ws.cell(row=1, column=c).fill = TITLE_FILL


def _header(ws, row, labels, widths=None):
    for i, lab in enumerate(labels, start=1):
        c = ws.cell(row=row, column=i, value=lab)
        c.font = _f(bold=True)
        c.fill = HEAD_FILL
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        c.border = BOX
    if widths:
        for i, w in enumerate(widths, start=1):
            ws.column_dimensions[ws.cell(row=row, column=i).column_letter].width = w
    ws.row_dimensions[row].height = 30


def _input(cell, fmt=None):
    cell.fill = INPUT_FILL
    cell.font = _f(color=BLUE)
    cell.border = BOX
    if fmt:
        cell.number_format = fmt


def _calc(cell, fmt=None, color=BLACK):
    cell.font = _f(color=color)
    cell.border = BOX
    if fmt:
        cell.number_format = fmt


# --------------------------------------------------------------------------
# sheets
# --------------------------------------------------------------------------

def _guide(ws):
    _title(ws, "Rebalance workbook - how to use it", 6)
    ws.column_dimensions["A"].width = 4
    ws.column_dimensions["B"].width = 100
    lines = [
        ("", ""),
        ("", "YELLOW cells are the only ones you type in. Everything else is a formula."),
        ("", ""),
        ("", "ROTATION DAY"),
        ("1", "PC, after the close: run the combined report. It fills the Target sheet (Symbol, Model, weight, price)."),
        ("2", "Setup: enter each account's CASH. Optional: cash to hold back, and the smallest trade worth making."),
        ("3", "Holdings: enter shares held in each account (carry them over from last time; update after trading)."),
        ("4", "Target: Track K rows (Model = K) default to Take = N. Set Y to act on one; My weight overrides its size."),
        ("5", "Target: Placement is Auto unless you choose. Roth / Trad pins a name to one account; Split shares it."),
        ("6", "Setup: the status must read READY. Each check says what to fix if not."),
        ("7", "Trade day, phone or iPad: Trades sheet. Sells first, then buys. Tick Done as each fills."),
        ("8", "Prices move: type the live price in Target column D and every share count updates."),
        ("9", "Afterwards: copy Plan's 'shares after' columns into Holdings, ready for next time."),
        ("", ""),
        ("", "PLACEMENT"),
        ("", "Auto  - held in one account: stays there. Held in both: keeps today's proportions. New: Split."),
        ("", "Split - divided between the accounts in proportion to each one's free cash, which keeps both fully invested."),
        ("", "Roth / Trad - the whole name in that account. Too many pins can overload one account: Setup will say so."),
        ("", ""),
        ("", "TRACK K (discretionary)"),
        ("", "Its rows show the model's recommended weight. Take = Y includes it; Track J names shrink pro-rata to make room."),
        ("", "A Track K name you hold stays listed while Take = Y, even after the model drops it. Set N to sell it."),
        ("", ""),
        ("", "EXAMPLE Target row (format only):"),
        ("", "Symbol MU | Model J | Model weight 6.25% | Price 120.50 | Take (blank for J) | My weight (blank) | Placement Auto"),
        ("", ""),
        ("", "Whole shares only, rounded down. Trades below the Setup minimum are skipped unless they close a position."),
        ("", "This file holds account data: it lives in private/, which is never committed."),
    ]
    for i, (n, text) in enumerate(lines, start=2):
        ws.cell(row=i, column=1, value=n).font = _f(bold=True)
        c = ws.cell(row=i, column=2, value=text)
        c.font = _f(bold=text.isupper() and len(text) < 40, size=11 if text.isupper() else 10)
        c.alignment = Alignment(wrap_text=True, vertical="top")
    ws.sheet_view.showGridLines = False


def _setup(ws):
    _title(ws, "Setup - accounts and checks", 4)
    ws.column_dimensions["A"].width = 2
    ws.column_dimensions["B"].width = 38
    ws.column_dimensions["C"].width = 16
    ws.column_dimensions["D"].width = 70
    rows = [
        (3, "INPUTS", None, None),
        (4, "Roth IRA cash ($)", None, USD2),
        (5, "Traditional IRA cash ($)", None, USD2),
        (6, "Roth cash to keep uninvested ($)", 0, USD),
        (7, "Traditional cash to keep uninvested ($)", 0, USD),
        (8, "Skip trades smaller than ($)", 100, USD),
    ]
    for r, label, default, fmt in rows:
        ws.cell(row=r, column=2, value=label).font = _f(bold=label.isupper())
        if fmt:
            c = ws.cell(row=r, column=3, value=default)
            _input(c, fmt)
    ws["C8"].comment = Comment("Assumption: $100 is a placeholder, not a measured figure - set your "
                               "own. Positions being closed always trade.", "model")

    calc = [
        (10, "VALUES", None, None),
        (11, "Roth holdings value", "=SUM(Holdings!$F$6:$F$50)", USD),
        (12, "Traditional holdings value", "=SUM(Holdings!$G$6:$G$50)", USD),
        (13, "Roth account value", "=N(C4)+C11", USD),
        (14, "Traditional account value", "=N(C5)+C12", USD),
        (15, "Household value", "=C13+C14", USD),
        (16, "Roth investable", "=MAX(0,C13-N(C6))", USD),
        (17, "Traditional investable", "=MAX(0,C14-N(C7))", USD),
        (18, "Household investable", "=C16+C17", USD),
    ]
    for r, label, formula, fmt in calc:
        ws.cell(row=r, column=2, value=label).font = _f(bold=label.isupper())
        if formula:
            _calc(ws.cell(row=r, column=3, value=formula), fmt)

    ws.cell(row=20, column=2, value="CHECKS").font = _f(bold=True)
    checks = [
        (21, "Target weights add to", "=Target!$I$3", PCT,
         '=IF(ABS(C21-1)<0.005,"OK","FIX: weights should add to 100% - check Take / My weight in Target")'),
        (22, "Split share going to Roth", "=Plan!$F$4", PCT,
         '=IF(Plan!$I$2=0,"OK (no Split names - any leftover stays as cash)",IF(AND(C22>=0,C22<=1),"OK",'
         '"FIX: pinned names overload one account - set one to Split or to the other account"))'),
        (23, "Roth cash after trades", "=N(C4)-SUM(Plan!$U$6:$U$95)", USD2,
         '=IF(C23<-1,"FIX: Roth is short of cash - move a buy to Trad or Split","OK")'),
        (24, "Traditional cash after trades", "=N(C5)-SUM(Plan!$V$6:$V$95)", USD2,
         '=IF(C24<-1,"FIX: Traditional is short of cash - move a buy to Roth or Split","OK")'),
        (25, "Names missing a price", '=COUNTIFS(Plan!$B$6:$B$95,"?*",Plan!$C$6:$C$95,0)', "0",
         '=IF(C25=0,"OK","FIX: type a price for "&C25&" name(s) - Target column D, or Holdings column D")'),
        (26, "Roth trades / Traditional trades", '=COUNT(Plan!$W$6:$X$95)&" / "&COUNT(Plan!$Y$6:$Z$95)', None,
         '="sells + buys per account"'),
        (27, "Longest list on the Trades sheet", "=MAX(0,MAX(Plan!$W$6:$Z$95))", "0",
         f'=IF(C27<={BLOCK},"OK","FIX: more trades than the Trades sheet shows ({BLOCK}) - read them from Plan")'),
    ]
    for r, label, formula, fmt, msg in checks:
        ws.cell(row=r, column=2, value=label)
        _calc(ws.cell(row=r, column=3, value=formula), fmt, GREEN)
        _calc(ws.cell(row=r, column=4, value=msg))
    ws["B28"] = "STATUS"
    ws["B28"].font = _f(bold=True, size=12)
    ws["C28"] = ('=IF(AND(LEFT(D21,2)="OK",LEFT(D22,2)="OK",LEFT(D23,2)="OK",'
                 'LEFT(D24,2)="OK",LEFT(D25,2)="OK",LEFT(D27,2)="OK"),"READY","CHECK")')
    ws["C28"].font = _f(bold=True, size=12)
    ws["C28"].border = BOX
    ws.conditional_formatting.add("C28", CellIsRule(operator="equal", formula=['"READY"'],
                                                    fill=PatternFill("solid", fgColor="C6EFCE")))
    ws.conditional_formatting.add("C28", CellIsRule(operator="equal", formula=['"CHECK"'],
                                                    fill=PatternFill("solid", fgColor="FFC7CE")))
    ws.conditional_formatting.add("D21:D27", FormulaRule(formula=['LEFT(D21,3)="FIX"'],
                                                         font=Font(name=FONT, color="C00000", bold=True)))
    ws.sheet_view.showGridLines = False


def _target(ws):
    _title(ws, "Target - what the models recommend (filled by the report, or typed)", 10)
    ws["A2"], ws["A3"] = "As of", "Source"
    for a in ("A2", "A3"):
        ws[a].font = _f(bold=True)
    _input(ws["B2"])
    _input(ws["B3"])
    ws["G3"] = "Track K total / all weights:"
    ws["G3"].font = _f(bold=True)
    _calc(ws["H3"], PCT)
    _calc(ws["I3"], PCT)
    ws["H3"] = f"=SUM(H{T0}:H{T1})"
    ws["I3"] = f"=SUM(I{T0}:I{T1})"
    _header(ws, 5, ["Symbol", "Model (J/K)", "Model weight", "Price", "Take? (K)",
                    "My weight (K, optional)", "Placement", "Track K used", "Final weight", "#"],
            [10, 8, 10, 11, 8, 11, 11, 10, 10, 5])
    dv_model = DataValidation(type="list", formula1='"J,K"', allow_blank=True)
    dv_take = DataValidation(type="list", formula1='"Y,N"', allow_blank=True)
    dv_place = DataValidation(type="list", formula1='"Auto,Roth,Trad,Split"', allow_blank=True)
    for dv in (dv_model, dv_take, dv_place):
        ws.add_data_validation(dv)
    for r in range(T0, T1 + 1):
        for col, fmt in (("A", None), ("B", None), ("C", PCT), ("D", USD2), ("E", None),
                         ("F", PCT), ("G", None)):
            _input(ws[f"{col}{r}"], fmt)
        dv_model.add(f"B{r}")
        dv_take.add(f"E{r}")
        dv_place.add(f"G{r}")
        ws[f"H{r}"] = (f'=IF(AND(A{r}<>"",B{r}="K"),IF(E{r}="Y",IF(F{r}<>"",F{r},C{r}),0),0)')
        ws[f"I{r}"] = f'=IF(A{r}="","",IF(B{r}="K",H{r},N(C{r})*(1-$H$3)))'
        ws[f"J{r}"] = f'=IF(A{r}<>"",MAX($J$5:J{r - 1})+1,"")'
        _calc(ws[f"H{r}"], PCT)
        _calc(ws[f"I{r}"], PCT)
        _calc(ws[f"J{r}"], "0", GREY)
    ws["C5"].comment = Comment("Track J rows: weight within Track J (they add to 100%). "
                               "Track K rows: the model's recommended share of the whole book.", "model")
    ws["I5"].comment = Comment("Track J weights shrink by the Track K total you take; "
                               "Track K rows use My weight, else the model weight, when Take = Y.", "model")
    ws.freeze_panes = "B6"


def _holdings(ws):
    _title(ws, "Holdings - shares held today (update after every trade day)", 9)
    _header(ws, 5, ["Symbol", "Roth shares", "Trad shares", "Price (only if not in Target)",
                    "Price used", "Roth $", "Trad $", "In target?", "#"],
            [10, 11, 11, 14, 11, 12, 12, 9, 5])
    for r in range(H0, H1 + 1):
        _input(ws[f"A{r}"])
        _input(ws[f"B{r}"], SH)
        _input(ws[f"C{r}"], SH)
        _input(ws[f"D{r}"], USD2)
        ws[f"E{r}"] = (f'=IF(A{r}="","",IFERROR(INDEX(Target!$D${T0}:$D${T1},'
                       f'MATCH(A{r},Target!$A${T0}:$A${T1},0)),N(D{r})))')
        ws[f"F{r}"] = f'=IF(A{r}="","",N(B{r})*E{r})'
        ws[f"G{r}"] = f'=IF(A{r}="","",N(C{r})*E{r})'
        ws[f"H{r}"] = f'=IF(A{r}="","",ISNUMBER(MATCH(A{r},Target!$A${T0}:$A${T1},0)))'
        ws[f"I{r}"] = f'=IF(A{r}="","",IF(H{r},"",MAX($I$5:I{r - 1})+1))'
        _calc(ws[f"E{r}"], USD2, GREEN)
        _calc(ws[f"F{r}"], USD)
        _calc(ws[f"G{r}"], USD)
        _calc(ws[f"H{r}"])
        _calc(ws[f"I{r}"], "0", GREY)
    ws["A3"] = "Totals"
    ws["A3"].font = _f(bold=True)
    ws["F3"] = f"=SUM(F{H0}:F{H1})"
    ws["G3"] = f"=SUM(G{H0}:G{H1})"
    _calc(ws["F3"], USD)
    _calc(ws["G3"], USD)
    ws.freeze_panes = "B6"


def _plan(ws):
    _title(ws, "Plan - the arithmetic (no typing here)", 28)
    ws["A2"], ws["C2"] = "Target names", f"=MAX(Target!$J${T0}:$J${T1})"
    ws["A2"].font = _f(bold=True)
    _calc(ws["C2"], "0")
    labels = {"E2": "Free Roth $", "E3": "Free Trad $", "E4": "Split share to Roth", "H2": "Split names $"}
    for k, v in labels.items():
        ws[k] = v
        ws[k].font = _f(bold=True)
    ws["F2"] = f"=Setup!$C$16-SUM(L{P0}:L{P1})"
    ws["F3"] = f"=Setup!$C$17-SUM(M{P0}:M{P1})"
    ws["F4"] = "=IF(I2=0,0,IFERROR(F2/(F2+F3),0))"
    ws["I2"] = f"=SUM(N{P0}:N{P1})"
    for k, fmt in (("F2", USD), ("F3", USD), ("F4", PCT), ("I2", USD)):
        _calc(ws[k], fmt)
    heads = ["k", "Symbol", "Price", "Final weight", "Household target $", "Roth shares now",
             "Trad shares now", "Roth $ now", "Trad $ now", "Placement chosen", "Placement used",
             "Fixed Roth $", "Fixed Trad $", "Split $", "Roth target $", "Trad target $",
             "Roth target shares", "Trad target shares", "Roth trade shares", "Trad trade shares",
             "Roth trade $", "Trad trade $", "Roth sell #", "Roth buy #", "Trad sell #", "Trad buy #",
             "Roth shares after", "Trad shares after"]
    _header(ws, 5, heads, [4, 9, 10, 9, 12, 9, 9, 11, 11, 10, 10, 11, 11, 11, 12, 12, 9, 9, 9, 9,
                           11, 11, 7, 7, 7, 7, 9, 9])
    tA, tD, tG, tI, tJ = (f"Target!${c}${T0}:${c}${T1}" for c in "ADGIJ")
    hA, hB, hC, hE, hI = (f"Holdings!${c}${H0}:${c}${H1}" for c in "ABCEI")
    for i, r in enumerate(range(P0, P1 + 1), start=1):
        f = {
            "A": i,
            "B": (f'=IF(A{r}<=$C$2,IFERROR(INDEX({tA},MATCH(A{r},{tJ},0)),""),'
                  f'IFERROR(INDEX({hA},MATCH(A{r}-$C$2,{hI},0)),""))'),
            "C": (f'=IF(B{r}="","",IFERROR(N(INDEX({tD},MATCH(B{r},{tA},0))),'
                  f'IFERROR(INDEX({hE},MATCH(B{r},{hA},0)),0)))'),
            "D": f'=IF(B{r}="","",IFERROR(N(INDEX({tI},MATCH(B{r},{tA},0))),0))',
            "E": f'=IF(B{r}="","",D{r}*Setup!$C$18)',
            "F": f'=IF(B{r}="","",SUMIF({hA},B{r},{hB}))',
            "G": f'=IF(B{r}="","",SUMIF({hA},B{r},{hC}))',
            "H": f'=IF(B{r}="","",F{r}*C{r})',
            "I": f'=IF(B{r}="","",G{r}*C{r})',
            "J": (f'=IF(B{r}="","",IFERROR(IF(INDEX({tG},MATCH(B{r},{tA},0))="","Auto",'
                  f'INDEX({tG},MATCH(B{r},{tA},0))),"Auto"))'),
            "K": (f'=IF(B{r}="","",IF(J{r}<>"Auto",J{r},IF(AND(F{r}>0,G{r}>0),"Keep",'
                  f'IF(F{r}>0,"Roth",IF(G{r}>0,"Trad","Split")))))'),
            "L": (f'=IF(B{r}="",0,IF(K{r}="Roth",E{r},IF(K{r}="Keep",'
                  f'IF(H{r}+I{r}>0,E{r}*H{r}/(H{r}+I{r}),0),0)))'),
            "M": (f'=IF(B{r}="",0,IF(K{r}="Trad",E{r},IF(K{r}="Keep",'
                  f'IF(H{r}+I{r}>0,E{r}*I{r}/(H{r}+I{r}),0),0)))'),
            "N": f'=IF(B{r}="",0,IF(K{r}="Split",E{r},0))',
            "O": f'=IF(B{r}="","",L{r}+N{r}*$F$4)',
            "P": f'=IF(B{r}="","",M{r}+N{r}*(1-$F$4))',
            "Q": f'=IF(B{r}="","",IF(C{r}>0,ROUNDDOWN(O{r}/C{r},0),F{r}))',
            "R": f'=IF(B{r}="","",IF(C{r}>0,ROUNDDOWN(P{r}/C{r},0),G{r}))',
            "S": f'=IF(B{r}="","",IF(AND(Q{r}>0,ABS((Q{r}-F{r})*C{r})<Setup!$C$8),0,Q{r}-F{r}))',
            "T": f'=IF(B{r}="","",IF(AND(R{r}>0,ABS((R{r}-G{r})*C{r})<Setup!$C$8),0,R{r}-G{r}))',
            "U": f'=IF(B{r}="","",S{r}*C{r})',
            "V": f'=IF(B{r}="","",T{r}*C{r})',
            "W": f'=IF(B{r}="","",IF(S{r}<0,MAX(W$5:W{r - 1})+1,""))',
            "X": f'=IF(B{r}="","",IF(S{r}>0,MAX(X$5:X{r - 1})+1,""))',
            "Y": f'=IF(B{r}="","",IF(T{r}<0,MAX(Y$5:Y{r - 1})+1,""))',
            "Z": f'=IF(B{r}="","",IF(T{r}>0,MAX(Z$5:Z{r - 1})+1,""))',
            "AA": f'=IF(B{r}="","",F{r}+S{r})',
            "AB": f'=IF(B{r}="","",G{r}+T{r})',
        }
        fmts = {"C": USD2, "D": PCT, "E": USD, "F": SH, "G": SH, "H": USD, "I": USD, "L": USD,
                "M": USD, "N": USD, "O": USD, "P": USD, "Q": SH, "R": SH, "S": SH, "T": SH,
                "U": USD, "V": USD, "W": "0", "X": "0", "Y": "0", "Z": "0", "AA": SH, "AB": SH}
        for col, val in f.items():
            c = ws[f"{col}{r}"]
            c.value = val
            _calc(c, fmts.get(col), GREY if col in ("A", "W", "X", "Y", "Z") else BLACK)
    ws.freeze_panes = "C6"


def _trades(ws):
    ws.column_dimensions["A"].width = 4
    ws.column_dimensions["B"].width = 10
    ws.column_dimensions["C"].width = 9
    ws.column_dimensions["D"].width = 12
    ws.column_dimensions["E"].width = 7
    ws["A1"] = "TRADES"
    ws["A1"].font = Font(name=FONT, size=16, bold=True, color="FFFFFF")
    for c in range(1, 6):
        ws.cell(row=1, column=c).fill = TITLE_FILL
    ws["A2"] = '="Status: "&Setup!$C$28'
    ws["A2"].font = _f(bold=True, size=13)
    ws["A3"] = '="As of: "&Target!$B$2'
    ws["A3"].font = _f(size=11)
    ws.conditional_formatting.add("A2", FormulaRule(formula=['ISNUMBER(SEARCH("CHECK",A2))'],
                                                    font=Font(name=FONT, color="C00000", bold=True, size=13)))
    blocks = [("ROTH IRA - SELL (do these first)", "S", "U", "W", "C00000"),
              ("ROTH IRA - BUY", "S", "U", "X", "006100"),
              ("TRADITIONAL IRA - SELL (do these first)", "T", "V", "Y", "C00000"),
              ("TRADITIONAL IRA - BUY", "T", "V", "Z", "006100")]
    dv_done = DataValidation(type="list", formula1='"done"', allow_blank=True)
    ws.add_data_validation(dv_done)
    row = 5
    for n, (title, sh_col, usd_col, idx_col, color) in enumerate(blocks):
        ws.cell(row=row, column=1, value=title).font = Font(name=FONT, size=13, bold=True, color=color)
        row += 1
        _header(ws, row, ["#", "Symbol", "Shares", "~ $", "Done"])
        row += 1
        idx = f"Plan!${idx_col}${P0}:${idx_col}${P1}"
        for k in range(1, BLOCK + 1):
            m = f"MATCH({k},{idx},0)"
            ws.cell(row=row, column=1, value=f'=IF(B{row}="","",{k})')
            ws.cell(row=row, column=2, value=f'=IFERROR(INDEX(Plan!$B${P0}:$B${P1},{m}),"")')
            ws.cell(row=row, column=3, value=f'=IFERROR(ABS(INDEX(Plan!${sh_col}${P0}:${sh_col}${P1},{m})),"")')
            ws.cell(row=row, column=4, value=f'=IFERROR(ABS(INDEX(Plan!${usd_col}${P0}:${usd_col}${P1},{m})),"")')
            for c, fmt in ((1, "0"), (2, None), (3, SH), (4, USD)):
                cell = ws.cell(row=row, column=c)
                cell.font = _f(size=12, bold=(c == 2))
                cell.border = BOX
                if fmt:
                    cell.number_format = fmt
            done = ws.cell(row=row, column=5)
            _input(done)
            dv_done.add(done)
            row += 1
        if n in (1, 3):
            acct = "Roth" if n == 1 else "Traditional"
            ref = "Setup!$C$23" if n == 1 else "Setup!$C$24"
            ws.cell(row=row, column=1, value=f"{acct} cash after trades:").font = _f(bold=True, size=11)
            c = ws.cell(row=row, column=4, value=f"={ref}")
            c.number_format = USD
            c.font = _f(bold=True, size=11)
            row += 1
        row += 1
    ws.sheet_view.showGridLines = False
    ws["E5"].comment = Comment("Clear the Done column before re-planning: the rows re-order when "
                               "inputs change.", "model")


def build_workbook(path: Path = DEFAULT_PATH, overwrite: bool = False) -> Path:
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists and holds account data - pass overwrite=True "
                              f"(--force) only if you mean to start over")
    path.parent.mkdir(parents=True, exist_ok=True)
    wb = Workbook()
    ws = wb.active
    ws.title = "Trades"
    for name in ("Guide", "Setup", "Target", "Holdings", "Plan"):
        wb.create_sheet(name)
    _guide(wb["Guide"])
    _setup(wb["Setup"])
    _target(wb["Target"])
    _holdings(wb["Holdings"])
    _plan(wb["Plan"])
    _trades(wb["Trades"])
    wb._sheets = [wb[n] for n in ("Trades", "Setup", "Target", "Holdings", "Plan", "Guide")]
    wb.active = 0
    wb.calculation = CalcProperties(fullCalcOnLoad=True)
    wb.save(path)
    return path


# --------------------------------------------------------------------------
# filling the Target sheet from the report
# --------------------------------------------------------------------------

def fill_target(rows: List[Dict], as_of: str, source: str,
                path: Path = DEFAULT_PATH) -> Dict[str, int]:
    """
    Write the recommendation into Target, keeping what was chosen by hand.

    `rows`: dicts with symbol, model ('J'/'K'), weight, price.  For each symbol
    already present, Take / My weight / Placement are carried over.  A Track K
    row the user has Take = Y on survives even when the model no longer
    recommends it (weight 0), so a held position is never dropped silently.
    """
    path = Path(path)
    wb = load_workbook(path)
    ws = wb["Target"]
    kept: Dict[str, Dict] = {}
    for r in range(T0, T1 + 1):
        sym = ws[f"A{r}"].value
        if sym:
            kept[str(sym).strip().upper()] = {
                "model": ws[f"B{r}"].value, "take": ws[f"E{r}"].value,
                "my": ws[f"F{r}"].value, "place": ws[f"G{r}"].value,
                "price": ws[f"D{r}"].value}
    out = [dict(r, symbol=r["symbol"].upper()) for r in rows]
    seen = {r["symbol"] for r in out}
    carried = 0
    for sym, k in kept.items():
        if sym not in seen and k["model"] == "K" and str(k["take"]).upper() == "Y":
            out.append({"symbol": sym, "model": "K", "weight": 0.0, "price": k["price"]})
            carried += 1
    if len(out) > T1 - T0 + 1:
        raise ValueError(f"{len(out)} target rows exceed the sheet's {T1 - T0 + 1}")
    for r in range(T0, T1 + 1):
        for col in "ABCDEFG":
            ws[f"{col}{r}"].value = None
    for i, row in enumerate(out):
        r = T0 + i
        prev = kept.get(row["symbol"], {})
        ws[f"A{r}"] = row["symbol"]
        ws[f"B{r}"] = row["model"]
        ws[f"C{r}"] = round(float(row["weight"]), 6)
        ws[f"D{r}"] = round(float(row["price"]), 2) if row.get("price") else None
        if row["model"] == "K":
            ws[f"E{r}"] = prev.get("take") or "N"
            ws[f"F{r}"] = prev.get("my")
        ws[f"G{r}"] = prev.get("place") or "Auto"
    ws["B2"] = as_of
    ws["B3"] = source
    wb.calculation = CalcProperties(fullCalcOnLoad=True)
    wb.save(path)
    return {"rows": len(out), "carried_k": carried}
