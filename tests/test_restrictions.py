"""
Tests for the compliance restriction list.

These are not tests of a modelling choice, so they are written to fail loudly
and specifically.  The two failure modes that matter are opposites:

  UNDER-BLOCKING  a restricted name reaches a recommendation.  Tested by
                  asserting the live list parses to the names DLR actually
                  sent, and that each key catches what it is supposed to.

  OVER-BLOCKING   an unrelated business is excluded because a name key is too
                  loose.  Tested against the specific near-misses that the
                  current needles could plausibly hit — Cencora for the reused
                  COR ticker, Cloudflare and VeriSign for the rejected
                  "Internet Services & Infrastructure" industry key, Marathon
                  for "MARA", Environmental for "IREN".

Run:  python -m pytest tests/test_restrictions.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from momentum.restrictions import (  # noqa: E402
    RestrictedNameError, check_symbols, describe, filter_screen,
    load_restrictions, match_names,
)


# --- the list parses to what was actually sent ----------------------------

def test_employer_is_present():
    """DLR was not on the list DLR sent.  It must be restricted anyway."""
    assert "DLR" in load_restrictions()["symbols"]


def test_every_us_symbol_on_the_dlr_list_is_restricted():
    sent = ["AMT", "APLD", "BTBT", "BTDR", "BXDC", "BIP", "CIFR", "CLSK",
            "CDP", "CORZ", "DBRG", "EQIX", "FRMI", "GLXY", "GDS", "HUT",
            "IREN", "IRM", "KEEL", "MARA", "RIOT", "WULF", "WYFI"]
    missing = sorted(set(sent) - set(load_restrictions()["symbols"]))
    assert not missing, f"dropped from the DLR list: {missing}"


def test_earlier_list_is_not_lost():
    """A 'more complete' list replaced the old one; nothing may fall out."""
    assert "VNET" in load_restrictions()["symbols"]


def test_every_foreign_line_is_carried():
    sent = ["A17U SP", "AJBU SP", "DCRU SP", "ME8U SP", "NTDU SP",
            "DGT AU", "NXT AU", "MP1 AU"]
    missing = sorted(set(sent) - set(load_restrictions()["foreign"]))
    assert not missing, f"dropped foreign lines: {missing}"


def test_every_entity_row_has_a_category():
    """Category is how the list is reviewed; a blank one hides a name."""
    blank = [e["value"] for e in load_restrictions()["entries"]
             if e["kind"] in ("symbol", "foreign-symbol") and not e["category"]]
    assert not blank, f"uncategorized: {blank}"


# --- the keys catch what they should --------------------------------------

@pytest.mark.parametrize("description,expected", [
    ("Equinix, Inc.", "Equinix"),
    ("Digital Realty Trust Inc", "Digital Realty Trust"),      # no comma
    ("Keppel DC REIT", "Keppel DC REIT"),                      # SG, no US line
    ("NTT DC REIT Investment Corporation", "NTT DC REIT"),
    ("CapitaLand Ascendas Real Estate Inv Trust", "CapitaLand Ascendas REIT"),
    ("Mapletree Industrial Trust", "Mapletree Industrial Trust"),
    ("DigiCo Infrastructure REIT", "DigiCo Infrastructure REIT"),
    ("Digital Core REIT", "Digital Core REIT"),
    ("NEXTDC Limited", "NEXTDC"),
    ("Megaport Ltd", "Megaport"),
    ("Hut 8 Corp.", "Hut 8"),
    ("WhiteFiber, Inc.", "WhiteFiber"),
])
def test_name_key_catches_restricted_issuers(description, expected):
    assert match_names([description])[0] == expected


def test_name_key_catches_a_derivative_the_symbol_list_cannot():
    """
    The case that justifies the name key existing.

    'Defiance Daily Target 2X Long IREN ETF' trades as IRE.  No symbol
    blocklist would catch it, and a 2x leveraged wrapper on a restricted
    issuer is more exposure to that issuer, not less.
    """
    assert match_names(["Defiance Daily Target 2X Long IREN ETF"])[0] == "IREN"


def test_industry_key_catches_a_ticker_not_on_any_list():
    frame = pd.DataFrame({"symbol": ["ZZZZ"],
                          "sub-industry": ["Data Center REITs"]})
    kept, dropped = filter_screen(frame)
    assert len(kept) == 0
    assert dropped["restricted_by"].iloc[0] == "industry: data center reits"


# --- and nothing else -----------------------------------------------------

@pytest.mark.parametrize("description", [
    "Cencora, Inc.",                 # owns COR, the ex-CyrusOne ticker
    "Cloudflare, Inc.",              # Internet Services & Infrastructure
    "VeriSign, Inc.",                # Internet Services & Infrastructure
    "Marathon Petroleum Corporation",   # near "MARA"
    "Marathon Oil Corporation",
    "Environmental Solutions Worldwide",  # near "IREN"
    "Apple Inc.",
    "Prologis, Inc.",                # an industrial REIT, deliberately allowed
])
def test_name_key_does_not_over_match(description):
    assert match_names([description])[0] is None


def test_tower_reits_are_blocked_by_symbol_as_well_as_industry():
    """
    The industry key alone would not bite.

    liquid_pool.csv carries no industry column, so only the symbol and name
    keys fire against it — which is where CCI and SBAC actually live.  The
    industry key is the forward-looking half; these two are the half that
    works today.
    """
    r = load_restrictions()
    assert "telecom tower reits" in r["industries"]
    for sym in ("AMT", "CCI", "SBAC"):
        assert sym in r["symbols"]


def test_telecom_operators_are_not_tower_reits():
    """Blocking towers must not reach the carriers that rent them."""
    assert match_names(["Verizon Communications Inc.",
                        "T-Mobile US, Inc.",
                        "AT&T Inc."]) == [None, None, None]


def test_internet_services_industry_is_not_a_key():
    """
    Rejected deliberately: in the current screen it holds Cloudflare and
    VeriSign and no restricted name.  If someone adds it, this fails.
    """
    assert "internet services & infrastructure" not in \
        load_restrictions()["industries"]


# --- failure behaviour ----------------------------------------------------

def test_check_symbols_raises_rather_than_filters():
    with pytest.raises(RestrictedNameError, match="EQIX"):
        check_symbols(["AAPL", "EQIX", "MSFT"], "a test book")


def test_missing_file_raises_rather_than_permitting_everything():
    with pytest.raises(RestrictedNameError, match="Refusing to run"):
        load_restrictions(REPO_ROOT / "no_such_restricted_file.csv")


def test_unknown_kind_raises(tmp_path):
    """A typo'd Kind must not silently drop the row it was meant to block."""
    bad = tmp_path / "restricted.csv"
    bad.write_text("Kind,Value,Name,Category,Reason\n"
                   "symbol,DLR,Digital Realty,employer,employer\n"
                   "sybmol,EQIX,Equinix,peer,typo\n", encoding="utf-8")
    with pytest.raises(RestrictedNameError, match="unknown Kind"):
        load_restrictions(bad)


def test_empty_symbol_list_raises(tmp_path):
    empty = tmp_path / "restricted.csv"
    empty.write_text("Kind,Value,Name,Category,Reason\n"
                     "sub-industry,Data Center REITs,,dc,x\n", encoding="utf-8")
    with pytest.raises(RestrictedNameError, match="zero restricted symbols"):
        load_restrictions(empty)


# --- the live data files are clean ----------------------------------------

@pytest.mark.parametrize("filename,symbol_col", [
    ("universe.csv", "Symbol"),
    ("snapshots", None),
])
def test_live_universe_holds_no_restricted_name(filename, symbol_col):
    if symbol_col is None:
        pytest.skip("directory, not a symbol file")
    path = REPO_ROOT / filename
    if not path.exists():
        pytest.skip(f"{filename} not present")
    check_symbols(pd.read_csv(path)[symbol_col], filename)


def test_describe_lists_every_entity():
    text = describe()
    r = load_restrictions()
    for sym in r["symbols"] + r["foreign"]:
        assert sym in text, f"{sym} missing from describe()"
