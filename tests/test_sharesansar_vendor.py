"""Statutory filing reader (sharesansar), offline only.

The filing arrives as three unlabelled tables. Misclassifying one would hand the
fundamentals analyst an income statement labelled "balance sheet" — plausible,
wrong, and invisible. Classification is therefore fingerprint-based and needs two
independent markers, not one.
"""

import pytest

from tradingagents.dataflows.sharesansar import _classify

BALANCE = [("Cash and cash equivalent", "7,081,291.00"),
           ("Due from Nepal Rastra Bank", "24,380,465.00"),
           ("Total assets", "1.00")]
INCOME = [("Interest income", "37,955,399.00"), ("Interest expense", "20,946,111.00"),
          ("Net interest income", "17,009,288.00")]
RATIOS = [("Basic Earnings Per Share(Annualized EPS)", "27.74"),
          ("Capital fund to RWA", "12.37"), ("Diluted Earnings Per Share", "27.74")]


@pytest.mark.parametrize(("rows", "expected"), [
    (BALANCE, "Balance sheet"), (INCOME, "Income statement"), (RATIOS, "Key ratios"),
])
def test_each_filing_table_is_identified(rows, expected):
    assert _classify(rows) == expected


def test_one_incidental_match_is_not_enough():
    """'Interest income' alone could appear anywhere; two markers are required."""
    assert _classify([("Interest income", "1")]) is None


def test_unknown_table_is_rejected_rather_than_guessed():
    assert _classify([("Broker", "5"), ("Quantity", "10")]) is None
    assert _classify([]) is None


def test_classification_is_case_insensitive():
    shouty = [(label.upper(), v) for label, v in BALANCE]
    assert _classify(shouty) == "Balance sheet"


def test_balance_and_income_are_not_confused():
    """Both mention cash-like and income-like words; the fingerprints must not overlap."""
    assert _classify(BALANCE) != _classify(INCOME)
