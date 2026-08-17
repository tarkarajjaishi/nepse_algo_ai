"""Quarterly financial statements for NEPSE companies, from nepsetrading.com.

NEPSE publishes no machine-readable financials — quarterly reports exist only as
PDF filings — so ``nepse.py`` returns an explicit "unavailable" sentinel for
statements. This vendor fills that gap from nepsetrading.com, which parses those
filings into tables.

**This is a stopgap, and it is fragile by construction.** It reads the
server-rendered HTML of ``/quarterly?symbol=<SYMBOL>``, because that site exposes
no public JSON API. Two consequences worth stating plainly:

1. The page markup is a Next.js render. A component refactor changes the DOM and
   this parser stops matching. It fails loud (a typed vendor error), never quiet.
2. The data is currently readable without authentication even though the page is
   sold as a subscription. If that is closed server-side, this breaks immediately.

The durable replacement is a JSON endpoint on the site's own API. When one
exists, keep the vendor functions and swap ``_fetch_tables`` for a client call —
nothing downstream needs to change.

Figures keep the Nepali numbering they are published in: **Ar** = arab (10^9),
**Cr** = crore (10^7). They are passed through verbatim rather than converted,
because a wrong scale factor in a financial report is worse than an unfamiliar
unit, and the header below tells the analyst how to read them.
"""

from __future__ import annotations

import os
import re
from typing import Annotated

import requests
from parsel import Selector

from .errors import NoMarketDataError, VendorNotConfiguredError

BASE_URL = os.environ.get("NEPSETRADING_BASE_URL", "https://nepsetrading.com").rstrip("/")
TIMEOUT_SECONDS = 45

# One fetch per symbol per process. The page is ~650 KB and this is somebody's
# production site; three statement tools asking for the same symbol must not
# become three downloads.
_page_cache: dict[str, list[tuple[list[str], list[list[str]]]]] = {}

UNITS_NOTE = (
    "Figures use Nepali numbering as published: Ar = arab (1,000,000,000), "
    "Cr = crore (10,000,000). Percentages are marked (%). Columns are fiscal "
    "years in Bikram Sambat (e.g. 82/83) at the same quarter, oldest first, "
    "with the final column the year-on-year change."
)

# First cell of the header row for the three statement tables on the page.
_ITEM_HEADER = "Item"
_YEAR = re.compile(r"\d{2}/\d{2}")
_QUARTER = re.compile(r"Q[1-4]")


def _cells(row) -> list[str]:
    return [" ".join(t.split()) for t in row.css("::text").getall() if t.strip()]


def _merge_period_header(header: list[str]) -> list[str]:
    """Join split fiscal-year/quarter header cells into one label per column.

    The page renders the year and the quarter as separate text nodes, so a raw
    read gives ``[Item, 78/79, Q3, 79/80, Q3, ...]`` — 12 headings for 7 columns
    of data. Left unmerged the markdown table misaligns and every figure sits
    under the wrong year, which is worse than having no table at all.
    """
    merged: list[str] = []
    i = 0
    while i < len(header):
        cell = header[i]
        nxt = header[i + 1] if i + 1 < len(header) else ""
        if _YEAR.fullmatch(cell) and _QUARTER.fullmatch(nxt):
            merged.append(f"{cell} {nxt}")
            i += 2
        else:
            merged.append(cell)
            i += 1
    return merged


def _fetch_tables(symbol: str) -> list[tuple[list[str], list[list[str]]]]:
    """Return every ``Item x fiscal-year`` table for ``symbol``.

    Swap this one function for an API client when a JSON endpoint exists.
    """
    symbol = symbol.strip().upper()
    if symbol in _page_cache:
        return _page_cache[symbol]

    url = f"{BASE_URL}/quarterly?symbol={symbol}"
    try:
        resp = requests.get(url, timeout=TIMEOUT_SECONDS)
    except requests.exceptions.ConnectionError as exc:
        raise VendorNotConfiguredError(
            f"cannot reach {BASE_URL} for quarterly financials: {exc}"
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise NoMarketDataError(symbol, None, f"quarterly fetch failed: {exc}") from exc

    if resp.status_code in (401, 403):
        # The figures are readable unauthenticated today; if that is closed this
        # is the branch that fires, and it should say so rather than look empty.
        raise VendorNotConfiguredError(
            f"nepsetrading.com now requires authentication for {symbol} "
            f"(HTTP {resp.status_code}) — this vendor needs an API endpoint or token"
        )
    if not resp.ok:
        raise NoMarketDataError(symbol, None, f"quarterly page returned HTTP {resp.status_code}")

    tables = []
    for table in Selector(resp.text).css("table"):
        rows = table.css("tr")
        if not rows:
            continue
        header = _cells(rows[0])
        if not header or header[0] != _ITEM_HEADER:
            continue
        body = [c for c in (_cells(r) for r in rows[1:]) if c]
        if body:
            tables.append((_merge_period_header(header), body))

    if not tables:
        raise NoMarketDataError(
            symbol, None,
            f"no quarterly tables found for {symbol} at {url} — the page layout "
            f"changed, or this symbol has no published statements",
        )
    _page_cache[symbol] = tables
    return tables


def _render(symbol: str, wanted: tuple[str, ...], title: str) -> str:
    """Render whichever table contains ``wanted`` row labels as markdown."""
    tables = _fetch_tables(symbol)
    for header, body in tables:
        labels = {row[0] for row in body}
        if not labels & set(wanted):
            continue
        lines = [
            f"# {title} for {symbol.upper()} (NEPSE)",
            "",
            f"Source: nepsetrading.com/quarterly?symbol={symbol.upper()}",
            "",
            "| " + " | ".join(header) + " |",
            "|" + "---|" * len(header),
        ]
        for row in body:
            padded = row + [""] * (len(header) - len(row))
            lines.append("| " + " | ".join(padded[:len(header)]) + " |")
        lines += ["", f"> {UNITS_NOTE}"]
        return "\n".join(lines)

    raise NoMarketDataError(
        symbol, None,
        f"no table for {title.lower()} — looked for rows {wanted}, "
        f"found {[r[0] for h, b in tables for r in b][:8]}",
    )


def get_nepsetrading_income_statement(
    ticker: Annotated[str, "NEPSE ticker, e.g. NABIL"],
    freq: Annotated[str, "reporting frequency; only quarterly is published"] = "quarterly",
    curr_date: Annotated[str, "current date, yyyy-mm-dd"] = "",
) -> str:
    """Quarterly income statement across the published fiscal years.

    Signature matches the ``get_income_statement`` vendor contract, which passes
    a reporting frequency between the ticker and the date. NEPSE companies file
    quarterly, so ``freq`` is accepted and noted rather than honoured — an annual
    request still returns the quarterly series, which is what exists.
    """
    return _render(
        ticker,
        ("Net Interest Income", "Net Profit", "Distributable Profit", "Impairment Charge"),
        "Income statement",
    )


def get_nepsetrading_balance_sheet(
    ticker: Annotated[str, "NEPSE ticker, e.g. NABIL"],
    freq: Annotated[str, "reporting frequency; only quarterly is published"] = "quarterly",
    curr_date: Annotated[str, "current date, yyyy-mm-dd"] = "",
) -> str:
    """Quarterly balance sheet across the published fiscal years."""
    return _render(
        ticker,
        ("Total Assets", "Deposits", "Total Liabilities", "Paid Up Capital",
         "Loans & Advances", "Reserves & Surplus"),
        "Balance sheet",
    )


def get_nepsetrading_ratios(
    ticker: Annotated[str, "NEPSE ticker, e.g. NABIL"],
    freq: Annotated[str, "reporting frequency; only quarterly is published"] = "quarterly",
    curr_date: Annotated[str, "current date, yyyy-mm-dd"] = "",
) -> str:
    """Per-quarter ratios: margins, EPS, growth, NPL, P/E and P/BV where published."""
    return _render(
        ticker,
        ("EPS (Annu.)", "EPS (TTM)", "Net Interest Margin (%)",
         "Net Profit Margin (%)", "Earnings Growth (%)"),
        "Key ratios",
    )


def demo() -> None:
    """Self-check. Offline assertions; add --live to hit the real page."""
    import sys

    html = """<table><tr><th>Item</th><th>81/82</th><th>82/83</th><th>YoY</th></tr>
      <tr><td>Net Profit</td><td>5.05 Ar</td><td>6.76 Ar</td><td>+33.92%</td></tr></table>
      <table><tr><th>Broker</th><th>Qty</th></tr><tr><td>ABC</td><td>10</td></tr></table>"""
    tables = []
    for t in Selector(html).css("table"):
        rows = t.css("tr")
        head = _cells(rows[0])
        if head and head[0] == _ITEM_HEADER:
            tables.append((head, [c for c in (_cells(r) for r in rows[1:]) if c]))
    # Only the Item table is picked up; broker/mutual-fund tables are ignored.
    assert len(tables) == 1, tables
    assert tables[0][1][0][0] == "Net Profit"
    assert "Ar" in tables[0][1][0][1], "units must survive verbatim, not be converted"
    print("offline checks passed")

    if "--live" in sys.argv:
        text = get_nepsetrading_income_statement("NABIL")
        assert "Net Profit" in text and "Ar" in text, text[:200]
        print(text.splitlines()[0], "|", len(text), "chars")
        bal = get_nepsetrading_balance_sheet("NABIL")
        assert "Total Assets" in bal
        print(bal.splitlines()[0])
        print(get_nepsetrading_ratios("NABIL").splitlines()[0])
        print("live checks passed")


if __name__ == "__main__":
    demo()
