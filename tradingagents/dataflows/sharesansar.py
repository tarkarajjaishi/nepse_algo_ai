"""Full statutory quarterly filings for NEPSE companies, from sharesansar.com.

``nepsetrading.py`` gives a five-year trend of four or five headline lines.
This gives the *filing itself* for the latest quarter — 35 balance-sheet lines,
33 income-statement lines and 18 regulatory ratios for a commercial bank — which
is what a fundamentals analyst actually needs to reason about asset quality and
capital adequacy. The two are complementary: trend there, depth here.

Preferred over the nepsetrading reader where both apply. sharesansar publishes
``robots.txt`` as ``User-agent: * / Disallow:`` — crawling explicitly permitted —
whereas the nepsetrading path depends on that site's paywall staying open, which
it should not.

Mechanics: the company page loads its report tabs by POSTing to
``/company-quarterly-report`` with the company id, symbol and sector it renders
into the DOM, plus a Laravel CSRF token. A bare POST returns HTTP 419, so the
token and session cookie are collected from the company page first — the same
sequence a browser performs.

**There is no cash-flow statement.** Nepali quarterly filings carry the balance
sheet, profit and loss, and ratios; cash flow appears only in the annual report
PDF. Verified against sharesansar, nepsetrading and merolagani — none exposes
"operating/investing/financing activities". ``get_cashflow`` keeps its sentinel.
"""

from __future__ import annotations

import os
from typing import Annotated

import requests
from parsel import Selector

from .errors import NoMarketDataError, VendorNotConfiguredError

BASE_URL = os.environ.get("SHARESANSAR_BASE_URL", "https://www.sharesansar.com").rstrip("/")
REPORT_PATH = "/company-quarterly-report"
TIMEOUT_SECONDS = 45
_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; tradingagents/0.3; +research)"}

# {symbol: [(caption, [(label, value), ...]), ...]}
_cache: dict[str, list[tuple[str, list[tuple[str, str]]]]] = {}

# The filing arrives as three unlabelled tables in a fixed order. Each is
# identified by a line only it contains, so a reordering upstream cannot
# silently swap the balance sheet for the income statement.
_FINGERPRINTS = (
    ("Balance sheet", ("cash and cash equivalent", "due from nepal rastra bank",
                       "total assets", "deposits from customers")),
    ("Income statement", ("interest income", "net interest income",
                          "profit for the period", "fees and commission income")),
    ("Key ratios", ("basic earnings per share", "capital fund to rwa",
                    "non performing loan", "diluted earnings per share")),
)


def _session_and_page(symbol: str) -> tuple[requests.Session, Selector]:
    session = requests.Session()
    session.headers.update(_HEADERS)
    url = f"{BASE_URL}/company/{symbol.strip().lower()}"
    try:
        resp = session.get(url, timeout=TIMEOUT_SECONDS)
    except requests.exceptions.ConnectionError as exc:
        raise VendorNotConfiguredError(f"cannot reach {BASE_URL}: {exc}") from exc
    except requests.exceptions.RequestException as exc:
        raise NoMarketDataError(symbol, None, f"sharesansar fetch failed: {exc}") from exc
    if resp.status_code == 404:
        raise NoMarketDataError(symbol, None, f"sharesansar has no company page at {url}")
    if not resp.ok:
        raise NoMarketDataError(symbol, None, f"sharesansar returned HTTP {resp.status_code}")
    return session, Selector(resp.text)


def _classify(rows: list[tuple[str, str]]) -> str | None:
    labels = " ".join(label.lower() for label, _ in rows)
    for caption, markers in _FINGERPRINTS:
        if sum(m in labels for m in markers) >= 2:
            return caption
    return None


def fetch_filing(symbol: str) -> list[tuple[str, list[tuple[str, str]]]]:
    """The latest quarterly filing as ``[(caption, [(label, value), ...]), ...]``."""
    symbol = symbol.strip().upper()
    if symbol in _cache:
        return _cache[symbol]

    session, page = _session_and_page(symbol)
    fields = {k: (page.css(f"#{k}::text").get() or "").strip()
              for k in ("companyid", "symbol", "sector")}
    if not fields["companyid"]:
        raise NoMarketDataError(
            symbol, None,
            "sharesansar company page carried no company id — its markup changed",
        )
    payload = {
        "company": fields["companyid"],
        "symbol": fields["symbol"] or symbol,
        "sector": fields["sector"],
        "_token": page.css('input[name="_token"]::attr(value)').get() or "",
    }
    resp = session.post(
        f"{BASE_URL}{REPORT_PATH}", data=payload,
        headers={"X-Requested-With": "XMLHttpRequest",
                 "Referer": f"{BASE_URL}/company/{symbol.lower()}"},
        timeout=TIMEOUT_SECONDS,
    )
    if resp.status_code == 419:
        raise NoMarketDataError(
            symbol, None,
            "sharesansar rejected the CSRF token (419) — the token or session "
            "handling on the company page changed",
        )
    if not resp.ok:
        raise NoMarketDataError(symbol, None, f"quarterly report HTTP {resp.status_code}")

    tables: list[tuple[str, list[tuple[str, str]]]] = []
    for table in Selector(resp.text).css("table"):
        rows: list[tuple[str, str]] = []
        header = ""
        for tr in table.css("tr"):
            cells = [" ".join(t.split()) for t in tr.css("::text").getall() if t.strip()]
            if len(cells) < 2:
                continue
            if not header and "particular" in cells[0].lower():
                header = cells[1]
                continue
            rows.append((cells[0], cells[1]))
        caption = _classify(rows) if rows else None
        if caption:
            tables.append((f"{caption} — {header}" if header else caption, rows))

    if not tables:
        raise NoMarketDataError(
            symbol, None,
            f"no recognisable filing tables for {symbol}; the report layout changed "
            f"or this company has filed nothing",
        )
    _cache[symbol] = tables
    return tables


def _render(symbol: str, caption_prefix: str) -> str:
    tables = fetch_filing(symbol)
    for caption, rows in tables:
        if not caption.lower().startswith(caption_prefix.lower()):
            continue
        lines = [
            f"# {caption} for {symbol.upper()} (NEPSE)",
            "",
            f"Source: sharesansar.com/company/{symbol.lower()} — statutory quarterly filing.",
            "",
            "| Particulars | Amount |",
            "|---|---:|",
        ]
        lines += [f"| {label} | {value} |" for label, value in rows]
        lines += [
            "",
            "> Figures are as filed, in thousands of NPR (Rs. in '000) unless the "
            "row is a ratio or a per-share amount. Values are passed through "
            "unconverted.",
        ]
        return "\n".join(lines)

    found = ", ".join(c for c, _ in tables) or "none"
    raise NoMarketDataError(
        symbol, None, f"no {caption_prefix.lower()} in the filing; found: {found}"
    )


def get_sharesansar_balance_sheet(
    ticker: Annotated[str, "NEPSE ticker, e.g. NABIL"],
    freq: Annotated[str, "reporting frequency; NEPSE files quarterly"] = "quarterly",
    curr_date: Annotated[str, "current date, yyyy-mm-dd"] = "",
) -> str:
    """Balance sheet as filed for the latest quarter."""
    return _render(ticker, "Balance sheet")


def get_sharesansar_income_statement(
    ticker: Annotated[str, "NEPSE ticker, e.g. NABIL"],
    freq: Annotated[str, "reporting frequency; NEPSE files quarterly"] = "quarterly",
    curr_date: Annotated[str, "current date, yyyy-mm-dd"] = "",
) -> str:
    """Profit and loss as filed for the latest quarter."""
    return _render(ticker, "Income statement")


def get_sharesansar_ratios(
    ticker: Annotated[str, "NEPSE ticker, e.g. NABIL"],
    freq: Annotated[str, "reporting frequency; NEPSE files quarterly"] = "quarterly",
    curr_date: Annotated[str, "current date, yyyy-mm-dd"] = "",
) -> str:
    """Regulatory ratios: EPS, capital adequacy, NPL, as filed."""
    return _render(ticker, "Key ratios")


def demo() -> None:
    """Self-check. Offline assertions; add --live to hit sharesansar."""
    import sys

    balance = [("Cash and cash equivalent", "7,081,291.00"), ("Total assets", "1.00")]
    income = [("Interest income", "1"), ("Net interest income", "2")]
    ratios = [("Basic Earnings Per Share(Annualized EPS)", "27.74"), ("Capital fund to RWA", "12.37")]
    assert _classify(balance) == "Balance sheet"
    assert _classify(income) == "Income statement"
    assert _classify(ratios) == "Key ratios"
    # One incidental match must not be enough to claim a table.
    assert _classify([("Interest income", "1")]) is None
    assert _classify([("Something else", "1")]) is None
    print("offline checks passed")

    if "--live" in sys.argv:
        tables = fetch_filing("NABIL")
        print(f"live: {len(tables)} tables")
        for caption, rows in tables:
            print(f"   {caption}  ({len(rows)} rows)")
        text = get_sharesansar_balance_sheet("NABIL")
        assert "Cash and cash equivalent" in text
        print(text.splitlines()[0])
        assert "Interest income" in get_sharesansar_income_statement("NABIL")
        print("live checks passed")


if __name__ == "__main__":
    demo()
