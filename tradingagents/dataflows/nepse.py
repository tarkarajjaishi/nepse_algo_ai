"""NEPSE (Nepal Stock Exchange) data vendor.

Backed by a self-hosted NepseAPI-Unofficial server::

    git clone https://github.com/surajrimal07/NepseAPI-Unofficial
    pip install -r requirements.txt
    python server.py          # FastAPI on port 8000

Point ``NEPSE_API_BASE_URL`` at it if it is not on ``http://localhost:8000``.
The project's public demo instance (nepseapi.surajrimal.dev) is dead — every
path returns a plain-text 404 — so a local server is the only working option.

Coverage against the vendor contract in ``interface.VENDOR_METHODS``:

    get_stock_data            /PriceVolumeHistory   daily OHLCV, ~365d window
    get_indicators            computed locally by stockstats from that OHLCV
    get_fundamentals          /CompanyDetails
    get_balance_sheet         unavailable ─┐
    get_cashflow              unavailable  │ NEPSE publishes quarterly reports
    get_income_statement      unavailable  │ as PDFs only — no JSON source.
    get_news                  unavailable  │ The API exposes no news endpoint.
    get_global_news           unavailable  │
    get_insider_transactions  unavailable ─┘

The unavailable ones return an explicit sentinel string instead of raising.
``news_data`` and ``fundamental_data`` are NOT in ``interface.OPTIONAL_CATEGORIES``,
so raising would abort the whole graph; a sentinel lets the run finish with the
gap stated plainly in the report, which is what stops the analyst inventing
numbers to fill the hole.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Annotated, Any

import pandas as pd
import requests

from .errors import NoMarketDataError, VendorNotConfiguredError, VendorRateLimitError

DEFAULT_BASE_URL = "http://localhost:8000"
BASE_URL_ENV = "NEPSE_API_BASE_URL"
TIMEOUT_SECONDS = 20

# NEPSE closes for long public-holiday runs (Dashain and Tihar together can shut
# the exchange for most of two weeks). The shared staleness guard's 10-day
# default false-fires across those, so the NEPSE path uses a wider budget.
#
# Do not assume which weekdays it trades: the exchange moved from Sunday-Thursday
# to Monday-Friday in April 2026 (last Sunday session 2026-04-05, first Friday
# 2026-04-10). Saturday is the only day never traded in the served history.
# Anything that needs the trading week must derive it from the returned bars.
# ponytail: fixed 21-day budget; a real Nepali trading calendar would be exact.
MAX_STALE_DAYS = 21

# NEPSE's price-history records use camelCase keys. Each framework column lists
# the candidate source keys in priority order; the lookup also retries
# case-insensitively before giving up, so a server that renames `openPrice` to
# `open_price` still resolves.
_REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    "Date": ("businessDate", "business_date", "date", "tradeDate"),
    "High": ("highPrice", "high", "high_price"),
    "Low": ("lowPrice", "low", "low_price"),
    "Close": ("closePrice", "close", "close_price", "lastTradedPrice"),
    "Volume": (
        "totalTradedQuantity",
        "totalTradeQuantity",
        "volume",
        "total_traded_quantity",
    ),
}

# NEPSE does not publish an opening price. Verified live against
# /PriceVolumeHistory on 2026-08-17: records carry only businessDate, totalTrades,
# totalTradedQuantity, totalTradedValue, highPrice, lowPrice, closePrice. Open is
# therefore optional rather than required — every indicator this framework asks
# for (SMA/EMA/MACD/RSI/Bollinger/ATR/VWMA) derives from close, high, low and
# volume, so nothing downstream needs it. Kept as a lookup in case a future
# server version adds it; never synthesised from the prior close, which would put
# a fabricated price in front of the analyst.
_OPTIONAL_FIELDS: dict[str, tuple[str, ...]] = {
    "Open": ("openPrice", "open", "open_price"),
}

_COLUMN_ORDER = ("Date", "Open", "High", "Low", "Close", "Volume")

_UNAVAILABLE = (
    "<unavailable: NEPSE has no {what} source. The NepseAPI server exposes no "
    "such endpoint, and NEPSE publishes this only as PDF filings. Treat {what} "
    "as unknown for this analysis — do not infer or estimate it.>"
)


def _base_url() -> str:
    return os.environ.get(BASE_URL_ENV, DEFAULT_BASE_URL).rstrip("/")


def _get(path: str, **params: Any) -> Any:
    """GET one NepseAPI endpoint, mapping transport failures to vendor errors."""
    url = f"{_base_url()}{path}"
    try:
        resp = requests.get(url, params=params or None, timeout=TIMEOUT_SECONDS)
    except requests.exceptions.ConnectionError as exc:
        # By far the most common failure: the user has not started the server.
        # Say so explicitly rather than surfacing a bare socket error.
        raise VendorNotConfiguredError(
            f"cannot reach the NepseAPI server at {_base_url()} — start it with "
            f"`python server.py` (see nepse.py header), or set {BASE_URL_ENV}. "
            f"Original error: {exc}"
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise NoMarketDataError(
            str(params.get("symbol", "")), None, f"NEPSE request to {path} failed: {exc}"
        ) from exc

    if resp.status_code == 429:
        raise VendorRateLimitError(f"NEPSE server rate-limited {path} (60 req/min cap)")
    if resp.status_code == 404:
        raise NoMarketDataError(
            str(params.get("symbol", "")),
            None,
            f"NEPSE server has no {path} route (is it the NepseAPI-Unofficial server?)",
        )
    if not resp.ok:
        raise NoMarketDataError(
            str(params.get("symbol", "")),
            None,
            f"NEPSE {path} returned HTTP {resp.status_code}",
        )
    try:
        return resp.json()
    except ValueError as exc:
        raise NoMarketDataError(
            str(params.get("symbol", "")), None, f"NEPSE {path} returned non-JSON: {exc}"
        ) from exc


def _pick(
    record: dict,
    candidates: tuple[str, ...],
    column: str,
    symbol: str,
    *,
    required: bool = True,
) -> Any:
    """Resolve one OHLCV column from a NEPSE record, tolerating key renames.

    Returns None for an absent optional column; raises for an absent required one.
    """
    for name in candidates:
        if record.get(name) is not None:
            return record[name]
    lowered = {k.lower(): v for k, v in record.items()}
    for name in candidates:
        if lowered.get(name.lower()) is not None:
            return lowered[name.lower()]
    if not required:
        return None
    # Listing the keys actually returned turns a schema drift into a one-line
    # fix (add the new key to _REQUIRED_FIELDS) instead of a debugging session.
    raise NoMarketDataError(
        symbol,
        None,
        f"NEPSE price record has no field for {column!r}; tried {candidates}, "
        f"record has {sorted(record)}",
    )


def _records(payload: Any) -> list[dict]:
    """Unwrap the list of rows from however the server nests it."""
    if isinstance(payload, dict):
        for key in ("content", "data", "results"):
            if isinstance(payload.get(key), list):
                return payload[key]
        return []
    return payload if isinstance(payload, list) else []


def _normalize_row(record: dict, symbol: str) -> dict:
    """One NEPSE price record -> framework column names, dropping absent optionals."""
    row = {c: _pick(record, n, c, symbol) for c, n in _REQUIRED_FIELDS.items()}
    for column, names in _OPTIONAL_FIELDS.items():
        value = _pick(record, names, column, symbol, required=False)
        if value is not None:
            row[column] = value
    return row


def fetch_nepse_ohlcv(symbol: str) -> pd.DataFrame:
    """Daily OHLCV for one NEPSE symbol as a Date/Open/High/Low/Close/Volume frame.

    Shape matches what ``stockstats_utils._clean_dataframe`` expects, so the
    existing indicator engine works on this untouched.

    The server calls ``getCompanyPriceVolumeHistory`` without dates, which
    defaults to the last 365 days capped at 500 rows — enough for a 200-SMA,
    not enough to backtest a date more than a year old.
    """
    symbol = symbol.strip().upper()
    # Accrue the benchmark series alongside price fetches — see
    # record_nepse_index_close for why it has to be built up over time.
    record_nepse_index_close()
    rows = _records(_get("/PriceVolumeHistory", symbol=symbol))
    if not rows:
        raise NoMarketDataError(
            symbol, None, "NEPSE /PriceVolumeHistory returned no rows"
        )

    frame = pd.DataFrame([_normalize_row(row, symbol) for row in rows])
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame = frame.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    if frame.empty:
        raise NoMarketDataError(symbol, None, "no NEPSE rows had a parseable date")
    # Stable OHLCV order, minus whatever this server does not serve.
    return frame[[c for c in _COLUMN_ORDER if c in frame.columns]]


def get_nepse_stock_data(
    symbol: Annotated[str, "NEPSE ticker symbol, e.g. NABIL"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
) -> str:
    """OHLCV over a date range, formatted like the other vendors' CSV output."""
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    frame = fetch_nepse_ohlcv(symbol)

    window = frame[(frame["Date"] >= start) & (frame["Date"] <= end)]
    if window.empty:
        available = f"{frame['Date'].min().date()} to {frame['Date'].max().date()}"
        raise NoMarketDataError(
            symbol,
            None,
            f"no NEPSE rows between {start_date} and {end_date}; the server only "
            f"serves ~1y of history (has {available})",
        )

    window = window.copy()
    window["Date"] = window["Date"].dt.strftime("%Y-%m-%d")
    for col in ("Open", "High", "Low", "Close"):
        if col in window.columns:
            window[col] = pd.to_numeric(window[col], errors="coerce").round(2)

    header = (
        f"# NEPSE data for {symbol.upper()} from {start_date} to {end_date}\n"
        f"# Total records: {len(window)}\n"
        f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"# Source: NepseAPI /PriceVolumeHistory at {_base_url()}\n"
    )
    if "Open" not in window.columns:
        # Say it plainly in the data the analyst reads, so a missing column is
        # never mistaken for a gap this vendor could have filled.
        header += (
            "# NOTE: NEPSE publishes no opening price. There is no Open column, "
            "and none was inferred. Do not reference or estimate the open.\n"
        )
    return header + "\n" + window.to_csv(index=False)


def get_nepse_fundamentals(
    symbol: Annotated[str, "NEPSE ticker symbol, e.g. NABIL"],
    curr_date: Annotated[str, "current date, yyyy-mm-dd"] = "",
) -> str:
    """Company profile and ratios from /CompanyDetails, rendered as markdown."""
    symbol = symbol.strip().upper()
    payload = _get("/CompanyDetails", symbol=symbol)
    if isinstance(payload, dict):
        # The endpoint nests the useful part under one of a few wrappers
        # depending on server version; fall back to the whole dict.
        for key in ("securityDailyTradeDto", "security", "content", "data"):
            if isinstance(payload.get(key), dict):
                payload = {**payload, **payload[key]}
    if not isinstance(payload, dict) or not payload:
        raise NoMarketDataError(symbol, None, "NEPSE /CompanyDetails returned nothing")

    lines = [
        f"# Fundamentals for {symbol} (NEPSE)",
        "",
        f"Retrieved {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
        f"from NepseAPI /CompanyDetails at {_base_url()}.",
        "",
        "| Field | Value |",
        "| --- | --- |",
    ]
    for key, value in sorted(payload.items()):
        if isinstance(value, (dict, list)) or value is None:
            continue
        lines.append(f"| {key} | {value} |")

    lines += [
        "",
        "> Scope note: this is NEPSE's listing/trading record for the company. "
        "Balance sheet, income statement, and cash flow are NOT available from "
        "any NEPSE JSON source — treat them as unknown, do not estimate them.",
    ]
    return "\n".join(lines)


BENCHMARK_NAME = "NEPSE Index"
_INDEX_HISTORY_FILE = "nepse-index-history.csv"
_index_recorded_this_process = False


def fetch_nepse_index_close() -> tuple[str, float]:
    """``(business_date, closing_value)`` for the NEPSE Index.

    Sourced from the intraday graph because it is the only *self-dating* index
    endpoint: every point is ``[epoch_seconds, value]``, so the session's last
    point is the close and carries its own date. Verified against the 2026-08-14
    session — last point 15:00 Asia/Kathmandu, NEPSE's closing bell.

    ``/NepseIndex`` is deliberately not used: it carries no date at all, and three
    mutually inconsistent price fields (``close`` 2651.22 vs ``previousClose``
    2643.84 vs ``currentValue`` 2643.83 on the same response), so picking one
    would be a guess that silently poisons every alpha figure downstream.
    """
    points = _get("/DailyNepseIndexGraph")
    if not isinstance(points, list) or not points:
        # The graph covers the current session only and is emptied overnight, so
        # it is legitimately blank between the reset and the opening bell. The
        # accumulator treats this as "nothing to record yet" and tries again on
        # the next fetch; it is not a failure of the endpoint.
        raise NoMarketDataError(
            BENCHMARK_NAME, None,
            "/DailyNepseIndexGraph is empty — no session yet today (the index "
            "graph only covers the current session). Retry after the opening bell.",
        )
    epoch, value = points[-1][0], points[-1][1]
    stamp = pd.Timestamp(int(epoch), unit="s", tz="UTC").tz_convert("Asia/Kathmandu")
    return stamp.strftime("%Y-%m-%d"), float(value)


def _index_history_path() -> str:
    from .config import get_config

    cache = get_config()["data_cache_dir"]
    os.makedirs(cache, exist_ok=True)
    return os.path.join(cache, _INDEX_HISTORY_FILE)


def record_nepse_index_close() -> None:
    """Append today's NEPSE Index close to the local history file, once per process.

    NEPSE serves no multi-day index history — the graph endpoints are intraday
    for the current session only, and indices are not securities, so
    /PriceVolumeHistory cannot reach them. Alpha therefore cannot be computed
    retroactively; this accrues the series forward instead. Because the
    reflection layer resolves a decision several days *after* it is made, a
    series started now yields real alpha within a normal holding period.

    Best-effort: a failure here must never break a price fetch.
    """
    global _index_recorded_this_process
    if _index_recorded_this_process:
        return
    _index_recorded_this_process = True
    try:
        business_date, close = fetch_nepse_index_close()
        path = _index_history_path()
        existing = (
            pd.read_csv(path)
            if os.path.exists(path)
            else pd.DataFrame(columns=["Date", "Close"])
        )
        if business_date in set(existing["Date"].astype(str)):
            return
        updated = pd.concat(
            [existing, pd.DataFrame([{"Date": business_date, "Close": close}])]
        ).sort_values("Date")
        updated.to_csv(path, index=False)
    except Exception:  # noqa: BLE001 — never break a price fetch over the benchmark
        pass


def nepse_index_return(start_date: str, end_date: str) -> float | None:
    """Fractional NEPSE Index return between two dates, or None if not recorded.

    Returns None rather than approximating from the nearest available dates: an
    alpha figure silently computed over the wrong window is worse than no alpha.
    """
    path = _index_history_path()
    if not os.path.exists(path):
        return None
    history = pd.read_csv(path)
    if history.empty:
        return None
    by_date = dict(zip(history["Date"].astype(str), history["Close"], strict=False))
    start_value, end_value = by_date.get(start_date), by_date.get(end_date)
    if start_value is None or end_value is None or not start_value:
        return None
    return float((end_value - start_value) / start_value)


def resolve_nepse_identity(symbol: str) -> dict[str, str]:
    """Company name and sector for a NEPSE symbol, for the identity guard.

    Supplies for NEPSE what ``agent_utils.resolve_instrument_identity`` gets from
    yfinance elsewhere. Without it that guard is inert on NEPSE tickers — yfinance
    404s on them — and the analysts pattern-match the chart to an invented company
    (#814), which is the exact failure the guard exists to stop.

    Best-effort by the same contract as the yfinance path: any failure returns
    ``{}`` so a run never blocks on identity.
    """
    try:
        payload = _get("/CompanyDetails", symbol=symbol.strip().upper())
    except Exception:  # noqa: BLE001 — fail open, never block the run
        return {}
    if not isinstance(payload, dict):
        return {}

    security = payload.get("security") or {}
    company = security.get("companyId") or {}
    sector = (company.get("sectorMaster") or {}).get("sectorDescription")
    name = security.get("securityName") or company.get("companyName")

    identity: dict[str, str] = {}
    if isinstance(name, str) and name.strip():
        identity["company_name"] = name.strip()
    if isinstance(sector, str) and sector.strip():
        identity["sector"] = sector.strip()
    if identity:
        identity["exchange"] = "NEPSE"
    return identity


def _unavailable_factory(what: str):
    """Build a vendor function that reports a hard data gap without raising."""

    def _unavailable(*_args: Any, **_kwargs: Any) -> str:
        return _UNAVAILABLE.format(what=what)

    _unavailable.__name__ = f"get_nepse_{what.replace(' ', '_')}"
    _unavailable.__doc__ = f"Always reports {what} as unavailable for NEPSE."
    return _unavailable


get_nepse_balance_sheet = _unavailable_factory("balance sheet")
get_nepse_cashflow = _unavailable_factory("cash flow")
get_nepse_income_statement = _unavailable_factory("income statement")
get_nepse_news = _unavailable_factory("company news")
get_nepse_global_news = _unavailable_factory("macro news")
get_nepse_insider_transactions = _unavailable_factory("insider transactions")


def demo() -> None:
    """Self-check. Offline by default; add --live to hit a running server."""
    import sys

    # The real shape, copied from a live /PriceVolumeHistory response: no
    # openPrice, newest row first.
    sample = [
        {
            "businessDate": "2026-08-14",
            "totalTrades": 314,
            "totalTradedQuantity": 22999,
            "totalTradedValue": 12655763.5,
            "highPrice": 552.0,
            "lowPrice": 549.0,
            "closePrice": 549.9,
        },
        {
            "businessDate": "2026-08-13",
            "totalTrades": 308,
            "totalTradedQuantity": 20477,
            "totalTradedValue": 11274469.6,
            "highPrice": 553.0,
            "lowPrice": 548.0,
            "closePrice": 550.0,
        },
    ]

    frame = pd.DataFrame([_normalize_row(r, "TEST") for r in sample])
    frame["Date"] = pd.to_datetime(frame["Date"])
    frame = frame.sort_values("Date").reset_index(drop=True)
    # A missing open must drop the column, never invent one.
    assert list(frame.columns) == ["Date", "High", "Low", "Close", "Volume"], frame.columns
    # Rows arrive newest-first; the indicator engine needs them oldest-first.
    assert frame["Date"].is_monotonic_increasing, "must be oldest-first for stockstats"
    assert frame.loc[0, "Close"] == 550.0, frame.loc[0, "Close"]
    assert frame.loc[0, "Volume"] == 20477, frame.loc[0, "Volume"]

    # A server that does serve an open must surface it, in the right slot.
    with_open = pd.DataFrame([_normalize_row({**sample[0], "openPrice": 551.0}, "T")])
    assert with_open.loc[0, "Open"] == 551.0

    # Key renames must still resolve, case-insensitively.
    assert _pick({"open_price": 1.0}, _OPTIONAL_FIELDS["Open"], "Open", "T") == 1.0
    assert _pick({"CLOSEPRICE": 2.0}, _REQUIRED_FIELDS["Close"], "Close", "T") == 2.0

    # An unmappable *required* field must name the keys it actually saw.
    try:
        _pick({"weird": 1}, _REQUIRED_FIELDS["Close"], "Close", "T")
        raise AssertionError("expected NoMarketDataError")
    except NoMarketDataError as exc:
        assert "weird" in str(exc), exc

    # Unwrapping handles both bare lists and the nested shapes.
    assert _records({"content": sample}) == sample
    assert _records(sample) == sample
    assert _records({"nothing": 1}) == []

    # Sentinels return text, never raise — a raise would abort the graph.
    assert "unavailable" in get_nepse_news("NABIL", "2026-08-16")
    assert "unavailable" in get_nepse_balance_sheet("NABIL")

    # Index return: exact date match only, None otherwise — never approximate.
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, _INDEX_HISTORY_FILE)
        pd.DataFrame(
            [
                {"Date": "2026-08-10", "Close": 2600.0},
                {"Date": "2026-08-14", "Close": 2652.0},
            ]
        ).to_csv(path, index=False)
        original = globals()["_index_history_path"]
        globals()["_index_history_path"] = lambda: path
        try:
            got = nepse_index_return("2026-08-10", "2026-08-14")
            assert got is not None and abs(got - 0.02) < 1e-9, got
            # A date with no recorded close must yield None, not a nearest match.
            assert nepse_index_return("2026-08-11", "2026-08-14") is None
            assert nepse_index_return("2026-08-10", "2026-08-15") is None
        finally:
            globals()["_index_history_path"] = original

    print("offline checks passed")

    if "--live" in sys.argv:
        symbol = "NABIL"
        live = fetch_nepse_ohlcv(symbol)
        assert not live.empty
        assert live["Date"].is_monotonic_increasing
        print(
            f"live: {len(live)} rows for {symbol}, "
            f"{live['Date'].min().date()} to {live['Date'].max().date()}"
        )
        print(get_nepse_fundamentals(symbol).splitlines()[0])
        print("live checks passed")


if __name__ == "__main__":
    demo()
