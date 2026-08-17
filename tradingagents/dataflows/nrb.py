"""Macro data for Nepal from Nepal Rastra Bank, the central bank.

FRED is the framework's macro vendor, but its series are US-centric and it needs
an API key, so a NEPSE run reports macro as unavailable. NRB publishes the
official NPR reference rates through a documented public API, which is the macro
series that actually moves Nepali equities: remittance inflows are around a
quarter of GDP and are denominated in USD and the Gulf currencies, and the INR
peg anchors import costs for the whole economy.

Only exchange rates are machine-readable. NRB's inflation, policy-rate and
liquidity figures are published as PDF macroeconomic reports, so those remain
genuinely unavailable rather than approximated from something else.

    https://www.nrb.org.np/api/forex/v1/rates?from=YYYY-MM-DD&to=YYYY-MM-DD
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta
from typing import Annotated

import requests

from .errors import NoMarketDataError, VendorNotConfiguredError

BASE_URL = os.environ.get("NRB_BASE_URL", "https://www.nrb.org.np").rstrip("/")
RATES_PATH = "/api/forex/v1/rates"
TIMEOUT_SECONDS = 45
DEFAULT_LOOKBACK_DAYS = 30

# Friendly aliases the agents are likely to ask for -> ISO 4217. The macro tool
# takes a free-text indicator name, so an unknown value must not fail the run;
# it falls back to the headline set below.
_ALIASES = {
    "usd": "USD", "dollar": "USD", "us dollar": "USD",
    "inr": "INR", "indian rupee": "INR", "india": "INR",
    "eur": "EUR", "euro": "EUR",
    "gbp": "GBP", "pound": "GBP", "sterling": "GBP",
    "aed": "AED", "dirham": "AED",
    "sar": "SAR", "riyal": "SAR",
    "qar": "QAR", "myr": "MYR", "krw": "KRW", "jpy": "JPY", "cny": "CNY",
}
# USD and INR anchor the currency; the Gulf and Malaysia currencies are where
# most remittance income is actually earned.
_HEADLINE = ("USD", "INR", "EUR", "AED", "SAR", "MYR")


def _fetch(from_date: str, to_date: str) -> list[dict]:
    url = f"{BASE_URL}{RATES_PATH}"
    params = {"from": from_date, "to": to_date, "per_page": 100, "page": 1}

    # NRB intermittently closes the connection without a response. Observed
    # twice within an hour, and it succeeds on an immediate retry, so one retry
    # turns a routine blip into a non-event rather than a missing macro report.
    last_exc: Exception | None = None
    resp = None
    for attempt in (1, 2):
        try:
            resp = requests.get(url, params=params, timeout=TIMEOUT_SECONDS)
            break
        except requests.exceptions.ConnectionError as exc:
            last_exc = exc
            if attempt == 2:
                raise VendorNotConfiguredError(
                    f"cannot reach NRB at {BASE_URL} after 2 attempts: {exc}"
                ) from exc
            time.sleep(1.5)
        except requests.exceptions.RequestException as exc:
            raise NoMarketDataError("NRB", None, f"forex request failed: {exc}") from exc
    if resp is None:                                    # pragma: no cover
        raise VendorNotConfiguredError(f"cannot reach NRB at {BASE_URL}: {last_exc}")
    if not resp.ok:
        raise NoMarketDataError("NRB", None, f"NRB returned HTTP {resp.status_code}")

    payload = resp.json()
    data = payload.get("data") if isinstance(payload, dict) else None
    days = (data or {}).get("payload") if isinstance(data, dict) else data
    if not isinstance(days, list) or not days:
        raise NoMarketDataError(
            "NRB", None, f"no NRB rates published between {from_date} and {to_date}"
        )
    return days


def _rate(day: dict, iso3: str) -> dict | None:
    for entry in day.get("rates") or []:
        if (entry.get("currency") or {}).get("iso3") == iso3:
            return entry
    return None


def get_macro_data(
    indicator: Annotated[str, "currency code or name, e.g. 'usd', 'inr'"],
    curr_date: Annotated[str, "current date, yyyy-mm-dd"],
    look_back_days: Annotated[int | None, "window length in days"] = None,
) -> str:
    """NPR reference rates from Nepal Rastra Bank over the lookback window.

    ``indicator`` selects a currency; anything unrecognised falls back to the
    headline set rather than failing, since the agents pass free text here.
    """
    days_back = look_back_days or DEFAULT_LOOKBACK_DAYS
    try:
        end = datetime.strptime(curr_date, "%Y-%m-%d")
    except (TypeError, ValueError) as exc:
        raise NoMarketDataError("NRB", None, f"bad date {curr_date!r}") from exc
    start = end - timedelta(days=days_back)

    days = _fetch(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    days.sort(key=lambda d: d.get("date", ""))
    first, last = days[0], days[-1]

    key = (indicator or "").strip().lower()
    wanted = (_ALIASES[key],) if key in _ALIASES else _HEADLINE

    lines = [
        f"# Nepal Rastra Bank reference rates to {curr_date}",
        "",
        f"Official NPR rates, {first.get('date')} to {last.get('date')} "
        f"({len(days)} publications).",
        "",
        "| Currency | Unit | Buy | Sell | Change over window |",
        "|---|---:|---:|---:|---:|",
    ]
    shown = 0
    for iso3 in wanted:
        now, then = _rate(last, iso3), _rate(first, iso3)
        if not now:
            continue
        shown += 1
        unit = (now.get("currency") or {}).get("unit", 1)
        try:
            move = (float(now["sell"]) - float(then["sell"])) / float(then["sell"]) * 100
            delta = f"{move:+.2f}%"
        except (TypeError, ValueError, KeyError, ZeroDivisionError):
            delta = "—"
        lines.append(
            f"| {iso3} ({(now.get('currency') or {}).get('name', '')}) | {unit} "
            f"| {now.get('buy', '—')} | {now.get('sell', '—')} | {delta} |"
        )

    if not shown:
        raise NoMarketDataError("NRB", None, f"NRB published no rate for {indicator!r}")

    lines += [
        "",
        "> Rates are NPR per unit of foreign currency, as published by the central "
        "bank. A rising rate means a weaker rupee. USD and INR anchor import costs "
        "(the INR peg especially); AED, SAR and MYR are the main remittance "
        "corridors, and remittances are roughly a quarter of Nepal's GDP.",
        "> NRB publishes inflation, policy rates and liquidity only as PDF reports, "
        "so those are not available here and must not be inferred from these rates.",
    ]
    return "\n".join(lines)


def demo() -> None:
    """Self-check. Offline assertions; add --live to hit the NRB API."""
    import sys

    day = {"date": "2026-08-14", "rates": [
        {"currency": {"iso3": "USD", "name": "U.S. Dollar", "unit": 1},
         "buy": "152.41", "sell": "153.01"}]}
    assert _rate(day, "USD")["sell"] == "153.01"
    assert _rate(day, "XXX") is None
    assert _ALIASES["dollar"] == "USD" and _ALIASES["indian rupee"] == "INR"
    print("offline checks passed")

    if "--live" in sys.argv:
        text = get_macro_data("usd", "2026-08-17", 30)
        assert "USD" in text and "NPR" in text
        print(text.splitlines()[0])
        print([ln for ln in text.splitlines() if ln.startswith("| USD")][0])
        head = get_macro_data("anything unrecognised", "2026-08-17", 14)
        assert "INR" in head, "unknown indicator must fall back to the headline set"
        print("fallback set rows:", sum(1 for ln in head.splitlines() if ln.startswith("| ")) - 1)
        print("live checks passed")


if __name__ == "__main__":
    demo()
