"""Nepali market news from merolagani.com.

NEPSE publishes no news feed, so ``nepse.py`` returns an "unavailable" sentinel
for both news tools. Merolagani runs the country's most-read market news desk and
renders its headline list server-side, which makes it readable without emulating
the site's ASP.NET postbacks.

Scope, and the reasons for it:

* **Market news works.** ``/NewsList.aspx`` server-renders the latest headlines
  with timestamps.
* **Per-company news does not have a public filter.** ``?symbol=`` is accepted
  and ignored — the company tab is a WebForms panel behind ``__doPostBack`` and
  VIEWSTATE round-trips. Rather than emulate that, ``get_merolagani_news``
  filters the market tape for the ticker and says plainly when nothing matched,
  handing back the market context instead of pretending to be a company feed.
* **Most headlines are in Nepali (Devanagari).** They are passed through
  untranslated: the models read Nepali, and a machine translation layer would add
  a distortion step to the one genuinely local signal in the pipeline. It does
  mean a Latin ticker rarely appears in the text, which is why company matching
  usually finds nothing — stated in the output rather than hidden.
* **Cash flow and insider transactions are still not available anywhere.** Cash
  flow lives inside the quarterly PDF filings; Nepal has no insider-transaction
  disclosure feed. Those keep their sentinels.
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timedelta
from typing import Annotated

import requests
from parsel import Selector

from .errors import NoMarketDataError, VendorNotConfiguredError

BASE_URL = os.environ.get("MEROLAGANI_BASE_URL", "https://merolagani.com").rstrip("/")
TIMEOUT_SECONDS = 45
# Identified, non-spoofing agent. The site rejects an empty User-Agent.
_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; tradingagents/0.3; +research)"}

_DATE = re.compile(r"([A-Z][a-z]{2} \d{1,2}, \d{4})(?:\s+(\d{1,2}:\d{2} [AP]M))?")
_news_cache: list[dict] | None = None


def _fetch_headlines() -> list[dict]:
    """Latest market headlines: title, published datetime, url. Cached per process."""
    global _news_cache
    if _news_cache is not None:
        return _news_cache

    url = f"{BASE_URL}/NewsList.aspx"
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=TIMEOUT_SECONDS)
    except requests.exceptions.ConnectionError as exc:
        raise VendorNotConfiguredError(f"cannot reach {BASE_URL}: {exc}") from exc
    except requests.exceptions.RequestException as exc:
        raise NoMarketDataError("", None, f"merolagani news fetch failed: {exc}") from exc
    if not resp.ok:
        raise NoMarketDataError("", None, f"merolagani returned HTTP {resp.status_code}")

    items: list[dict] = []
    for link in Selector(resp.text).css("a[href*='NewsDetail']"):
        title = " ".join(" ".join(link.css("::text").getall()).split())
        if not title:
            continue
        block = link.xpath("./ancestor::*[self::div or self::li][1]")
        text = " ".join(" ".join(block.css("::text").getall()).split())
        stamp = _DATE.search(text)
        href = link.attrib.get("href", "")
        items.append({
            "title": title,
            "published": " ".join(p for p in (stamp.group(1), stamp.group(2)) if p) if stamp else "",
            "url": f"{BASE_URL}{href}" if href.startswith("/") else href,
        })

    # One headline can appear as both image and text link; keep first occurrence.
    seen: set[str] = set()
    _news_cache = [i for i in items if not (i["title"] in seen or seen.add(i["title"]))]
    if not _news_cache:
        raise NoMarketDataError(
            "", None,
            f"no headlines parsed from {url} — the page layout changed",
        )
    return _news_cache


def _within(published: str, curr_date: str, days: int) -> bool:
    """True when a headline falls inside the lookback, or its date is unreadable.

    Unparseable dates are kept rather than dropped: losing a real headline is a
    worse failure than showing one slightly outside the window.
    """
    if not published or not curr_date:
        return True
    try:
        when = datetime.strptime(published.split(" ")[0] + " " + published.split(" ")[1].rstrip(",")
                                 + " " + published.split(" ")[2], "%b %d %Y")
        cutoff = datetime.strptime(curr_date, "%Y-%m-%d") - timedelta(days=days)
        return when >= cutoff
    except (ValueError, IndexError):
        return True


def _render(items: list[dict], heading: str, note: str) -> str:
    lines = [f"# {heading}", "", f"Source: {BASE_URL}/NewsList.aspx", ""]
    for item in items:
        lines.append(f"- [{item['published'] or 'undated'}] {item['title']}")
        if item["url"]:
            lines.append(f"    {item['url']}")
    lines += ["", f"> {note}"]
    return "\n".join(lines)


def get_merolagani_global_news(
    curr_date: Annotated[str, "current date, yyyy-mm-dd"],
    look_back_days: Annotated[int, "how many days back"] = 7,
) -> str:
    """Nepali market and macro headlines from the merolagani news desk."""
    items = [i for i in _fetch_headlines() if _within(i["published"], curr_date, look_back_days)]
    if not items:
        return (
            f"<no merolagani headlines in the {look_back_days} days before {curr_date}. "
            "The feed was read successfully; it simply had nothing in that window.>"
        )
    return _render(
        items,
        f"Nepali market news to {curr_date}",
        "Headlines are mostly in Nepali (Devanagari) and are passed through "
        "untranslated. This is the market-wide tape, not company-specific.",
    )


def get_merolagani_news(
    symbol: Annotated[str, "NEPSE ticker, e.g. NABIL"],
    curr_date: Annotated[str, "current date, yyyy-mm-dd"] = "",
    look_back_days: Annotated[int, "how many days back"] = 7,
) -> str:
    """Headlines mentioning ``symbol``, falling back to the market tape.

    Merolagani has no public per-company news filter, so this searches the market
    tape for the ticker. Most headlines are Nepali, where a Latin ticker seldom
    appears, so a nil result is common and is reported as such — the analyst is
    told it is looking at market context, not a company feed.
    """
    symbol = symbol.strip().upper()
    items = [i for i in _fetch_headlines() if _within(i["published"], curr_date, look_back_days)]
    pattern = re.compile(rf"\b{re.escape(symbol)}\b", re.IGNORECASE)
    hits = [i for i in items if pattern.search(i["title"])]

    if hits:
        return _render(
            hits,
            f"Merolagani headlines mentioning {symbol} to {curr_date}",
            "Matched by ticker against the market tape.",
        )
    if not items:
        return f"<merolagani had no headlines in the {look_back_days} days before {curr_date}>"
    return _render(
        items,
        f"No {symbol}-specific headlines — Nepali market tape to {curr_date}",
        f"IMPORTANT: nothing published mentioned {symbol} by ticker in this window, "
        "and merolagani offers no per-company filter. Treat the items below as "
        f"market background only — do NOT attribute any of them to {symbol}.",
    )


def demo() -> None:
    """Self-check. Offline assertions; add --live to hit the real site."""
    import sys

    assert _within("Aug 17, 2026 10:24 AM", "2026-08-17", 7)
    assert not _within("Aug 1, 2026 10:24 AM", "2026-08-17", 7)
    # An unreadable date must be kept, never silently dropped.
    assert _within("garbled", "2026-08-17", 7)
    assert _within("", "2026-08-17", 7)
    print("offline checks passed")

    if "--live" in sys.argv:
        heads = _fetch_headlines()
        assert heads and heads[0]["title"]
        print(f"live: {len(heads)} headlines, newest {heads[0]['published']}")
        text = get_merolagani_news("NABIL", "2026-08-17")
        assert "market" in text.lower()
        print("company view:", text.splitlines()[0])
        print("global view :", get_merolagani_global_news("2026-08-17").splitlines()[0])
        print("live checks passed")


if __name__ == "__main__":
    demo()
