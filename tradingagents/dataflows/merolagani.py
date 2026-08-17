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


def _parse_published(published: str) -> datetime | None:
    """'Aug 17, 2026 10:24 AM' -> datetime, or None when unreadable."""
    try:
        parts = published.split(" ")
        return datetime.strptime(f"{parts[0]} {parts[1].rstrip(',')} {parts[2]}", "%b %d %Y")
    except (ValueError, IndexError, AttributeError):
        return None


def _between(published: str, start_date: str, end_date: str) -> bool:
    """True when a headline falls in [start_date, end_date], inclusive.

    Anything unparseable — the headline's date or either bound — is kept rather
    than dropped: losing a real headline is a worse failure than showing one
    slightly outside the window.
    """
    when = _parse_published(published)
    if when is None:
        return True
    for bound, keep_if_before in ((start_date, False), (end_date, True)):
        try:
            edge = datetime.strptime(bound, "%Y-%m-%d")
        except (ValueError, TypeError):
            continue                       # unusable bound: do not filter on it
        if keep_if_before and when > edge + timedelta(days=1):
            return False
        if not keep_if_before and when < edge:
            return False
    return True


def _within(published: str, curr_date: str, days: int) -> bool:
    """True when a headline falls within ``days`` before ``curr_date``."""
    when = _parse_published(published)
    if when is None or not curr_date:
        return True
    try:
        cutoff = datetime.strptime(curr_date, "%Y-%m-%d") - timedelta(days=int(days))
    except (ValueError, TypeError):
        return True
    return when >= cutoff


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
    look_back_days: Annotated[int | None, "how many days back"] = None,
    limit: Annotated[int | None, "max headlines to return"] = None,
) -> str:
    """Nepali market and macro headlines from the merolagani news desk.

    Signature matches the ``get_global_news`` vendor contract, including the
    ``limit`` the router passes; omitting it crashed the news analyst.
    """
    look_back_days = look_back_days or 7
    items = [i for i in _fetch_headlines() if _within(i["published"], curr_date, look_back_days)]
    if limit:
        items = items[:limit]
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
    ticker: Annotated[str, "NEPSE ticker, e.g. NABIL"],
    start_date: Annotated[str, "window start, yyyy-mm-dd"] = "",
    end_date: Annotated[str, "window end, yyyy-mm-dd"] = "",
) -> str:
    """Headlines mentioning ``ticker``, falling back to the market tape.

    Signature matches the ``get_news`` vendor contract — ``(ticker, start_date,
    end_date)``, two date strings — not a lookback count. Getting that wrong fed
    a date where a day count belonged and crashed the sentiment analyst.

    Merolagani has no public per-company news filter, so this searches the market
    tape for the ticker. Most headlines are Nepali, where a Latin ticker seldom
    appears, so a nil result is common and is reported as such — the analyst is
    told it is looking at market context, not a company feed.
    """
    symbol = ticker.strip().upper()
    curr_date = end_date or start_date
    items = [i for i in _fetch_headlines() if _between(i["published"], start_date, end_date)]
    pattern = re.compile(rf"\b{re.escape(symbol)}\b", re.IGNORECASE)
    hits = [i for i in items if pattern.search(i["title"])]

    if hits:
        return _render(
            hits,
            f"Merolagani headlines mentioning {symbol} to {curr_date}",
            "Matched by ticker against the market tape.",
        )
    if not items:
        return f"<merolagani had no headlines between {start_date} and {end_date}>"
    return _render(
        items,
        f"No {symbol}-specific headlines — Nepali market tape to {curr_date}",
        f"IMPORTANT: nothing published mentioned {symbol} by ticker between "
        f"{start_date or 'the window start'} and {end_date or 'now'}, "
        "and merolagani offers no per-company filter. Treat the items below as "
        f"market background only — do NOT attribute any of them to {symbol}.",
    )


def demo() -> None:
    """Self-check. Offline assertions; add --live to hit the real site."""
    import sys

    # The bug that crashed the sentiment analyst: get_news passes two date
    # strings, so a day-count parameter received "2026-08-14" and blew up in
    # timedelta. Guard the real contract, not the one I assumed.
    assert _between("Aug 17, 2026 10:24 AM", "2026-08-10", "2026-08-17")
    assert not _between("Aug 1, 2026 09:00 AM", "2026-08-10", "2026-08-17")
    assert _between("garbled", "2026-08-10", "2026-08-17")
    assert _between("Aug 17, 2026 10:24 AM", "", "")      # unusable bounds: keep
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
