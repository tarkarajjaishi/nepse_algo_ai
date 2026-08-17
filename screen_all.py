"""Screen every listed NEPSE company and publish the result for the dashboard.

A full agent analysis costs roughly 60 LLM calls, so running one per company is
~38,800 calls — over a year of a free-tier budget and 75 hours of runtime. The
*deterministic* half of the pipeline has no such limit: prices and indicators are
computed locally, so the whole market can be screened in minutes for zero LLM
cost. That is what this does.

Indicators come from the same ``stockstats`` engine the market analyst uses, so a
value here is the value the agents would see, not a lookalike computed twice.

Pacing matters: the NepseAPI server allows 60 requests per rolling minute, and
647 symbols would trip that many times over. Requests are spaced accordingly and
a 429 is waited out rather than dropped.

    python screen_all.py            # every listed company (~11 min)
    python screen_all.py --limit 25 # quick sample
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from stockstats import wrap

from tradingagents.dataflows.errors import VendorError
from tradingagents.dataflows.nepse import _base_url, fetch_nepse_ohlcv

OUT = Path(__file__).parent / "screener.json"

# 60 requests / rolling 60s on the server. Leave headroom so a retry or the
# dashboard polling alongside us does not push the window over.
REQUESTS_PER_MINUTE = 50
_PACE_SECONDS = 60.0 / REQUESTS_PER_MINUTE

INDICATORS = ("rsi", "close_50_sma", "close_200_sma", "macd")

# Minimum bars before an indicator means anything. stockstats computes a
# "200-day average" from twelve bars without complaint, and thinly traded
# debentures then show RSI 100 and an SMA nowhere near their price. Publishing
# that is worse than publishing nothing, so each value is gated on its own
# lookback and emitted as null when the history cannot support it.
_MIN_BARS = {"rsi": 15, "close_50_sma": 50, "close_200_sma": 200, "macd": 26}


def _num(value):
    """Round for display, or None when stockstats could not produce a value."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(f) else round(f, 2)


def screen_one(symbol: str) -> dict:
    """Price, indicators and 52-week position for one symbol."""
    bars = fetch_nepse_ohlcv(symbol)
    if len(bars) < 2:
        raise VendorError(f"{symbol}: only {len(bars)} bars")

    frame = wrap(bars.copy())
    for name in INDICATORS:
        frame[name]                      # triggers computation

    def indicator(name):
        if len(bars) < _MIN_BARS[name]:
            return None
        return _num(frame[name].iloc[-1])

    last, prev = bars.iloc[-1], bars.iloc[-2]
    close, pclose = float(last["Close"]), float(prev["Close"])
    high52 = float(bars["High"].max())
    low52 = float(bars["Low"].min())
    span = high52 - low52

    return {
        "symbol": symbol,
        "close": round(close, 2),
        "change_pct": round((close - pclose) / pclose * 100, 2) if pclose else None,
        "rsi": indicator("rsi"),
        "sma50": indicator("close_50_sma"),
        "sma200": indicator("close_200_sma"),
        "macd": indicator("macd"),
        "high52": round(high52, 2),
        "low52": round(low52, 2),
        # Where the last close sits between the 52-week extremes, 0-100.
        "pos52": round((close - low52) / span * 100, 1) if span else None,
        "bars": len(bars),
        "last_traded": last["Date"].strftime("%Y-%m-%d"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, help="screen only the first N companies")
    args = ap.parse_args()

    listing = requests.get(f"{_base_url()}/CompanyList", timeout=60).json()
    companies = sorted(listing, key=lambda c: c["symbol"])
    if args.limit:
        companies = companies[: args.limit]

    total = len(companies)
    print(f"screening {total} companies at {REQUESTS_PER_MINUTE}/min "
          f"(~{total * _PACE_SECONDS / 60:.0f} min)")

    rows, skipped = [], []
    started = time.time()
    for i, company in enumerate(companies, 1):
        symbol = company["symbol"]
        deadline = time.time() + _PACE_SECONDS
        try:
            row = screen_one(symbol)
            row["name"] = company.get("companyName") or company.get("securityName") or symbol
            row["sector"] = ((company.get("sectorName") or "") or "—")
            rows.append(row)
        except Exception as exc:                       # noqa: BLE001
            # Debentures, promoter lines and freshly listed scrips often have no
            # usable history. Recorded, not silently dropped — a screener that
            # quietly covers 400 of 647 looks complete and is not.
            skipped.append({"symbol": symbol, "reason": f"{type(exc).__name__}: {str(exc)[:90]}"})
        if i % 50 == 0 or i == total:
            print(f"  {i}/{total}  kept={len(rows)} skipped={len(skipped)} "
                  f"{time.time() - started:.0f}s", flush=True)
        time.sleep(max(0.0, deadline - time.time()))

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "counts": {"listed": total, "screened": len(rows), "skipped": len(skipped)},
        "rows": rows,
        "skipped": skipped[:60],
    }
    OUT.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"\nwrote {OUT.name}: {len(rows)} screened, {len(skipped)} skipped, "
          f"{time.time() - started:.0f}s total")


if __name__ == "__main__":
    main()
