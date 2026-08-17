"""Publish the latest completed run to ``run_state.json`` for the dashboard.

The browser cannot read ``~/.tradingagents`` directly, so this flattens the newest
run's reports and the decision log into one JSON file inside the project, which
the static server already serves.

Every field is parsed out of what the agents actually wrote. Nothing is inferred:
a value the run never produced is emitted as null and the dashboard shows it as
"not stated" rather than filling the gap.

Run it after an analysis, or let ``run_nabil.py`` call it on completion.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from tradingagents.default_config import DEFAULT_CONFIG

OUT = Path(__file__).parent / "run_state.json"

# The graph's stages, in execution order, mapped to the report file each writes.
# Presence of the file is what marks a stage complete — the run either produced
# that agent's output or it did not.
STAGES = [
    ("Market analyst", "1_analysts/market.md"),
    ("Sentiment analyst", "1_analysts/sentiment.md"),
    ("News analyst", "1_analysts/news.md"),
    ("Fundamentals analyst", "1_analysts/fundamentals.md"),
    ("Bull researcher", "2_research/bull.md"),
    ("Bear researcher", "2_research/bear.md"),
    ("Research manager", "2_research/manager.md"),
    ("Trader", "3_trading/trader.md"),
    ("Risk team", "4_risk/neutral.md"),
    ("Portfolio manager", "5_portfolio/decision.md"),
]

# The Trader and Portfolio Manager render their structured proposal to markdown,
# so the numbers come back out by label. Tolerant of bold/spacing variations.
FIELDS = {
    "action": r"\*{0,2}(?:Action|Recommendation|Rating)\*{0,2}\s*[:|]\s*\*{0,2}\s*([A-Za-z]+)",
    "entry_price": r"\*{0,2}Entry(?:\s+Price)?\*{0,2}\s*[:|]\s*\*{0,2}\s*(?:Rs\.?|NPR|\$)?\s*([\d,]+\.?\d*)",
    "stop_loss": r"\*{0,2}Stop[- ]?Loss\*{0,2}\s*[:|]\s*\*{0,2}\s*(?:Rs\.?|NPR|\$)?\s*([\d,]+\.?\d*)",
    "price_target": r"\*{0,2}(?:Price\s+)?Target\*{0,2}\s*[:|]\s*\*{0,2}\s*(?:Rs\.?|NPR|\$)?\s*([\d,]+\.?\d*)",
    "time_horizon": r"\*{0,2}Time\s+Horizon\*{0,2}\s*[:|]\s*\*{0,2}\s*([^\n|]{2,60})",
    "position_sizing": r"\*{0,2}Position\s+Sizing\*{0,2}\s*[:|]\s*\*{0,2}\s*([^\n|]{2,160})",
}


def _num(raw):
    try:
        return float(str(raw).replace(",", ""))
    except (TypeError, ValueError):
        return None


def _extract(text: str) -> dict:
    out = {}
    for key, pattern in FIELDS.items():
        m = re.search(pattern, text, re.IGNORECASE)
        value = m.group(1).strip() if m else None
        out[key] = _num(value) if key in {"entry_price", "stop_loss", "price_target"} else value
    # "FINAL TRANSACTION PROPOSAL: **BUY**" is the line the pipeline is built around.
    final = re.search(r"FINAL TRANSACTION PROPOSAL:\s*\*{0,2}\s*([A-Za-z]+)", text, re.I)
    if final:
        out["action"] = final.group(1).strip()
    return out


def latest_run(results_dir: Path):
    """Newest report tree, wherever it sits under results_dir.

    ``save_reports`` writes ``reports/TICKER_YYYYmmdd_HHMMSS`` while the CLI uses
    its own layout, so this looks for the marker subdirectory the tree always has
    rather than assuming either shape.
    """
    if not results_dir.exists():
        return None, None, None
    trees = {
        marker.parent
        for pattern in ("**/1_analysts", "**/5_portfolio")
        for marker in results_dir.glob(pattern)
        if marker.is_dir()
    }
    if not trees:
        return None, None, None
    run_dir = max(trees, key=lambda p: p.stat().st_mtime)
    name = run_dir.name
    ticker = name.split("_")[0] if "_" in name else name
    stamp = datetime.fromtimestamp(run_dir.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    return ticker, stamp, run_dir


def build() -> dict:
    results = Path(DEFAULT_CONFIG["results_dir"])
    ticker, date, run_dir = latest_run(results)

    state = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ticker": ticker,
        "trade_date": date,
        "complete": False,
        "stages": [],
        "decision": None,
        "reports": {},
    }
    if not run_dir:
        state["message"] = "No completed analysis yet. The decision panel fills in once a run finishes."
        return state

    for name, rel in STAGES:
        path = run_dir / rel
        state["stages"].append({
            "name": name,
            "done": path.exists(),
            "chars": path.stat().st_size if path.exists() else 0,
        })

    # The two agents carry different halves of the answer and must be merged:
    #   Trader  (render_trader_proposal) -> action, entry_price, stop_loss, position_sizing
    #   Manager (render_pm_decision)     -> rating, price_target, time_horizon
    # Reading only one would silently drop entry and stop-loss, which are the
    # levels the whole pipeline exists to produce.
    decision_md = run_dir / "5_portfolio/decision.md"
    trader_md = run_dir / "3_trading/trader.md"

    trader_txt = trader_md.read_text(encoding="utf-8", errors="replace") if trader_md.exists() else ""
    pm_txt = decision_md.read_text(encoding="utf-8", errors="replace") if decision_md.exists() else ""

    if trader_txt or pm_txt:
        merged = _extract(trader_txt) if trader_txt else {}
        if pm_txt:
            for key, value in _extract(pm_txt).items():
                # The Portfolio Manager ratifies; its values win where it has one.
                if value is not None:
                    merged[key] = value
        state["decision"] = merged
        state["decision"]["excerpt"] = (pm_txt or trader_txt).strip()[:900]
        state["decision_source"] = (
            "trader + portfolio manager" if (trader_txt and pm_txt)
            else "portfolio manager" if pm_txt else "trader only"
        )
        state["complete"] = bool(pm_txt)

    for name, rel in STAGES:
        path = run_dir / rel
        if path.exists():
            state["reports"][name] = path.read_text(encoding="utf-8", errors="replace")[:6000]
    return state


def main() -> None:
    state = build()
    OUT.write_text(json.dumps(state, indent=1), encoding="utf-8")
    where = f"{state['ticker']} {state['trade_date']}" if state["ticker"] else "no runs yet"
    print(f"wrote {OUT.name} ({where}, complete={state['complete']})")


if __name__ == "__main__":
    main()
