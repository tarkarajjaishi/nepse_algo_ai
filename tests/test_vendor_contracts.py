"""Every vendor for a tool must accept the arguments the router passes it.

A signature mismatch type-checks fine and passes any test that calls the vendor
directly — it only detonates mid-run, inside an agent. The merolagani news vendor
took (symbol, curr_date, look_back_days) while get_news passes
(ticker, start_date, end_date), so a date string landed in a day-count parameter
and killed the sentiment analyst three minutes into a run.
"""

import inspect

import pytest

from tradingagents.dataflows.interface import VENDOR_METHODS

# How route_to_vendor is called for each tool, from agents/utils/*_tools.py.
CALLS = {
    "get_stock_data": ("NABIL", "2026-08-01", "2026-08-14"),
    "get_indicators": ("NABIL", "rsi", "2026-08-14", 5),
    "get_fundamentals": ("NABIL", "2026-08-14"),
    # Statements carry a reporting frequency between ticker and date.
    "get_balance_sheet": ("NABIL", "quarterly", "2026-08-14"),
    "get_cashflow": ("NABIL", "quarterly", "2026-08-14"),
    "get_income_statement": ("NABIL", "quarterly", "2026-08-14"),
    "get_news": ("NABIL", "2026-08-01", "2026-08-14"),
    "get_global_news": ("2026-08-14", 7, 10),
    "get_insider_transactions": ("NABIL",),
    "get_macro_indicators": ("usd", "2026-08-14", 30),
    "get_prediction_markets": ("nepal banking", 10),
}


@pytest.mark.parametrize(
    ("method", "vendor"),
    [(m, v) for m, vendors in VENDOR_METHODS.items() for v in vendors],
)
def test_vendor_accepts_the_routers_arguments(method, vendor):
    """Bind the real call signature without invoking it — no network needed."""
    fn = VENDOR_METHODS[method][vendor]
    args = CALLS[method]
    try:
        inspect.signature(fn).bind(*args)
    except TypeError as exc:
        pytest.fail(
            f"{vendor}.{method} cannot accept {args}: {exc}\n"
            f"  its signature is {inspect.signature(fn)}"
        )


def test_every_routed_tool_is_covered_here():
    """A new tool must gain a CALLS entry, or it escapes this check silently."""
    assert set(VENDOR_METHODS) == set(CALLS), set(VENDOR_METHODS) ^ set(CALLS)


@pytest.mark.parametrize("bound", ["", "not-a-date", None])
def test_merolagani_keeps_headlines_when_a_bound_is_unusable(bound):
    from tradingagents.dataflows.merolagani import _between
    assert _between("Aug 17, 2026 10:24 AM", bound, bound)


def test_merolagani_window_filters_on_real_dates():
    from tradingagents.dataflows.merolagani import _between
    assert _between("Aug 17, 2026 10:24 AM", "2026-08-10", "2026-08-17")
    assert not _between("Aug 1, 2026 09:00 AM", "2026-08-10", "2026-08-17")


class TestSentimentSourcesAreMarketAware:
    """StockTwits and Reddit have no NEPSE coverage. Querying them costs a round
    trip each, trips Reddit's per-IP rate limit — which degrades the next run for
    a ticker they *do* cover — and hands the model three failure placeholders
    that read like three attempted signals."""

    @staticmethod
    def _chatter_fn():
        """The closure the analyst node uses to pick its chatter sources."""
        from tradingagents.agents.analysts import sentiment_analyst as sa
        node = sa.create_sentiment_analyst(object())
        return next(
            cell.cell_contents for cell in node.__closure__ or []
            if getattr(cell.cell_contents, "__name__", "") == "_chatter_sources"
        )

    def test_nepse_skips_both_and_says_why(self, monkeypatch):
        from tradingagents.agents.analysts import sentiment_analyst as sa

        called = []
        monkeypatch.setattr("tradingagents.dataflows.config.get_config",
                            lambda: {"data_vendors": {"core_stock_apis": "nepse"}})
        monkeypatch.setattr(sa, "fetch_stocktwits_messages",
                            lambda *a, **k: called.append("st"))
        monkeypatch.setattr(sa, "fetch_reddit_posts", lambda *a, **k: called.append("rd"))

        stock, reddit = self._chatter_fn()("NABIL")
        assert "no NEPSE coverage" in stock and "no NEPSE coverage" in reddit
        assert "draw no inference" in stock
        assert called == [], "neither venue should be queried for a NEPSE ticker"

    def test_yahoo_markets_still_query_both(self, monkeypatch):
        from tradingagents.agents.analysts import sentiment_analyst as sa

        monkeypatch.setattr("tradingagents.dataflows.config.get_config",
                            lambda: {"data_vendors": {"core_stock_apis": "yfinance"}})
        monkeypatch.setattr(sa, "fetch_stocktwits_messages", lambda *a, **k: "ST")
        monkeypatch.setattr(sa, "fetch_reddit_posts", lambda *a, **k: "RD")

        assert self._chatter_fn()("NVDA") == ("ST", "RD")
