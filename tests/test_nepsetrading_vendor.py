"""Quarterly-statement vendor that parses nepsetrading.com (offline only)."""

import pytest

from tradingagents.dataflows.nepsetrading import _cells, _merge_period_header


class _Row:
    """Stands in for a parsel row selector."""

    def __init__(self, texts):
        self._texts = texts

    def css(self, _sel):
        return self

    def getall(self):
        return self._texts


class TestPeriodHeaderMerge:
    """The page splits '78/79' and 'Q3' into separate text nodes. Unmerged, the
    header has 12 cells against 7 of data and every figure lands under the wrong
    fiscal year — silently wrong numbers, which is worse than no table."""

    def test_year_quarter_pairs_become_one_column(self):
        raw = ["Item", "78/79", "Q3", "79/80", "Q3", "80/81", "Q3", "YoY"]
        assert _merge_period_header(raw) == [
            "Item", "78/79 Q3", "79/80 Q3", "80/81 Q3", "YoY",
        ]

    def test_header_width_matches_a_data_row(self):
        raw = ["Item", "81/82", "Q3", "82/83", "Q3", "YoY"]
        data = ["Net Profit", "5.05 Ar", "6.76 Ar", "+33.92%"]
        assert len(_merge_period_header(raw)) == len(data)

    def test_unpaired_cells_pass_through(self):
        assert _merge_period_header(["Item", "YoY"]) == ["Item", "YoY"]
        assert _merge_period_header(["Item", "78/79"]) == ["Item", "78/79"]

    def test_quarter_without_a_year_is_not_swallowed(self):
        assert _merge_period_header(["Item", "Q3", "YoY"]) == ["Item", "Q3", "YoY"]

    @pytest.mark.parametrize("quarter", ["Q1", "Q2", "Q3", "Q4"])
    def test_every_quarter_merges(self, quarter):
        assert _merge_period_header(["Item", "80/81", quarter]) == ["Item", f"80/81 {quarter}"]


def test_cells_strips_whitespace_and_empties():
    assert _cells(_Row(["  Net Profit ", "\n", "6.76 Ar", "   "])) == ["Net Profit", "6.76 Ar"]


def test_units_are_never_converted():
    """Ar/Cr are Nepali scale units. A wrong conversion factor in a financial
    report is far worse than an unfamiliar unit, so they pass through verbatim."""
    from tradingagents.dataflows.nepsetrading import UNITS_NOTE
    assert "arab" in UNITS_NOTE and "crore" in UNITS_NOTE
    assert _cells(_Row(["12.46 Ar"])) == ["12.46 Ar"]


class TestZeroPricePlaceholdersRejected:
    """NEPSE pads non-trading securities with close=0, volume=0 rows. A zero is
    not a price: fed to stockstats it drags every average toward zero, and a
    debenture quoted at 1000 came back with a '200-day average' of 155."""

    @staticmethod
    def _rows():
        # Shape copied from a real /PriceVolumeHistory response for a debenture.
        pad = [{"businessDate": f"2026-07-{d:02d}", "closePrice": 0, "highPrice": 0,
                "lowPrice": 0, "totalTradedQuantity": 0} for d in range(1, 6)]
        real = [{"businessDate": "2026-08-14", "closePrice": 1000.0, "highPrice": 1000.0,
                 "lowPrice": 1000.0, "totalTradedQuantity": 12}]
        return pad + real

    def test_placeholder_rows_are_dropped(self, monkeypatch):
        import tradingagents.dataflows.nepse as nepse

        monkeypatch.setattr(nepse, "_get", lambda *a, **k: self._rows())
        monkeypatch.setattr(nepse, "record_nepse_index_close", lambda: None)
        frame = nepse.fetch_nepse_ohlcv("ADBLB")
        assert len(frame) == 1
        assert float(frame["Close"].iloc[0]) == 1000.0

    def test_all_placeholder_history_raises_rather_than_returning_zeros(self, monkeypatch):
        import tradingagents.dataflows.nepse as nepse
        from tradingagents.dataflows.errors import NoMarketDataError

        monkeypatch.setattr(nepse, "_get", lambda *a, **k: self._rows()[:5])
        monkeypatch.setattr(nepse, "record_nepse_index_close", lambda: None)
        with pytest.raises(NoMarketDataError, match="zero-price placeholder"):
            nepse.fetch_nepse_ohlcv("DEAD")
