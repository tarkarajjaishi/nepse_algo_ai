"""The CLI must carry the NEPSE routing through to the run config.

`python -m cli.main` needs a real console for its prompts, so this covers
everything after them. _build_run_config shallow-copies DEFAULT_CONFIG and then
overwrites provider/model/url keys — a future edit that also rewrote the vendor
maps would silently send a NEPSE run at Yahoo, which fails as "no market data"
far from the cause.
"""

import os

import pytest

from cli.main import _build_run_config


@pytest.fixture
def nepse_env(monkeypatch):
    """Rebuild DEFAULT_CONFIG with the market switch on, as .env would."""
    import importlib

    import tradingagents.default_config as dc
    monkeypatch.setenv("TRADINGAGENTS_MARKET", "nepse")
    importlib.reload(dc)
    monkeypatch.setattr("cli.main.DEFAULT_CONFIG", dc.DEFAULT_CONFIG)
    yield dc.DEFAULT_CONFIG
    monkeypatch.setenv("TRADINGAGENTS_MARKET", "")
    importlib.reload(dc)


def selections(**over):
    base = {
        "ticker": "NABIL", "analysis_date": "2026-08-14", "research_depth": 1,
        "llm_provider": "google", "backend_url": None,
        "shallow_thinker": "gemini-3.1-flash-lite", "deep_thinker": "gemini-3.5-flash",
        "output_language": "English",
    }
    base.update(over)
    return base


def test_market_switch_survives_the_cli(nepse_env):
    cfg = _build_run_config(selections(), checkpoint=None)
    assert cfg["data_vendors"]["core_stock_apis"] == "nepse"
    assert cfg["data_vendors"]["technical_indicators"] == "nepse"
    assert cfg["data_vendors"]["macro_data"] == "nrb"


def test_per_tool_vendors_survive_the_cli(nepse_env):
    """Assert the routing intent, not an exact string — these are ordered chains
    and adding a fallback should not break the test that guards the switch."""
    cfg = _build_run_config(selections(), checkpoint=None)
    for tool in ("get_income_statement", "get_balance_sheet"):
        chain = [v.strip() for v in cfg["tool_vendors"][tool].split(",")]
        assert chain[0] == "sharesansar", f"{tool} should prefer the full filing"
        assert "nepsetrading" in chain, f"{tool} should keep the trend source as fallback"
    assert cfg["tool_vendors"]["get_news"] == "merolagani"


def test_backend_url_is_not_left_pointing_at_another_provider(nepse_env):
    """A stale OpenRouter URL forwarded to Gemini builds malformed requests."""
    cfg = _build_run_config(selections(backend_url=None), checkpoint=None)
    assert cfg["backend_url"] is None


def test_env_round_counts_win_over_the_depth_prompt(nepse_env, monkeypatch):
    monkeypatch.setenv("TRADINGAGENTS_MAX_DEBATE_ROUNDS", "1")
    cfg = _build_run_config(selections(research_depth=5), checkpoint=None)
    assert cfg["max_debate_rounds"] == nepse_env["max_debate_rounds"]


def test_depth_prompt_applies_when_env_is_unset(nepse_env, monkeypatch):
    monkeypatch.delenv("TRADINGAGENTS_MAX_DEBATE_ROUNDS", raising=False)
    monkeypatch.delenv("TRADINGAGENTS_MAX_RISK_ROUNDS", raising=False)
    cfg = _build_run_config(selections(research_depth=3), checkpoint=None)
    assert cfg["max_debate_rounds"] == 3


def test_cli_does_not_mutate_the_shared_default(nepse_env):
    before = dict(nepse_env["data_vendors"])
    _build_run_config(selections(llm_provider="openai"), checkpoint=None)
    assert nepse_env["data_vendors"] == before


def test_checkpoint_flag_only_applies_when_given(nepse_env):
    assert _build_run_config(selections(), checkpoint=True)["checkpoint_enabled"] is True
    assert _build_run_config(selections(), checkpoint=False)["checkpoint_enabled"] is False
    untouched = _build_run_config(selections(), checkpoint=None)
    assert untouched["checkpoint_enabled"] == nepse_env["checkpoint_enabled"]


def test_yahoo_default_is_unaffected_without_the_switch(monkeypatch):
    monkeypatch.setenv("TRADINGAGENTS_MARKET", "")
    import importlib

    import tradingagents.default_config as dc
    importlib.reload(dc)
    monkeypatch.setattr("cli.main.DEFAULT_CONFIG", dc.DEFAULT_CONFIG)
    cfg = _build_run_config(selections(), checkpoint=None)
    assert cfg["data_vendors"]["core_stock_apis"] == "yfinance"
    assert os.environ.get("TRADINGAGENTS_MARKET") == ""
