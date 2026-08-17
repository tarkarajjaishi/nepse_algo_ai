"""Cross-provider LLM failover (tradingagents.llm_clients.fallback)."""

import pytest

from tradingagents.llm_clients.fallback import (
    FallbackLLM,
    _is_actually_free,
    _is_usable,
    parse_fallback_specs,
    should_failover,
)


class _Boom(Exception):
    """Stand-in for a provider SDK error."""


class FakeLLM:
    """Minimal chat model: returns a fixed answer or raises a fixed error."""

    def __init__(self, name, error=None, structured=True, tools=True):
        self.name = name
        self.error = error
        self.structured = structured
        self.tools = tools
        self.calls = 0

    def invoke(self, *_args, **_kwargs):
        self.calls += 1
        if self.error:
            raise self.error
        return f"answer from {self.name}"

    def bind_tools(self, *_args, **_kwargs):
        if not self.tools:
            raise NotImplementedError(f"{self.name} has no tool support")
        return FakeLLM(f"{self.name}+tools", self.error)

    def with_structured_output(self, *_args, **_kwargs):
        if not self.structured:
            raise NotImplementedError(f"{self.name} has no structured output")
        return FakeLLM(f"{self.name}+structured", self.error)


def chain(*models):
    return FallbackLLM(list(models), [m.name for m in models])


class TestParseFallbackSpecs:
    def test_empty_values_yield_no_specs(self):
        for value in ("", None, [], "   "):
            assert parse_fallback_specs(value) == []

    def test_splits_on_first_colon_only(self):
        """Model IDs contain colons (``...lightning:free``); providers never do."""
        specs = parse_fallback_specs("openrouter:nvidia/nemotron-3.5-lightning:free")
        assert specs == [("openrouter", "nvidia/nemotron-3.5-lightning:free")]

    def test_multiple_entries_and_whitespace(self):
        specs = parse_fallback_specs(" google:gemini-3.5-flash , nvidia:openai/gpt-oss-120b ")
        assert specs == [
            ("google", "gemini-3.5-flash"),
            ("nvidia", "openai/gpt-oss-120b"),
        ]

    def test_accepts_a_list_as_well_as_a_string(self):
        assert parse_fallback_specs(["google:a", "nvidia:b"]) == [
            ("google", "a"),
            ("nvidia", "b"),
        ]

    def test_malformed_entries_are_skipped_not_raised(self):
        """A typo in a backup must not stop a run whose primary is fine."""
        assert parse_fallback_specs("google:ok,garbage,:nomodel,noprovider:") == [
            ("google", "ok")
        ]


class TestShouldFailover:
    @pytest.mark.parametrize(
        "message",
        [
            "Error code: 429 - rate limit exceeded",
            "RESOURCE_EXHAUSTED: quota exceeded for metric generate_content",
            "You exceeded your current quota, please check your plan",
            "Too Many Requests",
        ],
    )
    def test_detects_quota_messages_across_providers(self, message):
        assert should_failover(_Boom(message))

    @pytest.mark.parametrize(
        "message",
        [
            "Upstream error from Nvidia: Internal server error, code 502",
            "503 Service Unavailable",
            "504 gateway timeout",
            "The model is overloaded, please try again",
        ],
    )
    def test_dead_upstream_also_fails_over(self, message):
        """Auto-discovery ranks on the catalogue, so it can pick a model that is
        currently down (nemotron-ultra 502s). One dead model must not abort a run."""
        assert should_failover(_Boom(message))

    def test_detects_by_exception_class_name(self):
        class RateLimitError(Exception):
            pass

        assert should_failover(RateLimitError("slow down"))

    @pytest.mark.parametrize(
        "message",
        [
            "connection reset by peer",
            "invalid api key",
            "model not found",
            "401 unauthorized",
        ],
    )
    def test_config_faults_do_not_fail_over(self, message):
        """These fail identically on every provider; walking the chain would only
        bury the real error behind three more."""
        assert not should_failover(_Boom(message))


class TestFailover:
    def test_primary_used_when_healthy_and_backup_untouched(self):
        primary, backup = FakeLLM("p"), FakeLLM("b")
        assert chain(primary, backup).invoke("hi") == "answer from p"
        assert backup.calls == 0

    def test_quota_error_falls_over_to_backup(self):
        primary = FakeLLM("p", error=_Boom("Error code: 429 free-models-per-day"))
        backup = FakeLLM("b")
        assert chain(primary, backup).invoke("hi") == "answer from b"
        assert backup.calls == 1

    def test_walks_the_whole_chain(self):
        quota = _Boom("RESOURCE_EXHAUSTED")
        a, b, c = FakeLLM("a", quota), FakeLLM("b", quota), FakeLLM("c")
        assert chain(a, b, c).invoke("hi") == "answer from c"

    def test_mixed_quota_then_dead_upstream(self):
        """The realistic chain: primary out of quota, next model 502-ing."""
        a = FakeLLM("a", _Boom("429 free-models-per-day"))
        b = FakeLLM("b", _Boom("Upstream error from Nvidia: Internal server error"))
        c = FakeLLM("c")
        assert chain(a, b, c).invoke("hi") == "answer from c"

    def test_non_quota_error_raises_immediately(self):
        """A real bug must surface, not be masked by trying another provider."""
        primary = FakeLLM("p", error=_Boom("invalid api key"))
        backup = FakeLLM("b")
        with pytest.raises(_Boom, match="invalid api key"):
            chain(primary, backup).invoke("hi")
        assert backup.calls == 0

    def test_last_provider_error_propagates(self):
        quota = _Boom("429 quota")
        with pytest.raises(_Boom):
            chain(FakeLLM("a", quota), FakeLLM("b", quota)).invoke("hi")

    def test_single_model_chain_still_raises_its_quota_error(self):
        with pytest.raises(_Boom):
            chain(FakeLLM("only", _Boom("429 quota"))).invoke("hi")

    def test_empty_chain_rejected(self):
        with pytest.raises(ValueError):
            FallbackLLM([], [])


class TestChainSurvivesBinding:
    """The agents bind tools/schemas and keep the result, so binding must not
    flatten the chain down to just the primary."""

    def test_bind_tools_keeps_failover(self):
        primary = FakeLLM("p", error=_Boom("429 quota"))
        bound = chain(primary, FakeLLM("b")).bind_tools([])
        assert isinstance(bound, FallbackLLM)
        assert bound.invoke("hi") == "answer from b+tools"

    def test_with_structured_output_keeps_failover(self):
        primary = FakeLLM("p", error=_Boom("429 quota"))
        bound = chain(primary, FakeLLM("b")).with_structured_output(dict)
        assert bound.invoke("hi") == "answer from b+structured"

    def test_backup_lacking_structured_output_is_dropped_not_fatal(self):
        primary = FakeLLM("p")
        weak = FakeLLM("weak", structured=False)
        bound = chain(primary, weak).with_structured_output(dict)
        assert bound.invoke("hi") == "answer from p+structured"

    def test_primary_lacking_structured_output_propagates(self):
        """bind_structured() catches NotImplementedError to choose free text;
        swallowing it here would hide that decision from the agent."""
        with pytest.raises(NotImplementedError):
            chain(FakeLLM("p", structured=False), FakeLLM("b")).with_structured_output(dict)


def test_unknown_attributes_delegate_to_primary():
    primary = FakeLLM("p")
    assert chain(primary, FakeLLM("b")).name == "p"


def model_entry(mid, prompt="0", completion="0", tools=True, ctx=128000, out=("text",)):
    return {
        "id": mid,
        "pricing": {"prompt": prompt, "completion": completion},
        "supported_parameters": ["tools"] if tools else [],
        "architecture": {"output_modalities": list(out)},
        "context_length": ctx,
    }


class TestFreeModelDetection:
    def test_price_decides_not_the_name(self):
        """OpenRouter lists a paid model and its :free twin under near-identical
        names, so matching on ':free' would silently start spending money."""
        paid = model_entry("nvidia/nemotron-3.5-lightning", prompt="0.00000008")
        free = model_entry("nvidia/nemotron-3.5-lightning:free")
        assert not _is_actually_free(paid)
        assert _is_actually_free(free)

    def test_free_prompt_but_paid_completion_is_not_free(self):
        assert not _is_actually_free(model_entry("x", prompt="0", completion="0.5"))

    def test_unparseable_pricing_is_not_free(self):
        assert not _is_actually_free({"pricing": {"prompt": None, "completion": "0"}})
        assert not _is_actually_free({})

    def test_requires_tool_support(self):
        assert not _is_usable(model_entry("x", tools=False))

    def test_requires_text_output(self):
        assert not _is_usable(model_entry("x", out=("image",)))

    def test_free_tool_capable_text_model_is_usable(self):
        assert _is_usable(model_entry("x"))


class TestDiscovery:
    @staticmethod
    def _patch(monkeypatch, payload=None, exc=None):
        import tradingagents.llm_clients.fallback as mod

        mod._auto_cache.clear()

        class FakeResponse:
            def raise_for_status(self):
                pass

            def json(self):
                return {"data": payload or []}

        class FakeRequests:
            @staticmethod
            def get(*_a, **_kw):
                if exc:
                    raise exc
                return FakeResponse()

        monkeypatch.setitem(__import__("sys").modules, "requests", FakeRequests)
        return mod

    def test_ranks_by_context_length_and_applies_limit(self, monkeypatch):
        mod = self._patch(monkeypatch, [
            model_entry("small", ctx=8000),
            model_entry("huge", ctx=1000000),
            model_entry("mid", ctx=262144),
        ])
        assert mod.discover_openrouter_free_models(limit=2) == ["huge", "mid"]

    def test_filters_out_paid_and_toolless(self, monkeypatch):
        mod = self._patch(monkeypatch, [
            model_entry("paid", prompt="0.001", ctx=999999),
            model_entry("notools", tools=False, ctx=999999),
            model_entry("good", ctx=1000),
        ])
        assert mod.discover_openrouter_free_models() == ["good"]

    def test_network_failure_returns_empty_not_raise(self, monkeypatch):
        """Discovery is an optimisation; a blip must not stop a healthy run."""
        mod = self._patch(monkeypatch, exc=RuntimeError("dns is down"))
        assert mod.discover_openrouter_free_models() == []


class TestAutoExpansion:
    def test_auto_entry_expands_to_discovered_models(self, monkeypatch):
        import tradingagents.llm_clients.fallback as mod

        monkeypatch.setattr(mod, "discover_openrouter_free_models", lambda limit=3: ["a", "b"])
        assert mod.expand_auto_specs(parse_fallback_specs("openrouter:auto")) == [
            ("openrouter", "a"),
            ("openrouter", "b"),
        ]

    def test_explicit_pins_survive_alongside_auto(self, monkeypatch):
        import tradingagents.llm_clients.fallback as mod

        monkeypatch.setattr(mod, "discover_openrouter_free_models", lambda limit=3: ["a"])
        specs = parse_fallback_specs("google:gemini-x,openrouter:auto,nvidia:n1")
        assert mod.expand_auto_specs(specs) == [
            ("google", "gemini-x"),
            ("openrouter", "a"),
            ("nvidia", "n1"),
        ]

    def test_duplicates_dropped_keeping_first_position(self, monkeypatch):
        import tradingagents.llm_clients.fallback as mod

        monkeypatch.setattr(mod, "discover_openrouter_free_models", lambda limit=3: ["a", "b"])
        specs = parse_fallback_specs("openrouter:a,openrouter:auto")
        assert mod.expand_auto_specs(specs) == [("openrouter", "a"), ("openrouter", "b")]

    def test_no_auto_entry_makes_no_network_call(self, monkeypatch):
        import tradingagents.llm_clients.fallback as mod

        def explode(**_kw):
            raise AssertionError("discovery must not run without an auto entry")

        monkeypatch.setattr(mod, "discover_openrouter_free_models", explode)
        assert mod.expand_auto_specs(parse_fallback_specs("google:x")) == [("google", "x")]


class TestChainBuildSkipsSelfHop:
    """The fallback chain is shared by both LLM slots, so one slot's primary is
    routinely listed as the other's backup. Falling back to yourself is a
    guaranteed-dead hop against the budget you just exhausted."""

    @staticmethod
    def _labels(primary_model, fallbacks):
        from tradingagents.graph.trading_graph import TradingAgentsGraph

        graph = object.__new__(TradingAgentsGraph)   # skip the heavy graph build
        graph.config = {
            "llm_provider": "google",
            "backend_url": None,
            "llm_fallbacks": fallbacks,
        }
        built = graph._build_llm(primary_model, {})
        return getattr(built, "_labels", [f"google:{primary_model}"])

    def test_primary_not_repeated_as_its_own_fallback(self):
        labels = self._labels("gemini-3.1-flash-lite",
                              "google:gemini-3.1-flash-lite,google:gemini-3.5-flash")
        assert labels == ["google:gemini-3.1-flash-lite", "google:gemini-3.5-flash"]

    def test_other_slots_primary_is_kept_as_a_backup(self):
        """The whole point: the deep slot must be able to reach the quick
        model's separate per-model budget once its own is spent."""
        labels = self._labels("gemini-3.5-flash",
                              "google:gemini-3.1-flash-lite,google:gemini-3.5-flash")
        assert labels[0] == "google:gemini-3.5-flash"
        assert "google:gemini-3.1-flash-lite" in labels

    def test_chain_of_only_the_primary_returns_a_bare_client(self):
        built = self._labels("gemini-3.5-flash", "google:gemini-3.5-flash")
        assert built == ["google:gemini-3.5-flash"]


class TestUsableInLcelChains:
    """The analysts build `prompt | llm.bind_tools(tools)`. A wrapper that only
    supports .invoke() passes every unit test and still dies at the first agent
    with "Expected a Runnable, callable or dict" — so pipe composition is the
    behaviour worth pinning, not the method surface."""

    def test_wrapper_is_a_runnable(self):
        from langchain_core.runnables import Runnable
        assert isinstance(chain(FakeLLM("p")), Runnable)

    def test_composes_with_pipe_and_falls_over(self):
        from langchain_core.runnables import RunnableLambda

        primary = FakeLLM("p", error=_Boom("429 free-models-per-day"))
        piped = RunnableLambda(lambda x: x) | chain(primary, FakeLLM("b")).bind_tools([])
        assert piped.invoke("hi") == "answer from b+tools"

    def test_bound_chain_still_pipes(self):
        from langchain_core.runnables import RunnableLambda

        piped = chain(FakeLLM("p"), FakeLLM("b")).bind_tools([]) | RunnableLambda(str.upper)
        assert piped.invoke("hi") == "ANSWER FROM P+TOOLS"
