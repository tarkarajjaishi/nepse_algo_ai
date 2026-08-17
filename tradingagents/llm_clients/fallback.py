"""Cross-provider failover for the deep/quick LLM slots.

Every free LLM tier is too small to finish one analysis on its own (a run makes
roughly 50-70 calls). Budgets do not add up *within* a provider — OpenRouter's
50/day is shared across all its free models, so rotating models there buys
nothing — but they do add up *across* providers. This wraps the configured
primary model in a chain of backups on other providers and moves down the chain
whenever one reports its quota exhausted, so a run can span several free tiers.

Configure with ``llm_fallbacks`` (or ``TRADINGAGENTS_LLM_FALLBACKS``), a
comma-separated list of ``provider:model``::

    TRADINGAGENTS_LLM_FALLBACKS=google:gemini-3.1-flash-lite,nvidia:openai/gpt-oss-120b

Unset means no wrapping at all and the primary model is used directly, so the
default behaviour is byte-for-byte unchanged.

Use ``openrouter:auto`` to discover OpenRouter's free models at runtime instead
of naming them::

    TRADINGAGENTS_LLM_FALLBACKS=openrouter:auto

**Never hardcode an OpenRouter model ID.** That catalogue turns over constantly —
models are added and withdrawn, and a model's free variant can disappear while
the paid one remains. Worse, a name is not evidence of being free:
``nvidia/nemotron-3.5-lightning`` bills at $0.08/M while
``nvidia/nemotron-3.5-lightning:free`` is the free one, so the ``:free`` suffix is
a naming convention, not a guarantee. Selection here is driven entirely by the
live ``/models`` response — price actually zero, tool-calling supported, text
output — so the chain re-forms itself as the catalogue changes.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.runnables import Runnable

logger = logging.getLogger(__name__)

# Substrings that identify "this provider cannot serve me right now" across
# provider SDKs, which share no common exception type: OpenAI-style raises
# RateLimitError, Google raises ChatGoogleGenerativeAIError wrapping
# RESOURCE_EXHAUSTED, others surface a bare HTTP status. Matching on the rendered
# message is crude but is the only provider-agnostic signal available.
_FAILOVER_MARKERS = (
    # Out of quota / throttled.
    "429",
    "resource_exhausted",
    "rate limit",
    "ratelimit",
    "rate_limit",
    "quota",
    "insufficient_quota",
    "too many requests",
    # Upstream is broken or overloaded. Auto-discovery ranks purely on the
    # catalogue, so it will happily pick a model that is currently down —
    # nvidia/nemotron-3-ultra-550b-a55b:free advertises a 1M context and returns
    # 502 from NVIDIA. Without these, one dead model aborts the whole run.
    "502",
    "503",
    "504",
    "bad gateway",
    "service unavailable",
    "internal server error",
    "overloaded",
    "temporarily unavailable",
)


def should_failover(exc: BaseException) -> bool:
    """True when ``exc`` means "try the next provider" rather than "stop".

    Covers both exhausted quota and an upstream that is down. Deliberately broad:
    a false positive costs one wasted hop to a backup that would also have
    worked, while a false negative aborts the run — the failure this module
    exists to prevent.

    Deliberately does *not* match configuration faults (bad API key, unknown
    model, malformed request). Those are the same on every provider, so walking
    the chain would only bury the real error behind three more.

    Fires only *after* the SDK's own retry budget is spent (``llm_max_retries``),
    so a transient blip has already been waited out.
    """
    text = f"{type(exc).__name__} {exc}".lower()
    return any(marker in text for marker in _FAILOVER_MARKERS)


def parse_fallback_specs(raw: Any) -> list[tuple[str, str]]:
    """Parse ``"provider:model,provider:model"`` into ``[(provider, model), ...]``.

    Split on the *first* colon only: provider names never contain one, but model
    IDs routinely do (``nvidia/nemotron-3.5-lightning:free``). Accepts a list too,
    so the value can be set programmatically as well as through a flat env var.
    Malformed entries are skipped with a warning rather than raising — a typo in a
    backup should not stop a run whose primary is fine.
    """
    if not raw:
        return []
    entries = raw if isinstance(raw, (list, tuple)) else str(raw).split(",")

    specs: list[tuple[str, str]] = []
    for entry in entries:
        text = str(entry).strip()
        if not text:
            continue
        provider, separator, model = text.partition(":")
        if not separator or not provider.strip() or not model.strip():
            logger.warning(
                "Ignoring malformed LLM fallback %r (expected 'provider:model')", text
            )
            continue
        specs.append((provider.strip().lower(), model.strip()))
    return specs


OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
AUTO_MODEL = "auto"
_DEFAULT_AUTO_LIMIT = 3

# Discovery is one HTTP call shared by both LLM slots for the life of the process.
_auto_cache: dict[int, list[str]] = {}


def _is_actually_free(model: dict) -> bool:
    """True when both prompt and completion bill at exactly zero.

    Checked against the numbers, never the name: OpenRouter lists
    ``nvidia/nemotron-3.5-lightning`` (paid) alongside
    ``nvidia/nemotron-3.5-lightning:free``, so a ``:free`` suffix is a convention
    rather than a guarantee, and matching on it would silently start spending.
    """
    pricing = model.get("pricing") or {}
    try:
        return (
            float(pricing.get("prompt", 1)) == 0
            and float(pricing.get("completion", 1)) == 0
        )
    except (TypeError, ValueError):
        return False


def _is_usable(model: dict) -> bool:
    """Free alone is not enough — the framework needs tools and text output."""
    if not _is_actually_free(model):
        return False
    if "tools" not in (model.get("supported_parameters") or []):
        # Every analyst fetches its data through tool calls; a model without
        # tool support cannot do any useful work here.
        return False
    architecture = model.get("architecture") or {}
    return "text" in (architecture.get("output_modalities") or [])


def discover_openrouter_free_models(
    limit: int = _DEFAULT_AUTO_LIMIT, timeout: int = 15
) -> list[str]:
    """Current free, tool-capable OpenRouter model IDs, most capable first.

    Ranked by context length because the late graph stages (Research Manager,
    Trader, Risk, Portfolio Manager) carry every analyst report forward, so a
    short context is what actually breaks a run.

    Returns ``[]`` on any failure — discovery is an optimisation, and a network
    blip must not stop a run whose primary provider is healthy.
    """
    if limit in _auto_cache:
        return _auto_cache[limit]

    try:
        import requests

        response = requests.get(OPENROUTER_MODELS_URL, timeout=timeout)
        response.raise_for_status()
        models = response.json().get("data", [])
    except Exception as exc:  # noqa: BLE001 — discovery is best-effort
        logger.warning("Could not auto-discover OpenRouter free models: %s", exc)
        return []

    usable = [m for m in models if _is_usable(m)]
    usable.sort(key=lambda m: m.get("context_length") or 0, reverse=True)
    discovered = [m["id"] for m in usable[:limit] if m.get("id")]

    if discovered:
        logger.info(
            "Auto-discovered %d free OpenRouter models (of %d free, %d total): %s",
            len(discovered), sum(1 for m in models if _is_actually_free(m)),
            len(models), ", ".join(discovered),
        )
    else:
        logger.warning("OpenRouter has no free tool-capable models right now")
    _auto_cache[limit] = discovered
    return discovered


def expand_auto_specs(
    specs: list[tuple[str, str]], limit: int = _DEFAULT_AUTO_LIMIT
) -> list[tuple[str, str]]:
    """Replace each ``openrouter:auto`` entry with the models discovered now.

    Non-auto entries pass through untouched and keep their position, so an
    explicit pin and auto-discovery can be mixed in one chain. Duplicates are
    dropped, preserving first occurrence.
    """
    expanded: list[tuple[str, str]] = []
    for provider, model in specs:
        if provider == "openrouter" and model.lower() == AUTO_MODEL:
            expanded.extend(("openrouter", found)
                            for found in discover_openrouter_free_models(limit))
        else:
            expanded.append((provider, model))

    seen: set[tuple[str, str]] = set()
    return [s for s in expanded if not (s in seen or seen.add(s))]


class FallbackLLM(Runnable):
    """Chat model that moves to the next provider when one is out of quota.

    Wraps an ordered list of already-constructed chat models. ``bind_tools`` and
    ``with_structured_output`` re-wrap so the chain survives whatever the agents
    do to it — the agents hold whatever this returns and call ``invoke`` on it,
    so a wrapper that flattened at bind time would lose its backups.

    Subclasses ``Runnable`` because the agents compose it into LCEL pipelines
    (``prompt | llm.bind_tools(tools)``). A plain class is rejected there by
    ``coerce_to_runnable`` with "Expected a Runnable, callable or dict", which
    kills the run at the first analyst — passing ``.invoke()`` tests proves
    nothing about that path.

    Only the methods the agents actually use are overridden; everything else
    falls through to the primary so the wrapper stays transparent to callers
    that inspect model attributes.
    """

    def __init__(self, models: list[Any], labels: list[str]):
        if not models:
            raise ValueError("FallbackLLM needs at least one model")
        self._models = models
        self._labels = labels

    @property
    def primary(self) -> Any:
        return self._models[0]

    def _derive(self, method: str, *args: Any, **kwargs: Any) -> FallbackLLM:
        """Apply ``method`` to every model, keeping the chain intact.

        A backup that cannot do it (older models reject structured output) is
        dropped rather than allowed to break the chain. The *primary* is allowed
        to raise: callers such as ``agents.utils.structured.bind_structured``
        catch that to decide whether to use free-text generation instead.
        """
        derived_models = [getattr(self._models[0], method)(*args, **kwargs)]
        derived_labels = [self._labels[0]]
        for model, label in zip(self._models[1:], self._labels[1:], strict=False):
            try:
                derived_models.append(getattr(model, method)(*args, **kwargs))
                derived_labels.append(label)
            except (NotImplementedError, AttributeError, TypeError) as exc:
                logger.warning("Fallback %s does not support %s (%s); dropping it "
                               "from this chain", label, method, exc)
        return FallbackLLM(derived_models, derived_labels)

    def bind_tools(self, *args: Any, **kwargs: Any) -> FallbackLLM:
        return self._derive("bind_tools", *args, **kwargs)

    def with_structured_output(self, *args: Any, **kwargs: Any) -> FallbackLLM:
        return self._derive("with_structured_output", *args, **kwargs)

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        last_index = len(self._models) - 1
        for index, (model, label) in enumerate(
            zip(self._models, self._labels, strict=False)
        ):
            try:
                return model.invoke(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001 — re-raised unless it is quota
                if index == last_index or not should_failover(exc):
                    raise
                logger.warning(
                    "%s is out of quota (%s); falling over to %s",
                    label, str(exc)[:160], self._labels[index + 1],
                )
        raise AssertionError("unreachable: loop always returns or raises")

    def __getattr__(self, name: str) -> Any:
        # Only consulted for attributes this class does not define, so it cannot
        # shadow invoke/bind_tools/with_structured_output above.
        return getattr(self._models[0], name)

    def __repr__(self) -> str:
        return f"FallbackLLM({' -> '.join(self._labels)})"
