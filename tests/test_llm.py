"""Tests for the LLM configuration and client.

No real network calls are made here — the OpenAI client is monkeypatched
with a fake that returns a canned, realistically-shaped response so we can
verify Uma's own logic (config validation, usage extraction, cost math)
without depending on a live API key.
"""

from __future__ import annotations

import pytest

from uma.llm.client import LLMConfig, LLMConfigError, SYSTEM_PROMPT, UmaLLMClient


# --------------------------------------------------------------------------
# LLMConfig.from_env
# --------------------------------------------------------------------------


def test_llm_config_from_env_requires_api_key(monkeypatch):
    monkeypatch.delenv("UMA_LLM_API_KEY", raising=False)
    monkeypatch.setenv("UMA_LLM_MODEL", "gpt-4o-mini")
    with pytest.raises(LLMConfigError):
        LLMConfig.from_env()


def test_llm_config_from_env_requires_model(monkeypatch):
    monkeypatch.setenv("UMA_LLM_API_KEY", "sk-test")
    monkeypatch.delenv("UMA_LLM_MODEL", raising=False)
    with pytest.raises(LLMConfigError):
        LLMConfig.from_env()


def test_llm_config_from_env_defaults(monkeypatch):
    monkeypatch.setenv("UMA_LLM_API_KEY", "sk-test")
    monkeypatch.setenv("UMA_LLM_MODEL", "gpt-4o-mini")
    monkeypatch.delenv("UMA_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("UMA_LLM_TEMPERATURE", raising=False)
    monkeypatch.delenv("UMA_LLM_MAX_TOKENS", raising=False)
    monkeypatch.delenv("UMA_LLM_INPUT_PRICE_PER_1M", raising=False)
    monkeypatch.delenv("UMA_LLM_OUTPUT_PRICE_PER_1M", raising=False)

    config = LLMConfig.from_env()
    assert config.model == "gpt-4o-mini"
    assert config.base_url is None
    assert config.temperature == 0.0
    assert config.max_tokens == 400
    assert config.input_price_per_1m is None
    assert config.output_price_per_1m is None


def test_llm_config_generation_params_identical_for_equal_config():
    c1 = LLMConfig("gpt-4o-mini", "k", None, 0.0, 400, None, None)
    c2 = LLMConfig("gpt-4o-mini", "k", None, 0.0, 400, None, None)
    assert c1.generation_params() == c2.generation_params()


# --------------------------------------------------------------------------
# UmaLLMClient.generate (fake transport)
# --------------------------------------------------------------------------


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeUsage:
    def __init__(self, prompt_tokens, completion_tokens, total_tokens=None):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        if total_tokens is not None:
            self.total_tokens = total_tokens
        # else: attribute intentionally absent, matching providers whose
        # usage payload doesn't carry a total (exercises the getattr fallback).


class _FakeResponse:
    def __init__(self, model, content, prompt_tokens, completion_tokens, total_tokens=None):
        self.model = model
        self.choices = [_FakeChoice(content)]
        self.usage = _FakeUsage(prompt_tokens, completion_tokens, total_tokens)


class _FakeCompletions:
    def __init__(self, response):
        self._response = response
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return self._response


class _FakeChat:
    def __init__(self, completions):
        self.completions = completions


class _FakeOpenAI:
    """Stands in for openai.OpenAI, capturing constructor + call args."""

    instances: list["_FakeOpenAI"] = []
    # Overridable per-test via fake_openai.next_response = _FakeResponse(...)
    next_response: _FakeResponse | None = None

    def __init__(self, api_key=None, base_url=None):
        self.api_key = api_key
        self.base_url = base_url
        response = _FakeOpenAI.next_response or _FakeResponse("gpt-4o-mini", "The answer.", 120, 30)
        self._completions = _FakeCompletions(response)
        self.chat = _FakeChat(self._completions)
        _FakeOpenAI.instances.append(self)


@pytest.fixture()
def fake_openai(monkeypatch):
    _FakeOpenAI.instances.clear()
    _FakeOpenAI.next_response = None
    monkeypatch.setattr("openai.OpenAI", _FakeOpenAI)
    return _FakeOpenAI


def test_generate_returns_real_measured_usage(fake_openai):
    config = LLMConfig("gpt-4o-mini", "sk-test", None, 0.0, 400, None, None)
    client = UmaLLMClient(config)

    usage = client.generate(prompt="What is X?", context="X is Y.")

    assert usage.model == "gpt-4o-mini"
    assert usage.input_tokens == 120
    assert usage.output_tokens == 30
    assert usage.total_tokens == 150
    assert usage.answer == "The answer."
    assert usage.latency_sec >= 0
    assert usage.cost_usd is None
    assert usage.cost_is_estimate is True
    assert usage.cost_unavailable_reason is not None


def test_generate_computes_cost_when_pricing_configured(fake_openai):
    config = LLMConfig("gpt-4o-mini", "sk-test", None, 0.0, 400, input_price_per_1m=1.0, output_price_per_1m=2.0)
    client = UmaLLMClient(config)

    usage = client.generate(prompt="What is X?", context="X is Y.")

    expected_cost = (120 * 1.0 + 30 * 2.0) / 1_000_000.0
    assert usage.cost_usd == pytest.approx(expected_cost)
    assert usage.cost_is_estimate is True
    assert usage.cost_unavailable_reason is None


def test_generate_sends_system_and_user_messages(fake_openai):
    config = LLMConfig("gpt-4o-mini", "sk-test", None, 0.0, 400, None, None)
    client = UmaLLMClient(config)
    client.generate(prompt="What is X?", context="X is Y.")

    call_kwargs = fake_openai.instances[0]._completions.last_kwargs
    messages = call_kwargs["messages"]
    assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert "X is Y." in messages[1]["content"]
    assert "What is X?" in messages[1]["content"]
    assert call_kwargs["model"] == "gpt-4o-mini"
    assert call_kwargs["temperature"] == 0.0
    assert call_kwargs["max_tokens"] == 400


def test_two_generate_calls_with_same_config_use_identical_generation_params(fake_openai):
    """Simulates what the judge CLI does: one config/client, two calls."""
    config = LLMConfig("gpt-4o-mini", "sk-test", None, 0.0, 400, None, None)
    client = UmaLLMClient(config)

    client.generate(prompt="Q", context="full context")
    call_1 = dict(fake_openai.instances[0]._completions.last_kwargs)

    client.generate(prompt="Q", context="filtered context")
    call_2 = dict(fake_openai.instances[0]._completions.last_kwargs)

    # Same client -> same instance -> same underlying transport was reused.
    assert len(fake_openai.instances) == 1
    for key in ("model", "temperature", "max_tokens"):
        assert call_1[key] == call_2[key]
    # Only the context (embedded in the user message) is allowed to differ.
    assert call_1["messages"][1]["content"] != call_2["messages"][1]["content"]


# --------------------------------------------------------------------------
# Hidden/reasoning-token accounting (e.g. "thinking" model variants)
# --------------------------------------------------------------------------


def test_generate_prefers_provider_reported_total_over_local_sum(fake_openai):
    """Some models bill hidden reasoning tokens not in prompt+completion.

    When the provider's usage payload reports a larger total than
    prompt_tokens + completion_tokens, Uma must surface that larger total
    rather than silently understating it.
    """
    fake_openai.next_response = _FakeResponse(
        "reasoning-model", "Short answer.", prompt_tokens=12, completion_tokens=7, total_tokens=205
    )
    config = LLMConfig("reasoning-model", "sk-test", None, 0.0, 400, None, None)
    client = UmaLLMClient(config)

    usage = client.generate(prompt="Q", context="C")

    assert usage.input_tokens == 12
    assert usage.output_tokens == 7
    assert usage.total_tokens == 205  # not 19


def test_generate_costs_hidden_tokens_at_output_rate(fake_openai):
    fake_openai.next_response = _FakeResponse(
        "reasoning-model", "Short answer.", prompt_tokens=12, completion_tokens=7, total_tokens=205
    )
    config = LLMConfig(
        "reasoning-model", "sk-test", None, 0.0, 400, input_price_per_1m=1.0, output_price_per_1m=2.0
    )
    client = UmaLLMClient(config)

    usage = client.generate(prompt="Q", context="C")

    # Hidden tokens (205 - 12 = 193) priced at the output rate, not dropped.
    expected_cost = (12 * 1.0 + 193 * 2.0) / 1_000_000.0
    assert usage.cost_usd == pytest.approx(expected_cost)


def test_generate_falls_back_to_local_sum_when_provider_omits_total(fake_openai):
    fake_openai.next_response = _FakeResponse(
        "gpt-4o-mini", "Answer.", prompt_tokens=120, completion_tokens=30, total_tokens=None
    )
    config = LLMConfig("gpt-4o-mini", "sk-test", None, 0.0, 400, None, None)
    client = UmaLLMClient(config)

    usage = client.generate(prompt="Q", context="C")

    assert usage.total_tokens == 150
