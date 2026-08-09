"""Verifies the controlled A/B test's core fairness guarantee.

The baseline (WITHOUT UMA) and Uma (WITH UMA) experiment arms must be run
with the identical prompt, the identical model/generation settings, and the
identical *source* context object — the only thing allowed to differ is the
context string actually sent to the LLM (full vs. Uma-filtered).

The LLM transport is faked (see the fixture below) so this test exercises
Uma's own orchestration logic in :mod:`uma.cli.judge`, not network
variance. The cross-encoder filtering step is real.
"""

from __future__ import annotations

import pytest

from uma.cli.judge import run_baseline, run_with_uma
from uma.llm.client import LLMConfig, UmaLLMClient

QUERY = "What was Apple's revenue in fiscal year 2024?"
CONTEXT = (
    "Apple reported total net sales of $391.035 billion in fiscal year 2024. "
    "Apple was founded in 1976 by Steve Jobs and Steve Wozniak. "
    "Net income for fiscal year 2024 was $93.736 billion. "
    "Bananas are a good source of potassium."
)


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeUsage:
    def __init__(self, prompt_tokens, completion_tokens):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class _FakeResponse:
    def __init__(self, model, content, prompt_tokens, completion_tokens):
        self.model = model
        self.choices = [_FakeChoice(content)]
        self.usage = _FakeUsage(prompt_tokens, completion_tokens)


class _FakeCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        # Token counts loosely track the (fake) input so baseline vs Uma
        # calls are still distinguishable in assertions below.
        context_len = len(kwargs["messages"][1]["content"])
        return _FakeResponse(kwargs["model"], "A fake answer.", context_len, 20)


class _FakeChat:
    def __init__(self, completions):
        self.completions = completions


class _FakeOpenAI:
    instances = []

    def __init__(self, api_key=None, base_url=None):
        self.api_key = api_key
        self.base_url = base_url
        self._completions = _FakeCompletions()
        self.chat = _FakeChat(self._completions)
        _FakeOpenAI.instances.append(self)


@pytest.fixture()
def fake_openai(monkeypatch):
    _FakeOpenAI.instances.clear()
    monkeypatch.setattr("openai.OpenAI", _FakeOpenAI)
    return _FakeOpenAI


def test_baseline_and_uma_share_prompt_model_and_settings(fake_openai):
    config = LLMConfig(
        model="gpt-4o-mini",
        api_key="sk-test",
        base_url=None,
        temperature=0.0,
        max_tokens=400,
        input_price_per_1m=None,
        output_price_per_1m=None,
    )
    client = UmaLLMClient(config)

    baseline = run_baseline(client, QUERY, CONTEXT)
    uma_result, filter_result = run_with_uma(client, QUERY, CONTEXT, threshold=0.5, max_tokens=None)

    calls = client._client._completions.calls
    baseline_call, uma_call = calls[0], calls[1]

    # Identical model, temperature, max_tokens.
    for key in ("model", "temperature", "max_tokens"):
        assert baseline_call[key] == uma_call[key]

    # Identical system prompt.
    assert baseline_call["messages"][0] == uma_call["messages"][0]

    # Identical question embedded in the user message.
    assert QUERY in baseline_call["messages"][1]["content"]
    assert QUERY in uma_call["messages"][1]["content"]

    # Both experiments started from the exact same source context object.
    assert baseline.prompt == uma_result.prompt == QUERY
    assert baseline.model == uma_result.model == "gpt-4o-mini"

    # The ONLY permitted difference: the context text actually sent.
    assert baseline_call["messages"][1]["content"] != uma_call["messages"][1]["content"]
    assert filter_result.original_context == CONTEXT
    assert filter_result.filtered_context != CONTEXT
    assert filter_result.filtered_context in uma_call["messages"][1]["content"]
    assert CONTEXT in baseline_call["messages"][1]["content"]


def test_uma_result_carries_real_filter_metrics(fake_openai):
    config = LLMConfig("gpt-4o-mini", "sk-test", None, 0.0, 400, None, None)
    client = UmaLLMClient(config)

    uma_result, filter_result = run_with_uma(client, QUERY, CONTEXT, threshold=0.5, max_tokens=None)

    assert uma_result.filter_metrics is not None
    assert uma_result.filter_metrics.original_tokens == filter_result.metrics.original_tokens
    assert uma_result.filter_metrics.filtered_tokens <= uma_result.filter_metrics.original_tokens
