"""Tests for the Uma Calibrate engine (uma.core.calibrate).

No real network calls: the LLM transport is faked, same pattern as
test_llm.py, so these tests exercise Uma's own calibration logic (threshold
sweep, scoring, minimum-sufficient-context selection) deterministically.
The cross-encoder filtering step is real.
"""

from __future__ import annotations

from uma.core.calibrate import (
    BenchmarkCase,
    run_baseline_benchmark,
    run_benchmark_at_threshold,
    run_calibration,
    score_answer,
)
from uma.llm.client import LLMConfig, UmaLLMClient

CASES = [
    BenchmarkCase(
        question="What was Apple's total net sales for fiscal year 2024?",
        context=(
            "Apple reported total net sales of $391.035 billion in fiscal year 2024. "
            "Apple was founded in 1976 by Steve Jobs and Steve Wozniak. "
            "Bananas are a good source of potassium."
        ),
        expected_answer="391.035 billion",
    ),
    BenchmarkCase(
        question="In what year was Apple founded?",
        context=(
            "Apple reported total net sales of $391.035 billion in fiscal year 2024. "
            "Apple was founded in 1976 by Steve Jobs and Steve Wozniak. "
            "Bananas are a good source of potassium."
        ),
        expected_answer="1976",
    ),
]


# --------------------------------------------------------------------------
# score_answer
# --------------------------------------------------------------------------


def test_score_answer_case_insensitive_substring_match():
    assert score_answer("The revenue was $391.035 Billion.", "391.035 billion") is True


def test_score_answer_false_when_absent():
    assert score_answer("The revenue was $200 billion.", "391.035 billion") is False


def test_score_answer_strips_whitespace():
    assert score_answer("  Tim Cook is CEO.  ", "tim cook") is True


# --------------------------------------------------------------------------
# Fake LLM transport (same shape as test_llm.py's fakes)
# --------------------------------------------------------------------------


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
    """Returns a canned answer if the question text is found in a lookup table."""

    def __init__(self, answers_by_keyword: dict[str, str]):
        self._answers = answers_by_keyword
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        user_content = kwargs["messages"][1]["content"]
        # Match only against the question portion, not the context — both
        # test cases share one context blob that mentions both topics.
        question_part = user_content.rsplit("Question:", 1)[-1]
        answer = "I don't know."
        for keyword, canned in self._answers.items():
            if keyword in question_part:
                answer = canned
                break
        context_len = len(user_content)
        return _FakeResponse(kwargs["model"], answer, context_len, len(answer.split()))


class _FakeChat:
    def __init__(self, completions):
        self.completions = completions


class _FakeOpenAI:
    def __init__(self, api_key=None, base_url=None, answers_by_keyword=None):
        self._completions = _FakeCompletions(answers_by_keyword or {})
        self.chat = _FakeChat(self._completions)


def _make_client(monkeypatch, answers_by_keyword: dict[str, str]) -> UmaLLMClient:
    monkeypatch.setattr(
        "openai.OpenAI",
        lambda api_key=None, base_url=None: _FakeOpenAI(answers_by_keyword=answers_by_keyword),
    )
    config = LLMConfig("fake-model", "sk-test", None, 0.0, 400, None, None)
    return UmaLLMClient(config)


# --------------------------------------------------------------------------
# run_baseline_benchmark / run_benchmark_at_threshold
# --------------------------------------------------------------------------


def test_run_baseline_benchmark_scores_each_case(monkeypatch):
    client = _make_client(
        monkeypatch,
        {
            "founded": "Answer not related to revenue.",
            "net sales": "Total net sales were $391.035 billion.",
        },
    )
    result = run_baseline_benchmark(CASES, client)
    assert result.threshold is None
    assert len(result.cases) == 2
    # Full context always sent -> filtered_tokens == original_tokens.
    for case in result.cases:
        assert case.original_tokens == case.filtered_tokens


def test_run_benchmark_at_threshold_uses_filtered_context(monkeypatch):
    client = _make_client(
        monkeypatch,
        {
            "founded": "Apple was founded in 1976.",
            "net sales": "Total net sales were $391.035 billion.",
        },
    )
    result = run_benchmark_at_threshold(CASES, threshold=0.5, llm_client=client, max_tokens=None)
    assert result.threshold == 0.5
    for case in result.cases:
        assert case.filtered_tokens <= case.original_tokens


# --------------------------------------------------------------------------
# run_calibration + minimum_sufficient
# --------------------------------------------------------------------------


def test_run_calibration_always_correct_picks_most_aggressive_threshold(monkeypatch):
    """If every threshold gets 100% accuracy, minimum_sufficient is the
    threshold with the *lowest* average context kept — the most aggressive
    filtering that didn't cost any accuracy."""
    client = _make_client(
        monkeypatch,
        {
            "founded": "Apple was founded in 1976 by Steve Jobs.",
            "net sales": "Net sales were $391.035 billion.",
        },
    )
    result = run_calibration(CASES, thresholds=[0.1, 0.9], llm_client=client)

    assert result.baseline.accuracy == 1.0
    assert len(result.threshold_results) == 2
    assert result.best_accuracy == 1.0

    minimum = result.minimum_sufficient
    assert minimum is not None
    # threshold 0.9 keeps fewer/equal sentences than 0.1 for this content,
    # so it should be selected as the more aggressive option at equal accuracy.
    kept_by_threshold = {r.threshold: r.avg_context_kept_percent for r in result.threshold_results}
    assert minimum.avg_context_kept_percent == min(kept_by_threshold.values())


def test_run_calibration_prefers_higher_accuracy_over_aggressiveness(monkeypatch):
    """A threshold that breaks accuracy should never be picked as minimum
    sufficient, even if it filters more aggressively."""
    client = _make_client(
        monkeypatch,
        {
            "founded": "Apple was founded in 1976.",
            "net sales": "I don't have enough information to answer.",
        },
    )
    result = run_calibration(CASES, thresholds=[0.5], llm_client=client)
    # "net sales" case never answered correctly -> accuracy < 1.0 at every threshold.
    assert result.best_accuracy < 1.0
    minimum = result.minimum_sufficient
    assert minimum is not None
    assert minimum.accuracy == result.best_accuracy


def test_calibration_result_as_dict_shape(monkeypatch):
    client = _make_client(monkeypatch, {"founded": "1976", "net sales": "391.035 billion"})
    result = run_calibration(CASES, thresholds=[0.5], llm_client=client)
    d = result.as_dict()
    assert "baseline" in d
    assert "results" in d
    assert "best_accuracy" in d
    assert "minimum_sufficient_context" in d
    assert len(d["results"]) == 1
    assert d["results"][0]["threshold"] == 0.5
