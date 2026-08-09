"""Uma Calibrate — empirically finds the minimum sufficient context.

Everything upstream of this module answers "how much context did Uma
remove?" This module answers a different question: "how aggressively
*could* Uma filter before answer quality actually breaks?"

Given a small benchmark of (question, context, expected_answer) cases, this
sweeps a set of relevance thresholds, and for each one:

  1. Filters every case's context at that threshold (the real
     :func:`uma.core.filter.filter_context` engine — same code path as
     ``uma judge`` and the MCP tools, nothing new implemented here).
  2. Sends the filtered context + question to the real configured LLM.
  3. Scores the answer against ``expected_answer`` (exact substring match,
     case-insensitive — deliberately simple and inspectable, not an
     LLM-as-judge, so the number is reproducible and never itself a
     fabricated judgment call).

The threshold with the best accuracy *and* the lowest average context kept
is reported as the "minimum sufficient context" — the empirical answer to
"how much of the retrieved context did the model actually need?"

This module has no dependency on Render — it's pure Python, fully
unit-testable with a fake LLM client. :mod:`uma.render_workflows.workflow`
is a thin Render Workflows wrapper around exactly these functions, so the
threshold sweep can fan out in parallel; the logic itself lives in exactly
one place.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from uma.core.filter import filter_context
from uma.core.tokenizer import count_tokens
from uma.llm.client import UmaLLMClient

DEFAULT_THRESHOLDS: list[float] = [0.20, 0.35, 0.50, 0.65, 0.80]


@dataclass(frozen=True)
class BenchmarkCase:
    """One (question, context, expected_answer) entry from examples/benchmark.json."""

    question: str
    context: str
    expected_answer: str


@dataclass(frozen=True)
class CaseResult:
    """Outcome of running one benchmark case at one threshold (or unfiltered)."""

    question: str
    original_tokens: int
    filtered_tokens: int
    answer: str
    correct: bool

    @property
    def context_kept_percent(self) -> float:
        if self.original_tokens == 0:
            return 0.0
        return 100.0 * self.filtered_tokens / self.original_tokens


@dataclass(frozen=True)
class ThresholdResult:
    """Aggregated outcome of running the whole benchmark at one threshold.

    ``threshold`` is ``None`` for the WITHOUT-UMA baseline (full,
    unfiltered context) so it can be reported alongside the swept
    thresholds for comparison.
    """

    threshold: float | None
    cases: list[CaseResult] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        if not self.cases:
            return 0.0
        return sum(1 for c in self.cases if c.correct) / len(self.cases)

    @property
    def correct_count(self) -> int:
        return sum(1 for c in self.cases if c.correct)

    @property
    def avg_context_kept_percent(self) -> float:
        if not self.cases:
            return 0.0
        return sum(c.context_kept_percent for c in self.cases) / len(self.cases)

    def as_dict(self) -> dict:
        return {
            "threshold": self.threshold,
            "accuracy": round(self.accuracy, 4),
            "correct_count": self.correct_count,
            "total_cases": len(self.cases),
            "avg_context_kept_percent": round(self.avg_context_kept_percent, 2),
            "cases": [
                {
                    "question": c.question,
                    "original_tokens": c.original_tokens,
                    "filtered_tokens": c.filtered_tokens,
                    "context_kept_percent": round(c.context_kept_percent, 2),
                    "correct": c.correct,
                    "answer": c.answer,
                }
                for c in self.cases
            ],
        }


@dataclass(frozen=True)
class CalibrationResult:
    """Full calibration run: baseline + every swept threshold."""

    baseline: ThresholdResult
    threshold_results: list[ThresholdResult]

    @property
    def best_accuracy(self) -> float:
        accuracies = [r.accuracy for r in self.threshold_results]
        return max(accuracies, default=0.0)

    @property
    def minimum_sufficient(self) -> ThresholdResult | None:
        """The most aggressive threshold that still reaches best_accuracy.

        "Most aggressive" = lowest average context kept. Returns None if no
        threshold was swept (e.g. empty threshold list).
        """
        candidates = [r for r in self.threshold_results if r.accuracy == self.best_accuracy]
        if not candidates:
            return None
        return min(candidates, key=lambda r: r.avg_context_kept_percent)

    def as_dict(self) -> dict:
        minimum = self.minimum_sufficient
        return {
            "baseline": self.baseline.as_dict(),
            "results": [r.as_dict() for r in self.threshold_results],
            "best_accuracy": round(self.best_accuracy, 4),
            "minimum_sufficient_context": (
                {
                    "threshold": minimum.threshold,
                    "avg_context_kept_percent": round(minimum.avg_context_kept_percent, 2),
                    "accuracy": round(minimum.accuracy, 4),
                }
                if minimum is not None
                else None
            ),
        }


def score_answer(answer: str, expected_answer: str) -> bool:
    """Exact, case-insensitive substring match.

    Deliberately not an LLM-as-judge: this is a simple, deterministic,
    inspectable check ("did the key fact from expected_answer appear in the
    model's answer, verbatim modulo case"). It will produce false negatives
    on correct-but-differently-phrased answers — see the Limitations
    section in the README. That tradeoff is intentional: a scoring method
    that itself required trusting another LLM call would undermine the
    point of measuring accuracy in the first place.
    """
    return expected_answer.strip().lower() in answer.strip().lower()


def run_benchmark_at_threshold(
    cases: list[BenchmarkCase],
    threshold: float,
    llm_client: UmaLLMClient,
    max_tokens: int | None = None,
) -> ThresholdResult:
    """Run every benchmark case through Uma's real filter + the real LLM at one threshold."""
    results = []
    for case in cases:
        filter_result = filter_context(
            case.question, case.context, threshold=threshold, max_tokens=max_tokens, record_stats=False
        )
        usage = llm_client.generate(prompt=case.question, context=filter_result.filtered_context)
        results.append(
            CaseResult(
                question=case.question,
                original_tokens=filter_result.metrics.original_tokens,
                filtered_tokens=filter_result.metrics.filtered_tokens,
                answer=usage.answer,
                correct=score_answer(usage.answer, case.expected_answer),
            )
        )
    return ThresholdResult(threshold=threshold, cases=results)


def run_baseline_benchmark(cases: list[BenchmarkCase], llm_client: UmaLLMClient) -> ThresholdResult:
    """WITHOUT UMA: every case's full, unfiltered context goes straight to the LLM.

    Token counts here use the same tiktoken-based :func:`count_tokens` as
    the threshold sweep (via ``filter_context``'s metrics), not the
    provider's own reported ``input_tokens`` — different providers tokenize
    differently, and mixing the two bases would make "avg_context_kept_percent"
    compare apples to oranges between the baseline and the swept thresholds.
    """
    results = []
    for case in cases:
        usage = llm_client.generate(prompt=case.question, context=case.context)
        tokens = count_tokens(case.context)
        results.append(
            CaseResult(
                question=case.question,
                original_tokens=tokens,
                filtered_tokens=tokens,  # nothing was filtered
                answer=usage.answer,
                correct=score_answer(usage.answer, case.expected_answer),
            )
        )
    return ThresholdResult(threshold=None, cases=results)


def run_calibration(
    cases: list[BenchmarkCase],
    thresholds: list[float] | None = None,
    llm_client: UmaLLMClient | None = None,
    max_tokens: int | None = None,
) -> CalibrationResult:
    """Run the full calibration sweep sequentially (no Render, no asyncio).

    This is the local/CLI/test entry point. :mod:`uma.render_workflows.workflow`
    calls :func:`run_baseline_benchmark` / :func:`run_benchmark_at_threshold`
    directly instead, fanning them out in parallel across threads via
    ``asyncio.gather`` — same functions, different execution strategy.
    """
    thresholds = thresholds if thresholds is not None else DEFAULT_THRESHOLDS
    client = llm_client or UmaLLMClient()

    baseline = run_baseline_benchmark(cases, client)
    threshold_results = [
        run_benchmark_at_threshold(cases, t, client, max_tokens=max_tokens) for t in thresholds
    ]
    return CalibrationResult(baseline=baseline, threshold_results=threshold_results)
