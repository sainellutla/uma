"""Typed metrics containers shared across the CLI, MCP tools, and tests.

Every field here is a real measurement taken during a run — nothing in this
module fabricates a number. Latencies come from ``time.perf_counter()``
around the actual operation; token counts come from :mod:`uma.core.tokenizer`
or the LLM provider's own usage payload.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FilterMetrics:
    """Metrics produced by a single :func:`uma.core.filter.filter_context` call."""

    original_tokens: int
    filtered_tokens: int
    sentences_processed: int
    sentences_retained: int
    filtering_latency_ms: float

    @property
    def tokens_removed(self) -> int:
        return max(0, self.original_tokens - self.filtered_tokens)

    @property
    def reduction_percent(self) -> float:
        if self.original_tokens == 0:
            return 0.0
        return 100.0 * self.tokens_removed / self.original_tokens

    @property
    def sentences_removed(self) -> int:
        return max(0, self.sentences_processed - self.sentences_retained)

    def as_dict(self) -> dict:
        return {
            "original_tokens": self.original_tokens,
            "filtered_tokens": self.filtered_tokens,
            "tokens_removed": self.tokens_removed,
            "reduction_percent": round(self.reduction_percent, 2),
            "sentences_processed": self.sentences_processed,
            "sentences_retained": self.sentences_retained,
            "sentences_removed": self.sentences_removed,
            "filtering_latency_ms": round(self.filtering_latency_ms, 3),
        }


@dataclass
class LLMUsage:
    """Real usage numbers as reported by (or measured around) the LLM call."""

    model: str
    input_tokens: int
    output_tokens: int
    latency_sec: float
    answer: str
    cost_usd: float | None = None
    cost_is_estimate: bool = True
    cost_unavailable_reason: str | None = None
    # Some providers/models (e.g. "thinking"/reasoning variants) bill hidden
    # reasoning tokens that aren't reflected in prompt_tokens+completion_tokens.
    # When the provider's own usage payload includes a total, we trust that
    # over our own sum so token counts are never silently understated.
    reported_total_tokens: int | None = None

    @property
    def total_tokens(self) -> int:
        if self.reported_total_tokens is not None:
            return self.reported_total_tokens
        return self.input_tokens + self.output_tokens

    def as_dict(self) -> dict:
        return {
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "latency_sec": round(self.latency_sec, 4),
            "cost_usd": round(self.cost_usd, 6) if self.cost_usd is not None else None,
            "cost_is_estimate": self.cost_is_estimate,
            "cost_unavailable_reason": self.cost_unavailable_reason,
        }


@dataclass
class ExperimentResult:
    """Full result of one arm (baseline or Uma) of the controlled A/B test."""

    label: str
    prompt: str
    model: str
    llm: LLMUsage
    context_tokens_sent: int
    total_latency_sec: float
    filter_metrics: FilterMetrics | None = None

    @property
    def uma_latency_sec(self) -> float | None:
        if self.filter_metrics is None:
            return None
        return self.filter_metrics.filtering_latency_ms / 1000.0

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "model": self.model,
            "context_tokens_sent": self.context_tokens_sent,
            "total_latency_sec": round(self.total_latency_sec, 4),
            "llm": self.llm.as_dict(),
            "filter_metrics": self.filter_metrics.as_dict() if self.filter_metrics else None,
        }


@dataclass
class RuntimeStats:
    """Cumulative stats across every filter_context call in this process."""

    total_calls: int = 0
    total_sentences_processed: int = 0
    total_sentences_retained: int = 0
    total_original_tokens: int = 0
    total_filtered_tokens: int = 0
    total_filtering_latency_ms: float = 0.0
    history: list[FilterMetrics] = field(default_factory=list)

    def record(self, m: FilterMetrics) -> None:
        self.total_calls += 1
        self.total_sentences_processed += m.sentences_processed
        self.total_sentences_retained += m.sentences_retained
        self.total_original_tokens += m.original_tokens
        self.total_filtered_tokens += m.filtered_tokens
        self.total_filtering_latency_ms += m.filtering_latency_ms
        self.history.append(m)

    @property
    def average_reduction_percent(self) -> float:
        if self.total_original_tokens == 0:
            return 0.0
        removed = self.total_original_tokens - self.total_filtered_tokens
        return 100.0 * removed / self.total_original_tokens

    @property
    def average_filtering_latency_ms(self) -> float:
        if self.total_calls == 0:
            return 0.0
        return self.total_filtering_latency_ms / self.total_calls

    def as_dict(self) -> dict:
        return {
            "total_calls": self.total_calls,
            "total_sentences_processed": self.total_sentences_processed,
            "total_sentences_retained": self.total_sentences_retained,
            "total_original_tokens": self.total_original_tokens,
            "total_filtered_tokens": self.total_filtered_tokens,
            "average_reduction_percent": round(self.average_reduction_percent, 2),
            "average_filtering_latency_ms": round(self.average_filtering_latency_ms, 3),
        }
