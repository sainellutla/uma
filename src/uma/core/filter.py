"""The core Uma filtering engine.

    Context
     |
     v
    Sentence segmentation
     |
     v
    Cross-encoder
     |
     v
    Relevance score per sentence
     |
     v
    Threshold filtering
     |
     v
    Optional token budget
     |
     v
    Filtered context

This is the single implementation used by the terminal judge demo AND the
MCP tools (``uma_filter`` / ``uma_score``) — nothing here is duplicated
elsewhere.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from uma.core.metrics import FilterMetrics, RuntimeStats
from uma.core.scorer import ScoredSentence, score_sentences, split_sentences
from uma.core.tokenizer import count_tokens

# Process-wide runtime stats, updated by every filter_context() call.
# Exposed through the uma_stats MCP tool and available to the CLI.
RUNTIME_STATS = RuntimeStats()


@dataclass
class FilterResult:
    """Everything produced by one :func:`filter_context` call."""

    query: str
    original_context: str
    filtered_context: str
    threshold: float
    max_tokens: int | None
    scored_sentences: list[ScoredSentence]
    kept_indices: list[int]
    metrics: FilterMetrics

    @property
    def kept_sentences(self) -> list[ScoredSentence]:
        kept = set(self.kept_indices)
        return [s for s in self.scored_sentences if s.index in kept]

    @property
    def removed_sentences(self) -> list[ScoredSentence]:
        kept = set(self.kept_indices)
        return [s for s in self.scored_sentences if s.index not in kept]


def score_context(query: str, context: str) -> list[ScoredSentence]:
    """Segment ``context`` into sentences and score each against ``query``.

    Used directly by the ``uma_score`` MCP tool, and internally by
    :func:`filter_context`.
    """
    sentences = split_sentences(context)
    return score_sentences(query, sentences)


def filter_context(
    query: str,
    context: str,
    threshold: float = 0.5,
    max_tokens: int | None = None,
    *,
    record_stats: bool = True,
) -> FilterResult:
    """Filter ``context`` down to the sentences relevant to ``query``.

    Parameters
    ----------
    query:
        The user's question / retrieval query.
    context:
        The raw retrieved context (e.g. concatenated RAG chunks).
    threshold:
        Minimum sigmoid-normalized cross-encoder relevance score (0-1) a
        sentence must reach to be retained. Default 0.5 means "the model
        judges this sentence more relevant than not."
    max_tokens:
        If set, an optional hard token budget applied *after* threshold
        filtering. Sentences are kept in their original order and appended
        until the budget would be exceeded, so the final chunk of text is
        never truncated mid-sentence.
    record_stats:
        Whether to fold this call's metrics into the process-wide
        :data:`RUNTIME_STATS` (disable for read-only introspection/tests).
    """
    start = time.perf_counter()

    original_tokens = count_tokens(context)
    scored = score_context(query, context)

    kept_indices = [s.index for s in scored if s.score >= threshold]

    if max_tokens is not None:
        kept_indices = _apply_token_budget(scored, kept_indices, max_tokens)

    kept_set = set(kept_indices)
    filtered_text = " ".join(s.text for s in scored if s.index in kept_set)
    filtered_tokens = count_tokens(filtered_text)

    elapsed_ms = (time.perf_counter() - start) * 1000.0

    metrics = FilterMetrics(
        original_tokens=original_tokens,
        filtered_tokens=filtered_tokens,
        sentences_processed=len(scored),
        sentences_retained=len(kept_indices),
        filtering_latency_ms=elapsed_ms,
    )

    if record_stats:
        RUNTIME_STATS.record(metrics)

    return FilterResult(
        query=query,
        original_context=context,
        filtered_context=filtered_text,
        threshold=threshold,
        max_tokens=max_tokens,
        scored_sentences=scored,
        kept_indices=kept_indices,
        metrics=metrics,
    )


def _apply_token_budget(
    scored: list[ScoredSentence],
    kept_indices: list[int],
    max_tokens: int,
) -> list[int]:
    """Trim ``kept_indices`` (in original order) to fit within ``max_tokens``.

    Sentences are ranked by score (highest first) to decide what to drop
    when over budget, but the final retained set is re-emitted in original
    document order so the filtered context reads naturally.
    """
    kept_set = set(kept_indices)
    candidates = [s for s in scored if s.index in kept_set]

    if not candidates:
        return []

    total = sum(count_tokens(s.text) for s in candidates)
    if total <= max_tokens:
        return kept_indices

    # Greedily accept sentences in descending relevance order until the
    # budget is exhausted.
    by_score = sorted(candidates, key=lambda s: s.score, reverse=True)
    budget_remaining = max_tokens
    accepted: set[int] = set()
    for s in by_score:
        cost = count_tokens(s.text)
        if cost <= budget_remaining:
            accepted.add(s.index)
            budget_remaining -= cost

    return [i for i in kept_indices if i in accepted]
