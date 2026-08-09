"""Plain-Python implementations of Uma's three MCP tools.

These functions are the actual tool logic. :mod:`uma.mcp.server` only wraps
them with MCP tool decorators — it does not reimplement any of this. Keeping
them as plain functions (returning JSON-serializable dicts) also makes them
directly unit-testable without spinning up an MCP session.

All three tools call into :mod:`uma.core.filter`, which is the exact same
filtering engine used by the ``uma judge`` terminal demo.
"""

from __future__ import annotations

from uma.core.filter import RUNTIME_STATS, filter_context, score_context


def uma_filter(
    query: str,
    context: str,
    threshold: float = 0.5,
    max_tokens: int | None = None,
) -> dict:
    """Filter ``context`` down to the sentences relevant to ``query``.

    Runs the real local cross-encoder (no LLM involved) and returns the
    filtered context plus the exact metrics produced by that run.
    """
    result = filter_context(query, context, threshold=threshold, max_tokens=max_tokens)
    m = result.metrics
    return {
        "filtered_context": result.filtered_context,
        "original_tokens": m.original_tokens,
        "filtered_tokens": m.filtered_tokens,
        "tokens_removed": m.tokens_removed,
        "reduction_percent": round(m.reduction_percent, 2),
        "sentences_processed": m.sentences_processed,
        "sentences_kept": m.sentences_retained,
        "sentences_removed": m.sentences_removed,
        "filtering_latency_ms": round(m.filtering_latency_ms, 3),
    }


def uma_score(query: str, context: str) -> dict:
    """Return per-sentence cross-encoder relevance scores for ``context``.

    Scores are sigmoid-normalized to [0, 1]; higher means more relevant to
    ``query``. Does not apply any threshold or filtering — this is the raw
    scoring stage on its own, useful for inspecting why Uma kept or dropped
    a particular sentence.
    """
    scored = score_context(query, context)
    return {
        "query": query,
        "sentences": [
            {"index": s.index, "text": s.text, "score": round(s.score, 4)}
            for s in scored
        ],
        "sentences_scored": len(scored),
    }


def uma_stats() -> dict:
    """Return cumulative filtering statistics for this server process.

    Reflects every :func:`uma.core.filter.filter_context` call made so far
    in this process, including calls made via ``uma_filter`` and the
    terminal judge demo if they share a process.
    """
    return RUNTIME_STATS.as_dict()
