"""Tests for sentence segmentation, cross-encoder scoring, and filtering.

These exercise the real local cross-encoder (no mocking, no fabricated
scores) — per the project's requirement that filtering behavior is never
faked. The first test in this module that touches the model will trigger a
one-time weight download/load.
"""

from __future__ import annotations

from uma.core.filter import filter_context, score_context
from uma.core.scorer import split_sentences
from uma.core.tokenizer import count_tokens, truncate_to_tokens

QUERY = "What was Apple's revenue in fiscal year 2024?"

CONTEXT = (
    "Apple Inc. was founded in 1976 by Steve Jobs and Steve Wozniak. "
    "Apple reported total net sales of $391.035 billion in fiscal year 2024. "
    "The company is headquartered in Cupertino, California. "
    "Net income for fiscal year 2024 was $93.736 billion. "
    "Tim Cook has been CEO of Apple since 2011. "
    "Bananas are a good source of potassium."
)


# --------------------------------------------------------------------------
# Sentence segmentation (no model required)
# --------------------------------------------------------------------------


def test_split_sentences_basic():
    text = "This is one sentence. This is another one. And a third?"
    sentences = split_sentences(text)
    assert sentences == [
        "This is one sentence.",
        "This is another one.",
        "And a third?",
    ]


def test_split_sentences_survives_abbreviations_and_decimals():
    text = (
        "Apple Inc. reported revenue of $391.035 billion. "
        "Dr. Smith agreed with the U.S. filing."
    )
    sentences = split_sentences(text)
    assert len(sentences) == 2
    assert "391.035" in sentences[0]
    assert sentences[1].startswith("Dr. Smith")


def test_split_sentences_empty_input():
    assert split_sentences("") == []
    assert split_sentences("   ") == []


def test_split_sentences_preserves_order():
    text = "Alpha sentence one. Beta sentence two. Gamma sentence three."
    sentences = split_sentences(text)
    assert sentences[0].startswith("Alpha")
    assert sentences[1].startswith("Beta")
    assert sentences[2].startswith("Gamma")


# --------------------------------------------------------------------------
# Cross-encoder scoring (real model)
# --------------------------------------------------------------------------


def test_score_context_returns_one_score_per_sentence():
    scored = score_context(QUERY, CONTEXT)
    sentence_count = len(split_sentences(CONTEXT))
    assert len(scored) == sentence_count
    for s in scored:
        assert 0.0 <= s.score <= 1.0


def test_score_context_ranks_relevant_sentence_above_irrelevant():
    scored = score_context(QUERY, CONTEXT)
    by_text = {s.text: s.score for s in scored}
    revenue_sentence = next(t for t in by_text if "391.035" in t)
    banana_sentence = next(t for t in by_text if "Bananas" in t)
    assert by_text[revenue_sentence] > by_text[banana_sentence]


# --------------------------------------------------------------------------
# filter_context: threshold filtering, order preservation, token budget
# --------------------------------------------------------------------------


def test_filter_context_removes_irrelevant_sentences():
    result = filter_context(QUERY, CONTEXT, threshold=0.5, record_stats=False)
    assert "Bananas" not in result.filtered_context
    assert "391.035" in result.filtered_context


def test_filter_context_preserves_original_order():
    result = filter_context(QUERY, CONTEXT, threshold=0.0, record_stats=False)
    # threshold=0.0 keeps everything (all sigmoid scores are > 0)
    assert result.filtered_context.strip() == CONTEXT.strip() or len(result.kept_indices) == len(
        result.scored_sentences
    )
    indices = result.kept_indices
    assert indices == sorted(indices)


def test_filter_context_higher_threshold_keeps_fewer_or_equal_sentences():
    loose = filter_context(QUERY, CONTEXT, threshold=0.1, record_stats=False)
    strict = filter_context(QUERY, CONTEXT, threshold=0.9, record_stats=False)
    assert strict.metrics.sentences_retained <= loose.metrics.sentences_retained


def test_filter_context_metrics_are_consistent():
    result = filter_context(QUERY, CONTEXT, threshold=0.5, record_stats=False)
    m = result.metrics
    assert m.original_tokens == count_tokens(CONTEXT)
    assert m.filtered_tokens == count_tokens(result.filtered_context)
    assert m.tokens_removed == m.original_tokens - m.filtered_tokens
    assert m.sentences_removed == m.sentences_processed - m.sentences_retained
    assert m.filtering_latency_ms >= 0


def test_filter_context_respects_max_tokens_budget():
    result = filter_context(QUERY, CONTEXT, threshold=0.0, max_tokens=15, record_stats=False)
    assert result.metrics.filtered_tokens <= 15


def test_filter_context_empty_context():
    result = filter_context(QUERY, "", threshold=0.5, record_stats=False)
    assert result.filtered_context == ""
    assert result.metrics.original_tokens == 0
    assert result.metrics.sentences_processed == 0


# --------------------------------------------------------------------------
# tokenizer
# --------------------------------------------------------------------------


def test_count_tokens_empty():
    assert count_tokens("") == 0


def test_count_tokens_increases_with_length():
    assert count_tokens("hello world") < count_tokens("hello world " * 20)


def test_truncate_to_tokens_respects_budget():
    text = "one two three four five six seven eight nine ten"
    truncated = truncate_to_tokens(text, 3)
    assert count_tokens(truncated) <= 3


def test_truncate_to_tokens_noop_when_under_budget():
    text = "short text"
    assert truncate_to_tokens(text, 1000) == text
