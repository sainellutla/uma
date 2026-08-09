"""Tests for the metrics dataclasses (no model, no network)."""

from __future__ import annotations

from uma.core.metrics import ExperimentResult, FilterMetrics, LLMUsage, RuntimeStats


def test_filter_metrics_derived_fields():
    m = FilterMetrics(
        original_tokens=100,
        filtered_tokens=40,
        sentences_processed=10,
        sentences_retained=4,
        filtering_latency_ms=12.5,
    )
    assert m.tokens_removed == 60
    assert m.reduction_percent == 60.0
    assert m.sentences_removed == 6


def test_filter_metrics_zero_original_tokens_no_division_error():
    m = FilterMetrics(
        original_tokens=0,
        filtered_tokens=0,
        sentences_processed=0,
        sentences_retained=0,
        filtering_latency_ms=0.0,
    )
    assert m.reduction_percent == 0.0
    assert m.tokens_removed == 0


def test_filter_metrics_as_dict_has_expected_keys():
    m = FilterMetrics(
        original_tokens=100,
        filtered_tokens=40,
        sentences_processed=10,
        sentences_retained=4,
        filtering_latency_ms=12.5,
    )
    d = m.as_dict()
    for key in (
        "original_tokens",
        "filtered_tokens",
        "tokens_removed",
        "reduction_percent",
        "sentences_processed",
        "sentences_retained",
        "sentences_removed",
        "filtering_latency_ms",
    ):
        assert key in d


def test_llm_usage_total_tokens():
    usage = LLMUsage(
        model="gpt-4o-mini",
        input_tokens=100,
        output_tokens=50,
        latency_sec=1.2,
        answer="hello",
    )
    assert usage.total_tokens == 150


def test_llm_usage_cost_unavailable_by_default():
    usage = LLMUsage(
        model="gpt-4o-mini",
        input_tokens=100,
        output_tokens=50,
        latency_sec=1.2,
        answer="hello",
    )
    assert usage.cost_usd is None
    d = usage.as_dict()
    assert d["cost_usd"] is None


def test_experiment_result_uma_latency_none_for_baseline():
    usage = LLMUsage(model="m", input_tokens=1, output_tokens=1, latency_sec=0.1, answer="a")
    result = ExperimentResult(
        label="WITHOUT UMA",
        prompt="q",
        model="m",
        llm=usage,
        context_tokens_sent=10,
        total_latency_sec=0.1,
        filter_metrics=None,
    )
    assert result.uma_latency_sec is None


def test_experiment_result_uma_latency_present_when_filtered():
    usage = LLMUsage(model="m", input_tokens=1, output_tokens=1, latency_sec=0.1, answer="a")
    fm = FilterMetrics(
        original_tokens=100,
        filtered_tokens=40,
        sentences_processed=10,
        sentences_retained=4,
        filtering_latency_ms=25.0,
    )
    result = ExperimentResult(
        label="WITH UMA",
        prompt="q",
        model="m",
        llm=usage,
        context_tokens_sent=40,
        total_latency_sec=0.125,
        filter_metrics=fm,
    )
    assert result.uma_latency_sec == 0.025


def test_runtime_stats_accumulates_across_calls():
    stats = RuntimeStats()
    m1 = FilterMetrics(100, 50, 10, 5, 10.0)
    m2 = FilterMetrics(200, 100, 20, 10, 20.0)
    stats.record(m1)
    stats.record(m2)
    assert stats.total_calls == 2
    assert stats.total_original_tokens == 300
    assert stats.total_filtered_tokens == 150
    assert stats.average_reduction_percent == 50.0
    assert stats.average_filtering_latency_ms == 15.0
    assert len(stats.history) == 2
