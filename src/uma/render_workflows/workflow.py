"""Render Workflow tasks for Uma Calibrate.

    UMA CALIBRATE
          |
    Render Workflow
          |
  +-------+-------+-------+-------+-------+
  |       |       |       |       |       |
baseline  .20     .35     .50     .65     .80
  |       |       |       |       |       |
benchmark benchmark benchmark benchmark benchmark benchmark
  +-------+-------+-------+-------+-------+
          |
   COST x QUALITY CURVE
          |
  Minimum sufficient context

Every branch above is an independent task run (Render's supported parallel
pattern — see uma_calibrate() below, fanned out with asyncio.gather). Each
one runs the SAME logic in uma.core.calibrate — nothing is reimplemented
here; this file is a thin orchestration layer that hands off to the real,
independently-tested engine and reports back whatever it actually measured.

Deploy (Render Dashboard -> New -> Workflow, pointed at this repo):
    Runtime:       Python 3
    Build Command: pip install -e ".[render]"
    Start Command: python -m uma.render_workflows.workflow
    Env vars:      UMA_LLM_API_KEY, UMA_LLM_MODEL (required — same ones
                   `uma judge` uses), plus any optional UMA_LLM_* vars.

Local dev (no Render account required — Render's own CLI dev server):
    render workflows dev -- python -m uma.render_workflows.workflow
    render workflows tasks runs start --task uma_calibrate --local -o json -- '{}'
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from render_sdk import Retry, Workflows

from uma.core.calibrate import (
    DEFAULT_THRESHOLDS,
    BenchmarkCase,
    run_baseline_benchmark,
    run_benchmark_at_threshold,
)
from uma.llm.client import LLMConfig, UmaLLMClient

app = Workflows(
    default_retry=Retry(max_retries=2, wait_duration_ms=2000, backoff_scaling=1.5),
    default_timeout=300,
)


def _default_benchmark_path() -> Path:
    # src/uma/render_workflows/workflow.py -> parents[3] == repo root
    return Path(__file__).resolve().parents[3] / "examples" / "benchmark.json"


def _load_benchmark(benchmark_path: str | None) -> list[dict]:
    path = Path(benchmark_path) if benchmark_path else _default_benchmark_path()
    return json.loads(path.read_text(encoding="utf-8"))


@app.task
async def calibrate_baseline(benchmark: list[dict]) -> dict:
    """WITHOUT UMA: one independent task run — full, unfiltered context sent
    to the LLM for every benchmark case."""
    cases = [BenchmarkCase(**c) for c in benchmark]
    client = UmaLLMClient(LLMConfig.from_env())
    result = await asyncio.to_thread(run_baseline_benchmark, cases, client)
    return result.as_dict()


@app.task
async def calibrate_threshold(
    threshold: float, benchmark: list[dict], max_tokens: int | None = None
) -> dict:
    """WITH UMA at one threshold: one independent task run.

    The real filtering (local cross-encoder) and real LLM calls happen
    synchronously under the hood via uma.core.calibrate.run_benchmark_at_threshold;
    it's offloaded to a thread with asyncio.to_thread so several thresholds
    can run concurrently when the orchestrator below fans them out with
    asyncio.gather.
    """
    cases = [BenchmarkCase(**c) for c in benchmark]
    client = UmaLLMClient(LLMConfig.from_env())
    result = await asyncio.to_thread(run_benchmark_at_threshold, cases, threshold, client, max_tokens)
    return result.as_dict()


@app.task
async def uma_calibrate(
    thresholds: list[float] | None = None,
    benchmark_path: str | None = None,
    max_tokens: int | None = None,
) -> dict:
    """Uma Calibrate orchestrator: find the minimum sufficient context.

    Runs the WITHOUT-UMA baseline and every WITH-UMA threshold as
    independent, parallel task runs (fan-out), then fans back in to report
    which threshold reached the best observed accuracy while keeping the
    least context. Every number in the return value comes from an actual
    filter_context() call and an actual LLM call made during this run —
    nothing here is a target, an estimate, or asserted in advance.
    """
    thresholds = thresholds if thresholds is not None else DEFAULT_THRESHOLDS
    benchmark = _load_benchmark(benchmark_path)

    baseline_run = calibrate_baseline(benchmark)
    threshold_runs = [calibrate_threshold(t, benchmark, max_tokens) for t in thresholds]

    baseline_result, *threshold_results = await asyncio.gather(baseline_run, *threshold_runs)

    best_accuracy = max((r["accuracy"] for r in threshold_results), default=0.0)
    candidates = [r for r in threshold_results if r["accuracy"] == best_accuracy]
    minimum_sufficient = (
        min(candidates, key=lambda r: r["avg_context_kept_percent"]) if candidates else None
    )

    return {
        "benchmark_size": len(benchmark),
        "thresholds_tested": thresholds,
        "baseline": baseline_result,
        "results": threshold_results,
        "best_accuracy": round(best_accuracy, 4),
        "minimum_sufficient_context": (
            {
                "threshold": minimum_sufficient["threshold"],
                "avg_context_kept_percent": minimum_sufficient["avg_context_kept_percent"],
                "accuracy": minimum_sufficient["accuracy"],
            }
            if minimum_sufficient
            else None
        ),
    }


if __name__ == "__main__":
    app.start()
