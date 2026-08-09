"""``uma calibrate`` — local runner for Uma Calibrate.

This runs the exact same engine (:mod:`uma.core.calibrate`) that backs the
Render Workflow in :mod:`uma.render_workflows.workflow`, just sequentially
in this process instead of fanned out across parallel Render task runs.
Useful for local iteration on the benchmark/thresholds without needing a
deployed Render service, and as a fallback if Render isn't configured.

For the actual parallel, Render-hosted version, see
``src/uma/render_workflows/workflow.py`` and the README's "Uma Calibrate"
section.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from uma.core.calibrate import DEFAULT_THRESHOLDS, BenchmarkCase, run_calibration
from uma.llm.client import LLMConfig, LLMConfigError, UmaLLMClient

console = Console(highlight=False)


def _default_benchmark_path() -> Path:
    # src/uma/cli/calibrate.py -> parents[3] == repo root
    return Path(__file__).resolve().parents[3] / "examples" / "benchmark.json"


def _load_benchmark(path: str | None) -> list[BenchmarkCase]:
    p = Path(path) if path else _default_benchmark_path()
    raw = json.loads(p.read_text(encoding="utf-8"))
    return [BenchmarkCase(**c) for c in raw]


def run_calibrate(
    thresholds: list[float] | None = None,
    benchmark_path: str | None = None,
    max_tokens: int | None = None,
) -> int:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    try:
        config = LLMConfig.from_env()
    except LLMConfigError as exc:
        console.print(Panel(str(exc), title="MISSING CONFIGURATION", border_style="red"))
        return 1

    thresholds = thresholds or DEFAULT_THRESHOLDS
    cases = _load_benchmark(benchmark_path)

    # console.rule() FIRST, same reasoning as judge.py: constructing
    # UmaLLMClient imports the (large, sometimes slow-to-import-cold)
    # openai package, and doing that before anything is printed can make
    # the terminal look completely hung with zero feedback.
    console.rule("[bold]UMA CALIBRATE[/bold]")
    client = UmaLLMClient(config)
    console.print(
        Panel(
            f"Benchmark cases: {len(cases)}\n"
            f"Thresholds tested: {', '.join(str(t) for t in thresholds)}\n"
            f"Model: {config.model}",
            title="RUNNING",
            title_align="left",
            border_style="cyan",
        )
    )

    with console.status("[bold]Running WITHOUT-UMA baseline + threshold sweep...", spinner="dots"):
        result = run_calibration(cases, thresholds=thresholds, llm_client=client, max_tokens=max_tokens)

    table = Table(title="COST x QUALITY CURVE", show_lines=False)
    table.add_column("Threshold")
    table.add_column("Context kept", justify="right")
    table.add_column("Accuracy", justify="right")
    table.add_column("Correct", justify="right")

    table.add_row(
        "— (WITHOUT UMA)",
        "100.0%",
        f"{result.baseline.accuracy * 100:.0f}%",
        f"{result.baseline.correct_count}/{len(result.baseline.cases)}",
    )
    for r in result.threshold_results:
        table.add_row(
            f"{r.threshold:.2f}",
            f"{r.avg_context_kept_percent:.1f}%",
            f"{r.accuracy * 100:.0f}%",
            f"{r.correct_count}/{len(r.cases)}",
        )
    console.print(table)

    minimum = result.minimum_sufficient
    if minimum is not None:
        console.print(
            Panel(
                f"Threshold {minimum.threshold:.2f} retained only "
                f"{minimum.avg_context_kept_percent:.1f}% of context while matching "
                f"the best observed accuracy ({result.best_accuracy * 100:.0f}%, "
                f"{minimum.correct_count}/{len(minimum.cases)} correct).",
                title="MINIMUM SUFFICIENT CONTEXT",
                title_align="left",
                border_style="green",
            )
        )
    else:
        console.print(Panel("No threshold produced a scored result.", border_style="red"))

    return 0
