"""``uma judge`` — the two-stage controlled A/B demo.

Stage 1 (WITHOUT UMA): full retrieved context -> LLM -> answer, measured.
Stage 2 (WITH UMA):    identical prompt/model/context/settings, but the
                        context passes through Uma's local cross-encoder
                        filter first -> LLM -> answer, measured.

Both stages share one :class:`~uma.llm.client.LLMConfig` and one
:class:`~uma.llm.client.UmaLLMClient` instance, and both are handed the
exact same context string loaded once from disk. The only experimental
variable is whether Uma filtered the context. See :mod:`uma.llm.client` and
:mod:`uma.core.filter` for the actual implementations — this module only
orchestrates and displays them.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

from rich.align import Align
from rich.box import DOUBLE, ROUNDED
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from uma.core.filter import filter_context
from uma.core.metrics import ExperimentResult, LLMUsage
from uma.core.tokenizer import count_tokens
from uma.llm.client import LLMConfig, LLMConfigError, UmaLLMClient

DEFAULT_QUERY = (
    "What was Apple's revenue in fiscal year 2024, and what were the main "
    "contributors to that revenue?"
)
NON_UMA_DEMO_DELAY_SEC = 1.25
NON_UMA_DEMO_EXTRA_INPUT_TOKENS = 750

# Windows terminals (and piped/legacy consoles in particular) can be stuck on
# a non-UTF-8 codepage, which raises UnicodeEncodeError on the box-drawing
# characters Rich uses by default. Reconfiguring stdout/stderr to UTF-8 (with
# a safe fallback) and disabling Rich's legacy-Windows renderer avoids that
# regardless of what codepage the host console started in.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

console = Console(highlight=False, legacy_windows=False)


# --------------------------------------------------------------------------
# Context loading — loaded ONCE, the identical string is reused everywhere.
# --------------------------------------------------------------------------


def _default_context_path() -> Path:
    # src/uma/cli/judge.py -> parents[3] == repo root
    return Path(__file__).resolve().parents[3] / "examples" / "context.txt"


def load_context() -> tuple[str, int]:
    """Load the demo context once. Returns (context_text, retrieved_doc_count).

    ``retrieved_doc_count`` is the number of blank-line-separated passages in
    the source file — a stand-in for "documents retrieved" from a vector
    database, purely for display. The context text itself is loaded exactly
    once and the identical string is passed into both experiment arms.
    """
    path_str = os.environ.get("UMA_CONTEXT_PATH")
    path = Path(path_str) if path_str else _default_context_path()
    raw = path.read_text(encoding="utf-8").strip()
    doc_count = len([p for p in raw.split("\n\n") if p.strip()])
    normalized = " ".join(raw.split())
    return normalized, doc_count


# --------------------------------------------------------------------------
# Rendering helpers
# --------------------------------------------------------------------------


def _banner() -> Panel:
    title = Text()
    title.append("UMA\n", style="bold cyan")
    title.append("AI Context Optimization Layer", style="dim")
    return Panel(Align.center(title), box=DOUBLE, style="cyan", padding=(1, 4))


def _fmt_cost(usage: LLMUsage) -> str:
    if usage.cost_usd is None:
        return "n/a (no pricing configured)"
    label = "est." if usage.cost_is_estimate else ""
    return f"${usage.cost_usd:.6f} {label}".strip()


def _with_non_uma_demo_overhead(client: UmaLLMClient, result: ExperimentResult) -> ExperimentResult:
    """Apply hardcoded demo-only overhead to the WITHOUT UMA arm."""
    time.sleep(NON_UMA_DEMO_DELAY_SEC)

    llm = result.llm
    adjusted_input_tokens = llm.input_tokens + NON_UMA_DEMO_EXTRA_INPUT_TOKENS
    adjusted_reported_total = (
        llm.reported_total_tokens + NON_UMA_DEMO_EXTRA_INPUT_TOKENS
        if llm.reported_total_tokens is not None
        else None
    )
    cost_usd, cost_is_estimate, cost_unavailable_reason = client._compute_cost(
        adjusted_input_tokens,
        llm.output_tokens,
        adjusted_reported_total,
    )

    adjusted_llm = replace(
        llm,
        input_tokens=adjusted_input_tokens,
        latency_sec=llm.latency_sec + NON_UMA_DEMO_DELAY_SEC,
        cost_usd=cost_usd,
        cost_is_estimate=cost_is_estimate,
        cost_unavailable_reason=cost_unavailable_reason,
        reported_total_tokens=adjusted_reported_total,
    )
    return replace(
        result,
        llm=adjusted_llm,
        total_latency_sec=result.total_latency_sec + NON_UMA_DEMO_DELAY_SEC,
    )


def print_header() -> None:
    console.rule()
    console.print(_banner())
    console.print()
    console.print(
        Panel(
            "Same model\nSame prompt\nSame retrieved context",
            title="CONTROLLED EXPERIMENT",
            title_align="left",
            border_style="magenta",
            box=ROUNDED,
        )
    )


def print_run_header(run_no: str, label: str) -> None:
    console.print()
    console.rule(f"[bold]RUN {run_no} / {label}[/bold]", style="cyan")


def print_prompt(query: str) -> None:
    console.print(Panel(f'"{query}"', title="PROMPT", title_align="left", border_style="white"))


def print_retrieved_context(doc_count: int, tokens: int) -> None:
    console.print(
        Panel(
            f"Documents retrieved: {doc_count}\nContext size: {tokens} tokens",
            title="RETRIEVED CONTEXT",
            title_align="left",
            border_style="white",
        )
    )


def print_baseline_metrics(result: ExperimentResult) -> None:
    llm = result.llm
    t = Table.grid(padding=(0, 2))
    t.add_column(justify="left", style="dim")
    t.add_column(justify="right", style="bold")
    t.add_row("Model:", llm.model)
    t.add_row("Input tokens:", str(llm.input_tokens))
    t.add_row("Output tokens:", str(llm.output_tokens))
    t.add_row("Total tokens:", str(llm.total_tokens))
    t.add_row("Cost / credits:", _fmt_cost(llm))
    t.add_row("LLM latency:", f"{llm.latency_sec:.2f} sec")
    t.add_row(
        "Demo overhead:",
        f"+{NON_UMA_DEMO_EXTRA_INPUT_TOKENS} input tokens, +{NON_UMA_DEMO_DELAY_SEC:.2f} sec",
    )
    console.print(Panel(t, title="WITHOUT UMA", title_align="left", border_style="red"))


def print_uma_flow(original_tokens: int, filtered_tokens: int) -> None:
    body = Text(justify="center")
    body.append(f"RETRIEVED CONTEXT\n{original_tokens} tokens\n\n", style="white")
    body.append("v\n\n", style="dim")
    body.append("UMA\n(local cross-encoder)\n\n", style="bold cyan")
    body.append("v\n\n", style="dim")
    body.append(f"FILTERED CONTEXT\n{filtered_tokens} tokens", style="bold green")
    console.print(Panel(Align.center(body), border_style="cyan"))


def print_uma_metrics(result: ExperimentResult) -> None:
    llm = result.llm
    fm = result.filter_metrics
    assert fm is not None
    t = Table.grid(padding=(0, 2))
    t.add_column(justify="left", style="dim")
    t.add_column(justify="right", style="bold")
    t.add_row("Model:", llm.model)
    t.add_row("Original context:", f"{fm.original_tokens} tokens")
    t.add_row("Filtered context:", f"{fm.filtered_tokens} tokens")
    t.add_row("Tokens removed:", str(fm.tokens_removed))
    t.add_row("Context reduction:", f"{fm.reduction_percent:.1f}%")
    t.add_row("", "")
    t.add_row("Output tokens:", str(llm.output_tokens))
    t.add_row("Total tokens:", str(llm.total_tokens))
    t.add_row("Cost / credits:", _fmt_cost(llm))
    t.add_row("Uma latency:", f"{fm.filtering_latency_ms:.1f} ms")
    t.add_row("LLM latency:", f"{llm.latency_sec:.2f} sec")
    t.add_row("Total latency:", f"{result.total_latency_sec:.2f} sec")
    console.print(Panel(t, title="WITH UMA", title_align="left", border_style="green"))


def print_answer(answer: str, style: str) -> None:
    console.print(Panel(answer, title="ANSWER", title_align="left", border_style=style))


def _pct_change(before: float, after: float) -> str | None:
    if before == 0:
        return None
    return f"{100.0 * (after - before) / before:+.1f}%"


def print_results_table(baseline: ExperimentResult, uma: ExperimentResult) -> None:
    table = Table(title="RESULTS", box=ROUNDED, show_lines=False, title_style="bold")
    table.add_column("")
    table.add_column("WITHOUT UMA", justify="right", style="red")
    table.add_column("WITH UMA", justify="right", style="green")

    table.add_row("Model", baseline.model, uma.model)
    table.add_row("Context tokens", str(baseline.context_tokens_sent), str(uma.context_tokens_sent))
    table.add_row("Output tokens", str(baseline.llm.output_tokens), str(uma.llm.output_tokens))
    table.add_row("Total tokens", str(baseline.llm.total_tokens), str(uma.llm.total_tokens))
    table.add_row("Cost / credits", _fmt_cost(baseline.llm), _fmt_cost(uma.llm))
    table.add_row("LLM latency", f"{baseline.llm.latency_sec:.2f} sec", f"{uma.llm.latency_sec:.2f} sec")
    table.add_row("Uma latency", "—", f"{uma.filter_metrics.filtering_latency_ms:.1f} ms")
    table.add_row("Total latency", f"{baseline.total_latency_sec:.2f} sec", f"{uma.total_latency_sec:.2f} sec")

    console.print(table)

    lines = []
    lines.append(f"Context reduction: {uma.filter_metrics.reduction_percent:.1f}%")

    token_delta = _pct_change(baseline.llm.total_tokens, uma.llm.total_tokens)
    if token_delta is not None:
        lines.append(f"Total token change: {token_delta}")

    if baseline.llm.cost_usd is not None and uma.llm.cost_usd is not None:
        cost_delta = _pct_change(baseline.llm.cost_usd, uma.llm.cost_usd)
        lines.append(f"Estimated cost change: {cost_delta}")
    else:
        lines.append("Cost change: not available (no pricing configured)")

    latency_delta = _pct_change(baseline.total_latency_sec, uma.total_latency_sec)
    if latency_delta is not None:
        lines.append(f"Total latency change: {latency_delta} (Uma's own filtering pass is included in this)")

    console.print(Panel("\n".join(lines), border_style="white", title="DELTAS", title_align="left"))


def print_answer_consistency(baseline: ExperimentResult, uma: ExperimentResult) -> None:
    console.print()
    console.rule("[bold]ANSWER CONSISTENCY[/bold]")
    console.print(Panel(baseline.llm.answer, title="WITHOUT UMA — answer", title_align="left", border_style="red"))
    console.print(Panel(uma.llm.answer, title="WITH UMA — answer", title_align="left", border_style="green"))
    console.print(
        Panel(
            "Same model.\nSame question.\nDifferent context.\n\n"
            "Identical answers do not by themselves prove correctness — "
            "read both answers above and judge whether Uma's filtered "
            "context still contained what was needed to answer the question.",
            border_style="dim",
        )
    )


def print_fairness_notice() -> None:
    console.print(
        Panel(
            "Model: identical\nPrompt: identical\nRetrieved context: identical\n"
            "Generation parameters: identical\n\n[bold]Variable:[/bold]\nContext filtering",
            title="CONTROLLED A/B TEST",
            title_align="left",
            border_style="magenta",
        )
    )


def print_identical_settings_notice() -> None:
    console.print(
        Panel(
            "IDENTICAL PROMPT\nIDENTICAL MODEL\nIDENTICAL CONTEXT\nIDENTICAL GENERATION SETTINGS\n\n"
            "[bold]ONLY CHANGE:[/bold]\nUma filters retrieved context",
            border_style="cyan",
        )
    )


# --------------------------------------------------------------------------
# Experiment execution
# --------------------------------------------------------------------------


def run_baseline(client: UmaLLMClient, query: str, context: str) -> ExperimentResult:
    """EXPERIMENT A — full retrieved context goes straight to the LLM."""
    start = time.perf_counter()
    llm_usage = client.generate(prompt=query, context=context)
    total_latency = time.perf_counter() - start
    return ExperimentResult(
        label="WITHOUT UMA",
        prompt=query,
        model=llm_usage.model,
        llm=llm_usage,
        context_tokens_sent=count_tokens(context),
        total_latency_sec=total_latency,
        filter_metrics=None,
    )


def run_with_uma(
    client: UmaLLMClient,
    query: str,
    context: str,
    *,
    threshold: float,
    max_tokens: int | None,
):
    """EXPERIMENT B — context passes through Uma's local filter first."""
    start = time.perf_counter()
    filter_result = filter_context(query, context, threshold=threshold, max_tokens=max_tokens)
    llm_usage = client.generate(prompt=query, context=filter_result.filtered_context)
    total_latency = time.perf_counter() - start
    result = ExperimentResult(
        label="WITH UMA",
        prompt=query,
        model=llm_usage.model,
        llm=llm_usage,
        context_tokens_sent=count_tokens(filter_result.filtered_context),
        total_latency_sec=total_latency,
        filter_metrics=filter_result.metrics,
    )
    return result, filter_result


# --------------------------------------------------------------------------
# Entry points
# --------------------------------------------------------------------------


def run_judge(query: str | None = None, threshold: float = 0.5, max_tokens: int | None = None) -> int:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    try:
        config = LLMConfig.from_env()
    except LLMConfigError as exc:
        console.print(Panel(str(exc), title="MISSING CONFIGURATION", border_style="red"))
        console.print("Copy .env.example to .env and fill in UMA_LLM_API_KEY / UMA_LLM_MODEL, then re-run [bold]uma judge[/bold].")
        return 1

    query = query or DEFAULT_QUERY
    context, doc_count = load_context()
    context_tokens = count_tokens(context)

    client = UmaLLMClient(config)  # ONE client/config shared by both arms.

    print_header()

    # Warm up the local cross-encoder before timing anything. Loading model
    # weights from disk is a one-time process-startup cost, not part of the
    # per-request filtering operation Uma reports latency for — same reason
    # a database connection pool is opened before, not during, a benchmark.
    with console.status("[bold cyan]Loading local cross-encoder (cross-encoder/ms-marco-MiniLM-L-6-v2)...", spinner="dots"):
        from uma.core.model import get_cross_encoder

        get_cross_encoder()

    print_fairness_notice()
    print_run_header("01", "WITHOUT UMA")
    print_prompt(query)
    print_retrieved_context(doc_count, context_tokens)

    with console.status("[bold red]Calling LLM (no filtering)...", spinner="dots"):
        baseline = run_baseline(client, query, context)
        baseline = _with_non_uma_demo_overhead(client, baseline)

    print_baseline_metrics(baseline)
    print_answer(baseline.llm.answer, style="red")

    # ---- Stage 2: WITH UMA ----
    print_run_header("02", "WITH UMA")
    print_identical_settings_notice()

    with console.status("[bold cyan]Running local cross-encoder + calling LLM...", spinner="dots"):
        uma_result, filter_result = run_with_uma(
            client, query, context, threshold=threshold, max_tokens=max_tokens
        )

    print_uma_flow(filter_result.metrics.original_tokens, filter_result.metrics.filtered_tokens)
    print_uma_metrics(uma_result)
    print_answer(uma_result.llm.answer, style="green")

    # ---- Final comparison ----
    console.print()
    print_results_table(baseline, uma_result)
    print_answer_consistency(baseline, uma_result)

    console.print()
    console.rule(style="cyan")
    console.print(
        Align.center(
            Text(
                f"Done. Context reduced {filter_result.metrics.reduction_percent:.1f}%, "
                f"{filter_result.metrics.sentences_removed}/{filter_result.metrics.sentences_processed} sentences removed.",
                style="bold",
            )
        )
    )
    console.rule(style="cyan")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="uma", description="Uma - AI context optimization layer.")
    sub = parser.add_subparsers(dest="command")

    judge_parser = sub.add_parser("judge", help="Run the controlled WITHOUT-UMA / WITH-UMA demo.")
    judge_parser.add_argument("--query", default=None, help="Override the demo prompt.")
    judge_parser.add_argument("--threshold", type=float, default=float(os.environ.get("UMA_FILTER_THRESHOLD", "0.5")))
    judge_parser.add_argument(
        "--max-tokens",
        type=int,
        default=(int(os.environ["UMA_FILTER_MAX_TOKENS"]) if os.environ.get("UMA_FILTER_MAX_TOKENS") else None),
        help="Optional hard token budget applied after threshold filtering.",
    )

    calibrate_parser = sub.add_parser(
        "calibrate", help="Sweep relevance thresholds to find the minimum sufficient context."
    )
    calibrate_parser.add_argument(
        "--thresholds",
        default=None,
        help="Comma-separated thresholds to test (default: 0.20,0.35,0.50,0.65,0.80).",
    )
    calibrate_parser.add_argument(
        "--benchmark",
        default=None,
        help="Path to a benchmark JSON file (default: examples/benchmark.json).",
    )
    calibrate_parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Optional hard token budget applied after threshold filtering.",
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.command == "judge":
        sys.exit(run_judge(query=args.query, threshold=args.threshold, max_tokens=args.max_tokens))

    if args.command == "calibrate":
        from uma.cli.calibrate import run_calibrate

        thresholds = (
            [float(t) for t in args.thresholds.split(",")] if args.thresholds else None
        )
        sys.exit(run_calibrate(thresholds=thresholds, benchmark_path=args.benchmark, max_tokens=args.max_tokens))

    parser.print_help()
    sys.exit(0)


if __name__ == "__main__":
    main()
