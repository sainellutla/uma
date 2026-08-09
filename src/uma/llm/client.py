"""A configurable, OpenAI-compatible LLM client.

Both arms of the controlled experiment (baseline and Uma) go through this
exact same client, constructed from the exact same :class:`LLMConfig`, so
model / temperature / max_tokens can never silently drift between runs.

Configuration is environment-variable driven — no API keys are ever
hardcoded:

    UMA_LLM_API_KEY               required
    UMA_LLM_MODEL                 required
    UMA_LLM_BASE_URL              optional (defaults to OpenAI's API)
    UMA_LLM_TEMPERATURE           optional (default 0.0)
    UMA_LLM_MAX_TOKENS            optional (default 400)
    UMA_LLM_INPUT_PRICE_PER_1M    optional, USD per 1M input tokens
    UMA_LLM_OUTPUT_PRICE_PER_1M   optional, USD per 1M output tokens

If pricing env vars are not set, Uma reports raw token counts and marks
cost as unavailable rather than guessing at a number.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass

from uma.core.metrics import LLMUsage

# Retry-with-backoff on transient rate-limit errors. Uma Calibrate and
# uma_calibrate (the Render Workflow) make many sequential real LLM calls
# per benchmark run, which routinely hits free-tier per-minute request caps
# — this is a real-world condition worth handling, not a hack for one demo.
# The wait actually happens (this is not simulated), so it's reflected
# honestly in the measured latency for whichever call triggered it.
MAX_RATE_LIMIT_RETRIES = 3
RATE_LIMIT_BACKOFF_SEC = 20.0

SYSTEM_PROMPT = (
    "You are a precise assistant. Answer the user's question using ONLY the "
    "information in the provided context. Be concise. If the context does "
    "not contain the answer, say so explicitly."
)


class LLMConfigError(RuntimeError):
    """Raised when required LLM configuration is missing or invalid."""


@dataclass(frozen=True)
class LLMConfig:
    """Generation configuration. Identical instances are used by both arms."""

    model: str
    api_key: str
    base_url: str | None
    temperature: float
    max_tokens: int
    input_price_per_1m: float | None
    output_price_per_1m: float | None

    @classmethod
    def from_env(cls) -> "LLMConfig":
        api_key = os.environ.get("UMA_LLM_API_KEY", "").strip()
        model = os.environ.get("UMA_LLM_MODEL", "").strip()

        if not api_key:
            raise LLMConfigError(
                "UMA_LLM_API_KEY is not set. Export it or add it to a .env "
                "file (see .env.example)."
            )
        if not model:
            raise LLMConfigError(
                "UMA_LLM_MODEL is not set. Export it or add it to a .env "
                "file (see .env.example)."
            )

        base_url = os.environ.get("UMA_LLM_BASE_URL", "").strip() or None
        temperature = float(os.environ.get("UMA_LLM_TEMPERATURE", "0.0"))
        max_tokens = int(os.environ.get("UMA_LLM_MAX_TOKENS", "400"))

        input_price = _optional_float(os.environ.get("UMA_LLM_INPUT_PRICE_PER_1M"))
        output_price = _optional_float(os.environ.get("UMA_LLM_OUTPUT_PRICE_PER_1M"))

        return cls(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            input_price_per_1m=input_price,
            output_price_per_1m=output_price,
        )

    def generation_params(self) -> dict:
        """The subset of config that must be byte-identical across both arms."""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }


def _optional_float(value: str | None) -> float | None:
    if value is None or not value.strip():
        return None
    try:
        return float(value)
    except ValueError:
        return None


class UmaLLMClient:
    """Thin wrapper around the OpenAI-compatible chat completions API."""

    def __init__(self, config: LLMConfig | None = None):
        self.config = config or LLMConfig.from_env()

        # Imported lazily so modules that don't need a live LLM (filtering,
        # MCP scoring tools, most tests) never require the `openai` package
        # to be importable/configured.
        from openai import OpenAI

        self._client = OpenAI(api_key=self.config.api_key, base_url=self.config.base_url)

    def generate(self, *, prompt: str, context: str) -> LLMUsage:
        """Run one real chat completion and return real, measured usage.

        ``prompt`` and ``context`` are combined into the user message. This
        is the single call path used by both the baseline and Uma
        experiments — the only thing that differs between calls is the
        ``context`` string passed in.
        """
        user_message = f"Context:\n{context}\n\nQuestion: {prompt}"

        params = self.config.generation_params()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]

        start = time.perf_counter()
        response = self._create_with_retry(params, messages)
        # Deliberately includes any real retry wait above — that time was
        # actually spent getting this response, so hiding it would misstate
        # the measured latency.
        latency_sec = time.perf_counter() - start

        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0
        reported_total = getattr(usage, "total_tokens", None) if usage else None
        answer = (response.choices[0].message.content or "").strip()

        cost_usd, is_estimate, unavailable_reason = self._compute_cost(
            input_tokens, output_tokens, reported_total
        )

        return LLMUsage(
            model=response.model or self.config.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_sec=latency_sec,
            answer=answer,
            cost_usd=cost_usd,
            cost_is_estimate=is_estimate,
            cost_unavailable_reason=unavailable_reason,
            reported_total_tokens=reported_total,
        )

    def _create_with_retry(self, params: dict, messages: list[dict]):
        """Call chat.completions.create(), retrying on rate limits.

        Free/low tiers of many providers (Gemini's free tier included, at
        15 requests/minute for some models) reject bursts of sequential
        calls — exactly what Uma Calibrate's per-threshold benchmark sweep
        does. Retries with a fixed backoff up to MAX_RATE_LIMIT_RETRIES
        times before giving up and raising.
        """
        from openai import RateLimitError

        attempt = 0
        while True:
            try:
                return self._client.chat.completions.create(
                    model=params["model"],
                    temperature=params["temperature"],
                    max_tokens=params["max_tokens"],
                    messages=messages,
                )
            except RateLimitError:
                attempt += 1
                if attempt > MAX_RATE_LIMIT_RETRIES:
                    raise
                wait_sec = RATE_LIMIT_BACKOFF_SEC * attempt
                print(
                    f"[uma] rate limited, retrying in {wait_sec:.0f}s "
                    f"(attempt {attempt}/{MAX_RATE_LIMIT_RETRIES})",
                    file=sys.stderr,
                )
                time.sleep(wait_sec)

    def _compute_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        reported_total: int | None = None,
    ) -> tuple[float | None, bool, str | None]:
        """Return (cost_usd, is_estimate, unavailable_reason).

        OpenAI-compatible chat completion responses do not include a cost
        field, so any cost figure Uma shows is necessarily calculated from
        user-supplied per-token pricing, not read from the provider. It is
        always labeled ``cost_is_estimate=True`` for that reason.

        Some models (reasoning/"thinking" variants) bill hidden tokens that
        inflate ``usage.total_tokens`` beyond prompt_tokens+completion_tokens.
        Those hidden tokens are generation-side, so when present they're
        priced at the output rate rather than silently dropped.
        """
        if self.config.input_price_per_1m is None or self.config.output_price_per_1m is None:
            return (
                None,
                True,
                "No pricing configured (set UMA_LLM_INPUT_PRICE_PER_1M / "
                "UMA_LLM_OUTPUT_PRICE_PER_1M) — showing token counts only.",
            )
        billed_output_tokens = output_tokens
        if reported_total is not None:
            billed_output_tokens = max(output_tokens, reported_total - input_tokens)

        cost = (
            input_tokens * self.config.input_price_per_1m
            + billed_output_tokens * self.config.output_price_per_1m
        ) / 1_000_000.0
        return cost, True, None
