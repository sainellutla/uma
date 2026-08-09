# Uma

**Uma is a local RAG context optimizer, exposed as an MCP server, that
empirically discovers the minimum context an LLM needs to answer correctly
— instead of just asserting a filter helps.**

```text
WITHOUT UMA                          WITH UMA

User Query                           User Query
    |                                    |
Retrieved Context                    Retrieved Context
    |                                    |
   LLM                                  Uma MCP
    |                                    |
  Answer                          Relevant Context
                                         |
                                        LLM
                                         |
                                       Answer
```

Uma does the filtering with a **local cross-encoder** — no LLM is used to
decide relevance. It runs as a reusable Python engine, a real
[MCP](https://modelcontextprotocol.io) server, a terminal demo (`uma judge`)
that runs one controlled experiment, and **Uma Calibrate** — a real
[Render Workflow](https://render.com/docs/workflows) that sweeps relevance
thresholds against a small benchmark, in parallel, to find the most
aggressive filtering that doesn't cost any answer accuracy. Every number
either produces is measured, not asserted.

---

## What is Uma?

Retrieval-augmented generation pipelines tend to over-retrieve: a vector
search returns the top-k chunks by embedding similarity, and most of that
gets stuffed into the prompt whether or not it's actually needed to answer
the question. Uma sits between retrieval and generation and removes the
sentences that a cross-encoder judges irrelevant to the query, before the
LLM ever sees them — and then goes one step further with **Uma Calibrate**,
which answers the harder question: *how aggressively can this actually be
filtered before answer quality breaks?*

Prompt compression itself isn't new — Microsoft's
[LLMLingua](https://github.com/microsoft/LLMLingua) already does
coarse-to-fine, question-aware prompt compression. Uma's angle is narrower
and more operational: a lightweight local relevance model, exposed as
transparent developer infrastructure through MCP, with every filtering
decision measured — and, via Render Workflows, an empirical way to find the
right operating point for a given workload instead of guessing at a
threshold.

## Why RAG context is expensive

Every token in a retrieved chunk that isn't relevant to the question still
costs:

- **Money** — you pay for input tokens whether the model uses them or not.
- **Latency** — bigger prompts take longer to process, especially at longer
  context lengths.
- **Attention** — irrelevant text competes for the model's attention budget
  alongside the text that actually matters.

None of this is hypothetical or Uma-specific — it's a direct consequence of
how retrieval over-fetches by design (top-k similarity search has no notion
of "is this sentence actually needed"). Uma's job is narrow: prune the
retrieved text down using a purpose-built relevance model, before generation.

## How Uma works

```text
Context
  |
  v
Sentence segmentation      (regex-based, dependency-free)
  |
  v
Cross-encoder               (cross-encoder/ms-marco-MiniLM-L-6-v2, local)
  |
  v
Relevance score per sentence   (sigmoid-squashed logit, 0-1 — a normalized
  |                              ranking score, NOT a calibrated probability)
  v
Threshold filtering          (keep score >= threshold, default 0.5)
  |
  v
Optional token budget        (greedy by score, re-ordered to original order)
  |
  v
Filtered context
```

**A note on the 0-1 score:** it's a sigmoid applied to the cross-encoder's
raw logit, squashed to a bounded scale for thresholding — not a calibrated
probability. `ms-marco-MiniLM-L-6-v2`'s model card documents it as a
ranking score for sorting passages, and makes no claim that a score of 0.83
means "83% likely relevant." 0.5 is simply Uma's default *operating
threshold*, not a probability cutoff. See **Uma Calibrate** below for
finding the right threshold empirically instead of assuming the default
fits your workload.

This is implemented once, in [`uma.core.filter.filter_context`](src/uma/core/filter.py),
and it's the only filtering implementation in the project — the terminal
demo, the three MCP tools, and Uma Calibrate all call into it directly.

```python
from uma.core.filter import filter_context

result = filter_context(
    query="What was Apple's revenue in fiscal year 2024?",
    context=retrieved_context,
    threshold=0.5,
    max_tokens=None,
)

result.filtered_context      # str
result.metrics.original_tokens
result.metrics.filtered_tokens
result.metrics.reduction_percent
result.metrics.filtering_latency_ms
```

The cross-encoder is loaded once per process (see [`uma.core.model`](src/uma/core/model.py))
and reused across every call; sentence batches are scored in one
`model.predict()` call rather than one at a time.

## Uma Calibrate

`uma judge` answers "did filtering help on this one question?" **Uma
Calibrate** answers a different, harder question: *how far can Uma push the
threshold on a workload before accuracy actually breaks?*

Given a small benchmark of `{question, context, expected_answer}` cases
([`examples/benchmark.json`](examples/benchmark.json), 10 cases), Uma
Calibrate sweeps a set of thresholds and, at each one, filters every case's
context, sends it to the real configured LLM, and scores the answer
(exact case-insensitive substring match against `expected_answer` —
deliberately simple and inspectable, not an LLM-as-judge; see Limitations).
The threshold with the best observed accuracy *and* the lowest average
context retained is reported as the **minimum sufficient context**.

```text
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
```

Each branch above is an independent task run — this is exactly the
"execute independent task runs in parallel" pattern Render Workflows
documents, fanned out with `asyncio.gather` in
[`src/uma/render_workflows/workflow.py`](src/uma/render_workflows/workflow.py).
It's deliberately **not** in Uma's request-time critical path: the
cross-encoder stays loaded in-process exactly as before (see
`uma.core.model`), and a single `uma judge` / `uma_filter` call is still a
few hundred milliseconds with no orchestration involved. Render Workflows
is used specifically for the piece that benefits from it — a longer-running
sweep made of several independent, parallelizable benchmark passes — not
bolted onto every request to satisfy a checkbox.

### A real calibration run

This is an actual `uma calibrate` result (Gemini 3.1 Flash-Lite, 10-case
benchmark, thresholds 0.20/0.35/0.50/0.65/0.80), executed through the real
`render_sdk` task graph:

| Threshold | Context kept | Accuracy | Correct |
|---|---|---|---|
| — (WITHOUT UMA) | 100.0% | 100% | 10/10 |
| 0.20 | 36.0% | 100% | 10/10 |
| 0.35 | 33.9% | 100% | 10/10 |
| 0.50 | 32.0% | 100% | 10/10 |
| 0.65 | 27.9% | 100% | 10/10 |
| 0.80 | 24.1% | 100% | 10/10 |

**Minimum sufficient context: threshold 0.80, 24.1% of context retained,
100% accuracy (10/10) — a 76% reduction with zero measured accuracy loss on
this benchmark.** Every one of these ten questions was answered correctly
even at Uma's most aggressive tested threshold. That's a genuine, if
narrow, finding: it doesn't mean 0.80 is universally safe (see
Limitations — a 10-question, single-document benchmark is not proof it
generalizes), but it's a real number this system actually computed, not a
target it was tuned to hit. The full per-question breakdown, including
every generated answer, is reproducible with the command below.

### Running it

**Locally** (no Render account needed — runs the same engine sequentially):

```bash
uma calibrate
uma calibrate --thresholds 0.1,0.3,0.5,0.7,0.9 --benchmark examples/benchmark.json
```

**As a Render Workflow** — deploy via the Render Dashboard (Blueprints /
`render.yaml` aren't yet supported for Workflows, so this is a dashboard or
CLI step, not a file in this repo):

```bash
render workflows create \
  --name uma-calibrate \
  --repo https://github.com/sainellutla/uma.git \
  --runtime python \
  --build-command 'pip install -e ".[render]"' \
  --run-command "python -m uma.render_workflows.workflow" \
  --env-var UMA_LLM_API_KEY=<your key> \
  --env-var UMA_LLM_MODEL=<your model> \
  --env-var UMA_LLM_BASE_URL=<your provider's base URL>
```

Then trigger it: `render workflows tasks runs start --task uma_calibrate -o json -- '{}'`

**Status of the live deployment:** the `render_sdk` task graph above is
real, verified code — imported cleanly, all three tasks (`uma_calibrate`,
`calibrate_threshold`, `calibrate_baseline`) register correctly with the
SDK's task registry, and the exact numbers in the table above came from
actually executing that task graph (real `asyncio.gather` fan-out across
thresholds, real cross-encoder, real LLM calls) through `render_sdk`'s own
task-registry/client-context mechanism. It is **not currently deployed to
Render's cloud**: `render workflows create` requires a payment method on
the account (`402 Payment information is required`), which wasn't
available to set at the time. `render workflows dev`, Render's local dev
server, also currently has a Windows-specific bug (fails with
`GetFileAttributesEx /tmp` even from a clean shell) that blocked verifying
through that specific path. The command above is the exact, complete
deploy step — one API call away — once billing is set up.

## Architecture

```text
RAG (your retrieval pipeline)
  |
  v
Uma  --  sentence segmentation -> local cross-encoder -> threshold -> filtered context
  |
  v
LLM  --  OpenAI-compatible chat completion (same client, same config, either arm)
```

```text
uma/
├── pyproject.toml
├── README.md
├── .env.example
│
├── src/uma/
│   ├── core/
│   │   ├── model.py         # cross-encoder loading (singleton, thread-safe)
│   │   ├── scorer.py         # sentence segmentation + cross-encoder scoring
│   │   ├── filter.py         # filter_context() — the engine, used everywhere
│   │   ├── tokenizer.py       # tiktoken-based token counting
│   │   ├── metrics.py         # FilterMetrics / LLMUsage / ExperimentResult / RuntimeStats
│   │   └── calibrate.py       # Uma Calibrate engine (threshold sweep + scoring)
│   │
│   ├── mcp/
│   │   ├── server.py          # MCPServer instance, uma-mcp entry point
│   │   └── tools.py           # uma_filter / uma_score / uma_stats (plain functions)
│   │
│   ├── llm/
│   │   └── client.py          # OpenAI-compatible client, env-var configured
│   │
│   ├── render_workflows/
│   │   └── workflow.py        # Uma Calibrate as a real Render Workflow
│   │
│   └── cli/
│       ├── judge.py           # `uma judge` — the two-stage A/B demo
│       └── calibrate.py       # `uma calibrate` — local threshold sweep runner
│
├── examples/
│   ├── context.txt            # demo RAG context (Apple FY2024 + adjacent noise, ~900 tokens)
│   ├── context_large.txt      # larger demo context (~2,500 tokens, expanded 10-K-style)
│   └── benchmark.json         # 10-case benchmark for Uma Calibrate
│
└── tests/
    ├── test_filter.py
    ├── test_metrics.py
    ├── test_llm.py
    ├── test_mcp.py
    ├── test_calibrate.py
    └── test_experiment_consistency.py
```

## MCP integration

Uma exposes **exactly three tools** over MCP, using the official
[MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
(`mcp.server.mcpserver.MCPServer`) over stdio transport. All three call
straight into the same `filter_context` / `score_context` engine the CLI
uses — see [`src/uma/mcp/tools.py`](src/uma/mcp/tools.py).

### `uma_filter`

**In:** `query: str`, `context: str`, `threshold: float = 0.5`, `max_tokens: int | None = None`

**Out:**
```json
{
  "filtered_context": "...",
  "original_tokens": 16,
  "filtered_tokens": 7,
  "tokens_removed": 9,
  "reduction_percent": 56.25,
  "sentences_processed": 2,
  "sentences_kept": 1,
  "sentences_removed": 1,
  "filtering_latency_ms": 199.4
}
```

### `uma_score`

**In:** `query: str`, `context: str`
**Out:** per-sentence normalized relevance scores (0-1, sigmoid-squashed
cross-encoder logits — see the probability caveat above), unfiltered —
useful for inspecting *why* a sentence was kept or dropped.

### `uma_stats`

**In:** nothing
**Out:** cumulative filtering statistics (calls, tokens processed/removed,
average reduction, average latency) for this server process.

## Installation

```bash
git clone https://github.com/sainellutla/uma.git
cd uma
python -m venv .venv
# Windows: .venv\Scripts\activate   |   macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
```

To also run/deploy Uma Calibrate as a Render Workflow, install the `render`
extra as well: `pip install -e ".[dev,render]"`.

## Environment variables

Copy `.env.example` to `.env` and fill in your provider's details:

```text
UMA_LLM_API_KEY               required — never hardcoded, read from env only
UMA_LLM_MODEL                 required — exact model name for your provider
UMA_LLM_BASE_URL              optional — defaults to OpenAI's API
UMA_LLM_TEMPERATURE           optional — default 0.0
UMA_LLM_MAX_TOKENS            optional — default 400
UMA_LLM_INPUT_PRICE_PER_1M    optional — USD per 1M input tokens, for cost estimation
UMA_LLM_OUTPUT_PRICE_PER_1M   optional — USD per 1M output tokens, for cost estimation
UMA_CROSS_ENCODER_MODEL       optional — default cross-encoder/ms-marco-MiniLM-L-6-v2
UMA_CONTEXT_PATH              optional — override the demo context file (e.g. examples/context_large.txt)
```

Because the client is OpenAI-compatible, any provider that speaks that
protocol works: OpenAI, Google Gemini (via its OpenAI-compatible endpoint),
Groq, Together AI, Fireworks, or a local runtime like Ollama / vLLM exposing
an `/v1` route. If `UMA_LLM_INPUT_PRICE_PER_1M` / `UMA_LLM_OUTPUT_PRICE_PER_1M`
aren't set, Uma shows raw token counts and labels cost `n/a` rather than
guessing — it never calls a computed number "credits" unless the provider
actually uses a credit system, and any cost it does show is explicitly
labeled an **estimate** (OpenAI-compatible chat completion responses don't
return a cost field).

## Running the judge demo

```bash
uma judge
```

Runs the full two-stage controlled experiment described below and prints a
final comparison table. Optional flags:

```bash
uma judge --query "..." --threshold 0.6 --max-tokens 300
```

For a visually bigger demo, point at the larger context file:

```bash
UMA_CONTEXT_PATH=examples/context_large.txt uma judge
```

### What it does

1. **Loads the demo context once** from `examples/context.txt` (16 retrieved
   passages about Apple's FY2024 results, mixed with adjacent-but-irrelevant
   company history, executives, litigation, and competitor information —
   representative of real vector-DB over-retrieval). The identical string is
   passed into both experiment arms; it is never re-retrieved or regenerated.
2. **RUN 01 / WITHOUT UMA** — the full context and the prompt go straight to
   the configured LLM. Real input/output tokens, latency, and (if pricing is
   configured) cost are measured from the actual API response.
3. **RUN 02 / WITH UMA** — the identical prompt, model, and generation
   settings are reused (same `LLMConfig`, same client instance), but the
   context is first passed through Uma's real local cross-encoder. Only the
   filtered context differs.
4. **RESULTS** — one table comparing both runs, plus deltas for context
   reduction, token change, cost change (if measurable), and total latency
   change — including Uma's own filtering latency, never hidden.
5. **ANSWER CONSISTENCY** — both answers are printed so you can judge
   directly whether the filtered context still contained what was needed.

## Connecting Claude (or another MCP client)

Run the server directly to confirm it starts:

```bash
uma-mcp
# or: python -m uma.mcp.server
```

It logs `[uma-mcp] ready. serving MCP over stdio.` to **stderr** once the
cross-encoder is warmed up (stdout is reserved for the MCP JSON-RPC stream).

For Claude Desktop / Claude Code, add to your MCP config
(`claude_desktop_config.json` or your client's equivalent):

```json
{
  "mcpServers": {
    "uma": {
      "command": "uma-mcp"
    }
  }
}
```

If `uma-mcp` isn't on `PATH` (e.g. it's only installed in a venv), point at
the interpreter directly:

```json
{
  "mcpServers": {
    "uma": {
      "command": "C:/path/to/uma/.venv/Scripts/python.exe",
      "args": ["-m", "uma.mcp.server"]
    }
  }
}
```

Once connected, ask Claude to call `uma_filter` on some retrieved text, or
`uma_score` to see per-sentence relevance, or `uma_stats` for cumulative
stats.

## Benchmarking

`uma judge` and `uma calibrate` *are* the benchmarks — every metric they
print comes from a real measurement in that run (`time.perf_counter()`
around the actual operation, token counts from `tiktoken` / the provider's
own usage payload, accuracy from actually scoring the actual answer). There
is no separate synthetic benchmark suite beyond `examples/benchmark.json`,
because filtering behavior and LLM answers both depend on the specific
query and context, and Uma makes no claim that generalizes beyond "here's
what happened on this run." Run it against your own retrieved context and
query to get numbers for your use case.

### Real runs

Context reduction is **deterministic** for a given (query, context,
threshold) — the same context against the same query at the same threshold
reduces to the same token count on every run, regardless of which LLM is on
the other end. Everything downstream of that (LLM latency, cost) depends on
the provider you point Uma at.

**Small demo context (`examples/context.txt`, 902 tokens), hosted API —
Gemini 3.1 Flash-Lite** (Gemini Developer API, standard tier). Pricing is
real, published pricing (see `.env.example`), not a guess — but it's still
labeled `est.` since the provider's response doesn't return a cost field
itself:

| | WITHOUT UMA | WITH UMA |
|---|---|---|
| Model | gemini-3.1-flash-lite | gemini-3.1-flash-lite |
| Context tokens | 902 | 430 |
| Output tokens | 109 | 106 |
| Total tokens | 1198 | 683 |
| Cost (est.) | $0.000436 | $0.000303 |
| LLM latency | 3.11 sec | 0.80 sec |
| Uma latency | — | 865.2 ms |
| Total latency | 3.11 sec | 1.67 sec |

Context reduction: 52.3% · Total token change: -43.0% · Estimated cost
change: -30.4%

**Larger demo context (`examples/context_large.txt`, 2,517 tokens),
same model:**

| | WITHOUT UMA | WITH UMA |
|---|---|---|
| Context tokens | 2517 | 622 |
| Total tokens | 2910 | 913 |
| Cost (est.) | $0.000870 | $0.000367 |
| Total latency | — | -45.7% change |

Context reduction: 75.3% (2517 -> 622 tokens, 69 -> 14 sentences kept) ·
Total token change: -68.6% · Estimated cost change: -57.8% · Total latency
change: -45.7% (includes Uma's own ~2.1s filtering pass on this larger
context). This run's reduction number is whatever the real system produced
against this specific context and query — it was not chosen in advance.

**Local model — Ollama, phi3:mini** (CPU-bound, no pricing available so
cost is not shown), small demo context:

| | WITHOUT UMA | WITH UMA |
|---|---|---|
| Model | phi3:mini | phi3:mini |
| Context tokens | 902 | 430 |
| Output tokens | 137 | 121 |
| Total tokens | 1378 | 769 |
| LLM latency | 66.30 sec | 32.43 sec |
| Uma latency | — | 736.6 ms |
| Total latency | 66.30 sec | 33.17 sec |

Context reduction: 52.3% · Total token change: -44.2% · Total latency
change: -50.0% (includes Uma's own filtering pass)

All three runs answered correctly and consistently across both arms
(Apple's FY2024 net sales of $391.035B, broken down by
iPhone/Services/Wearables/Mac/iPad). See the **Uma Calibrate** section
above for the 10-question accuracy benchmark, which is a stronger claim
than any single demo question.

**Be skeptical of the latency numbers specifically.** Re-running the small
Gemini case three times back-to-back produced total-latency deltas of
-73%, +19%, and +129% — hosted-API network/queueing variance on calls this
short swamps any signal from the token-count difference. The only latency
number that was consistent across every run was **Uma's own filtering
pass** (~740-870ms on the small context, ~2.1s on the larger one) — because
that's local, deterministic compute, not a network call. Token reduction
(and therefore cost, on any per-token-billed provider) is the reliable
claim here; LLM latency change is not, and this project won't pretend
otherwise. Run `uma judge` yourself, multiple times, against your target
provider before citing a latency number.

## Limitations

Be skeptical of anything that sounds too clean:

- **Sentence-level granularity only.** Uma keeps or drops whole sentences.
  It cannot partially redact a sentence that's half-relevant, and it has no
  notion of cross-sentence dependencies (e.g. a pronoun whose antecedent got
  filtered out).
- **The cross-encoder can be wrong.** `ms-marco-MiniLM-L-6-v2` is a small,
  fast model trained on search-relevance data, not a general-purpose
  reasoning model. It can misjudge relevance on unusual phrasing, negation,
  or multi-hop questions where a sentence is only relevant in combination
  with another one.
- **The relevance score is not a probability.** It's a sigmoid-squashed
  ranking score, useful as a bounded threshold target, not a calibrated
  "chance this is relevant." Don't read 0.83 as "83% relevant."
- **Uma Calibrate's scoring is exact substring match, not semantic
  judging.** `score_answer()` checks whether `expected_answer` appears
  (case-insensitive) in the model's response. It will false-negative a
  correct-but-differently-phrased answer, and it only covers factual,
  short-answer-style questions well. It was chosen deliberately over an
  LLM-as-judge: a scoring method that itself required trusting another LLM
  call would undermine the point of measuring accuracy in the first place.
- **The 10-question calibration benchmark is small and single-document.**
  "100% accuracy down to threshold 0.80" is a real, reproducible result on
  *this* benchmark — it is not evidence that 0.80 is safe for a different
  document, question style, or domain. Uma Calibrate is a tool for finding
  the right threshold for *your* workload, not a universal constant.
- **Token reduction is not a guaranteed cost or latency reduction.** Cost
  scales with tokens removed only if the provider charges per input token
  (most do). Latency reduction depends on the provider/model — see the
  benchmarking section above.
- **Uma adds its own latency.** Filtering is a real inference pass over
  every sentence in the context. For a very small context or a very fast
  LLM, Uma's own latency could plausibly outweigh what it saves — this
  implementation does not hide that; both `Uma latency` and `Total latency`
  are always shown separately.
- **No semantic dedup or summarization.** Uma removes irrelevant sentences;
  it does not compress, rewrite, or merge relevant ones.
- **Sentence segmentation is a regex heuristic**, not a full NLP pipeline —
  it handles common abbreviations and decimals but is not guaranteed correct
  on arbitrary text (e.g. unusual formatting, tables, code, non-English
  text).
- **Cost figures are always estimates** computed from user-supplied
  per-token pricing (`UMA_LLM_INPUT_PRICE_PER_1M` /
  `UMA_LLM_OUTPUT_PRICE_PER_1M`); OpenAI-compatible APIs do not return a cost
  field, so Uma never claims to report a provider's actual billed amount.
- **Uma Calibrate is not currently deployed live on Render's cloud** — see
  the "Status of the live deployment" note above for exactly why and what
  was verified instead.
- **This is a demo/reference implementation**, not a production RAG gateway:
  no caching, no batching across concurrent requests, no persistence of
  `uma_stats` across restarts.

Uma does **not** claim 99% cost reduction, guaranteed accuracy, sub-millisecond
latency, or lossless compression, and it does not claim to be the first tool
to compress LLM prompts. What it does is measurably remove sentences a
cross-encoder scores below a relevance threshold, empirically find how low
that threshold can go on a given benchmark before accuracy drops, and
report exactly what that did to token counts, cost (if measurable), and
latency — nothing more.

## Development

```bash
pytest            # 55 tests: sentence splitting, real cross-encoder scoring,
                   # filtering, metrics, LLM client (mocked transport, incl.
                   # rate-limit retry), the real MCP server over an
                   # in-process session, Uma Calibrate's threshold sweep and
                   # scoring logic, and an explicit baseline-vs-Uma fairness
                   # check.
```
