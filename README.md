# Uma

**Uma is an AI infrastructure layer and MCP server that filters retrieved RAG
context down to what's actually relevant, before it reaches an LLM.**

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
[MCP](https://modelcontextprotocol.io) server, and a terminal demo
(`uma judge`) that runs one controlled experiment: the same prompt, the same
model, the same retrieved context, run twice — once with the full context,
once with Uma's filtered context — with every number measured, not asserted.

---

## What is Uma?

Retrieval-augmented generation pipelines tend to over-retrieve: a vector
search returns the top-k chunks by embedding similarity, and most of that
gets stuffed into the prompt whether or not it's actually needed to answer
the question. Uma sits between retrieval and generation and removes the
sentences that a cross-encoder judges irrelevant to the query, before the
LLM ever sees them.

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
Relevance score per sentence   (sigmoid-normalized logit, 0-1)
  |
  v
Threshold filtering          (keep score >= threshold, default 0.5)
  |
  v
Optional token budget        (greedy by score, re-ordered to original order)
  |
  v
Filtered context
```

This is implemented once, in [`uma.core.filter.filter_context`](src/uma/core/filter.py),
and it's the only filtering implementation in the project — the terminal
demo and the three MCP tools all call into it directly.

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
│   │   ├── model.py       # cross-encoder loading (singleton, thread-safe)
│   │   ├── scorer.py       # sentence segmentation + cross-encoder scoring
│   │   ├── filter.py       # filter_context() — the engine, used everywhere
│   │   ├── tokenizer.py     # tiktoken-based token counting
│   │   └── metrics.py       # FilterMetrics / LLMUsage / ExperimentResult / RuntimeStats
│   │
│   ├── mcp/
│   │   ├── server.py        # MCPServer instance, uma-mcp entry point
│   │   └── tools.py         # uma_filter / uma_score / uma_stats (plain functions)
│   │
│   ├── llm/
│   │   └── client.py        # OpenAI-compatible client, env-var configured
│   │
│   └── cli/
│       └── judge.py         # `uma judge` — the two-stage A/B demo
│
├── examples/
│   └── context.txt          # demo RAG context (Apple FY2024 + adjacent noise)
│
└── tests/
    ├── test_filter.py
    ├── test_metrics.py
    ├── test_llm.py
    ├── test_mcp.py
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
**Out:** per-sentence relevance scores (0-1, sigmoid-normalized cross-encoder
logits), unfiltered — useful for inspecting *why* a sentence was kept or
dropped.

### `uma_stats`

**In:** nothing
**Out:** cumulative filtering statistics (calls, tokens processed/removed,
average reduction, average latency) for this server process.

## Installation

```bash
git clone <this repo>
cd uma
python -m venv .venv
# Windows: .venv\Scripts\activate   |   macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
```

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
UMA_CONTEXT_PATH              optional — override the demo context file
```

Because the client is OpenAI-compatible, any provider that speaks that
protocol works: OpenAI itself, Groq, Together AI, Fireworks, or a local
runtime like Ollama / vLLM exposing an `/v1` route. If
`UMA_LLM_INPUT_PRICE_PER_1M` / `UMA_LLM_OUTPUT_PRICE_PER_1M` aren't set, Uma
shows raw token counts and labels cost `n/a` rather than guessing — it never
calls a computed number "credits" unless the provider actually uses a credit
system, and any cost it does show is explicitly labeled an **estimate**
(OpenAI-compatible chat completion responses don't return a cost field).

## Running the judge demo

```bash
uma judge
```

Runs the full two-stage controlled experiment described below and prints a
final comparison table. Optional flags:

```bash
uma judge --query "..." --threshold 0.6 --max-tokens 300
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

`uma judge` *is* the benchmark — every metric it prints comes from a real
measurement in that run (`time.perf_counter()` around the actual operation,
token counts from `tiktoken` / the provider's own usage payload). There is no
separate synthetic benchmark suite, because the filtering behavior and the
LLM's answer both depend on the specific query and context, and Uma makes no
claim that generalizes beyond "here's what happened on this run with this
context." Run it against your own retrieved context and query to get numbers
for your use case.

### Real runs

Context reduction is **deterministic** for a given (query, context,
threshold) — the same 902-token demo context against the same query at
threshold 0.5 reduces to 430 tokens (52.3% reduction, 16/29 sentences
removed) on every run, every time, regardless of which LLM is on the other
end. Everything downstream of that (LLM latency, cost) depends on the
provider you point Uma at, so here are two actual runs against two very
different backends.

**Hosted API — Gemini 3.1 Flash-Lite** (Gemini Developer API, standard
tier). Pricing is real, published pricing (see `.env.example`), not a
guess — but it's still labeled `est.` since the provider's response doesn't
return a cost field itself:

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

**Local model — Ollama, phi3:mini** (CPU-bound, no pricing available so
cost is not shown):

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

Both backends answered correctly and consistently across both arms
(Apple's FY2024 net sales of $391.035B, broken down by
iPhone/Services/Wearables/Mac/iPad).

**Be skeptical of the latency numbers specifically.** Re-running the Gemini
case three times back-to-back produced total-latency deltas of -73%, +19%,
and +129% — hosted-API network/queueing variance on calls this short swamps
any signal from the token-count difference. The only latency number that
was consistent across every run was **Uma's own filtering pass, ~740-870ms
every time** — because that's local, deterministic compute, not a network
call. Token reduction (and therefore cost, on any per-token-billed
provider) is the reliable claim here; LLM latency change is not, and this
project won't pretend otherwise. Run `uma judge` yourself, multiple times,
against your target provider before citing a latency number.

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
- **This is a demo/reference implementation**, not a production RAG gateway:
  no caching, no batching across concurrent requests, no persistence of
  `uma_stats` across restarts.

Uma does **not** claim 99% cost reduction, guaranteed accuracy, sub-millisecond
latency, or lossless compression. What it does is measurably remove
sentences a cross-encoder scores below a relevance threshold, and report
exactly what that did to token counts, cost (if measurable), and latency —
nothing more.

## Development

```bash
pytest            # 42 tests: sentence splitting, real cross-encoder scoring,
                   # filtering, metrics, LLM client (mocked transport), the
                   # real MCP server over an in-process session, and an
                   # explicit baseline-vs-Uma fairness check.
```
