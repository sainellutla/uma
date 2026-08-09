"""Tests for the MCP tool implementations and the MCP server itself.

The tool-function tests call :mod:`uma.mcp.tools` directly. The server
tests drive the actual :class:`mcp.server.mcpserver.MCPServer` instance
in-process (``list_tools`` / ``call_tool``) to prove the three tools are
really registered and really invokable through MCP, not just present as
plain functions.
"""

from __future__ import annotations

import json

import pytest

from uma.core.filter import RUNTIME_STATS
from uma.mcp import tools as uma_tools
from uma.mcp.server import mcp

QUERY = "What is the capital of France?"
CONTEXT = "Paris is the capital of France. Bananas are a good source of potassium."


# --------------------------------------------------------------------------
# Tool functions (same engine as the CLI — no separate implementation)
# --------------------------------------------------------------------------


def test_uma_filter_tool_shape():
    result = uma_tools.uma_filter(query=QUERY, context=CONTEXT, threshold=0.5)
    expected_keys = {
        "filtered_context",
        "original_tokens",
        "filtered_tokens",
        "tokens_removed",
        "reduction_percent",
        "sentences_processed",
        "sentences_kept",
        "sentences_removed",
        "filtering_latency_ms",
    }
    assert expected_keys.issubset(result.keys())
    assert "Bananas" not in result["filtered_context"]
    assert "Paris" in result["filtered_context"]


def test_uma_filter_tool_uses_core_engine_directly():
    from uma.core.filter import filter_context

    direct = filter_context(QUERY, CONTEXT, threshold=0.5, record_stats=False)
    via_tool = uma_tools.uma_filter(query=QUERY, context=CONTEXT, threshold=0.5)
    assert via_tool["filtered_context"] == direct.filtered_context
    assert via_tool["original_tokens"] == direct.metrics.original_tokens
    assert via_tool["sentences_kept"] == direct.metrics.sentences_retained


def test_uma_score_tool_returns_per_sentence_scores():
    result = uma_tools.uma_score(query=QUERY, context=CONTEXT)
    assert result["query"] == QUERY
    assert result["sentences_scored"] == 2
    scores = {s["text"]: s["score"] for s in result["sentences"]}
    paris_text = next(t for t in scores if "Paris" in t)
    banana_text = next(t for t in scores if "Bananas" in t)
    assert scores[paris_text] > scores[banana_text]


def test_uma_stats_tool_reflects_recorded_calls():
    before = uma_tools.uma_stats()["total_calls"]
    from uma.core.filter import filter_context

    filter_context(QUERY, CONTEXT, threshold=0.5)  # record_stats=True by default
    after = uma_tools.uma_stats()["total_calls"]
    assert after == before + 1


# --------------------------------------------------------------------------
# The actual MCP server: exactly 3 tools, real invocation
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mcp_server_exposes_exactly_three_tools():
    registered = await mcp.list_tools()
    names = {t.name for t in registered}
    assert names == {"uma_filter", "uma_score", "uma_stats"}


@pytest.mark.asyncio
async def test_mcp_call_tool_uma_filter_end_to_end():
    result = await mcp.call_tool(
        "uma_filter",
        {"query": QUERY, "context": CONTEXT, "threshold": 0.5},
    )
    assert result.is_error is False
    payload = json.loads(result.content[0].text)
    assert "Bananas" not in payload["filtered_context"]
    assert payload["original_tokens"] > 0
    assert payload["filtering_latency_ms"] >= 0


@pytest.mark.asyncio
async def test_mcp_call_tool_uma_score_end_to_end():
    result = await mcp.call_tool("uma_score", {"query": QUERY, "context": CONTEXT})
    assert result.is_error is False
    payload = json.loads(result.content[0].text)
    assert payload["sentences_scored"] == 2


@pytest.mark.asyncio
async def test_mcp_call_tool_uma_stats_end_to_end():
    result = await mcp.call_tool("uma_stats", {})
    assert result.is_error is False
    payload = json.loads(result.content[0].text)
    assert "total_calls" in payload
    assert payload["total_calls"] >= RUNTIME_STATS.total_calls - 1
