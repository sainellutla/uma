"""Uma's MCP server.

Exposes exactly three tools — ``uma_filter``, ``uma_score``, ``uma_stats`` —
over the Model Context Protocol, using the official MCP Python SDK
(``mcp.server.mcpserver.MCPServer``) and stdio transport for local clients
such as Claude Desktop / Claude Code.

Every tool call here goes straight to :mod:`uma.mcp.tools`, which in turn
calls :mod:`uma.core.filter` — the identical engine the ``uma judge`` CLI
uses. There is no separate/duplicate filtering implementation for MCP.

Run directly:

    uma-mcp

or:

    python -m uma.mcp.server
"""

from __future__ import annotations

from mcp.server.mcpserver import MCPServer

from uma.mcp import tools as _tools

mcp = MCPServer(
    name="uma",
    title="Uma — AI Context Optimization Layer",
    instructions=(
        "Uma filters retrieved RAG context down to the sentences relevant "
        "to a query, using a local cross-encoder (no LLM). Call uma_filter "
        "to get filtered context ready to hand to an LLM, uma_score to "
        "inspect per-sentence relevance, and uma_stats for cumulative "
        "filtering statistics for this server process."
    ),
    version="0.1.0",
)


@mcp.tool()
def uma_filter(
    query: str,
    context: str,
    threshold: float = 0.5,
    max_tokens: int | None = None,
) -> dict:
    """Filter retrieved context down to sentences relevant to a query.

    Runs sentence segmentation, then a local cross-encoder
    (cross-encoder/ms-marco-MiniLM-L-6-v2 by default) to score every
    sentence's relevance to the query, keeps sentences scoring at or above
    `threshold` (0-1, default 0.5), optionally trims to `max_tokens`, and
    returns the filtered context along with real measured metrics.
    """
    return _tools.uma_filter(query=query, context=context, threshold=threshold, max_tokens=max_tokens)


@mcp.tool()
def uma_score(query: str, context: str) -> dict:
    """Return per-sentence cross-encoder relevance scores for a context.

    Segments `context` into sentences and scores each one's relevance to
    `query` with the local cross-encoder, without applying any threshold.
    Useful for inspecting why Uma would keep or drop a given sentence.
    """
    return _tools.uma_score(query=query, context=context)


@mcp.tool()
def uma_stats() -> dict:
    """Return cumulative Uma filtering statistics for this server process."""
    return _tools.uma_stats()


def main() -> None:
    """Entry point for the `uma-mcp` console script. Runs over stdio."""
    import sys

    # Warm up the local cross-encoder before accepting connections, so the
    # first uma_filter/uma_score call a client makes reports real inference
    # latency rather than one-time model-loading latency. This is real work
    # (loading real model weights), just moved earlier — never skip it.
    from uma.core.model import get_cross_encoder

    print("[uma-mcp] loading local cross-encoder...", file=sys.stderr)
    get_cross_encoder()
    print("[uma-mcp] ready. serving MCP over stdio.", file=sys.stderr)

    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
