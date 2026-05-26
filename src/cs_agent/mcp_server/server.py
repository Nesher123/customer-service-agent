"""FastMCP streamable-HTTP server exposing the Bitext data-analyst tools.

Why this exists
---------------
Task 3 of the assignment asks for at least three of the agent's tools to
be reachable over the Model Context Protocol. We expose six (every
structured tool — ``summarize`` is intentionally absent because it's
LLM-backed and would couple any MCP client to a Nebius API key).

Single source of truth
----------------------
The MCP wrappers below do NOT re-implement any tool logic. Each one
delegates to ``cs_agent.tools.registry.TOOLS_BY_NAME[<name>].invoke(...)``,
the same registry the in-process LangGraph agent uses. So a behaviour
change to a tool (e.g. a new filter) automatically propagates to both
transports without any manual sync step.

Schema design choice
--------------------
We use **flat function parameters** rather than a single Pydantic-args
wrapper, because flat params produce idiomatic MCP schemas
(``call_tool("count_rows", {"category": "REFUND"})``) instead of nested
ones (``{"args": {"category": "REFUND"}}``). The LLM-artifact cleaning
that lives in ``cs_agent.tools.schemas.LLMToolBase`` is unnecessary here
— MCP clients are programmatic and send well-typed JSON.

Run
---
::

    uv run cs-agent-mcp                       # 127.0.0.1:8765/mcp
    uv run cs-agent-mcp --port 9000           # custom port
    uv run cs-agent-mcp --host 0.0.0.0        # bind all interfaces (use with care)

Or, in tests / verifiers, attach an in-memory ``Client(mcp)`` directly to
this module's ``mcp`` instance — no port, no HTTP, same JSON-RPC contract.
"""

from __future__ import annotations

import argparse
import sys
from typing import Annotated, Literal

from fastmcp import FastMCP
from pydantic import Field

from cs_agent.tools.registry import TOOLS_BY_NAME

mcp = FastMCP(
    name="cs-agent-tools",
    instructions=(
        "Read-only data tools over the Bitext customer-support training "
        "dataset. Useful for listing categories/intents, sampling rows, "
        "and computing simple distributions or counts. All filters are "
        "case-insensitive. The dataset has ~27K rows across 11 categories."
    ),
)


@mcp.tool
def list_categories() -> list[str]:
    """List the sorted set of distinct categories present in the dataset.

    Use this first when you don't know the exact spelling of a category
    label, e.g. before calling ``count_rows`` with a ``category`` filter.

    Returns:
        Sorted list of category strings, e.g. ``["ACCOUNT", "CANCEL", "REFUND", ...]``.
    """
    return TOOLS_BY_NAME["list_categories"].invoke({})


@mcp.tool
def list_intents(
    category: Annotated[
        str | None,
        Field(description="Optional category to scope the listing (e.g. 'REFUND'). Case-insensitive."),
    ] = None,
) -> list[str]:
    """List the sorted set of distinct intents in the dataset.

    Pass ``category`` to scope to one category — typically the second
    call after ``list_categories`` when answering "how many <X> requests?"
    questions.

    Returns:
        Sorted list of intent strings; empty list if the category does not exist.
    """
    return TOOLS_BY_NAME["list_intents"].invoke({"category": category})


@mcp.tool
def get_distribution(
    group_by: Annotated[
        Literal["category", "intent"],
        Field(description="Which column to group by: 'category' (high-level) or 'intent' (specific)."),
    ],
    scope_category: Annotated[
        str | None,
        Field(
            description=(
                "Optional. Restrict the distribution to one category. Useful with "
                "group_by='intent' to get the intent distribution within a single category."
            ),
        ),
    ] = None,
) -> dict[str, int]:
    """Return row counts grouped by category or intent.

    Examples:
        ``group_by='category'`` → ``{"ACCOUNT": 1957, "REFUND": 2992, ...}``.
        ``group_by='intent', scope_category='ACCOUNT'`` → ``{"create_account": 547, ...}``.
    """
    return TOOLS_BY_NAME["get_distribution"].invoke({"group_by": group_by, "scope_category": scope_category})


@mcp.tool
def count_rows(
    category: Annotated[
        str | None,
        Field(description="Optional category filter (e.g. 'REFUND'). Case-insensitive."),
    ] = None,
    intent: Annotated[
        str | None,
        Field(description="Optional intent filter (e.g. 'track_refund'). Case-insensitive."),
    ] = None,
    keyword: Annotated[
        str | None,
        Field(
            description="Optional case-insensitive substring to match against the user 'instruction' column."
        ),
    ] = None,
) -> int:
    """Count rows matching the optional filters.

    All three filters are AND-combined; passing none returns the full
    dataset row count.
    """
    return TOOLS_BY_NAME["count_rows"].invoke({"category": category, "intent": intent, "keyword": keyword})


@mcp.tool
def get_examples(
    category: Annotated[
        str | None,
        Field(description="Optional category filter (e.g. 'REFUND'). Case-insensitive."),
    ] = None,
    intent: Annotated[
        str | None,
        Field(description="Optional intent filter (e.g. 'track_refund'). Case-insensitive."),
    ] = None,
    keyword: Annotated[
        str | None,
        Field(
            description="Optional case-insensitive substring to match against the user 'instruction' column."
        ),
    ] = None,
    n: Annotated[
        int,
        Field(ge=1, le=50, description="Number of examples to return (1-50). Default 5."),
    ] = 5,
    columns: Annotated[
        list[str] | None,
        Field(
            description=(
                "Optional subset of columns to return. Allowed: 'instruction', 'response', "
                "'category', 'intent', 'flags'. Default: ['category', 'intent', 'instruction']."
            ),
        ),
    ] = None,
) -> list[dict]:
    """Return up to ``n`` example rows matching the optional filters."""
    return TOOLS_BY_NAME["get_examples"].invoke(
        {
            "category": category,
            "intent": intent,
            "keyword": keyword,
            "n": n,
            "columns": columns,
        }
    )


@mcp.tool
def search_by_keyword(
    keyword: Annotated[
        str,
        Field(
            min_length=1,
            description="Substring to match (case-insensitive) against the user 'instruction' column.",
        ),
    ],
    n: Annotated[
        int,
        Field(ge=1, le=50, description="Number of matches to return (1-50). Default 10."),
    ] = 10,
) -> list[dict]:
    """Return rows whose user-instruction text contains ``keyword`` (case-insensitive).

    Use for paraphrased / fuzzy queries like "people wanting their money back"
    → ``keyword="money back"``.
    """
    return TOOLS_BY_NAME["search_by_keyword"].invoke({"keyword": keyword, "n": n})


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: start the FastMCP server on streamable-HTTP.

    Wired up via ``[project.scripts].cs-agent-mcp`` so the install also
    yields a top-level ``cs-agent-mcp`` command.
    """
    p = argparse.ArgumentParser(
        prog="cs-agent-mcp",
        description=(
            "FastMCP server exposing six Bitext data-analyst tools over streamable-HTTP. "
            "Connects with any MCP client (FastMCP Client, Cursor, Claude Desktop, etc.)."
        ),
    )
    p.add_argument("--host", default="127.0.0.1", help="Bind address. Default: 127.0.0.1 (local-only).")
    p.add_argument("--port", type=int, default=8765, help="Bind port. Default: 8765.")
    p.add_argument("--path", default="/mcp", help="HTTP path prefix. Default: /mcp.")
    args = p.parse_args(argv)

    # FastMCP's "http" transport is the streamable-HTTP one (formerly
    # "streamable HTTP"). Other accepted aliases include "streamable-http".
    mcp.run(transport="http", host=args.host, port=args.port, path=args.path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
