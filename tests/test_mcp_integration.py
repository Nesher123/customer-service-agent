"""End-to-end MCP round-trip tests using FastMCP's in-memory transport.

A FastMCP ``Client`` attached directly to the ``mcp`` instance speaks the
same JSON-RPC 2.0 contract as a remote streamable-HTTP client, but skips
the network — so these tests are fast, deterministic, and don't need
a free port.

We hit three real tools end-to-end (``list_categories``, ``count_rows``,
``get_examples``) plus one error path. They touch the live Pandas
DataFrame, so we mark them ``integration`` to keep the fast suite
(``-m "not integration"``) zero-disk.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from fastmcp import Client

from cs_agent.data.loader import get_df
from cs_agent.mcp_server.server import mcp

pytestmark = pytest.mark.integration


def _call(tool: str, args: dict[str, Any]) -> Any:
    """Run a tool through an in-memory ``Client`` and return ``.data``."""

    async def _inner() -> Any:
        async with Client(mcp) as client:
            result = await client.call_tool(tool, args)
            return result.data

    return asyncio.run(_inner())


def test_list_categories_returns_real_categories() -> None:
    """The dataset's 11 known categories must come back through MCP unchanged."""
    cats = _call("list_categories", {})
    assert isinstance(cats, list)
    expected = {
        "ACCOUNT",
        "CANCEL",
        "CONTACT",
        "DELIVERY",
        "FEEDBACK",
        "INVOICE",
        "ORDER",
        "PAYMENT",
        "REFUND",
        "SHIPPING",
        "SUBSCRIPTION",
    }
    assert set(cats) == expected, f"category drift: got {sorted(cats)}"


def test_count_rows_total_matches_dataframe() -> None:
    """No-filter count_rows should equal len(df) — proves the wrapper isn't
    silently shadowing the registry impl."""
    total = _call("count_rows", {})
    assert total == len(get_df())


def test_count_rows_with_category_filter() -> None:
    """count_rows({category='REFUND'}) → 2992 in the live snapshot."""
    n = _call("count_rows", {"category": "REFUND"})
    assert n == 2992


def test_get_examples_respects_n_and_filter() -> None:
    """get_examples returns exactly N rows that all match the category filter."""
    rows = _call("get_examples", {"category": "REFUND", "n": 3})
    assert isinstance(rows, list) and len(rows) == 3
    for r in rows:
        assert r["category"] == "REFUND", r


def test_search_by_keyword_substring_match() -> None:
    """search_by_keyword('refund') returns at least one row containing 'refund'."""
    rows = _call("search_by_keyword", {"keyword": "refund", "n": 5})
    assert isinstance(rows, list) and rows, "expected at least one match"
    assert any("refund" in r.get("instruction", "").lower() for r in rows)


def test_invalid_args_surface_as_error() -> None:
    """search_by_keyword's ``keyword`` is required (min_length=1).

    FastMCP/Pydantic validation should refuse the call rather than crash
    the server — an MCP client sees a structured error result. We pass
    ``raise_on_error=False`` to inspect the error result directly instead
    of catching the default ``ToolError`` exception.
    """

    async def _inner() -> Any:
        async with Client(mcp) as client:
            return await client.call_tool("search_by_keyword", {"keyword": ""}, raise_on_error=False)

    result = asyncio.run(_inner())
    assert result.is_error, "empty keyword must fail validation, not return data"
