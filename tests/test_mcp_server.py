"""Fast unit tests for the FastMCP server module (Task 3).

Asserts the registration shape of ``cs_agent.mcp_server.server.mcp`` without
opening a socket or calling any tool body. The integration test in
``tests/test_mcp_integration.py`` covers end-to-end round-trips.

We wrap ``mcp.list_tools()`` (which is async) with ``asyncio.run`` so this
file stays sync and we avoid pulling in ``pytest-asyncio`` just for setup.
"""

from __future__ import annotations

import asyncio

from fastmcp.tools import FunctionTool

from cs_agent.mcp_server.server import mcp

EXPECTED_TOOLS = {
    "list_categories",
    "list_intents",
    "get_distribution",
    "count_rows",
    "get_examples",
    "search_by_keyword",
}


def _list_tools() -> list[FunctionTool]:
    """Synchronous wrapper around ``mcp.list_tools()``."""
    return asyncio.run(mcp.list_tools())


def test_at_least_three_tools_registered() -> None:
    """Task 3 requires >=3 tools; we expose all six structured ones."""
    tools = _list_tools()
    assert len(tools) >= 3, f"got {len(tools)} tools: {[t.name for t in tools]}"


def test_expected_tool_names_present() -> None:
    """All six structured Bitext tools must be wrapped."""
    names = {t.name for t in _list_tools()}
    missing = EXPECTED_TOOLS - names
    assert not missing, f"missing MCP tools: {missing}"


def test_summarize_is_not_exposed_over_mcp() -> None:
    """``summarize`` is intentionally absent — it's LLM-backed (needs Nebius)."""
    names = {t.name for t in _list_tools()}
    assert "summarize" not in names, "summarize should not be exposed over MCP; it requires a Nebius API key"


def test_tool_descriptions_are_non_empty() -> None:
    """Every exposed tool needs a description so MCP clients can render it."""
    for t in _list_tools():
        assert t.description and t.description.strip(), f"tool {t.name!r} has an empty description"


def test_count_rows_schema_has_optional_filter_fields() -> None:
    """Schema fidelity: ``count_rows`` advertises ``category``/``intent``/``keyword``."""
    by_name = {t.name: t for t in _list_tools()}
    schema = by_name["count_rows"].parameters
    assert schema["type"] == "object"
    props = schema["properties"]
    for field in ("category", "intent", "keyword"):
        assert field in props, f"count_rows schema missing '{field}': {list(props)}"
    assert schema.get("required", []) == [], (
        f"count_rows must have no required fields, got {schema.get('required')}"
    )


def test_search_by_keyword_schema_marks_keyword_required() -> None:
    """``search_by_keyword`` exposes ``keyword`` as a required string."""
    by_name = {t.name: t for t in _list_tools()}
    schema = by_name["search_by_keyword"].parameters
    assert "keyword" in schema["required"], schema
    assert schema["properties"]["keyword"]["type"] == "string"


def test_get_distribution_schema_has_enum_group_by() -> None:
    """``group_by`` is a Literal['category','intent'] → JSON enum."""
    by_name = {t.name: t for t in _list_tools()}
    schema = by_name["get_distribution"].parameters
    group_by = schema["properties"]["group_by"]
    assert set(group_by["enum"]) == {"category", "intent"}, group_by
    assert "group_by" in schema["required"]


def test_server_metadata() -> None:
    """Server identity is set so MCP clients know what they connected to."""
    assert mcp.name == "cs-agent-tools"
    assert mcp.instructions and "Bitext" in mcp.instructions
