---
name: task-3-mcp
overview: Focused sub-plan for Task 3 (20 pts) — wrap >=3 of the existing data-analyst tools in a FastMCP streamable-HTTP server that reuses the same Python implementations the agent uses. Adds a working README client snippet and an in-process verifier. No breaking changes to the agent, CLI, or memory subsystems.
todos:
  - id: t3-server-module
    content: Step 1 — mcp_server/server.py with FastMCP("cs-agent-tools") + thin @mcp.tool wrappers around >=3 existing data tools (list_categories, list_intents, get_distribution, count_rows, get_examples, search_by_keyword)
    status: completed
  - id: t3-entrypoint
    content: Step 2 — main() entrypoint with --transport / --host / --port flags + cs-agent-mcp console script in pyproject
    status: completed
  - id: t3-test-unit
    content: Step 3a — tests/test_mcp_server.py — fast unit tests asserting tool registration count, names, and schema fidelity
    status: completed
  - id: t3-test-int
    content: Step 3b — tests/test_mcp_integration.py — in-memory FastMCP Client roundtrip exercising 3 tools end-to-end
    status: completed
  - id: t3-verify
    content: Step 4 — scripts/verify_task3.py — spins the server in-process via FastMCP's in-memory transport, calls 3 tools, asserts plausible responses, prints a per-tool PASS/FAIL table
    status: completed
  - id: t3-readme
    content: Step 5 — README "MCP server" section with run command (streamable-http) + 15-line FastMCP Client snippet calling list_categories
    status: completed
isProject: false
---

# Task 3 — MCP Server (sub-plan)

## Goal

Expose at least three of the existing Bitext data-analyst tools over the
[Model Context Protocol](https://modelcontextprotocol.io/) so that an
external MCP client (Claude Desktop, Cursor, a Python script, another
LangGraph agent, etc.) can invoke them over HTTP. The MCP server reuses
the *same* Python implementations the in-process agent uses — MCP is
purely a transport.

## Grading map (20 pts)

- Server with ≥3 tools (10) — `src/cs_agent/mcp_server/server.py`
- Streamable-HTTP transport (5) — `main()` entrypoint + console script
- README client example (5) — runnable Python snippet using FastMCP `Client`

## Locked decisions

- **Library:** [FastMCP 2.x](https://gofastmcp.com/) (already in
  `pyproject.toml` `optional-dependencies.mcp`). FastMCP gives us
  Python-native `@mcp.tool` decoration, automatic JSON-Schema generation
  from Pydantic / type hints, streamable-HTTP transport, and an
  in-memory `Client` that we use for tests + the verifier (no
  subprocess + no network round-trip). Alternatives considered:
  the official `mcp` SDK (more boilerplate, no in-memory transport).

- **Tools to expose:** all six structured tools — `list_categories`,
  `list_intents`, `get_distribution`, `count_rows`, `get_examples`,
  `search_by_keyword`. We deliberately exclude `summarize` because
  it's LLM-backed and would require the MCP client to carry a Nebius
  key just to call it (and would couple the MCP server to Nebius
  uptime). Six is well above the >=3 floor and exercises every
  Pydantic-schema shape we have (no-args, optional-args, multi-arg,
  enum, list-of-strings).

- **Implementation reuse:** thin `@mcp.tool` wrappers that delegate to
  `cs_agent.tools.registry.TOOLS_BY_NAME[<name>].invoke(args)`. This
  preserves the Single-Source-of-Truth principle from the master plan:
  if we change a tool's behaviour, both the agent and the MCP server
  pick it up automatically. We do NOT re-implement any tool body in
  the MCP module.

- **Transport:** **streamable-http** (the modern MCP transport, formerly
  "streamable HTTP"). FastMCP serves it on a single endpoint with
  bidirectional streaming, which is what Cursor / Claude Desktop /
  most external clients expect today. We do NOT support stdio
  transport — it would be an extra surface area for negligible gain.

- **Default address:** `http://127.0.0.1:8765/mcp` — local-only by
  default. The `--host`/`--port` flags let a grader change either.

- **Console script:** `cs-agent-mcp` registered via
  `[project.scripts]` so the command is `uv run cs-agent-mcp` —
  symmetric with `uv run cs-agent`.

- **Auth:** none. The server is local-only by default. Documented as
  a known limitation in the README; FastMCP supports auth backends
  if we ever need it.

## Architecture delta

```mermaid
flowchart LR
    subgraph existing [Existing in-process agent (unchanged)]
      cli["cli.py"] --> graph["LangGraph"]
      graph --> tools_in["TOOLS_BY_NAME\n(BaseTool registry)"]
    end

    subgraph new [NEW Task 3]
      mcp_server["mcp_server/server.py\n(FastMCP, streamable-http)"]
      ext_client["External MCP client\n(FastMCP Client / Cursor / Claude / ...)"]
    end

    tools_in --- mcp_server
    ext_client -- "tools/list, tools/call" --> mcp_server
```

The arrow `tools_in --- mcp_server` is shared Python imports — the
MCP wrappers literally `from cs_agent.tools.registry import TOOLS_BY_NAME`.
No subprocess, no IPC, no duplicated implementations.

## Files this task creates / modifies

```
src/cs_agent/mcp_server/
├── __init__.py                      # NEW — namespace marker
└── server.py                        # NEW (Step 1+2) — FastMCP instance, wrappers, main()

pyproject.toml                       # MODIFIED — adds [project.scripts].cs-agent-mcp

tests/
├── test_mcp_server.py               # NEW (Step 3a) — fast unit tests, no network
└── test_mcp_integration.py          # NEW (Step 3b) — in-memory Client round-trip,
                                      # marked 'integration' (no Nebius, but exercises
                                      # the live DataFrame so we keep the marker)

scripts/verify_task3.py              # NEW (Step 4) — multi-tool MCP smoke verifier

README.md                            # MODIFIED — new "MCP server" section + client snippet

# UNCHANGED on purpose:
#   - src/cs_agent/agent/**       (Task 1+2 core)
#   - src/cs_agent/memory/**      (Task 2)
#   - src/cs_agent/cli.py         (Task 1+2)
#   - src/cs_agent/tools/**       (impls reused as-is via TOOLS_BY_NAME)
#   - tests/test_tools*.py, test_router.py, test_agent_integration.py
#   - tests/test_checkpoint.py, test_episodic_integration.py
#   - tests/test_profile*.py
#   - scripts/verify_task1.py, scripts/verify_task2.py
```

No new dependencies — `fastmcp>=2.0` is already in
`pyproject.toml`'s `optional-dependencies.mcp` extra.

## Step-by-step (estimated 2–3 hrs total)

### Step 1 — `mcp_server/server.py` (~45 min)

Skeleton:

```python
# src/cs_agent/mcp_server/server.py
"""FastMCP streamable-HTTP server exposing the Bitext data-analyst tools.

Re-uses the same tool implementations the in-process LangGraph agent uses.
The MCP wrappers are intentionally thin: each one is a Pydantic-typed
function whose body delegates to TOOLS_BY_NAME[<name>].invoke(args).
"""
from __future__ import annotations

from fastmcp import FastMCP

from cs_agent.tools.registry import TOOLS_BY_NAME
from cs_agent.tools.schemas import (
    ListIntentsArgs, DistributionArgs, CountRowsArgs,
    GetExamplesArgs, SearchByKeywordArgs,
)

mcp = FastMCP(
    name="cs-agent-tools",
    instructions=(
        "Read-only data tools over the Bitext customer-support training "
        "dataset. Useful for listing categories/intents, sampling rows, "
        "and computing simple distributions or counts."
    ),
)


@mcp.tool
def list_categories() -> list[str]:
    """Return the sorted list of distinct categories in the dataset."""
    return TOOLS_BY_NAME["list_categories"].invoke({})


@mcp.tool
def list_intents(category: str | None = None) -> list[str]:
    """List all intents, optionally scoped to one category."""
    return TOOLS_BY_NAME["list_intents"].invoke({"category": category})


@mcp.tool
def get_distribution(args: DistributionArgs) -> dict[str, int]:
    """Row counts grouped by category or intent."""
    return TOOLS_BY_NAME["get_distribution"].invoke(args.model_dump())


@mcp.tool
def count_rows(args: CountRowsArgs) -> int:
    """Count rows matching optional category/intent/keyword filters."""
    return TOOLS_BY_NAME["count_rows"].invoke(args.model_dump())


@mcp.tool
def get_examples(args: GetExamplesArgs) -> list[dict]:
    """Return up to N example rows matching optional filters."""
    return TOOLS_BY_NAME["get_examples"].invoke(args.model_dump())


@mcp.tool
def search_by_keyword(args: SearchByKeywordArgs) -> list[dict]:
    """Substring search over the user-instruction column."""
    return TOOLS_BY_NAME["search_by_keyword"].invoke(args.model_dump())
```

Notes:
- We intentionally use the existing Pydantic args schemas
  (`DistributionArgs`, etc.) so the MCP-tool JSON-schema is identical
  to the LangChain-tool JSON-schema. Two transports, one schema.
- For the two zero/one-arg cases (`list_categories`, `list_intents`)
  we drop the wrapper class and use plain typed parameters — FastMCP
  picks them up cleanly.
- `summarize` is intentionally absent; documented in the README.

### Step 2 — `main()` entrypoint + console script (~20 min)

Append to `server.py`:

```python
def main(argv: list[str] | None = None) -> int:
    """Start the FastMCP server on streamable-HTTP."""
    import argparse
    p = argparse.ArgumentParser(prog="cs-agent-mcp")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--path", default="/mcp")
    args = p.parse_args(argv)
    mcp.run(transport="http", host=args.host, port=args.port, path=args.path)
    return 0


if __name__ == "__main__":
    main()
```

`pyproject.toml` change:

```toml
[project.scripts]
cs-agent     = "cs_agent.cli:main"
cs-agent-mcp = "cs_agent.mcp_server.server:main"   # NEW
```

After `uv sync`, the grader can `uv run cs-agent-mcp --port 8765`.

### Step 3 — Tests (~45 min, two files)

**Step 3a — `tests/test_mcp_server.py`** (fast, no network)

Verifies registration shape without going over the wire:

```python
# tests/test_mcp_server.py — fast, deterministic, no LLM, no HTTP
import pytest
from cs_agent.mcp_server.server import mcp


@pytest.mark.asyncio  # FastMCP introspection APIs are async
async def test_at_least_three_tools_registered():
    tools = await mcp.list_tools()
    assert len(tools) >= 3, f"got {len(tools)} tools registered: {[t.name for t in tools]}"


@pytest.mark.asyncio
async def test_expected_tool_names_present():
    tools = {t.name for t in await mcp.list_tools()}
    expected = {"list_categories", "list_intents", "get_distribution",
                "count_rows", "get_examples", "search_by_keyword"}
    missing = expected - tools
    assert not missing, f"missing MCP tools: {missing}"


@pytest.mark.asyncio
async def test_summarize_is_NOT_exposed_over_mcp():
    """summarize is LLM-backed and intentionally not exposed."""
    tools = {t.name for t in await mcp.list_tools()}
    assert "summarize" not in tools


@pytest.mark.asyncio
async def test_tool_descriptions_are_not_empty():
    for t in await mcp.list_tools():
        assert t.description and t.description.strip(), (
            f"tool {t.name!r} has an empty description"
        )
```

(`pytest-asyncio` will be added to the dev dependency group if it isn't
already there — small `[dependency-groups].dev` edit, not a runtime dep.)

**Step 3b — `tests/test_mcp_integration.py`** (in-memory `Client`,
marked `integration`)

Exercises the wire format end-to-end without spawning a subprocess.
FastMCP's `Client(server)` connects directly to the in-process server
object — no HTTP, but the same JSON-RPC contract.

```python
# tests/test_mcp_integration.py
import pytest
from fastmcp import Client
from cs_agent.mcp_server.server import mcp

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


async def test_list_categories_returns_real_categories():
    async with Client(mcp) as client:
        result = await client.call_tool("list_categories", {})
        cats = result.data
        assert isinstance(cats, list)
        assert "REFUND" in cats
        assert "ACCOUNT" in cats


async def test_get_examples_returns_filtered_rows():
    async with Client(mcp) as client:
        result = await client.call_tool(
            "get_examples", {"args": {"category": "REFUND", "n": 3}}
        )
        rows = result.data
        assert isinstance(rows, list) and len(rows) == 3
        assert all(r["category"] == "REFUND" for r in rows)


async def test_count_rows_matches_dataframe_total():
    """Sanity check: count_rows() with no filters equals len(df)."""
    from cs_agent.data.loader import get_df
    async with Client(mcp) as client:
        result = await client.call_tool("count_rows", {"args": {}})
        assert result.data == len(get_df())
```

### Step 4 — `scripts/verify_task3.py` (~25 min)

Mirrors the structure of `verify_task2.py`. Per-tool case (name, args,
predicate). Uses the in-memory `Client` so no port juggling.

```python
# scripts/verify_task3.py
"""End-to-end MCP smoke verifier (3 tools, in-memory transport)."""
from __future__ import annotations
import asyncio
import sys
from dataclasses import dataclass
from typing import Any, Callable

from fastmcp import Client
from cs_agent.mcp_server.server import mcp


@dataclass
class Case:
    label: str
    tool: str
    args: dict
    predicate: Callable[[Any], bool]
    notes: str = ""


CASES: list[Case] = [
    Case("list-categories", "list_categories", {},
         lambda data: "REFUND" in data,
         "Must include REFUND."),
    Case("count-refund", "count_rows", {"args": {"category": "REFUND"}},
         lambda data: data == 2992,
         "Live dataset has 2992 REFUND rows."),
    Case("examples-shipping", "get_examples",
         {"args": {"category": "SHIPPING", "n": 2}},
         lambda data: isinstance(data, list) and len(data) == 2
                      and all(r["category"] == "SHIPPING" for r in data),
         "Schema + filter sanity."),
]


async def main() -> int:
    print("Building MCP server (in-memory transport)…\n")
    n_pass = 0
    async with Client(mcp) as client:
        for i, c in enumerate(CASES, 1):
            print(f"  [{i}/{len(CASES)}] {c.label:<22} ", end="", flush=True)
            try:
                res = await client.call_tool(c.tool, c.args)
                ok = bool(c.predicate(res.data))
            except Exception as exc:  # noqa: BLE001
                ok, res = False, repr(exc)
            mark = "✓" if ok else "✗"
            n_pass += int(ok)
            print(mark)
    print(f"\nResult: {n_pass}/{len(CASES)} passed")
    return 0 if n_pass == len(CASES) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
```

### Step 5 — README "MCP server" section (~25 min)

Insert between "Walkthrough examples" and "Architecture overview":

```markdown
## MCP server (Task 3)

Six of the seven Bitext data tools are exposed as a FastMCP server over
streamable HTTP. `summarize` is intentionally excluded because it requires
a Nebius API key to run.

### Run the server

```bash
uv run cs-agent-mcp                       # binds 127.0.0.1:8765/mcp
uv run cs-agent-mcp --host 0.0.0.0 --port 9000
```

### Connect from Python

```python
import asyncio
from fastmcp import Client

async def main():
    async with Client("http://127.0.0.1:8765/mcp") as client:
        cats = (await client.call_tool("list_categories", {})).data
        n_refunds = (await client.call_tool(
            "count_rows", {"args": {"category": "REFUND"}}
        )).data
        print(f"{len(cats)} categories; {n_refunds} REFUND rows")

asyncio.run(main())
```

### Connect from Cursor / Claude Desktop

Add to your client's MCP server config:

```json
{
  "cs-agent-tools": {
    "url": "http://127.0.0.1:8765/mcp",
    "transport": "http"
  }
}
```
```

## Acceptance criteria

- `uv run cs-agent-mcp` starts a streamable-HTTP server on `127.0.0.1:8765`
  and prints a "running" banner.
- A FastMCP `Client` connecting to that URL can list ≥3 tools and call
  `list_categories`, `count_rows`, and `get_examples` successfully.
- `uv run python scripts/verify_task3.py` exits 0 with `Result: 3/3 passed`.
- `uv run python -m pytest tests/test_mcp_server.py -v` passes (fast).
- `uv run python -m pytest tests/test_mcp_integration.py -m integration -v`
  passes (in-memory; ~5s; no Nebius needed).
- `uv run python -m pytest -m "not integration"` still green for ALL pre-existing
  fast tests; no Task-1 or Task-2 regressions.
- `ruff format/check` clean. `fon check` clean (any new same-name `main`
  duplicate gets a documented exception in `.fon/check/config.yaml`).
- README section walks a grader from cold-clone to a successful client
  call in <2 minutes.

## Out of scope

- Authentication / API keys on the MCP server (local-only by default;
  documented).
- stdio transport (streamable-http is sufficient and is what modern MCP
  clients use).
- Streaming partial tool results (FastMCP supports it; not needed for
  these read-only tools).
- Exposing `summarize` (would couple MCP clients to a Nebius key).
- Wrapping the entire LangGraph agent as one giant MCP "agent" tool —
  the assignment asks for individual *data* tools.
- Containerising the MCP server (out of scope for the assignment).
- LangGraph-on-top-of-MCP via `langchain-mcp-adapters.MultiServerMCPClient`
  — interesting follow-up but not required by the brief.
