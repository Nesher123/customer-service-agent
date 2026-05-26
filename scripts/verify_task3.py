"""Verify Task 3 (MCP server) end-to-end via FastMCP's in-memory transport.

We attach a ``Client`` directly to the in-process ``mcp`` instance — same
JSON-RPC 2.0 contract as a remote streamable-HTTP client, but no port,
no subprocess, no flaky CI ports. Each case asserts a structural
predicate on the tool's response (not just "did it not throw"), so
silent regressions in the underlying registry are caught.

Run with::

    uv run python scripts/verify_task3.py
"""

from __future__ import annotations

import asyncio
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from fastmcp import Client

from cs_agent.mcp_server.server import mcp


@dataclass
class Case:
    """One MCP tool invocation plus a predicate over its ``.data`` payload."""

    label: str
    tool: str
    args: dict[str, Any]
    predicate: Callable[[Any], bool]
    notes: str = ""


@dataclass
class CaseResult:
    """Outcome of running one ``Case``."""

    case: Case
    raw: Any = None
    passed: bool = False
    error: str | None = None
    elapsed_s: float = 0.0


CASES: list[Case] = [
    Case(
        label="list-categories",
        tool="list_categories",
        args={},
        predicate=lambda data: (
            isinstance(data, list) and "REFUND" in data and "ACCOUNT" in data and len(data) >= 5
        ),
        notes="11 known Bitext categories must round-trip through MCP unchanged.",
    ),
    Case(
        label="count-refund",
        tool="count_rows",
        args={"category": "REFUND"},
        predicate=lambda data: data == 2992,
        notes="Live snapshot of the dataset has exactly 2992 REFUND rows.",
    ),
    Case(
        label="examples-shipping",
        tool="get_examples",
        args={"category": "SHIPPING", "n": 2},
        predicate=lambda data: (
            isinstance(data, list)
            and len(data) == 2
            and all(isinstance(r, dict) and r.get("category") == "SHIPPING" for r in data)
        ),
        notes="Schema + filter sanity: returns exactly N rows, all matching the filter.",
    ),
    Case(
        label="distribution-by-category",
        tool="get_distribution",
        args={"group_by": "category"},
        predicate=lambda data: isinstance(data, dict) and "REFUND" in data and sum(data.values()) > 20_000,
        notes="Top-level distribution sums roughly to total row count.",
    ),
    Case(
        label="search-money-back",
        tool="search_by_keyword",
        args={"keyword": "money back", "n": 3},
        predicate=lambda data: (
            isinstance(data, list)
            and len(data) <= 3
            and all("money back" in r.get("instruction", "").lower() for r in data)
        ),
        notes="Case-insensitive substring search over user instructions.",
    ),
]


@dataclass
class RunReport:
    """Aggregated run state for the whole verifier."""

    results: list[CaseResult] = field(default_factory=list)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def total(self) -> int:
        return len(self.results)


async def _run_one(client: Client, case: Case) -> CaseResult:
    """Call one MCP tool and evaluate its predicate."""
    res = CaseResult(case=case)
    t0 = time.time()
    try:
        out = await client.call_tool(case.tool, case.args)
        res.raw = out.data
        res.passed = bool(case.predicate(out.data))
    except Exception as exc:  # noqa: BLE001 — verifier must keep going on individual case failures
        res.error = repr(exc)
        res.passed = False
    res.elapsed_s = time.time() - t0
    return res


async def _run_all() -> RunReport:
    """Open one in-memory ``Client`` and run every case."""
    report = RunReport()
    async with Client(mcp) as client:
        for i, case in enumerate(CASES, 1):
            print(f"  [{i}/{len(CASES)}] {case.label:<24} ", end="", flush=True)
            r = await _run_one(client, case)
            report.results.append(r)
            mark = "✓" if r.passed else "✗"
            suffix = "" if r.passed else f"  ({r.error or 'predicate failed'})"
            print(f"{mark}  ({r.elapsed_s:.2f}s){suffix}", flush=True)
    return report


def _truncate(s: str, n: int = 200) -> str:
    s = s.replace("\n", " ").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def print_summary(report: RunReport) -> None:
    """Print the per-case PASS/FAIL table."""
    print("\n" + "=" * 90)
    print(f"{'#':>2}  {'PASS':<5} {'LABEL':<26} {'TOOL':<22} {'TIME':>7}")
    print("=" * 90)
    for i, r in enumerate(report.results, 1):
        mark = "✓" if r.passed else "✗"
        print(f"{i:>2}  {mark:<5} {r.case.label:<26} {r.case.tool:<22} {r.elapsed_s:>6.2f}s")
    print("=" * 90)
    print(f"\nResult: {report.passed}/{report.total} passed, {report.total - report.passed} failed")


def print_details(report: RunReport) -> None:
    """Print per-case args, response preview, and notes (handy when a case fails)."""
    print("\n" + "—" * 90)
    print("DETAILS PER CASE")
    print("—" * 90)
    for i, r in enumerate(report.results, 1):
        print(f"\n[{i}] {r.case.label}  →  {r.case.tool}({r.case.args})")
        if r.case.notes:
            print(f"    notes:    {r.case.notes}")
        if r.error:
            print(f"    ERROR:    {r.error}")
        else:
            print(f"    response: {_truncate(repr(r.raw))}")


async def main_async() -> int:
    print(f"Verifying Task 3 (MCP server) — {len(CASES)} cases via in-memory transport.\n", flush=True)
    report = await _run_all()
    print_summary(report)
    print_details(report)
    return 0 if report.passed == report.total else 1


def main() -> int:
    """Entry point so callers can ``import scripts.verify_task3 as v; v.main()``."""
    return asyncio.run(main_async())


if __name__ == "__main__":
    sys.exit(main())
