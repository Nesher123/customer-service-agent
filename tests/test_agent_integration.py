"""Integration tests: run the compiled graph against each verifier case.

These mirror ``scripts/verify_task1.py`` 1:1 — the cases live in that script
and we import them here so we never drift between the README's smoke command
and what CI checks. Each case becomes a separate parametrized test, which
gives a clean per-case PASS/FAIL line in pytest output.

These tests **call live Nebius LLMs**, so they are marked ``integration`` and
skipped by the default ``pytest -m "not integration"`` run. To execute::

    uv run python -m pytest -m integration tests/test_agent_integration.py
"""

from __future__ import annotations

import pytest

from cs_agent.agent.graph import build_graph

# Imported from ``scripts/verify_task1.py`` (added to pytest pythonpath).
from scripts.verify_task1 import CASES, run_case

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def graph():
    """Compile the graph once for the whole module — saves repeated setup."""
    return build_graph()


@pytest.mark.parametrize("case", CASES, ids=[c.label for c in CASES])
def test_verifier_case(case, graph):
    """Each verifier case becomes its own pytest test, with the same checks
    that ``scripts/verify_task1.py`` enforces."""
    result = run_case(graph, case)

    if result.error is not None:
        pytest.fail(f"graph raised: {result.error}")

    assert result.route_ok, (
        f"route mismatch: got {result.route!r}, expected one of {list(case.expected_routes)}"
    )
    assert result.tools_ok, (
        f"none of the required tools were called. "
        f"called: {[name for name, _ in result.tool_calls]!r}, "
        f"required any of: {list(case.must_call_any_of)!r}"
    )
    assert result.answer_ok, (
        f"final answer is missing required substrings {list(case.answer_must_contain)!r}. "
        f"answer was: {result.final_answer[:300]!r}"
    )
