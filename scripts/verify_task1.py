"""Verify Task 1 by running the 8 example queries from the assignment brief.

For each query we capture:
- the router's classification,
- every tool call the agent made (name + args),
- the final user-facing answer.

We then check the result against an expected route and an "at least one of
these tools must have been called" set, and print a pass/fail summary.

Run with::

    uv run python scripts/verify_task1.py

This script is also reusable as a CI smoke test (no flakiness expected modulo
Nebius outages — both the router and the agent fall back gracefully on
transient failures).
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, HumanMessage

from cs_agent.agent.graph import build_graph


@dataclass
class Case:
    label: str
    query: str
    expected_route: str | tuple[str, ...]
    must_call_any_of: tuple[str, ...] = ()
    answer_must_contain: tuple[str, ...] = ()
    notes: str = ""

    @property
    def expected_routes(self) -> tuple[str, ...]:
        """Always-tuple form of expected_route, used for matching."""
        if isinstance(self.expected_route, str):
            return (self.expected_route,)
        return self.expected_route


CASES: list[Case] = [
    Case(
        label="categories",
        query="What categories exist in the dataset?",
        expected_route="structured",
        must_call_any_of=("list_categories",),
        answer_must_contain=("ACCOUNT", "REFUND"),
    ),
    Case(
        label="refund-count",
        query="How many refund requests did we get?",
        expected_route="structured",
        must_call_any_of=("count_rows", "get_distribution"),
        answer_must_contain=("2992",),
        notes="Expected: count_rows(category='REFUND')=2992 (potentially with intent chaining first).",
    ),
    Case(
        label="shipping-examples",
        query="Show me 5 examples of the SHIPPING category.",
        expected_route="structured",
        must_call_any_of=("get_examples",),
        # The answer must reference the SHIPPING category. After the LLM-artifact
        # cleaning was added in schemas.py, this case should resolve in 1 call.
        answer_must_contain=("shipping",),
        notes="Live data has 'SHIPPING' — agent may call list_categories first to confirm.",
    ),
    Case(
        label="complaint-summary",
        query="Summarize how agents respond to complaint intents.",
        expected_route="unstructured",
        must_call_any_of=("summarize",),
    ),
    Case(
        label="money-back-search",
        query="Show me examples of people wanting their money back.",
        expected_route="structured",
        must_call_any_of=("search_by_keyword", "get_examples"),
        notes="Paraphrased query — keyword search is the natural fit.",
    ),
    Case(
        label="account-distribution",
        query="What is the distribution of intents in the ACCOUNT category?",
        expected_route="structured",
        must_call_any_of=("get_distribution",),
        answer_must_contain=("create_account",),
    ),
    Case(
        label="oos-crm",
        query="What's the best CRM software for handling complaints?",
        expected_route="out_of_scope",
        answer_must_contain=("outside the scope",),
    ),
    Case(
        label="oos-france",
        query="Who is the president of France?",
        expected_route="out_of_scope",
        answer_must_contain=("outside the scope",),
    ),
    Case(
        label="greeting",
        query="hi",
        # A bare greeting must NOT be declined. Either route through the agent
        # (structured/unstructured) is acceptable; the agent's GREETINGS rule
        # in the system prompt produces a warm response without calling tools.
        expected_route=("structured", "unstructured"),
        answer_must_contain=("Bitext",),
        notes="Greetings should land in agent path, not the OOS decline.",
    ),
    Case(
        label="compound",
        query=("How many refund requests did we get? And summarize how agents respond to complaints."),
        # Compound queries can land in either route — both are acceptable.
        expected_route=("structured", "unstructured"),
        # Known Llama 3.3 70B quirk: for compound questions the model often
        # describes the tool calls as JSON inside the message content instead of
        # emitting them via the function-calling protocol. We assert routing
        # only; the detail block reveals the actual tool-call behaviour.
        notes=(
            "Edge case: Llama 3.3 70B sometimes textualises multiple tool calls "
            "instead of emitting them. Workaround: split into two turns."
        ),
    ),
]


@dataclass
class Result:
    case: Case
    route: str | None = None
    tool_calls: list[tuple[str, dict]] = field(default_factory=list)
    final_answer: str = ""
    iterations: int = 0
    elapsed_s: float = 0.0
    error: str | None = None

    @property
    def route_ok(self) -> bool:
        return self.route in self.case.expected_routes

    @property
    def tools_ok(self) -> bool:
        if not self.case.must_call_any_of:
            return True
        called = {name for name, _ in self.tool_calls}
        return any(t in called for t in self.case.must_call_any_of)

    @property
    def answer_ok(self) -> bool:
        # Strip thousands separators so "2,992" matches a required "2992".
        normalized = self.final_answer.replace(",", "")
        return all(s.lower() in normalized.lower() for s in self.case.answer_must_contain)

    @property
    def passed(self) -> bool:
        return self.error is None and self.route_ok and self.tools_ok and self.answer_ok


def run_case(graph, case: Case) -> Result:
    res = Result(case=case)
    initial = {
        "messages": [HumanMessage(case.query)],
        "iterations": 0,
        "user_id": "verifier",
        "route": None,
    }
    t0 = time.time()
    try:
        out = graph.invoke(initial)
    except Exception as exc:  # noqa: BLE001 — script must keep going
        res.error = repr(exc)
        res.elapsed_s = time.time() - t0
        return res
    res.elapsed_s = time.time() - t0
    res.route = out.get("route")
    res.iterations = out.get("iterations", 0)
    for m in out["messages"]:
        if isinstance(m, AIMessage) and m.tool_calls:
            for tc in m.tool_calls:
                res.tool_calls.append((tc["name"], dict(tc.get("args") or {})))
        if isinstance(m, AIMessage) and m.content and not m.tool_calls:
            res.final_answer = str(m.content)
    return res


def _truncate(s: str, n: int = 160) -> str:
    s = s.replace("\n", " ").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def print_summary(results: list[Result]) -> None:
    print("\n" + "=" * 100)
    print(f"{'#':>2}  {'PASS':<5} {'LABEL':<22} {'ROUTE':<13} {'CALLS':<5} {'TIME':>6}  {'TOOLS USED'}")
    print("=" * 100)
    for i, r in enumerate(results, 1):
        mark = "✓" if r.passed else "✗"
        route_disp = (r.route or "?") if r.route_ok else f"{r.route}≠{'/'.join(r.case.expected_routes)}"
        tool_names = ", ".join(name for name, _ in r.tool_calls) or "-"
        print(
            f"{i:>2}  {mark:<5} {r.case.label:<22} {route_disp:<13} "
            f"{len(r.tool_calls):>5} {r.elapsed_s:>5.1f}s  {tool_names}"
        )
    print("=" * 100)
    n_pass = sum(1 for r in results if r.passed)
    n_fail = len(results) - n_pass
    print(f"\nResult: {n_pass}/{len(results)} passed, {n_fail} failed")


def print_details(results: list[Result]) -> None:
    print("\n" + "—" * 100)
    print("DETAILS PER CASE")
    print("—" * 100)
    for i, r in enumerate(results, 1):
        print(f"\n[{i}] {r.case.label}: {r.case.query!r}")
        if r.case.notes:
            print(f"    notes:        {r.case.notes}")
        if isinstance(r.case.expected_route, str):
            expected_disp = repr(r.case.expected_route)
        else:
            expected_disp = f"one of {list(r.case.expected_routes)}"
        print(f"    route:        {r.route!r} (expected {expected_disp}) {'OK' if r.route_ok else 'FAIL'}")
        print(f"    iterations:   {r.iterations}")
        print(f"    tool calls ({len(r.tool_calls)}):")
        for name, args in r.tool_calls:
            arg_pairs = ", ".join(f"{k}={v!r}" for k, v in args.items() if v is not None)
            print(f"      - {name}({arg_pairs})")
        if r.case.must_call_any_of:
            print(
                f"    tools check:  must include any of {list(r.case.must_call_any_of)} "
                f"-> {'OK' if r.tools_ok else 'FAIL'}"
            )
        print(f"    final answer: {_truncate(r.final_answer, 240)}")
        if r.case.answer_must_contain:
            print(
                f"    answer check: must contain {list(r.case.answer_must_contain)} "
                f"-> {'OK' if r.answer_ok else 'FAIL'}"
            )
        if r.error:
            print(f"    ERROR:        {r.error}")


def main() -> int:
    print("Building graph…", flush=True)
    graph = build_graph()
    print(f"Running {len(CASES)} verification cases against the live graph...\n", flush=True)

    results: list[Result] = []
    for i, case in enumerate(CASES, 1):
        print(f"  [{i}/{len(CASES)}] {case.label!s:24s} ", end="", flush=True)
        r = run_case(graph, case)
        results.append(r)
        mark = "✓" if r.passed else "✗"
        print(f"{mark}  ({r.elapsed_s:.1f}s, {len(r.tool_calls)} tool calls)", flush=True)

    print_summary(results)
    print_details(results)
    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
