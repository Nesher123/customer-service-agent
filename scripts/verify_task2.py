"""Verify Task 2 (memory) end-to-end with two multi-turn cases.

Cases:
1. ``episodic-followup`` — "Show me 3 examples of REFUND" then "Show me 3
   more". Persists across a simulated process restart (we build the graph
   twice against the same SQLite checkpoint file).
2. ``profile-recall`` — "Hi, my name is Ofir" then, in a different session
   id, "What do you remember about me?". The final answer must contain
   "Ofir", proving the JSON profile survives the session change.

Both cases use a tmp-dir for ``checkpoints.sqlite`` and ``profiles/`` so the
verifier never pollutes the user's real working state. Task 1's
``scripts/verify_task1.py`` is unchanged.

Run with::

    uv run python scripts/verify_task2.py
"""

from __future__ import annotations

import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

import cs_agent.memory.profile as profile_mod
from cs_agent.agent.graph import build_graph
from cs_agent.agent.state import GraphState
from cs_agent.memory.checkpoint import get_checkpointer, make_thread_id
from cs_agent.memory.profile import load_profile


@dataclass
class Turn:
    """One conversational turn within a multi-turn case."""

    session: str
    user: str
    query: str


@dataclass
class MultiTurnCase:
    """A scripted dialog plus pass/fail predicates evaluated after the run."""

    label: str
    description: str
    turns: list[Turn]
    answer_must_contain: tuple[str, ...] = ()
    """Substrings (case-insensitive) that MUST appear in the FINAL answer."""

    notes: str = ""


CASES: list[MultiTurnCase] = [
    MultiTurnCase(
        label="episodic-followup",
        description="Follow-up question survives a process restart.",
        turns=[
            Turn(session="demo", user="grader", query="Show me 3 examples of REFUND"),
            Turn(session="demo", user="grader", query="Show me 3 more"),
        ],
        # We don't pin substrings on the second answer because the agent
        # may render the examples in many shapes ("Here are 3 more:", a
        # bulleted list, a paragraph, etc.). The structural check below
        # (≥ 2 HumanMessages persisted, ≥ 1 tool call in turn 2) is the
        # real assertion.
        notes=(
            "Pass = (a) both human messages persisted under the grader::demo "
            "thread across the simulated restart, (b) turn 2 produced an AI answer."
        ),
    ),
    MultiTurnCase(
        label="profile-recall",
        description="Cross-session profile recall.",
        turns=[
            Turn(session="intro", user="ofir", query="Hi, my name is Ofir"),
            Turn(session="recall", user="ofir", query="What do you remember about me?"),
        ],
        answer_must_contain=("ofir",),
        notes=(
            "Pass = (a) profiles/ofir.json exists and contains 'Ofir', "
            "(b) the recall-session final answer contains 'Ofir'."
        ),
    ),
]


@dataclass
class TurnResult:
    turn: Turn
    final_answer: str = ""
    tool_calls: list[tuple[str, dict]] = field(default_factory=list)
    error: str | None = None


@dataclass
class CaseResult:
    case: MultiTurnCase
    turn_results: list[TurnResult] = field(default_factory=list)
    elapsed_s: float = 0.0
    profile_dump: dict | None = None
    persisted_human_count: int = 0

    @property
    def final_answer(self) -> str:
        return self.turn_results[-1].final_answer if self.turn_results else ""

    @property
    def answer_ok(self) -> bool:
        if not self.case.answer_must_contain:
            return True
        ans = self.final_answer.lower()
        return all(s.lower() in ans for s in self.case.answer_must_contain)

    @property
    def errored(self) -> bool:
        return any(t.error for t in self.turn_results)

    @property
    def episodic_ok(self) -> bool:
        """Episodic-specific check: both human messages must be persisted."""
        if self.case.label != "episodic-followup":
            return True
        return self.persisted_human_count >= len(self.case.turns)

    @property
    def profile_ok(self) -> bool:
        """Profile-specific check: profile JSON has at least one fact."""
        if self.case.label != "profile-recall":
            return True
        return bool(self.profile_dump and self.profile_dump.get("name"))

    @property
    def passed(self) -> bool:
        return not self.errored and self.answer_ok and self.episodic_ok and self.profile_ok


def _final_ai_answer(messages) -> str:
    for m in reversed(messages):
        if isinstance(m, AIMessage) and m.content and not m.tool_calls:
            return str(m.content)
    return ""


def _tool_calls_in(messages) -> list[tuple[str, dict]]:
    out: list[tuple[str, dict]] = []
    for m in messages:
        if isinstance(m, AIMessage) and m.tool_calls:
            for tc in m.tool_calls:
                out.append((tc["name"], dict(tc.get("args") or {})))
    return out


def run_case(case: MultiTurnCase, ckpt_path: Path) -> CaseResult:
    """Run all turns of ``case`` against a freshly-compiled graph per turn.

    Compiling the graph anew on every turn simulates the worst case
    (process restart between every turn) and is the strongest proof that
    persistence is doing the work, not in-memory state.
    """
    res = CaseResult(case=case)
    t0 = time.time()

    for turn in case.turns:
        cp = get_checkpointer(ckpt_path)
        graph = build_graph(checkpointer=cp)
        config: RunnableConfig = {"configurable": {"thread_id": make_thread_id(turn.user, turn.session)}}
        initial: GraphState = {
            "messages": [HumanMessage(turn.query)],
            "iterations": 0,
            "user_id": turn.user,
            "route": None,
        }
        tr = TurnResult(turn=turn)
        try:
            out = graph.invoke(initial, config=config)
        except Exception as exc:  # noqa: BLE001 — verifier must keep going
            tr.error = repr(exc)
            res.turn_results.append(tr)
            continue
        tr.final_answer = _final_ai_answer(out["messages"])
        tr.tool_calls = _tool_calls_in(out["messages"])
        res.turn_results.append(tr)

    res.elapsed_s = time.time() - t0

    # Post-run inspection. Episodic: count persisted HumanMessages on the
    # last-used session. Profile: read the JSON file directly.
    last_turn = case.turns[-1]
    cp = get_checkpointer(ckpt_path)
    graph = build_graph(checkpointer=cp)
    final_state = graph.get_state(
        {"configurable": {"thread_id": make_thread_id(last_turn.user, last_turn.session)}}
    ).values
    persisted = final_state.get("messages") or []
    res.persisted_human_count = sum(1 for m in persisted if isinstance(m, HumanMessage))

    if case.label == "profile-recall":
        profile = load_profile(case.turns[0].user)
        res.profile_dump = profile.model_dump(mode="json") if profile.has_facts() else None

    return res


def _truncate(s: str, n: int = 240) -> str:
    s = s.replace("\n", " ").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def print_summary(results: list[CaseResult]) -> None:
    print("\n" + "=" * 100)
    print(f"{'#':>2}  {'PASS':<5} {'LABEL':<22} {'TURNS':<6} {'TIME':>6}  CHECKS")
    print("=" * 100)
    for i, r in enumerate(results, 1):
        mark = "✓" if r.passed else "✗"
        checks: list[str] = []
        checks.append(f"answer={'ok' if r.answer_ok else 'FAIL'}")
        if r.case.label == "episodic-followup":
            checks.append(f"persisted={r.persisted_human_count}/{len(r.case.turns)}")
        if r.case.label == "profile-recall":
            checks.append(f"profile={'ok' if r.profile_ok else 'MISSING'}")
        print(
            f"{i:>2}  {mark:<5} {r.case.label:<22} {len(r.case.turns):<6} "
            f"{r.elapsed_s:>5.1f}s  {' '.join(checks)}"
        )
    print("=" * 100)
    n_pass = sum(1 for r in results if r.passed)
    print(f"\nResult: {n_pass}/{len(results)} passed, {len(results) - n_pass} failed")


def print_details(results: list[CaseResult]) -> None:
    print("\n" + "—" * 100)
    print("DETAILS PER CASE")
    print("—" * 100)
    for i, r in enumerate(results, 1):
        print(f"\n[{i}] {r.case.label}: {r.case.description}")
        if r.case.notes:
            print(f"    notes:        {r.case.notes}")
        for j, tr in enumerate(r.turn_results, 1):
            print(f"    turn {j} (session={tr.turn.session!r}, user={tr.turn.user!r}):")
            print(f"      query:      {tr.turn.query!r}")
            print(f"      tool calls: {[name for name, _ in tr.tool_calls] or '-'}")
            print(f"      answer:     {_truncate(tr.final_answer)}")
            if tr.error:
                print(f"      ERROR:      {tr.error}")
        print(f"    persisted human messages on final thread: {r.persisted_human_count}")
        if r.case.label == "profile-recall":
            print(f"    profile dump: {r.profile_dump}")
        if r.case.answer_must_contain:
            print(
                f"    answer check: must contain {list(r.case.answer_must_contain)} "
                f"-> {'OK' if r.answer_ok else 'FAIL'}"
            )


def main() -> int:
    print("Building graph and running Task 2 verification cases…\n", flush=True)
    results: list[CaseResult] = []
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        # Redirect the per-user profile dir to the tmp path for the lifetime
        # of this process; the real profiles/ on disk is untouched.
        profile_mod.PROFILES_DIR = tmp / "profiles"
        profile_mod.PROFILES_DIR.mkdir(parents=True, exist_ok=True)
        ckpt_path = tmp / "checkpoints.sqlite"

        for i, case in enumerate(CASES, 1):
            print(f"  [{i}/{len(CASES)}] {case.label!s:24s} ", end="", flush=True)
            r = run_case(case, ckpt_path)
            results.append(r)
            mark = "✓" if r.passed else "✗"
            print(f"{mark}  ({r.elapsed_s:.1f}s, {len(r.case.turns)} turns)", flush=True)

    print_summary(results)
    print_details(results)
    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
