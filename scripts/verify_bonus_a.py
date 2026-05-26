"""Verify Bonus A (Streamlit UI) plumbing without spinning up a browser.

Four cases, all offline-safe (the live-turn case is skipped without a Nebius
key):

1. ``module-import``      — ``cs_agent.ui.streamlit_app`` and its rendering
                            module both import cleanly; the documented public
                            attrs (``render_app``, ``main``, the helpers) exist.
2. ``chunk-translation``  — ``chunk_to_reasoning_steps`` produces the right
                            ``(steps, final, kind)`` triple for every chunk
                            shape we expect from the live LangGraph stream
                            (router, agent-tool-call, tools, agent-final,
                            decline, fallback).
3. ``history-replay``     — ``messages_to_turns`` correctly groups a persisted
                            ``BaseMessage`` list into one ``RenderedTurn`` per
                            human turn, pairing tool calls with their results
                            via ``tool_call_id``.
4. ``live-turn``          — (optional) end-to-end run: build the graph against
                            a tmp ``SqliteSaver``, drive one canned query
                            through it, and confirm the helpers translate
                            every chunk into something renderable.

Run with::

    uv run python scripts/verify_bonus_a.py
"""

from __future__ import annotations

import importlib
import os
import sys
import tempfile
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

# scripts/ is not a package — make sibling _verifier_runner.py importable.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _verifier_runner import Case, run_main  # noqa: E402


def check_module_import() -> str:
    """Import both UI modules and assert the documented public surface exists."""
    rendering = importlib.import_module("cs_agent.ui.rendering")
    app = importlib.import_module("cs_agent.ui.streamlit_app")
    required_app = ("render_app", "main")
    required_rendering = (
        "chunk_to_reasoning_steps",
        "messages_to_turns",
        "RenderedTurn",
        "ReasoningStep",
    )
    missing = [n for n in required_app if not hasattr(app, n)] + [
        n for n in required_rendering if not hasattr(rendering, n)
    ]
    if missing:
        raise AssertionError(f"missing public attrs: {missing}")
    return f"{len(required_app) + len(required_rendering)} public attrs present"


def check_chunk_translation() -> str:
    """Exercise every chunk shape produced by the live LangGraph stream."""
    from cs_agent.ui.rendering import chunk_to_reasoning_steps

    tool_call = AIMessage(
        content="",
        tool_calls=[{"name": "count_rows", "args": {"category": "REFUND"}, "id": "abc"}],
    )
    tool_result = ToolMessage(content="2992", tool_call_id="abc", name="count_rows")
    cases: list[tuple[str, dict, int, str | None, str]] = [
        ("router", {"router": {"route": "structured"}}, 1, None, "normal"),
        ("agent-tool-call", {"agent": {"messages": [tool_call]}}, 1, None, "normal"),
        ("tools", {"tools": {"messages": [tool_result]}}, 1, None, "normal"),
        (
            "agent-final",
            {"agent": {"messages": [AIMessage("There are 2992 refund rows.")]}},
            0,
            "There are 2992 refund rows.",
            "normal",
        ),
        ("decline", {"decline": {"messages": [AIMessage("Out of scope.")]}}, 0, "Out of scope.", "decline"),
        (
            "fallback",
            {"fallback": {"messages": [AIMessage("ran out of budget")]}},
            0,
            "ran out of budget",
            "fallback",
        ),
    ]
    for label, chunk, n_steps, expected_final, expected_kind in cases:
        steps, final, kind = chunk_to_reasoning_steps(chunk)
        if len(steps) != n_steps:
            raise AssertionError(f"{label}: steps={len(steps)} expected {n_steps}")
        if final != expected_final:
            raise AssertionError(f"{label}: final={final!r} expected {expected_final!r}")
        if kind != expected_kind:
            raise AssertionError(f"{label}: kind={kind!r} expected {expected_kind!r}")
    return f"{len(cases)} chunk shapes assert ok"


def check_history_replay() -> str:
    """Replay a synthetic 2-turn conversation through ``messages_to_turns``."""
    from cs_agent.ui.rendering import messages_to_turns

    msgs = [
        HumanMessage("How many refunds?"),
        AIMessage(
            content="",
            tool_calls=[
                {"name": "count_rows", "args": {"category": "REFUND"}, "id": "t1"},
            ],
        ),
        ToolMessage(content="2992", tool_call_id="t1", name="count_rows"),
        AIMessage("There are 2992 refund rows."),
        HumanMessage("Show me 3 more"),
        AIMessage("Here are 3 more…"),
    ]
    turns = messages_to_turns(msgs)
    if len(turns) != 2:
        raise AssertionError(f"expected 2 turns, got {len(turns)}")
    if turns[0].user_query != "How many refunds?":
        raise AssertionError(f"turn0.user_query={turns[0].user_query!r}")
    if not turns[0].answer.startswith("There are 2992"):
        raise AssertionError(f"turn0.answer={turns[0].answer!r}")
    if len(turns[0].steps) != 2:
        raise AssertionError(f"turn0.steps={len(turns[0].steps)} expected 2")
    if turns[1].user_query != "Show me 3 more":
        raise AssertionError(f"turn1.user_query={turns[1].user_query!r}")
    if len(turns[1].steps) != 0:
        raise AssertionError(f"turn1.steps={len(turns[1].steps)} expected 0")
    return "2 turns reconstructed (steps=2/0)"


def check_live_turn() -> str:
    """End-to-end run against a tmp SqliteSaver; gated on Nebius credentials."""
    if not os.getenv("NEBIUS_API_KEY"):
        return "skipped (NEBIUS_API_KEY not set)"

    from cs_agent.agent.graph import build_graph
    from cs_agent.agent.state import GraphState
    from cs_agent.memory.checkpoint import get_checkpointer
    from cs_agent.ui.rendering import chunk_to_reasoning_steps

    with tempfile.TemporaryDirectory() as d:
        cp = get_checkpointer(Path(d) / "checkpoints.sqlite")
        graph = build_graph(checkpointer=cp)
        config: RunnableConfig = {"configurable": {"thread_id": "verify-bonus-a"}}
        initial: GraphState = {
            "messages": [HumanMessage("What categories exist in the dataset?")],
            "iterations": 0,
            "user_id": "verifier",
            "route": None,
        }
        steps_count = 0
        final: str | None = None
        for chunk in graph.stream(initial, config=config, stream_mode="updates"):
            s, f, _ = chunk_to_reasoning_steps(chunk)
            steps_count += len(s)
            if f is not None:
                final = f
        if final is None:
            raise AssertionError("no final answer produced")
        if steps_count == 0:
            raise AssertionError("no reasoning steps captured")
        return f"streamed {steps_count} steps; final answer {len(final)} chars"


CASES: list[Case] = [
    Case(
        label="module-import",
        description="cs_agent.ui.* import cleanly without Streamlit runtime calls.",
        fn=check_module_import,
        notes="render_app() is gated under if __name__=='__main__'; import must not fire it.",
    ),
    Case(
        label="chunk-translation",
        description="chunk_to_reasoning_steps covers every node shape from the stream.",
        fn=check_chunk_translation,
        notes="router / agent-tool-call / tools / agent-final / decline / fallback.",
    ),
    Case(
        label="history-replay",
        description="messages_to_turns groups by HumanMessage and pairs tool calls.",
        fn=check_history_replay,
        notes="Simulates a 2-turn conversation pulled from the checkpointer.",
    ),
    Case(
        label="live-turn",
        description="end-to-end run through the graph + UI helpers.",
        fn=check_live_turn,
        notes="Skipped without NEBIUS_API_KEY; otherwise runs one canned query.",
    ),
]


def main() -> int:
    """Entry point so callers can ``import scripts.verify_bonus_a as v; v.main()``."""
    return run_main(CASES, banner="Verifying Bonus A (Streamlit UI plumbing)", label_width=22)


if __name__ == "__main__":
    sys.exit(main())
