"""Per-case check functions for ``verify_bonus_b.py``.

Kept in a sibling private module so the driver stays small. Each function
returns a short success-detail string on pass and raises ``AssertionError``
on failure; the shared ``_verifier_runner`` turns that contract into a
PASS/FAIL row.
"""

from __future__ import annotations

import importlib
import os
import tempfile
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig


def _make_fake_suggestion_llm(query: str, rationale: str) -> Any:
    """Return an LLM stub whose ``with_structured_output(Suggestion).invoke`` returns the fixture."""
    from cs_agent.agent.recommender import Suggestion

    class _Structured:
        def invoke(self, _messages: Any) -> Suggestion:
            return Suggestion(query=query, rationale=rationale)

    class _LLM:
        def with_structured_output(self, schema: type) -> Any:  # noqa: ARG002
            return _Structured()

    return _LLM()


def _make_fake_intent_llm(intent: str, refinement: str | None = None) -> Any:
    """Return an LLM stub whose ``with_structured_output(RecommenderIntent).invoke`` returns the fixture."""
    from cs_agent.agent.recommender import RecommenderIntent

    class _Structured:
        def invoke(self, _messages: Any) -> RecommenderIntent:
            return RecommenderIntent(intent=intent, refinement=refinement)  # type: ignore[arg-type]

    class _LLM:
        def with_structured_output(self, schema: type) -> Any:  # noqa: ARG002
            return _Structured()

    return _LLM()


def check_recommender_module_import() -> str:
    """Import recommender + state and assert the documented public surface exists."""
    import typing as _typing

    rec = importlib.import_module("cs_agent.agent.recommender")
    state = importlib.import_module("cs_agent.agent.state")

    required_rec = ("recommender_node", "route_from_recommender", "Suggestion", "RecommenderIntent")
    missing = [n for n in required_rec if not hasattr(rec, n)]
    if missing:
        raise AssertionError(f"missing public attrs in recommender: {missing}")

    route_args = _typing.get_args(state.Route)
    if "recommend" not in route_args:
        raise AssertionError(f"'recommend' not in Route literal: {route_args!r}")
    return f"{len(required_rec)} attrs present; Route includes 'recommend'"


def check_router_short_circuit() -> str:
    """``router_node`` must return route=recommend without calling the LLM when pending is set."""
    from cs_agent.agent import router as router_mod

    calls: list[Any] = []

    def _fail_llm() -> Any:
        calls.append("router_llm")
        raise AssertionError("router_llm must NOT be called when pending_query is set")

    orig = router_mod.get_router_llm
    router_mod.get_router_llm = _fail_llm  # type: ignore[assignment]
    try:
        update = router_mod.router_node(
            {
                "messages": [HumanMessage("yes please go ahead")],
                "pending_query": "Show me 5 examples from the REFUND category.",
            }
        )
    finally:
        router_mod.get_router_llm = orig  # type: ignore[assignment]

    if update.get("route") != "recommend":
        raise AssertionError(f"expected route='recommend', got {update.get('route')!r}")
    if calls:
        raise AssertionError(f"router LLM was called: {calls!r}")
    return "router → recommend, LLM untouched"


def check_suggest_shape() -> str:
    """First-call recommender produces an AIMessage and sets pending_query to the suggestion."""
    from cs_agent.agent import recommender as rec_mod

    fake_query = "What is the distribution of intents in the REFUND category?"
    fake_rationale = "your interest in refund data"
    orig = rec_mod.get_agent_llm
    rec_mod.get_agent_llm = lambda: _make_fake_suggestion_llm(fake_query, fake_rationale)  # type: ignore[assignment]
    try:
        update = rec_mod.recommender_node(
            {
                "messages": [HumanMessage("what should I query next?")],
                "user_id": "anon",
                "pending_query": None,
            }
        )
    finally:
        rec_mod.get_agent_llm = orig  # type: ignore[assignment]

    if update.get("pending_query") != fake_query:
        raise AssertionError(f"pending_query={update.get('pending_query')!r} expected {fake_query!r}")
    msgs = update.get("messages") or []
    if not (len(msgs) == 1 and isinstance(msgs[0], AIMessage)):
        raise AssertionError(f"expected 1 AIMessage, got {msgs!r}")
    if fake_query not in str(msgs[0].content):
        raise AssertionError(f"suggested query missing from AIMessage: {msgs[0].content!r}")
    return f"pending_query set to {fake_query!r}; AIMessage emitted"


def check_confirm_routes_to_agent() -> str:
    """Confirmation clears pending and appends a synthetic HumanMessage; edge → agent."""
    from cs_agent.agent import recommender as rec_mod

    pending = "Show me 5 examples from the REFUND category."
    orig = rec_mod.get_router_llm
    rec_mod.get_router_llm = lambda: _make_fake_intent_llm("confirm")  # type: ignore[assignment]
    try:
        update = rec_mod.recommender_node(
            {
                "messages": [
                    HumanMessage("what should I query next?"),
                    AIMessage(f'Based on …, you might want to: "{pending}". …'),
                    HumanMessage("yes please"),
                ],
                "user_id": "anon",
                "pending_query": pending,
            }
        )
    finally:
        rec_mod.get_router_llm = orig  # type: ignore[assignment]

    if update.get("pending_query") is not None:
        raise AssertionError(f"pending_query={update.get('pending_query')!r} expected None")
    if update.get("iterations") != 0:
        raise AssertionError(f"iterations={update.get('iterations')!r} expected 0")
    msgs = update.get("messages") or []
    if not (len(msgs) == 1 and isinstance(msgs[0], HumanMessage)):
        raise AssertionError(f"expected 1 synthetic HumanMessage, got {msgs!r}")
    if str(msgs[0].content) != pending:
        raise AssertionError(f"synthetic message={msgs[0].content!r} expected {pending!r}")

    next_node = rec_mod.route_from_recommender(
        {
            "messages": [
                HumanMessage("what should I query next?"),
                AIMessage("..."),
                HumanMessage("yes please"),
                HumanMessage(pending),
            ],
            "pending_query": None,
        }
    )
    if next_node != "agent":
        raise AssertionError(f"route_from_recommender={next_node!r} expected 'agent'")
    return "pending cleared; synthetic HumanMessage appended; edge → agent"


def check_refine_regenerates() -> str:
    """Refinement regenerates the suggestion; edge falls through to profile."""
    from cs_agent.agent import recommender as rec_mod

    old_pending = "What is the distribution of intents in the REFUND category?"
    new_pending = "Show me 5 examples from the REFUND category."
    orig_router = rec_mod.get_router_llm
    orig_agent = rec_mod.get_agent_llm
    rec_mod.get_router_llm = lambda: _make_fake_intent_llm(  # type: ignore[assignment]
        "refine", refinement="examples instead"
    )
    rec_mod.get_agent_llm = lambda: _make_fake_suggestion_llm(  # type: ignore[assignment]
        new_pending, "your earlier interest in REFUND"
    )
    try:
        update = rec_mod.recommender_node(
            {
                "messages": [
                    HumanMessage("what should I query next?"),
                    AIMessage(f'Based on …, you might want to: "{old_pending}". …'),
                    HumanMessage("I'd rather see examples"),
                ],
                "user_id": "anon",
                "pending_query": old_pending,
            }
        )
    finally:
        rec_mod.get_router_llm = orig_router  # type: ignore[assignment]
        rec_mod.get_agent_llm = orig_agent  # type: ignore[assignment]

    if update.get("pending_query") != new_pending:
        raise AssertionError(f"pending_query={update.get('pending_query')!r} expected {new_pending!r}")
    msgs = update.get("messages") or []
    if not (len(msgs) == 1 and isinstance(msgs[0], AIMessage)):
        raise AssertionError(f"expected 1 AIMessage, got {msgs!r}")
    if new_pending not in str(msgs[0].content):
        raise AssertionError(f"new suggestion missing: {msgs[0].content!r}")

    next_node = rec_mod.route_from_recommender({"pending_query": new_pending, "messages": msgs})
    if next_node != "profile":
        raise AssertionError(f"route_from_recommender={next_node!r} expected 'profile'")
    return f"pending regenerated to {new_pending!r}; edge → profile"


def check_reject_clears() -> str:
    """Rejection clears pending_query and emits an acknowledgement."""
    from cs_agent.agent import recommender as rec_mod

    orig = rec_mod.get_router_llm
    rec_mod.get_router_llm = lambda: _make_fake_intent_llm("reject")  # type: ignore[assignment]
    try:
        update = rec_mod.recommender_node(
            {
                "messages": [
                    HumanMessage("what should I query next?"),
                    AIMessage("..."),
                    HumanMessage("no, cancel"),
                ],
                "user_id": "anon",
                "pending_query": "anything",
            }
        )
    finally:
        rec_mod.get_router_llm = orig  # type: ignore[assignment]

    if update.get("pending_query") is not None:
        raise AssertionError(f"pending_query={update.get('pending_query')!r} expected None")
    msgs = update.get("messages") or []
    if not (len(msgs) == 1 and isinstance(msgs[0], AIMessage)):
        raise AssertionError(f"expected 1 AIMessage, got {msgs!r}")

    next_node = rec_mod.route_from_recommender({"pending_query": None, "messages": msgs})
    if next_node != "profile":
        raise AssertionError(f"route_from_recommender={next_node!r} expected 'profile'")
    return "pending cleared; AIMessage emitted; edge → profile"


def check_live_flow() -> str:
    """End-to-end suggest → confirm against the real graph; gated on Nebius credentials."""
    if not os.getenv("NEBIUS_API_KEY"):
        return "skipped (NEBIUS_API_KEY not set)"

    from cs_agent.agent.graph import build_graph
    from cs_agent.agent.state import GraphState
    from cs_agent.memory.checkpoint import get_checkpointer

    with tempfile.TemporaryDirectory() as d:
        cp = get_checkpointer(Path(d) / "checkpoints.sqlite")
        graph = build_graph(checkpointer=cp)
        config: RunnableConfig = {"configurable": {"thread_id": "verify-bonus-b"}}

        initial: GraphState = {
            "messages": [HumanMessage("What should I query next?")],
            "iterations": 0,
            "user_id": "verifier",
            "route": None,
        }
        graph.invoke(initial, config=config)
        state_after_suggest = graph.get_state(config).values
        pending = state_after_suggest.get("pending_query")
        if not pending:
            raise AssertionError("pending_query was not set after the suggestion turn")

        confirm: GraphState = {
            "messages": [HumanMessage("yes, do it")],
            "iterations": 0,
            "user_id": "verifier",
            "route": None,
        }
        graph.invoke(confirm, config=config)
        state_after_confirm = graph.get_state(config).values
        if state_after_confirm.get("pending_query") is not None:
            raise AssertionError(
                f"pending_query={state_after_confirm.get('pending_query')!r} expected None after confirm"
            )

        messages = state_after_confirm.get("messages") or []
        tool_calls = sum(1 for m in messages if isinstance(m, AIMessage) and m.tool_calls)
        if tool_calls == 0:
            raise AssertionError("agent did not execute any tools after confirmation")
        return f"suggested {pending!r}; confirmed → {tool_calls} tool call(s); pending cleared"
