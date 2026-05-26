"""Compile the full LangGraph StateGraph for the customer-service agent.

Topology (Bonus B-complete):

    START
      |
      v
    router  ── out_of_scope ─►  decline ──► END
      |       ── recommend ───►  recommender ──┐
      |       ── profile_recall ► profile_recall ──► END
      |                                        ├── confirm ──► agent
      |                                        └── suggest / refine / reject ──► profile
      ▼
    agent  ◄────────────────┐
      | tool_calls?         │
      ├── yes ──► tools ────┘
      ├── fallback (loop / max-iter) ──► fallback ──► profile ──► END
      └── no (final answer) ────────────────────────► profile ──► END

The ``profile`` node is the per-user profile updater (Task 2b). It runs
unconditionally after a successful or fallback agent turn, but its FIRST
action is a cheap regex gate that returns immediately for non-personal
turns. Out-of-scope declines bypass the profile entirely — those messages
carry no user-relevant information.

The ``recommender`` node is Bonus B. It is reached either because the user
asked for a query suggestion ("what should I query next?") or because there
is already a ``pending_query`` in state — the router short-circuits to
``recommend`` whenever the latter holds, so the recommender owns every
follow-up turn until the suggestion is consumed (confirm) or dropped
(reject).

A ``checkpointer`` may be passed through. Task 1 left it as ``None`` (no
persistence). Task 2a passes a ``SqliteSaver`` so messages survive across
process restarts.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from cs_agent.agent.nodes import (
    agent_node,
    decline_node,
    fallback_node,
    profile_recall_node,
    profile_update_node,
    should_continue,
)
from cs_agent.agent.recommender import recommender_node, route_from_recommender
from cs_agent.agent.router import route_from_router, router_node
from cs_agent.agent.state import GraphState
from cs_agent.tools.registry import DATA_TOOLS


def build_graph(checkpointer: Any | None = None):
    """Compile the agent graph. Pass a checkpointer in Task 2a; ``None`` in Task 1."""
    builder = StateGraph(GraphState)

    builder.add_node("router", router_node)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(DATA_TOOLS))
    builder.add_node("decline", decline_node)
    builder.add_node("profile_recall", profile_recall_node)
    builder.add_node("fallback", fallback_node)
    builder.add_node("recommender", recommender_node)
    builder.add_node("profile", profile_update_node)

    builder.add_edge(START, "router")
    builder.add_conditional_edges(
        "router",
        route_from_router,
        {
            "agent": "agent",
            "decline": "decline",
            "recommender": "recommender",
            "profile_recall": "profile_recall",
        },
    )
    # Recommender either hands off to the agent (on confirmation, after
    # appending a synthetic HumanMessage with the resolved query) or falls
    # through to the profile updater on the way out (suggest / refine / reject).
    builder.add_conditional_edges(
        "recommender",
        route_from_recommender,
        {"agent": "agent", "profile": "profile"},
    )
    # Successful agent turns and fallbacks both pass through the profile
    # updater on the way out. The updater is a cheap no-op when the latest
    # human message has no personal-info markers.
    builder.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "fallback": "fallback", "end": "profile"},
    )
    builder.add_edge("tools", "agent")
    builder.add_edge("profile_recall", END)
    builder.add_edge("fallback", "profile")
    builder.add_edge("profile", END)
    builder.add_edge("decline", END)

    return builder.compile(checkpointer=checkpointer)
