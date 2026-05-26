"""Shared graph state shape.

Kept deliberately small. ``messages`` uses LangGraph's ``add_messages`` reducer
so that node updates *append* messages instead of replacing the list.

Fields:
- messages, route, iterations, user_id (Tasks 1 & 2)
- pending_query (Bonus B — Query Recommender)
"""

from __future__ import annotations

from typing import Annotated, Literal

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

Route = Literal["structured", "unstructured", "out_of_scope", "recommend", "profile_recall"]


class GraphState(TypedDict, total=False):
    """LangGraph state for the customer-service data-analyst agent."""

    messages: Annotated[list[BaseMessage], add_messages]
    """Conversation history. ``add_messages`` reducer appends, never overwrites."""

    route: Route | None
    """Output of ``router_node``. Drives the conditional edge after the router."""

    iterations: int
    """Number of times ``agent_node`` has run for the current turn. Reset per turn."""

    user_id: str
    """Stable identifier for the human user (separate from thread_id). Used
    by the user-profile module in Task 2b."""

    pending_query: str | None
    """Bonus B (Query Recommender): the suggested query the user has not yet
    confirmed. ``None`` means there is no outstanding suggestion. When set,
    ``router_node`` short-circuits to ``recommend`` regardless of the user's
    latest message so the recommender can interpret the reply as confirm /
    refine / reject. Persisted by the SqliteSaver checkpointer per
    ``thread_id`` so a pending suggestion survives a process restart."""
