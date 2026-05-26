"""Bonus B — Query Recommender node and routing.

Two phases keyed by ``pending_query``:

- No pending → suggest: ask the agent LLM for ONE concrete next query
  (informed by profile + recent turns), stash it in ``pending_query``,
  emit an AIMessage asking the user to confirm / refine / pick something
  else.
- Pending set → classify the user's reply with the router LLM:
  - **confirm** → clear ``pending_query``, append a synthetic
    ``HumanMessage`` with the resolved query, hand off to ``agent_node``.
  - **refine** → regenerate using the refinement as context.
  - **reject** → clear ``pending_query`` and acknowledge.

``pending_query`` is the single source of truth; the router short-circuits
to ``recommend`` whenever it is set, and the SqliteSaver persists it per
``thread_id`` so a pending suggestion survives a process restart.
"""

from __future__ import annotations

import json
import logging
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from cs_agent.agent.prompts import (
    RECOMMENDER_INTENT_SYSTEM,
    RECOMMENDER_SUGGEST_SYSTEM,
)
from cs_agent.agent.state import GraphState
from cs_agent.llm import get_agent_llm, get_router_llm
from cs_agent.memory.profile import load_profile

logger = logging.getLogger(__name__)

RECENT_TURNS_WINDOW = 10
"""Number of trailing messages fed to the suggestion LLM as conversation context."""

REJECT_MESSAGE = "Got it — dropping the suggestion. What would you like to know?"

SUGGEST_TEMPLATE = (
    'Based on {rationale}, you might want to: "{query}". '
    "Want me to run it, refine it, or pick something else?"
)

REFINE_TEMPLATE = 'Then I\'d suggest: "{query}". Should I go ahead?'

FALLBACK_SUGGEST_MESSAGE = (
    "I couldn't think of a tailored suggestion right now. "
    'A good general starting point is: "What categories exist in the dataset?". '
    "Want me to run it?"
)
FALLBACK_SUGGEST_QUERY = "What categories exist in the dataset?"


class Suggestion(BaseModel):
    """Structured output from the suggestion LLM."""

    query: str = Field(
        ...,
        min_length=3,
        description=(
            "A concrete next query, phrased as the USER would type it, e.g. "
            "'Show me 5 examples from the REFUND category.' Must be answerable "
            "by ONE of the dataset tools."
        ),
    )
    rationale: str = Field(
        ...,
        min_length=3,
        description=(
            "One short clause (no terminal period) explaining why this query "
            "is a good next step, anchored in the profile or prior turns."
        ),
    )


class RecommenderIntent(BaseModel):
    """Structured output from the intent classifier (confirm / refine / reject)."""

    intent: Literal["confirm", "refine", "reject"] = Field(
        ...,
        description=(
            "User's intent in response to the outstanding query suggestion. "
            "'confirm' = run the suggested query; 'refine' = ask for a "
            "different suggestion; 'reject' = drop the suggestion."
        ),
    )
    refinement: str | None = Field(
        default=None,
        description=(
            "When intent='refine', the user's refinement instruction "
            "(verbatim or paraphrased). Null for 'confirm' and 'reject'."
        ),
    )


def _last_human(messages: list[BaseMessage]) -> HumanMessage | None:
    """Return the most recent ``HumanMessage`` in ``messages`` (or ``None``)."""
    return next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)


def _format_recent_messages(messages: list[BaseMessage], limit: int = RECENT_TURNS_WINDOW) -> str:
    """Render the trailing slice of the conversation for the suggestion prompt.

    We keep only ``HumanMessage`` and ``AIMessage`` (with content) — tool
    calls and tool results would be noise here. The suggestion LLM cares
    about WHAT the user asked and WHAT the agent answered, not HOW.
    """
    lines: list[str] = []
    for m in messages[-limit:]:
        if isinstance(m, HumanMessage):
            lines.append(f"USER: {str(m.content).strip()}")
        elif isinstance(m, AIMessage) and m.content and not m.tool_calls:
            lines.append(f"AGENT: {str(m.content).strip()}")
    return "\n".join(lines) if lines else "(no prior turns)"


def _suggest(
    llm: BaseChatModel,
    user_id: str,
    messages: list[BaseMessage],
    refinement: str | None = None,
) -> Suggestion:
    """Ask the suggestion LLM for a single concrete next query.

    Inputs: the user's distilled profile + the last few turns + optional
    refinement instruction. Output: a validated ``Suggestion``.
    """
    profile = load_profile(user_id)
    payload = {
        "user_profile": profile.render_for_prompt(),
        "recent_conversation": _format_recent_messages(messages),
        "refinement_instruction": refinement,
    }
    structured = llm.with_structured_output(Suggestion)
    raw = structured.invoke(
        [
            SystemMessage(RECOMMENDER_SUGGEST_SYSTEM),
            HumanMessage(json.dumps(payload, ensure_ascii=False)),
        ]
    )
    if isinstance(raw, Suggestion):
        return raw
    return Suggestion.model_validate(raw)


def _classify_intent(llm: BaseChatModel, pending: str, user_reply: str) -> RecommenderIntent:
    """Classify the user's reply to the outstanding suggestion."""
    payload = {
        "pending_suggestion": pending,
        "user_reply": user_reply,
    }
    structured = llm.with_structured_output(RecommenderIntent)
    raw = structured.invoke(
        [
            SystemMessage(RECOMMENDER_INTENT_SYSTEM),
            HumanMessage(json.dumps(payload, ensure_ascii=False)),
        ]
    )
    if isinstance(raw, RecommenderIntent):
        return raw
    return RecommenderIntent.model_validate(raw)


def _emit_initial_suggestion(state: GraphState) -> dict:
    """Handle the first call: produce a suggestion and stash it in pending_query.

    On LLM failure we fall back to a generic "what categories exist?" prompt
    rather than crashing the turn — the user can always type ``no`` and ask
    something else.
    """
    user_id = state.get("user_id") or "anon"
    messages = state.get("messages") or []
    try:
        suggestion = _suggest(get_agent_llm(), user_id, messages)
    except Exception as exc:  # noqa: BLE001 — LLM/parse errors must not break the turn
        logger.warning("recommender suggest failed (%s); using fallback suggestion", exc)
        return {
            "messages": [AIMessage(FALLBACK_SUGGEST_MESSAGE)],
            "pending_query": FALLBACK_SUGGEST_QUERY,
        }

    text = SUGGEST_TEMPLATE.format(rationale=suggestion.rationale.rstrip("."), query=suggestion.query)
    return {
        "messages": [AIMessage(text)],
        "pending_query": suggestion.query,
    }


def _handle_confirm(pending: str) -> dict:
    """User said yes — clear pending and append the resolved query as a new turn.

    The synthetic ``HumanMessage`` becomes the agent's question. ``iterations``
    is reset so ``agent_node`` gets its full ReAct budget for this dispatch.
    """
    logger.info("recommender confirm — dispatching pending query to agent: %r", pending)
    return {
        "messages": [HumanMessage(pending)],
        "pending_query": None,
        "iterations": 0,
    }


def _handle_refine(
    state: GraphState,
    refinement: str | None,
) -> dict:
    """User asked for a different suggestion — regenerate and replace pending."""
    user_id = state.get("user_id") or "anon"
    messages = state.get("messages") or []
    try:
        suggestion = _suggest(get_agent_llm(), user_id, messages, refinement=refinement)
    except Exception as exc:  # noqa: BLE001 — keep the loop alive on LLM failure
        logger.warning("recommender refine failed (%s); using fallback suggestion", exc)
        return {
            "messages": [AIMessage(FALLBACK_SUGGEST_MESSAGE)],
            "pending_query": FALLBACK_SUGGEST_QUERY,
        }

    text = REFINE_TEMPLATE.format(query=suggestion.query)
    return {
        "messages": [AIMessage(text)],
        "pending_query": suggestion.query,
    }


def _handle_reject() -> dict:
    """User cancelled the suggestion — clear pending and acknowledge."""
    return {
        "messages": [AIMessage(REJECT_MESSAGE)],
        "pending_query": None,
    }


def recommender_node(state: GraphState) -> dict:
    """Bonus B entry point. Two phases keyed by ``pending_query``.

    - No ``pending_query`` → first call: generate a suggestion.
    - ``pending_query`` set → follow-up: classify the user's reply as
      confirm / refine / reject and dispatch.

    Returns a partial state update. The conditional edge
    ``route_from_recommender`` reads the update to decide whether to hand
    off to ``agent`` (confirmation) or fall through to ``profile`` (any other
    outcome).
    """
    pending = state.get("pending_query")
    if not pending:
        return _emit_initial_suggestion(state)

    messages = state.get("messages") or []
    last_human = _last_human(messages)
    if last_human is None:
        # Defensive: pending_query is set but there's no user message to
        # classify. Treat as if the user re-asked for a suggestion.
        logger.warning("recommender: pending_query set but no HumanMessage; re-suggesting")
        return _emit_initial_suggestion(state)

    user_reply = str(last_human.content)
    try:
        decision = _classify_intent(get_router_llm(), pending, user_reply)
    except Exception as exc:  # noqa: BLE001 — fall back to a refine path
        logger.warning("recommender intent classification failed (%s); refining", exc)
        return _handle_refine(state, refinement=user_reply)

    if decision.intent == "confirm":
        return _handle_confirm(pending)
    if decision.intent == "reject":
        return _handle_reject()
    return _handle_refine(state, refinement=decision.refinement or user_reply)


def route_from_recommender(state: GraphState) -> Literal["agent", "profile"]:
    """Conditional edge after ``recommender_node``.

    The recommender signals "execute now" by clearing ``pending_query`` AND
    appending a synthetic ``HumanMessage`` (the resolved query). Any other
    outcome — initial suggestion, refinement, rejection — ends the turn via
    the profile updater on the way out.
    """
    if state.get("pending_query"):
        return "profile"
    messages = state.get("messages") or []
    last = messages[-1] if messages else None
    if isinstance(last, HumanMessage):
        return "agent"
    return "profile"
