"""Graph nodes other than the router itself.

- ``agent_node``: one step of the ReAct loop. Calls the agent LLM with all
  tools bound. Honours ``MAX_ITERATIONS`` as a graceful fallback.
- ``decline_node``: terminal node for out-of-scope queries.
- ``should_continue``: conditional-edge function deciding whether the agent
  asked for a tool call (loop back to the tool node) or is done (END).
- ``profile_update_node``: post-agent node that updates the per-user JSON
  profile when the latest human message looks personal-info-bearing.
"""

from __future__ import annotations

import json
import logging
from typing import Literal

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from cs_agent.agent.prompts import build_agent_system
from cs_agent.agent.state import GraphState
from cs_agent.config import MAX_ITERATIONS
from cs_agent.llm import get_agent_llm, get_router_llm
from cs_agent.memory.profile import (
    UserProfile,
    is_personal_info_bearing,
    load_profile,
    now_utc,
    save_profile,
)
from cs_agent.tools.registry import DATA_TOOLS

logger = logging.getLogger(__name__)

DECLINE_MESSAGE = (
    "That question is outside the scope of this customer-service data agent. "
    "I can help with the Bitext customer-support dataset — categories, intents, "
    "examples, distributions, or summaries. Try one of those?"
)

MAX_ITER_MESSAGE = (
    "I couldn't reach a confident answer within my reasoning budget "
    "(max {max_iter} steps). Could you rephrase the question or break it into "
    "smaller pieces?"
)

LOOP_FALLBACK_TEMPLATE = "Based on the {tool_name} tool result:\n\n{content}"

PROFILE_UPDATE_SYSTEM = """\
You maintain a small structured profile for ONE user across sessions.

You will receive (a) the user's CURRENT profile as JSON and (b) the LATEST
turn — the human message and the agent's reply. Return the FULL UPDATED
profile that incorporates anything new the user volunteered.

Rules:
- Preserve the existing user_id verbatim.
- Add new facts; refine existing ones if the user contradicted them; drop
  items the user retracts ("forget that I…"). Never invent facts the user
  did not state.
- Keep `notable_facts` short (one short sentence per item) and de-duplicated.
- Use lowercase short phrases for `topics_of_interest` (e.g. "refunds",
  "complaints").
- If the latest turn contains NO new personal info, return the current
  profile unchanged.
- Return ONLY a JSON object matching the UserProfile schema. No prose."""


def _is_loop(state: GraphState) -> tuple[bool, ToolMessage | None]:
    """Detect a tool-call loop: the last AIMessage emits exactly the same tool
    call (name + args) as one of the previous AIMessages in this turn.

    Returns (is_loop, last_tool_message).
    """
    messages = state.get("messages") or []
    last = messages[-1] if messages else None
    if not (isinstance(last, AIMessage) and last.tool_calls):
        return False, None

    # Build a signature for the latest tool calls
    def sig(ai: AIMessage) -> tuple:
        return tuple((tc["name"], tuple(sorted((tc.get("args") or {}).items()))) for tc in ai.tool_calls)

    latest_sig = sig(last)
    for prev in messages[:-1]:
        if isinstance(prev, AIMessage) and prev.tool_calls and sig(prev) == latest_sig:
            # Find the most recent ToolMessage to surface as the answer
            last_tool_msg = next(
                (m for m in reversed(messages) if isinstance(m, ToolMessage)),
                None,
            )
            return True, last_tool_msg
    return False, None


def agent_node(state: GraphState) -> dict:
    """One step of the ReAct loop.

    Each visit:
    1. Checks the iteration budget; emits a graceful fallback if exceeded.
    2. Builds a route-aware system prompt (structured/unstructured steering).
    3. Invokes the agent LLM with all tools bound. The LLM either emits a
       final answer (no tool calls) or one or more ``tool_calls`` that the
       graph will execute via ``ToolNode`` and then route back here.

    Per-turn contract (Task 2a): the caller is expected to pass
    ``iterations=0`` and ``route=None`` in the per-turn invoke dict so leftover
    fields from a previously-checkpointed turn don't leak into this one.
    The ``state.get("iterations", 0)`` default below is a defensive guard for
    external callers that forget the reset; it does NOT replace the contract.
    """
    iterations = state.get("iterations", 0)
    if iterations >= MAX_ITERATIONS:
        logger.info("max_iterations (%d) reached; emitting fallback", MAX_ITERATIONS)
        return {
            "messages": [AIMessage(MAX_ITER_MESSAGE.format(max_iter=MAX_ITERATIONS))],
            "iterations": iterations + 1,
        }

    route = state.get("route")
    user_id = state.get("user_id") or "anon"
    # ``build_agent_system`` reads the per-user profile from disk lazily, so
    # the latest profile (including updates from the previous turn) is
    # injected on every agent step.
    system_prompt = build_agent_system(route=route, user_id=user_id)

    llm_with_tools = get_agent_llm().bind_tools(DATA_TOOLS)
    messages = state.get("messages") or []
    response = llm_with_tools.invoke([SystemMessage(system_prompt), *messages])

    return {"messages": [response], "iterations": iterations + 1}


def decline_node(state: GraphState) -> dict:
    """Terminal node for out-of-scope queries — never calls a tool, never an LLM."""
    return {"messages": [AIMessage(DECLINE_MESSAGE)]}


def profile_recall_node(state: GraphState) -> dict:
    """Answer meta-questions from the persisted profile — no tools, no LLM."""
    user_id = state.get("user_id") or "anon"
    return {"messages": [AIMessage(load_profile(user_id).render_recall_answer())]}


def should_continue(state: GraphState) -> Literal["tools", "fallback", "end"]:
    """Conditional edge after agent_node. Four outcomes (third route below
    short-circuits a stuck-loop pattern that some open-source models exhibit).

    - "tools": the LLM emitted tool_calls and we still have iteration budget,
      AND the call is not a duplicate of an earlier call in this turn.
    - "fallback": either (a) we ran out of iteration budget mid tool-call,
      OR (b) the LLM is repeating a previously-issued tool call verbatim
      (a clear loop signal). Either way we emit a graceful answer.
    - "end": the LLM produced a final natural-language answer. Done.
    """
    messages = state.get("messages") or []
    if not messages:
        return "end"
    last = messages[-1]
    has_pending_tool_call = isinstance(last, AIMessage) and bool(last.tool_calls)

    if not has_pending_tool_call:
        return "end"

    is_loop, _ = _is_loop(state)
    if is_loop:
        logger.info("loop detected — short-circuiting to fallback")
        return "fallback"

    if state.get("iterations", 0) >= MAX_ITERATIONS:
        return "fallback"
    return "tools"


def fallback_node(state: GraphState) -> dict:
    """Emit a graceful answer when we abandoned the ReAct loop early.

    If we hit the loop because the LLM was re-calling the same tool, we
    surface the tool's most recent result as the user-facing answer (the
    LLM had the answer; it just refused to use it). Otherwise we fall back
    to the generic "ran out of budget" message.
    """
    is_loop, last_tool = _is_loop(state)
    if is_loop and last_tool is not None:
        logger.info("loop fallback: surfacing %s tool result as final answer", last_tool.name)
        text = LOOP_FALLBACK_TEMPLATE.format(
            tool_name=last_tool.name,
            content=str(last_tool.content).strip(),
        )
        return {"messages": [AIMessage(text)]}
    return {"messages": [AIMessage(MAX_ITER_MESSAGE.format(max_iter=MAX_ITERATIONS))]}


def _last_human_message(state: GraphState) -> HumanMessage | None:
    """Return the most recent ``HumanMessage`` in the conversation, or None."""
    messages = state.get("messages") or []
    return next(
        (m for m in reversed(messages) if isinstance(m, HumanMessage)),
        None,
    )


def _last_agent_answer(state: GraphState) -> str:
    """Return the agent's most recent user-facing reply (no tool-call placeholder).

    Used to give the profile-update LLM the FULL latest turn (human + agent)
    so it can disambiguate references like "the second one" or
    "yes, that one".
    """
    messages = state.get("messages") or []
    for m in reversed(messages):
        if isinstance(m, AIMessage) and m.content and not m.tool_calls:
            return str(m.content)
    return ""


def profile_update_node(state: GraphState) -> dict:
    """Update the per-user profile if the latest human message is personal-info-bearing.

    Pipeline:
    1. Cheap regex gate (``is_personal_info_bearing``). On a miss we return
       immediately — no LLM call. This keeps the cost of dataset Q&A turns
       (the dominant case) at exactly zero extra tokens.
    2. Load the current profile from disk.
    3. Ask the small router LLM to return a fully updated profile via
       structured output (Pydantic schema = ``UserProfile``).
    4. Stamp ``last_updated`` and persist atomically.

    Failure handling: if step 3 raises (Nebius timeout, schema violation,
    etc.) we log a warning and leave the profile untouched. The agent's
    user-facing answer has already been emitted upstream, so a profile-update
    miss doesn't block the response.

    Returns ``{}`` — the profile lives on disk, NOT in graph state, so this
    node never mutates the checkpointed conversation.
    """
    last_human = _last_human_message(state)
    if last_human is None:
        return {}

    text = str(last_human.content)
    if not is_personal_info_bearing(text):
        return {}

    user_id = state.get("user_id") or "anon"
    current = load_profile(user_id)

    try:
        structured = get_router_llm().with_structured_output(UserProfile)
        payload = json.dumps(
            {
                "current_profile": current.model_dump(mode="json"),
                "latest_turn": {
                    "user": text,
                    "agent": _last_agent_answer(state),
                },
            },
            ensure_ascii=False,
        )
        updated_raw = structured.invoke([SystemMessage(PROFILE_UPDATE_SYSTEM), HumanMessage(payload)])
        # ``with_structured_output`` may return either a UserProfile instance
        # or a dict (depending on backend); normalise.
        if isinstance(updated_raw, UserProfile):
            updated = updated_raw
        else:
            updated = UserProfile.model_validate(updated_raw)
    except Exception as exc:  # noqa: BLE001 — LLM/parse errors must not break the turn
        logger.warning("profile update LLM failed (%s); leaving profile untouched", exc)
        return {}

    # Stamp the canonical user_id (LLM may echo it back wrong) and the
    # update time, then persist.
    updated.user_id = user_id
    updated.last_updated = now_utc()
    try:
        save_profile(updated)
    except OSError as exc:
        logger.warning("profile save failed (%s); profile NOT persisted", exc)

    return {}
