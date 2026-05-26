"""Pure-Python helpers shared between the Streamlit app and its verifier.

These helpers translate LangGraph stream chunks (live turns) or persisted
``BaseMessage`` lists (resumed history) into small renderable data structures
(``ReasoningStep`` / ``RenderedTurn``). They are deliberately Streamlit-free
so ``scripts/verify_bonus_a.py`` can import and exercise them offline without
spinning up a Streamlit runtime.

The Streamlit-specific rendering (chat bubbles, status expanders, sidebar)
lives in ``cs_agent.ui.streamlit_app``.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

TOOL_PREVIEW_CHARS = 240
"""Max chars shown of any single tool result before we ellipsise it."""

AnswerKind = Literal["normal", "decline", "fallback", "recommend"]


@dataclass
class ReasoningStep:
    """One row inside a per-turn reasoning expander."""

    kind: Literal["router", "tool_call", "tool_result", "recommender"]
    text: str


@dataclass
class RenderedTurn:
    """Snapshot of a single conversation turn used for (re-)rendering."""

    user_query: str
    steps: list[ReasoningStep] = field(default_factory=list)
    answer: str = ""
    answer_kind: AnswerKind = "normal"


def format_args(args: dict[str, Any] | None) -> str:
    """Render a tool-call's arg dict compactly (mirrors the CLI helper)."""
    if not args:
        return ""
    parts: list[str] = []
    for k, v in args.items():
        if v is None:
            continue
        parts.append(f"{k}={v!r}" if isinstance(v, str) else f"{k}={v}")
    return ", ".join(parts)


def truncate(text: str, limit: int = TOOL_PREVIEW_CHARS) -> str:
    """Trim ``text`` to ``limit`` chars with an ellipsis suffix."""
    text = text.strip()
    return text if len(text) <= limit else text[:limit].rstrip() + " …"


def chunk_to_reasoning_steps(
    chunk: dict[str, dict[str, Any]],
) -> tuple[list[ReasoningStep], str | None, AnswerKind]:
    """Translate one LangGraph ``stream_mode="updates"`` chunk into renderable parts.

    Returns ``(steps, final_answer, kind)``:

    - ``steps``        — reasoning rows to append to the expander (may be empty).
    - ``final_answer`` — terminal user-facing answer iff this chunk produced one
                         (a ``decline``/``fallback`` node, or an ``agent`` message
                         with content and no tool calls); otherwise ``None``.
    - ``kind``         — ``"decline"`` / ``"fallback"`` / ``"normal"`` for styling.
    """
    steps: list[ReasoningStep] = []
    final: str | None = None
    kind: AnswerKind = "normal"

    for node_name, update in chunk.items():
        if node_name == "router":
            route = update.get("route")
            if route:
                steps.append(ReasoningStep(kind="router", text=f"router → **{route}**"))
        elif node_name == "agent":
            for msg in update.get("messages", []):
                if not isinstance(msg, AIMessage):
                    continue
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        args = format_args(tc.get("args"))
                        steps.append(ReasoningStep(kind="tool_call", text=f"→ `{tc['name']}({args})`"))
                elif msg.content:
                    final = str(msg.content)
        elif node_name == "tools":
            for msg in update.get("messages", []):
                if not isinstance(msg, ToolMessage):
                    continue
                preview = truncate(str(msg.content))
                steps.append(ReasoningStep(kind="tool_result", text=f"← `{msg.name}` → {preview}"))
        elif node_name in {"decline", "fallback", "profile_recall"}:
            for msg in update.get("messages", []):
                if isinstance(msg, AIMessage) and msg.content:
                    final = str(msg.content)
                    kind = (
                        "decline"
                        if node_name == "decline"
                        else "fallback"
                        if node_name == "fallback"
                        else "normal"
                    )
        elif node_name == "recommender":
            # Bonus B. The recommender emits one of:
            # - an AIMessage suggestion/refinement/rejection (terminal for the
            #   turn) — render it as the assistant's final answer styled as
            #   "recommend".
            # - a synthetic HumanMessage carrying the resolved query that the
            #   agent will execute next. We record it in the reasoning trace
            #   so the user can see what the agent was dispatched to do.
            pending = update.get("pending_query")
            for msg in update.get("messages", []):
                if isinstance(msg, HumanMessage):
                    steps.append(
                        ReasoningStep(
                            kind="recommender",
                            text=f"recommender → dispatching `{str(msg.content).strip()}`",
                        )
                    )
                elif isinstance(msg, AIMessage) and msg.content:
                    final = str(msg.content)
                    kind = "recommend"
                    label = "awaiting confirmation" if pending else "suggestion dropped"
                    steps.append(
                        ReasoningStep(
                            kind="recommender",
                            text=f"recommender → **{label}**",
                        )
                    )

    return steps, final, kind


def messages_to_turns(messages: Iterable[BaseMessage]) -> list[RenderedTurn]:
    """Reconstruct rendered turns from a persisted message history.

    The checkpointer only retains ``BaseMessage`` objects; node identities and
    router decisions are gone, so reconstructed turns surface only tool-call /
    tool-result reasoning rows. The terminal ``AIMessage`` with content becomes
    the turn's ``answer``.

    Bonus B confirmation flow: a confirm turn produces TWO consecutive
    ``HumanMessage`` objects (the user's "yes" followed by the synthetic
    resolved query that the recommender hands to the agent). Treat the
    second HumanMessage as a continuation of the same turn — record it as
    a "recommender dispatched" reasoning step rather than starting a new
    turn that would orphan the user's "yes" with no answer.
    """
    turns: list[RenderedTurn] = []
    current: RenderedTurn | None = None
    pending: dict[str, str] = {}

    for msg in messages:
        if isinstance(msg, HumanMessage):
            # Recommender confirmation: previous turn has no answer yet and
            # no tool activity — fold this synthetic HumanMessage into it.
            if current is not None and not current.answer and not current.steps:
                current.steps.append(
                    ReasoningStep(
                        kind="recommender",
                        text=f"recommender → dispatched `{str(msg.content).strip()}`",
                    )
                )
                pending = {}
                continue
            if current is not None:
                turns.append(current)
            current = RenderedTurn(user_query=str(msg.content))
            pending = {}
            continue
        if current is None:
            continue
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                args = format_args(tc.get("args"))
                current.steps.append(ReasoningStep(kind="tool_call", text=f"→ `{tc['name']}({args})`"))
                tc_id = tc.get("id")
                if tc_id:
                    pending[tc_id] = tc["name"]
        elif isinstance(msg, ToolMessage):
            preview = truncate(str(msg.content))
            name = pending.pop(msg.tool_call_id, "") or msg.name or "tool"
            current.steps.append(ReasoningStep(kind="tool_result", text=f"← `{name}` → {preview}"))
        elif isinstance(msg, AIMessage) and msg.content:
            current.answer = str(msg.content)

    if current is not None:
        turns.append(current)
    return turns
