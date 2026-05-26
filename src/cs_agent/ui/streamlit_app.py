"""Streamlit chat UI for the customer-service data-analyst agent (Bonus A).

Run with::

    uv run cs-agent-ui
    # or
    uv run streamlit run src/cs_agent/ui/streamlit_app.py

The sidebar exposes a **session id** and a **user id**. Episodic memory is
scoped to the pair (``make_thread_id(user, session)``); the profile is keyed
by user alone (Task 2b). The app reuses the same compiled graph + ``SqliteSaver``
checkpointer that the CLI uses, so a conversation started in one transport
resumes in the other when both ids match.

Design notes
------------
- The compiled graph and checkpointer are created exactly once per process
  via ``@st.cache_resource``; Streamlit's full-script-rerun model would
  otherwise re-open the SQLite connection on every keystroke.
- Per-turn invocation matches the contract in ``cs_agent.cli``: pass only the
  new ``HumanMessage`` plus an explicit ``iterations=0, route=None`` reset.
  The checkpointer owns the running history.
- For *live* turns we stream ``stream_mode="updates"`` and render each chunk
  inside an ``st.status("reasoning…")`` expander followed by the final
  answer bubble. For *resumed* history (after a session switch) we only have
  the persisted ``BaseMessage`` list available, so the reasoning view is
  reconstructed from ``AIMessage.tool_calls`` + ``ToolMessage`` pairs only —
  router decisions are not retained in the checkpoint.
- Pure-Python helpers (``chunk_to_reasoning_steps``, ``messages_to_turns``,
  ``RenderedTurn``, …) live in ``cs_agent.ui.rendering`` so they can be
  imported and unit-tested without a Streamlit runtime
  (see ``scripts/verify_bonus_a.py``).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import streamlit as st
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from cs_agent.agent.graph import build_graph
from cs_agent.agent.state import GraphState
from cs_agent.memory.checkpoint import get_checkpointer, make_thread_id
from cs_agent.ui.rendering import (
    AnswerKind,
    RenderedTurn,
    chunk_to_reasoning_steps,
    messages_to_turns,
)

logger = logging.getLogger(__name__)

PAGE_TITLE = "CS Agent — Bitext"


@st.cache_resource(show_spinner=False)
def _build_cached_graph():
    """Compile the graph once per process with a persistent SqliteSaver.

    ``check_same_thread=False`` is already set inside ``get_checkpointer``,
    and ``SqliteSaver`` holds its own lock — safe across Streamlit's reruns.
    """
    checkpointer = get_checkpointer()
    return build_graph(checkpointer=checkpointer)


def _read_snapshot(graph, config: RunnableConfig) -> tuple[list[RenderedTurn], int]:
    """Read the persisted state once; return ``(turns, prior_human_count)``."""
    try:
        snapshot = graph.get_state(config)
    except Exception:  # noqa: BLE001 — sidebar banner must never crash the app
        return [], 0
    if snapshot is None:
        return [], 0
    messages = (snapshot.values or {}).get("messages") or []
    turns = messages_to_turns(messages)
    n_human = sum(1 for m in messages if isinstance(m, HumanMessage))
    return turns, n_human


def _emit_answer(target: Any, text: str, kind: AnswerKind) -> None:
    """Render the final answer; ``target`` may be ``st`` or an ``st.empty()``."""
    if kind == "decline":
        target.warning(text)
    elif kind == "fallback":
        target.info(text)
    elif kind == "recommend":
        # Cyan-ish callout for Bonus B suggestions / drops, so the user
        # immediately sees this is a recommendation awaiting their reply
        # rather than a finished answer.
        target.info(text)
    else:
        target.markdown(text)


def _render_turn(turn: RenderedTurn) -> None:
    """Render one historical (or just-completed) turn as chat bubbles."""
    with st.chat_message("user"):
        st.markdown(turn.user_query)
    with st.chat_message("assistant"):
        if turn.steps:
            with st.expander("reasoning", expanded=False):
                for step in turn.steps:
                    st.markdown(f"- {step.text}")
        if turn.answer:
            _emit_answer(st, turn.answer, turn.answer_kind)


def _stream_turn(
    graph,
    config: RunnableConfig,
    user_id: str,
    question: str,
) -> RenderedTurn:
    """Invoke the graph for one new turn and render it live."""
    initial: GraphState = {
        "messages": [HumanMessage(question)],
        "iterations": 0,
        "user_id": user_id,
        "route": None,
    }
    turn = RenderedTurn(user_query=question)

    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        status = st.status("reasoning…", expanded=True)
        answer_slot = st.empty()
        try:
            for chunk in graph.stream(initial, config=config, stream_mode="updates"):
                steps, final, kind = chunk_to_reasoning_steps(chunk)
                turn.steps.extend(steps)
                for step in steps:
                    status.markdown(f"- {step.text}")
                if final is not None:
                    turn.answer = final
                    turn.answer_kind = kind
        except Exception as exc:  # noqa: BLE001 — one bad turn must not kill the UI
            logger.exception("agent turn failed")
            status.update(label="error", state="error", expanded=True)
            answer_slot.error(f"agent error: {exc!r}")
            return turn

        status.update(label="reasoning", state="complete", expanded=False)
        if turn.answer:
            _emit_answer(answer_slot, turn.answer, turn.answer_kind)
    return turn


def _sidebar(graph) -> tuple[str, str, bool]:
    """Render the sidebar; return ``(session_id, user_id, reload_clicked)``."""
    with st.sidebar:
        st.header("session")
        session_id = st.text_input(
            "session id",
            value=st.session_state.get("session_id_input", "default"),
            help=(
                "Conversation name within this user. Combined with user id to "
                "scope chat history — changing it switches chats for the same user."
            ),
            key="session_id_input",
        )
        user_id = st.text_input(
            "user id",
            value=st.session_state.get("user_id_input", "anon"),
            help=(
                "Profile key (Task 2b) and half of the episodic-memory scope. "
                "Changing user id starts a fresh chat for that user even if the "
                "session name stays the same. One user can own many sessions."
            ),
            key="user_id_input",
        )
        reload_clicked = st.button("Switch / reload session", use_container_width=True)

        thread_id = make_thread_id(user_id, session_id)
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
        _, prior = _read_snapshot(graph, config)
        if prior:
            st.success(f"resumed **{user_id}** / **{session_id}** ({prior} prior turns)")
        else:
            st.info(f"new chat for **{user_id}** / **{session_id}** (no prior turns)")
        st.caption("Conversation state persists in `checkpoints.sqlite`.")
    return session_id, user_id, reload_clicked


def render_app() -> None:
    """Top-level Streamlit page renderer."""
    st.set_page_config(page_title=PAGE_TITLE, layout="wide")
    st.title("Customer Service Data Analyst")
    st.caption("LangGraph ReAct agent over the Bitext customer-support dataset.")

    graph = _build_cached_graph()
    session_id, user_id, reload_clicked = _sidebar(graph)
    thread_id = make_thread_id(user_id, session_id)
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

    # Reload persisted history when the user/session pair changes, the user
    # clicks reload, or we've never rendered anything yet.
    active_thread = st.session_state.get("active_thread")
    if reload_clicked or active_thread != thread_id or "history" not in st.session_state:
        history, _ = _read_snapshot(graph, config)
        st.session_state.history = history
        st.session_state.active_thread = thread_id

    for turn in st.session_state.history:
        _render_turn(turn)

    question = st.chat_input("Ask about the Bitext dataset…")
    if not question:
        return

    new_turn = _stream_turn(graph, config, user_id, question)
    st.session_state.history.append(new_turn)


def main() -> int:
    """Console-script entry point.

    Spawns ``streamlit run`` on this file so ``uv run cs-agent-ui`` works
    without the user having to remember the source path.
    """
    from streamlit.web import cli as stcli

    app_path = str(Path(__file__).resolve())
    sys.argv = ["streamlit", "run", app_path]
    return stcli.main()


if __name__ == "__main__":
    render_app()
