"""Live integration tests for Task 2b user profile.

End-to-end: in session A the user introduces themselves; in session B
(same ``--user``, different ``thread_id``) the user asks "what do you
remember about me?" — the agent must answer from the persisted profile.

These tests exercise:
- The cheap regex gate (the introduction message must trip it).
- The router LLM's structured-output call against ``UserProfile``.
- The atomic JSON write to ``profiles/<user_id>.json``.
- The agent's prompt-injection path: the persisted profile is rendered
  into the system prompt and the agent answers WITHOUT calling any tool.

Marked ``integration`` because they call live Nebius LLMs. Run with::

    uv run python -m pytest -m integration tests/test_profile_integration.py
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage

import cs_agent.memory.profile as profile_mod
from cs_agent.agent.graph import build_graph
from cs_agent.memory.checkpoint import get_checkpointer
from cs_agent.memory.profile import load_profile, profile_path

pytestmark = pytest.mark.integration


@pytest.fixture
def isolated_state(tmp_path, monkeypatch):
    """Redirect the profile dir AND the checkpoint file to a tmp dir.

    Yields a dict ``{"profiles_dir": ..., "checkpoint": ...}`` for tests that
    want to inspect the underlying files.
    """
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir()
    monkeypatch.setattr(profile_mod, "PROFILES_DIR", profiles_dir)
    return {
        "profiles_dir": profiles_dir,
        "checkpoint": tmp_path / "ckpt.sqlite",
    }


def _final_ai_answer(messages) -> str:
    """Return the last user-facing AI answer (no tool-call placeholder)."""
    for m in reversed(messages):
        if isinstance(m, AIMessage) and m.content and not m.tool_calls:
            return str(m.content)
    return ""


def test_personal_info_turn_writes_profile_to_disk(isolated_state):
    """A single 'my name is Ofir' turn must materialise ``profiles/ofir.json``
    on disk with ``name == 'Ofir'``."""
    cp = get_checkpointer(isolated_state["checkpoint"])
    g = build_graph(checkpointer=cp)

    g.invoke(
        {
            "messages": [HumanMessage("Hi, my name is Ofir")],
            "iterations": 0,
            "user_id": "ofir",
            "route": None,
        },
        config={"configurable": {"thread_id": "intro"}},
    )

    p = profile_path("ofir")
    assert p.exists(), f"expected profile file at {p}"

    profile = load_profile("ofir")
    assert profile.has_facts(), "profile should have at least one fact populated"
    # Pin the most important field strictly.
    assert profile.name and "ofir" in profile.name.lower(), (
        f"expected name to contain 'Ofir', got {profile.name!r}"
    )


def test_profile_recall_across_sessions(isolated_state):
    """Cross-session recall: introduce in session A, ask 'what do you
    remember' in session B. The answer must contain 'Ofir'.

    Both sessions share the SAME checkpoint file (mirrors what happens when
    the CLI is launched twice with different ``--session`` values but the
    same ``--user``); the profile is keyed by user_id, so session B sees
    session A's facts.
    """
    cp = get_checkpointer(isolated_state["checkpoint"])
    g = build_graph(checkpointer=cp)

    # Session A: introduce.
    g.invoke(
        {
            "messages": [HumanMessage("Hi, my name is Ofir")],
            "iterations": 0,
            "user_id": "ofir",
            "route": None,
        },
        config={"configurable": {"thread_id": "session-a"}},
    )

    profile = load_profile("ofir")
    assert profile.has_facts(), (
        "profile must be populated after the introduction turn — otherwise "
        "the recall test below isn't testing recall, it's testing a coincidence"
    )

    # Session B: ask for recall. Different thread_id => fresh conversation
    # history, so the only way the agent can know the name is via the
    # injected profile block in the system prompt.
    out = g.invoke(
        {
            "messages": [HumanMessage("What do you remember about me?")],
            "iterations": 0,
            "user_id": "ofir",
            "route": None,
        },
        config={"configurable": {"thread_id": "session-b"}},
    )

    answer = _final_ai_answer(out["messages"])
    assert answer, f"expected an AI answer; got messages: {[type(m).__name__ for m in out['messages']]}"
    assert "ofir" in answer.lower(), (
        f"expected answer to contain 'Ofir' (recall from profile); got: {answer[:300]!r}"
    )


def test_dataset_question_does_not_touch_profile(isolated_state):
    """A pure dataset question (gate misses) must NOT create a profile file.

    This is the cost-of-Q&A guarantee: dataset turns don't pay the
    profile-update LLM round-trip.
    """
    cp = get_checkpointer(isolated_state["checkpoint"])
    g = build_graph(checkpointer=cp)

    g.invoke(
        {
            "messages": [HumanMessage("How many refund requests did we get?")],
            "iterations": 0,
            "user_id": "anon",
            "route": None,
        },
        config={"configurable": {"thread_id": "qa-only"}},
    )

    assert not profile_path("anon").exists(), (
        "profile file should NOT exist after a non-personal turn — "
        "this would mean the gate is firing too eagerly and burning tokens"
    )
