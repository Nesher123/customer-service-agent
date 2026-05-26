"""Fast unit tests for the per-user profile (Task 2b).

What's covered:
- ``is_personal_info_bearing`` — the cheap regex gate that decides whether
  to invoke the LLM at all.
- ``UserProfile`` schema: empty defaults, ``has_facts`` semantics,
  ``render_for_prompt`` formatting.
- ``load_profile`` / ``save_profile`` — JSON roundtrip + missing-file +
  corrupt-file handling.
- ``profile_update_node`` — verifies the gate gates (no LLM call on
  non-personal turns) and that the LLM result is persisted on personal
  turns. The LLM is mocked, so this stays fast and deterministic.

Live cross-session recall is exercised by
``tests/test_profile_integration.py::test_profile_recall_across_sessions``.
"""

from __future__ import annotations

import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import cs_agent.agent.nodes as nodes_mod
import cs_agent.memory.profile as profile_mod
from cs_agent.agent.graph import build_graph
from cs_agent.agent.nodes import profile_recall_node, profile_update_node
from cs_agent.agent.state import GraphState
from cs_agent.memory.checkpoint import get_checkpointer
from cs_agent.memory.profile import (
    UserProfile,
    is_personal_info_bearing,
    is_profile_recall_question,
    load_profile,
    profile_path,
    save_profile,
)


@pytest.fixture
def isolated_profiles(tmp_path, monkeypatch):
    """Redirect ``PROFILES_DIR`` to a tmp dir for the duration of one test."""
    monkeypatch.setattr(profile_mod, "PROFILES_DIR", tmp_path)
    return tmp_path


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "msg",
    [
        "Hi, my name is Ofir",
        "Call me Ofir, please",
        "I prefer concise answers",
        "I love long detailed examples",
        "I hate verbose responses",
        "Remember that I work in data engineering",
        "Note that I'm only interested in REFUND",
        "I work as a data engineer",
        "My role is staff engineer",
        "I'm a developer",
    ],
)
def test_gate_fires_on_personal_info(msg):
    assert is_personal_info_bearing(msg), f"gate missed: {msg!r}"


@pytest.mark.parametrize(
    "msg",
    [
        "How many refund requests?",
        "Summarize the FEEDBACK category",
        "Show me 3 examples of REFUND",
        "What is the distribution of intents in ACCOUNT?",
        "List the categories",
        "hi",
        "thanks",
        "what can you do?",
    ],
)
def test_gate_does_not_fire_on_dataset_questions(msg):
    assert not is_personal_info_bearing(msg), f"gate falsely fired on: {msg!r}"


def test_gate_handles_empty_and_non_string_inputs():
    assert not is_personal_info_bearing("")
    assert not is_personal_info_bearing("   ")
    # Defensive: non-strings should return False rather than raise.
    assert not is_personal_info_bearing(None)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "msg",
    [
        "What do you remember about me?",
        "what do u know about me?",
        "waht do u know about me?",
        "Do you know my name?",
        "Remind me what I told you",
        "What facts do you have about me?",
    ],
)
def test_recall_gate_fires_on_meta_questions(msg):
    assert is_profile_recall_question(msg), f"recall gate missed: {msg!r}"


@pytest.mark.parametrize(
    "msg",
    [
        "How many refund requests?",
        "What do customers know about password recovery?",
        "Summarize the FEEDBACK category",
        "my name is Ofir",
    ],
)
def test_recall_gate_does_not_fire_on_dataset_or_intro(msg):
    assert not is_profile_recall_question(msg), f"recall gate falsely fired on: {msg!r}"


# ---------------------------------------------------------------------------
# Schema + render
# ---------------------------------------------------------------------------


def test_empty_profile_has_no_facts_and_renders_explicit_empty():
    p = UserProfile(user_id="nobody")
    assert not p.has_facts()
    assert "no prior facts" in p.render_for_prompt().lower()


def test_render_includes_all_populated_fields():
    p = UserProfile(
        user_id="ofir",
        name="Ofir",
        role="data engineer",
        topics_of_interest=["refunds", "complaints"],
        preferences={"answer_length": "concise"},
        notable_facts=["uses pyspark daily"],
    )
    out = p.render_for_prompt()
    assert "Ofir" in out
    assert "data engineer" in out
    assert "refunds" in out and "complaints" in out
    assert "answer_length=concise" in out
    assert "uses pyspark daily" in out


def test_has_facts_single_field_is_enough():
    """Even just a name should count as a non-empty profile."""
    p = UserProfile(user_id="u", name="Ofir")
    assert p.has_facts()


def test_render_recall_answer_empty_profile():
    answer = UserProfile(user_id="u").render_recall_answer()
    assert "don't have any prior facts" in answer.lower()


def test_render_recall_answer_includes_name_and_role():
    answer = UserProfile(
        user_id="ofir",
        name="Ofir",
        role="data engineer",
        preferences={"answer_length": "concise"},
    ).render_recall_answer()
    assert "Ofir" in answer
    assert "data engineer" in answer
    assert "concise" in answer


def test_profile_recall_node_reads_disk_without_llm(isolated_profiles):
    save_profile(
        UserProfile(
            user_id="ofir",
            name="Ofir",
            role="data engineer",
        )
    )
    state: GraphState = {
        "messages": [HumanMessage("what do u know about me?")],
        "user_id": "ofir",
    }
    out = profile_recall_node(state)
    answer = out["messages"][0].content
    assert "Ofir" in answer
    assert "data engineer" in answer


def test_graph_profile_recall_end_to_end_offline(isolated_profiles, tmp_path):
    """Full graph path for recall questions must not call tools or LLMs."""
    save_profile(
        UserProfile(
            user_id="ofir",
            name="Ofir",
            role="data engineer",
        )
    )
    graph = build_graph(checkpointer=get_checkpointer(tmp_path / "ckpt.sqlite"))
    out = graph.invoke(
        {
            "messages": [HumanMessage("what do u know about me?")],
            "iterations": 0,
            "user_id": "ofir",
            "route": None,
        },
        config={"configurable": {"thread_id": "offline-recall"}},
    )
    answer = next(
        str(m.content)
        for m in reversed(out["messages"])
        if isinstance(m, AIMessage) and m.content and not m.tool_calls
    )
    assert "Ofir" in answer
    assert not any(isinstance(m, ToolMessage) for m in out["messages"])


# ---------------------------------------------------------------------------
# Load / save
# ---------------------------------------------------------------------------


def test_load_profile_missing_returns_empty(isolated_profiles):
    p = load_profile("nobody")
    assert p.user_id == "nobody"
    assert p.name is None
    assert p.notable_facts == []
    assert not p.has_facts()


def test_save_then_load_roundtrip(isolated_profiles):
    save_profile(
        UserProfile(
            user_id="u1",
            name="Ofir",
            topics_of_interest=["refunds"],
            notable_facts=["a"],
        )
    )
    p = load_profile("u1")
    assert p.name == "Ofir"
    assert p.topics_of_interest == ["refunds"]
    assert "a" in p.notable_facts


def test_save_writes_atomic_no_leftover_tmp(isolated_profiles):
    save_profile(UserProfile(user_id="atomic", name="X"))
    final = profile_path("atomic")
    assert final.exists()
    # The atomic-write helper renames tmp -> final, so no .tmp should linger.
    leftover = list(isolated_profiles.glob("*.tmp"))
    assert leftover == [], f"unexpected leftover tmp files: {leftover}"


def test_load_profile_handles_corrupt_json(isolated_profiles):
    """A bad JSON file must not blow up the agent."""
    p = profile_path("broken")
    p.write_text("{this is not json", encoding="utf-8")

    profile = load_profile("broken")
    assert profile.user_id == "broken"
    assert not profile.has_facts()


# ---------------------------------------------------------------------------
# profile_update_node
# ---------------------------------------------------------------------------


class _FakeStructuredLLM:
    def __init__(self, decision: UserProfile | dict | None = None, error: Exception | None = None):
        self._decision = decision
        self._error = error
        self.invoke_calls: list = []

    def invoke(self, messages):
        self.invoke_calls.append(messages)
        if self._error is not None:
            raise self._error
        return self._decision


class _FakeLLM:
    def __init__(self, structured: _FakeStructuredLLM):
        self._structured = structured
        self.with_structured_output_calls: list = []

    def with_structured_output(self, schema):
        self.with_structured_output_calls.append(schema)
        return self._structured


def test_update_node_skips_when_gate_misses(isolated_profiles, monkeypatch):
    """A non-personal turn must NOT invoke the router LLM at all."""
    structured = _FakeStructuredLLM(error=AssertionError("LLM should not be called"))
    fake_llm = _FakeLLM(structured)
    monkeypatch.setattr(nodes_mod, "get_router_llm", lambda: fake_llm)

    state: GraphState = {
        "messages": [HumanMessage("How many refund requests?")],
        "user_id": "u1",
    }
    result = profile_update_node(state)

    assert result == {}
    assert structured.invoke_calls == []  # gate short-circuited
    assert not profile_path("u1").exists()  # nothing persisted


def test_update_node_skips_when_no_human_message(isolated_profiles, monkeypatch):
    """If there's no HumanMessage in state, the node must early-return."""
    structured = _FakeStructuredLLM(error=AssertionError("LLM should not be called"))
    monkeypatch.setattr(nodes_mod, "get_router_llm", lambda: _FakeLLM(structured))

    assert profile_update_node({"messages": [], "user_id": "u1"}) == {}
    assert structured.invoke_calls == []


def test_update_node_invokes_llm_and_persists(isolated_profiles, monkeypatch):
    """A personal turn must (1) call the LLM, (2) save the returned profile."""
    returned = UserProfile(user_id="u1", name="Ofir", notable_facts=["likes refunds"])
    structured = _FakeStructuredLLM(decision=returned)
    fake_llm = _FakeLLM(structured)
    monkeypatch.setattr(nodes_mod, "get_router_llm", lambda: fake_llm)

    state: GraphState = {
        "messages": [
            HumanMessage("Hi, my name is Ofir"),
            AIMessage("Hi Ofir, how can I help with the dataset?"),
        ],
        "user_id": "u1",
    }
    result = profile_update_node(state)

    assert result == {}
    assert fake_llm.with_structured_output_calls == [UserProfile]
    assert len(structured.invoke_calls) == 1
    # The payload sent to the LLM must be JSON containing both the current
    # profile and the latest turn.
    msgs = structured.invoke_calls[0]
    assert isinstance(msgs[1], HumanMessage)
    payload = json.loads(str(msgs[1].content))
    assert "current_profile" in payload and "latest_turn" in payload
    assert payload["latest_turn"]["user"] == "Hi, my name is Ofir"

    # Profile must be persisted with user_id pinned to the canonical value
    # AND last_updated stamped (i.e. the node actually ran the persist path).
    persisted = load_profile("u1")
    assert persisted.user_id == "u1"
    assert persisted.name == "Ofir"
    assert persisted.last_updated is not None


def test_update_node_pins_user_id_even_if_llm_returns_wrong_one(isolated_profiles, monkeypatch):
    """LLMs sometimes echo a different user_id back. The node must overwrite
    it with the canonical ``state['user_id']`` so the file is saved at the
    right path."""
    returned = UserProfile(user_id="DIFFERENT", name="Ofir")
    structured = _FakeStructuredLLM(decision=returned)
    monkeypatch.setattr(nodes_mod, "get_router_llm", lambda: _FakeLLM(structured))

    state: GraphState = {
        "messages": [HumanMessage("My name is Ofir")],
        "user_id": "u1",
    }
    profile_update_node(state)

    assert profile_path("u1").exists()
    assert not profile_path("DIFFERENT").exists()
    assert load_profile("u1").user_id == "u1"


def test_update_node_swallows_llm_errors_and_leaves_profile_untouched(isolated_profiles, monkeypatch):
    """If the LLM raises, the agent's final answer (already emitted upstream)
    must not be undone — we just skip the profile update silently."""
    save_profile(UserProfile(user_id="u1", name="Ofir"))  # pre-existing profile
    structured = _FakeStructuredLLM(error=RuntimeError("nebius timeout"))
    monkeypatch.setattr(nodes_mod, "get_router_llm", lambda: _FakeLLM(structured))

    state: GraphState = {
        "messages": [HumanMessage("My name is Ofira (typo)")],
        "user_id": "u1",
    }
    result = profile_update_node(state)

    assert result == {}
    # Pre-existing profile is preserved.
    assert load_profile("u1").name == "Ofir"


def test_update_node_accepts_dict_response_from_llm(isolated_profiles, monkeypatch):
    """Some structured-output backends return a dict instead of the model
    instance. The node must validate it transparently."""
    returned_dict = {
        "user_id": "u1",
        "name": "Ofir",
        "topics_of_interest": ["refunds"],
        "preferences": {},
        "notable_facts": [],
    }
    structured = _FakeStructuredLLM(decision=returned_dict)
    monkeypatch.setattr(nodes_mod, "get_router_llm", lambda: _FakeLLM(structured))

    profile_update_node(
        {
            "messages": [HumanMessage("My name is Ofir, I like refunds")],
            "user_id": "u1",
        }
    )

    p = load_profile("u1")
    assert p.name == "Ofir"
    assert "refunds" in p.topics_of_interest
