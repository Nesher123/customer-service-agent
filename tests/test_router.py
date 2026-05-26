"""Unit tests for the router and its fallback behaviour.

These tests use ``monkeypatch`` to replace the router and agent LLM factories
with mocks, so they don't make real Nebius calls and run in milliseconds.

What's covered:
- ``_classify_with`` returns a typed ``RouterDecision`` when the LLM cooperates.
- ``classify`` retries with the agent model when the primary router fails.
- ``router_node`` defaults to ``'structured'`` (NOT ``'out_of_scope'``) when
  both LLMs fail — the design choice that prevents legitimate questions from
  being declined during transient Nebius outages.
- ``router_node`` defaults to ``'structured'`` when there is no HumanMessage.
- ``route_from_router`` maps OOS → decline and everything else → agent.

Live integration of the router is exercised in
``tests/test_agent_integration.py::test_verifier_cases``.
"""

from __future__ import annotations

from typing import cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage

from cs_agent.agent import router as router_mod
from cs_agent.agent.router import (
    RouterDecision,
    _classify_with,
    classify,
    route_from_router,
    router_node,
)
from cs_agent.agent.state import GraphState


class _FakeStructuredLLM:
    """Stand-in for ``llm.with_structured_output(RouterDecision)`` — returns
    a fixed RouterDecision (or raises a configured exception) when ``invoke``
    is called."""

    def __init__(self, decision: RouterDecision | None = None, error: Exception | None = None):
        self._decision = decision
        self._error = error
        self.invoke_calls: list = []

    def invoke(self, messages):
        self.invoke_calls.append(messages)
        if self._error is not None:
            raise self._error
        return self._decision


class _FakeLLM:
    """Stand-in for a ChatOpenAI: just records ``with_structured_output`` calls
    and returns a pre-configured ``_FakeStructuredLLM``."""

    def __init__(self, structured: _FakeStructuredLLM):
        self._structured = structured
        self.with_structured_output_calls: list = []

    def with_structured_output(self, schema):
        self.with_structured_output_calls.append(schema)
        return self._structured


# ---------------------------------------------------------------------------
# _classify_with
# ---------------------------------------------------------------------------


def test_classify_with_returns_decision():
    decision = RouterDecision(route="structured", reason="x")
    llm = _FakeLLM(_FakeStructuredLLM(decision=decision))

    result = _classify_with(cast(BaseChatModel, llm), "How many refund requests?")

    assert isinstance(result, RouterDecision)
    assert result.route == "structured"
    assert llm.with_structured_output_calls == [RouterDecision]


def test_classify_with_propagates_llm_errors():
    """If the underlying LLM raises, ``_classify_with`` should re-raise so the
    caller (``classify``) can decide whether to fall back."""
    llm = _FakeLLM(_FakeStructuredLLM(error=RuntimeError("nebius timeout")))

    try:
        _classify_with(cast(BaseChatModel, llm), "anything")
    except RuntimeError as exc:
        assert "nebius timeout" in str(exc)
    else:
        raise AssertionError("expected RuntimeError to propagate")


# ---------------------------------------------------------------------------
# classify (router-LLM → agent-LLM fallback)
# ---------------------------------------------------------------------------


def test_classify_uses_router_llm_when_healthy(monkeypatch):
    primary_decision = RouterDecision(route="unstructured", reason="primary")
    primary_llm = _FakeLLM(_FakeStructuredLLM(decision=primary_decision))
    backup_decision = RouterDecision(route="structured", reason="should-not-be-used")
    backup_llm = _FakeLLM(_FakeStructuredLLM(decision=backup_decision))

    monkeypatch.setattr(router_mod, "get_router_llm", lambda: primary_llm)
    monkeypatch.setattr(router_mod, "get_agent_llm", lambda: backup_llm)

    result = classify("Summarize FEEDBACK")

    assert result.route == "unstructured"
    assert primary_llm.with_structured_output_calls == [RouterDecision]
    # Backup must NOT have been touched on the happy path.
    assert backup_llm.with_structured_output_calls == []


def test_classify_falls_back_to_agent_llm_when_router_fails(monkeypatch):
    primary_llm = _FakeLLM(_FakeStructuredLLM(error=RuntimeError("router 404")))
    backup_decision = RouterDecision(route="out_of_scope", reason="from agent llm")
    backup_llm = _FakeLLM(_FakeStructuredLLM(decision=backup_decision))

    monkeypatch.setattr(router_mod, "get_router_llm", lambda: primary_llm)
    monkeypatch.setattr(router_mod, "get_agent_llm", lambda: backup_llm)

    result = classify("Who is the president of France?")

    assert result.route == "out_of_scope"
    assert primary_llm.with_structured_output_calls == [RouterDecision]
    assert backup_llm.with_structured_output_calls == [RouterDecision]


def test_classify_raises_when_both_models_fail(monkeypatch):
    primary_llm = _FakeLLM(_FakeStructuredLLM(error=RuntimeError("router down")))
    backup_llm = _FakeLLM(_FakeStructuredLLM(error=RuntimeError("agent down")))

    monkeypatch.setattr(router_mod, "get_router_llm", lambda: primary_llm)
    monkeypatch.setattr(router_mod, "get_agent_llm", lambda: backup_llm)

    try:
        classify("anything")
    except RuntimeError as exc:
        assert "agent down" in str(exc)
    else:
        raise AssertionError("expected the second RuntimeError to surface")


# ---------------------------------------------------------------------------
# router_node — final last-resort defaults
# ---------------------------------------------------------------------------


def test_router_node_returns_classified_route(monkeypatch):
    primary_decision = RouterDecision(route="unstructured", reason="ok")
    primary_llm = _FakeLLM(_FakeStructuredLLM(decision=primary_decision))
    monkeypatch.setattr(router_mod, "get_router_llm", lambda: primary_llm)
    monkeypatch.setattr(router_mod, "get_agent_llm", lambda: primary_llm)

    state: GraphState = {"messages": [HumanMessage("Summarize complaints")]}
    update = router_node(state)
    assert update == {"route": "unstructured"}


def test_router_node_defaults_to_structured_when_both_fail(monkeypatch):
    """The crucial UX choice: if every model is down, route to structured so
    the question reaches the agent loop instead of being unfairly declined."""
    failing = _FakeLLM(_FakeStructuredLLM(error=RuntimeError("nebius down")))
    monkeypatch.setattr(router_mod, "get_router_llm", lambda: failing)
    monkeypatch.setattr(router_mod, "get_agent_llm", lambda: failing)

    state: GraphState = {"messages": [HumanMessage("How many refund requests?")]}
    update = router_node(state)

    assert update == {"route": "structured"}


def test_router_node_defaults_to_structured_when_no_human_message():
    state: GraphState = {"messages": [AIMessage("only AI messages here")]}
    assert router_node(state) == {"route": "structured"}


def test_router_node_handles_empty_messages():
    assert router_node({"messages": []}) == {"route": "structured"}


# ---------------------------------------------------------------------------
# route_from_router (the conditional edge)
# ---------------------------------------------------------------------------


def test_route_from_router_oos_goes_to_decline():
    assert route_from_router({"route": "out_of_scope"}) == "decline"


def test_route_from_router_structured_goes_to_agent():
    assert route_from_router({"route": "structured"}) == "agent"


def test_route_from_router_unstructured_goes_to_agent():
    assert route_from_router({"route": "unstructured"}) == "agent"


def test_route_from_router_missing_route_goes_to_agent():
    """A missing/None route should NOT be treated as out_of_scope."""
    assert route_from_router({}) == "agent"
    assert route_from_router({"route": None}) == "agent"


def test_router_node_short_circuits_profile_recall_without_llm(monkeypatch):
    def _fail_llm():
        raise AssertionError("router LLM must not run for profile recall")

    monkeypatch.setattr(router_mod, "get_router_llm", _fail_llm)
    monkeypatch.setattr(router_mod, "get_agent_llm", _fail_llm)

    state: GraphState = {"messages": [HumanMessage("what do u know about me?")]}
    assert router_node(state) == {"route": "profile_recall"}


def test_route_from_router_profile_recall_goes_to_profile_recall_node():
    assert route_from_router({"route": "profile_recall"}) == "profile_recall"
