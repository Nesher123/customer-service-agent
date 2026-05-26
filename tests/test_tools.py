"""Unit tests for the catalog/filter/search tools.

We monkeypatch ``loader.get_df`` to return a tiny fixture DataFrame so the
tests are fast, deterministic, and don't depend on the HuggingFace download.

The LLM-backed ``summarize`` tool is exercised separately via integration
tests (it makes real Nebius calls).
"""

from __future__ import annotations

import pandas as pd
import pytest

from cs_agent.data import loader
from cs_agent.tools.catalog import get_distribution, list_categories, list_intents
from cs_agent.tools.filter import count_rows, get_examples, search_by_keyword
from cs_agent.tools.registry import DATA_TOOLS, TOOLS_BY_NAME


def _make_fixture_df() -> pd.DataFrame:
    """A 7-row toy version of the Bitext schema with deterministic values."""
    return pd.DataFrame(
        {
            "flags": ["B", "Q", "B", "P", "B", "B", "K"],
            "instruction": [
                "I want my money back",
                "track my refund please",
                "where is my refund",
                "create an account for me",
                "cancel my order",
                "where is my package",
                "I'm not happy with the service",
            ],
            "category": [
                "REFUND",
                "REFUND",
                "REFUND",
                "ACCOUNT",
                "ORDER",
                "DELIVERY",
                "FEEDBACK",
            ],
            "intent": [
                "get_refund",
                "track_refund",
                "track_refund",
                "create_account",
                "cancel_order",
                "track_order",
                "complaint",
            ],
            "response": [f"resp{i}" for i in range(7)],
        }
    )


@pytest.fixture(autouse=True)
def patch_get_df(monkeypatch):
    """Replace loader.get_df with our fixture for every tool test."""
    fixture = _make_fixture_df()
    monkeypatch.setattr(loader, "get_df", lambda: fixture)
    return fixture


def test_registry_is_complete():
    expected = {
        "list_categories",
        "list_intents",
        "get_distribution",
        "count_rows",
        "get_examples",
        "search_by_keyword",
        "summarize",
    }
    assert {t.name for t in DATA_TOOLS} == expected
    assert set(TOOLS_BY_NAME.keys()) == expected


def test_every_tool_has_description_and_args_schema():
    for t in DATA_TOOLS:
        assert t.description and len(t.description) > 30, f"{t.name} description too short"
        assert t.args_schema is not None, f"{t.name} missing args_schema"


def test_list_categories_returns_sorted_unique():
    result = list_categories.invoke({})
    assert result == sorted({"REFUND", "ACCOUNT", "ORDER", "DELIVERY", "FEEDBACK"})


def test_list_intents_no_filter_returns_all_unique():
    result = list_intents.invoke({})
    assert result == sorted(
        {"get_refund", "track_refund", "create_account", "cancel_order", "track_order", "complaint"}
    )


def test_list_intents_scoped_to_refund():
    result = list_intents.invoke({"category": "REFUND"})
    assert result == ["get_refund", "track_refund"]


def test_list_intents_is_case_insensitive():
    assert list_intents.invoke({"category": "refund"}) == list_intents.invoke({"category": "REFUND"})


def test_list_intents_unknown_category_returns_empty():
    assert list_intents.invoke({"category": "NOSUCH"}) == []


def test_get_distribution_by_category():
    result = get_distribution.invoke({"group_by": "category"})
    assert result == {"REFUND": 3, "ACCOUNT": 1, "ORDER": 1, "DELIVERY": 1, "FEEDBACK": 1}


def test_get_distribution_intent_within_refund():
    result = get_distribution.invoke({"group_by": "intent", "scope_category": "REFUND"})
    assert result == {"track_refund": 2, "get_refund": 1}


def test_count_rows_no_filters_counts_all():
    assert count_rows.invoke({}) == 7


def test_count_rows_by_category():
    assert count_rows.invoke({"category": "REFUND"}) == 3


def test_count_rows_by_intent():
    assert count_rows.invoke({"intent": "track_refund"}) == 2


def test_count_rows_by_category_and_keyword():
    # only one REFUND row mentions "money"
    assert count_rows.invoke({"category": "REFUND", "keyword": "money"}) == 1


def test_count_rows_filters_are_case_insensitive():
    assert count_rows.invoke({"category": "refund"}) == 3
    assert count_rows.invoke({"intent": "TRACK_REFUND"}) == 2


def test_count_rows_no_match_returns_zero():
    assert count_rows.invoke({"keyword": "nonexistent_phrase_xyz"}) == 0


def test_get_examples_default_columns_and_limit():
    result = get_examples.invoke({"category": "REFUND", "n": 2})
    assert len(result) == 2
    assert all(set(row.keys()) == {"category", "intent", "instruction"} for row in result)
    assert all(row["category"] == "REFUND" for row in result)


def test_get_examples_with_explicit_columns():
    result = get_examples.invoke({"intent": "complaint", "columns": ["instruction", "response"]})
    assert len(result) == 1
    assert set(result[0].keys()) == {"instruction", "response"}


def test_get_examples_caps_at_dataset_size():
    # only 3 REFUND rows, but caller asked for 10
    result = get_examples.invoke({"category": "REFUND", "n": 10})
    assert len(result) == 3


def test_get_examples_unknown_column_raises():
    with pytest.raises(ValueError, match="Unknown columns"):
        get_examples.invoke({"columns": ["nope"]})


def test_get_examples_no_match_returns_empty():
    assert get_examples.invoke({"category": "NOSUCH"}) == []


def test_search_by_keyword_case_insensitive():
    result = search_by_keyword.invoke({"keyword": "MONEY BACK"})
    assert len(result) == 1
    assert "money back" in result[0]["instruction"].lower()


def test_search_by_keyword_multiple_matches():
    result = search_by_keyword.invoke({"keyword": "refund", "n": 5})
    # 2 instructions contain 'refund'
    assert len(result) == 2
    assert all("refund" in r["instruction"].lower() for r in result)


def test_search_by_keyword_no_match_returns_empty():
    assert search_by_keyword.invoke({"keyword": "blockchain"}) == []


def test_llm_artifact_null_strings_are_treated_as_none():
    """Llama 3.x sometimes emits 'null' / 'None' strings instead of true null
    for optional fields. The LLMToolBase pre-validator should drop them."""
    # All three of these would have raised pydantic ValidationError before the
    # pre-validator was added.
    assert count_rows.invoke({"category": "REFUND", "intent": "null", "keyword": ""}) == 3
    assert count_rows.invoke({"category": "REFUND", "intent": "None"}) == 3


def test_llm_artifact_string_int_is_coerced():
    """Pydantic's default lax mode coerces '5' -> 5 for int fields."""
    result = get_examples.invoke({"category": "REFUND", "n": "2"})
    assert len(result) == 2


def test_llm_artifact_json_string_columns_is_parsed():
    """Llama may emit columns as a JSON-encoded string. LLMToolBase parses it."""
    result = get_examples.invoke({"category": "REFUND", "columns": '["category", "instruction"]', "n": 1})
    assert len(result) == 1
    assert set(result[0].keys()) == {"category", "instruction"}
