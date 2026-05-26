"""Integration tests: tools must agree with the live Bitext DataFrame.

These run against the real parquet cache (downloaded once on first invocation
of :func:`cs_agent.data.loader.get_df`). They are slower than the fixture
tests in ``test_tools.py`` but catch a different class of bug:

- Schema drift upstream (Bitext renames a category, drops an intent, ...).
- Normalisation regressions (parquet loaded but ``category`` not upper-cased).
- Filter logic that happens to work on the toy fixture but breaks at scale.

Run only the integration tests:    ``pytest -m integration``
Skip them in fast loops:           ``pytest -m "not integration"``
Run everything (default):          ``pytest``
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from cs_agent.data import loader
from cs_agent.tools.catalog import get_distribution, list_categories, list_intents
from cs_agent.tools.filter import count_rows, get_examples, search_by_keyword

if TYPE_CHECKING:
    import pandas as pd

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def real_df() -> pd.DataFrame:
    """The full live Bitext DataFrame, loaded once for the whole module."""
    return loader.get_df()


# ---------------------------------------------------------------------------
# Row counts must agree with len(df)
# ---------------------------------------------------------------------------


def test_count_rows_no_filters_equals_dataframe_length(real_df):
    assert count_rows.invoke({}) == len(real_df)


def test_per_category_counts_sum_to_total(real_df):
    """Every row belongs to exactly one category — the per-category counts must partition the dataset."""
    cats = list_categories.invoke({})
    total = sum(count_rows.invoke({"category": c}) for c in cats)
    assert total == len(real_df)


def test_per_intent_counts_sum_to_total(real_df):
    """Every row also belongs to exactly one intent."""
    intents = list_intents.invoke({})
    total = sum(count_rows.invoke({"intent": i}) for i in intents)
    assert total == len(real_df)


def test_intents_within_each_category_partition_that_category(real_df):
    """For every category, summing rows over its intents reproduces the category's row count."""
    for cat in list_categories.invoke({}):
        cat_total = count_rows.invoke({"category": cat})
        intent_sum = sum(
            count_rows.invoke({"category": cat, "intent": i}) for i in list_intents.invoke({"category": cat})
        )
        assert intent_sum == cat_total, (
            f"intent counts in {cat} sum to {intent_sum}, but the category itself has {cat_total} rows"
        )


# ---------------------------------------------------------------------------
# Catalog tools agree with raw pandas
# ---------------------------------------------------------------------------


def test_list_categories_matches_unique(real_df):
    assert list_categories.invoke({}) == sorted(real_df["category"].unique().tolist())


def test_list_intents_no_filter_matches_unique(real_df):
    assert list_intents.invoke({}) == sorted(real_df["intent"].unique().tolist())


def test_list_intents_per_category_matches_unique(real_df):
    for cat in list_categories.invoke({}):
        expected = sorted(real_df.loc[real_df["category"] == cat, "intent"].unique().tolist())
        assert list_intents.invoke({"category": cat}) == expected


def test_get_distribution_by_category_matches_value_counts(real_df):
    expected = real_df["category"].value_counts().to_dict()
    expected = {k: int(v) for k, v in expected.items()}
    assert get_distribution.invoke({"group_by": "category"}) == expected


def test_get_distribution_intent_within_category_matches_value_counts(real_df):
    """Spot-check distribution-by-intent for two well-known categories."""
    for cat in ("REFUND", "ACCOUNT"):
        if cat not in real_df["category"].unique():
            pytest.skip(f"{cat} not in current dataset version")
        expected = real_df.loc[real_df["category"] == cat, "intent"].value_counts().to_dict()
        expected = {k: int(v) for k, v in expected.items()}
        actual = get_distribution.invoke({"group_by": "intent", "scope_category": cat})
        assert actual == expected


# ---------------------------------------------------------------------------
# Schema sanity
# ---------------------------------------------------------------------------


def test_normalisation_was_applied(real_df):
    """category column should be all-uppercase, intent column all-lowercase."""
    assert (real_df["category"] == real_df["category"].str.upper()).all()
    assert (real_df["intent"] == real_df["intent"].str.lower()).all()


def test_dataset_has_expected_minimum_size(real_df):
    """Bitext is ~27K rows — guard against silent truncation in the parquet write."""
    assert len(real_df) >= 25_000


# ---------------------------------------------------------------------------
# Returned examples are real rows (no fabrication)
# ---------------------------------------------------------------------------


def test_get_examples_returns_real_dataframe_rows(real_df):
    examples = get_examples.invoke(
        {"category": "REFUND", "intent": "get_refund", "n": 3, "columns": ["instruction"]}
    )
    instructions_in_df = set(real_df["instruction"].astype(str))
    for ex in examples:
        assert ex["instruction"] in instructions_in_df


def test_search_by_keyword_returns_real_matches(real_df):
    """Every row returned by search_by_keyword must actually contain the keyword."""
    for kw in ("money back", "password", "cancel"):
        rows = search_by_keyword.invoke({"keyword": kw, "n": 5})
        for r in rows:
            assert kw.lower() in r["instruction"].lower()
