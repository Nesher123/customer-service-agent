"""Filtering and aggregation tools over the Bitext DataFrame.

All three tools share the same composable filter set (category / intent /
keyword), all optional. The LLM chains them naturally: e.g. for "how many
refund requests?" it can either call ``count_rows(category='REFUND')`` directly
or first ``list_intents('REFUND')`` and then sum across each intent — both are
valid and the choice exposes whether the agent reasons before acting.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.tools import tool

from cs_agent.data import loader
from cs_agent.tools.schemas import (
    CountRowsArgs,
    GetExamplesArgs,
    SearchByKeywordArgs,
)

if TYPE_CHECKING:
    import pandas as pd

DEFAULT_EXAMPLE_COLUMNS: list[str] = ["category", "intent", "instruction"]
ALLOWED_COLUMNS: set[str] = {"flags", "instruction", "response", "category", "intent"}


def _apply_filters(
    df: pd.DataFrame,
    category: str | None,
    intent: str | None,
    keyword: str | None,
) -> pd.DataFrame:
    """Apply optional filters in a stable order. None means no filter for that field."""
    if category is not None:
        df = df.loc[df["category"] == category.upper()]
    if intent is not None:
        df = df.loc[df["intent"] == intent.lower()]
    if keyword is not None:
        df = df.loc[df["instruction"].str.contains(keyword, case=False, na=False, regex=False)]
    return df


@tool("count_rows", args_schema=CountRowsArgs)
def count_rows(
    category: str | None = None,
    intent: str | None = None,
    keyword: str | None = None,
) -> int:
    """Count rows in the dataset matching any combination of optional filters.

    Returns 0 if no rows match. Pass NO arguments to count every row in the dataset.

    Examples:
    - count_rows(category='REFUND')                       → all refund-related rows
    - count_rows(intent='get_refund')                     → just 'get refund' requests
    - count_rows(category='ACCOUNT', keyword='password')  → password-related account rows
    """
    df = _apply_filters(loader.get_df(), category, intent, keyword)
    return int(len(df))


@tool("get_examples", args_schema=GetExamplesArgs)
def get_examples(
    category: str | None = None,
    intent: str | None = None,
    keyword: str | None = None,
    n: int = 5,
    columns: list[str] | None = None,
) -> list[dict]:
    """Return up to ``n`` example rows matching the optional filters.

    Sampling is deterministic (random_state=42) so identical calls return the
    same examples — handy for follow-ups like "show me 3 more" once the agent
    has memory (Task 2a). Tools have no state of their own, so "more" is
    achieved by the agent passing a different ``n`` or different filters.

    Args:
        category, intent, keyword: standard composable filters (all optional).
        n: how many examples to return (1-50). Default 5.
        columns: subset of columns to include in each returned row. Default
            ['category', 'intent', 'instruction']. Add 'response' to also see
            the agent's reply, 'flags' for linguistic-variation tags.

    Returns:
        List of dicts, each representing one row. Empty list if no rows match.
    """
    df = _apply_filters(loader.get_df(), category, intent, keyword)
    if df.empty:
        return []

    selected = columns or DEFAULT_EXAMPLE_COLUMNS
    bad = [c for c in selected if c not in ALLOWED_COLUMNS]
    if bad:
        raise ValueError(f"Unknown columns requested: {bad}. Allowed: {sorted(ALLOWED_COLUMNS)}.")

    sample_n = min(n, len(df))
    rows = df.sample(n=sample_n, random_state=42).loc[:, selected]
    return rows.to_dict(orient="records")


@tool("search_by_keyword", args_schema=SearchByKeywordArgs)
def search_by_keyword(keyword: str, n: int = 10) -> list[dict]:
    """Search user instructions for a substring (case-insensitive) and return matches.

    This is the right tool for paraphrased questions where the user describes
    the *content* of the request rather than its category/intent label. For
    example, "show me people wanting their money back" → keyword='money back'.

    Args:
        keyword: substring to match against the user 'instruction' column.
        n: number of matches to return (1-50). Default 10.

    Returns:
        List of dicts with 'category', 'intent', and 'instruction' for each match.
    """
    df = loader.get_df()
    mask = df["instruction"].str.contains(keyword, case=False, na=False, regex=False)
    matches = df[mask]
    if matches.empty:
        return []
    sample_n = min(n, len(matches))
    rows = matches.sample(n=sample_n, random_state=42).loc[:, DEFAULT_EXAMPLE_COLUMNS]
    return rows.to_dict(orient="records")
