"""Discovery tools: enumerate the dataset's categories and intents.

These are usually the agent's *first* tool calls when it doesn't yet know
the exact spelling of a category or intent. Their job is to give the LLM a
ground-truth list to pick from before it invokes filtering tools.
"""

from __future__ import annotations

from typing import Literal

from langchain_core.tools import tool

from cs_agent.data import loader
from cs_agent.tools.schemas import DistributionArgs, ListIntentsArgs, NoArgs


@tool("list_categories", args_schema=NoArgs)
def list_categories() -> list[str]:
    """Return the sorted list of distinct high-level categories present in the dataset.

    Use this FIRST when the user asks "what categories exist" or whenever you
    are about to filter by category and are not 100% sure of the exact label
    (the live dataset's labels may differ from common assumptions, e.g. it
    has 'SHIPPING' rather than 'SHIPPING_ADDRESS').

    Returns:
        Sorted list of category strings, e.g. ['ACCOUNT', 'CANCEL', 'REFUND', ...].
    """
    df = loader.get_df()
    return sorted(df["category"].unique().tolist())


@tool("list_intents", args_schema=ListIntentsArgs)
def list_intents(category: str | None = None) -> list[str]:
    """Return the sorted list of distinct intents in the dataset.

    Pass a ``category`` to scope the listing to one category (recommended when
    answering questions like "how many refund requests?" — first list intents
    in REFUND, then count rows for each).

    Args:
        category: Optional category to filter by. Case-insensitive.

    Returns:
        Sorted list of intent strings, e.g. ['check_refund_policy', 'get_refund', 'track_refund'].
        Empty list if the category does not exist.
    """
    df = loader.get_df()
    if category is not None:
        df = df[df["category"] == category.upper()]
    return sorted(df.loc[:, "intent"].unique().tolist())


@tool("get_distribution", args_schema=DistributionArgs)
def get_distribution(
    group_by: Literal["category", "intent"],
    scope_category: str | None = None,
) -> dict[str, int]:
    """Return the row-count distribution grouped by category or intent.

    Use this for questions like:
    - "What is the distribution of intents in the ACCOUNT category?"
      → group_by='intent', scope_category='ACCOUNT'
    - "How many rows in each category?"
      → group_by='category'

    Args:
        group_by: 'category' or 'intent'.
        scope_category: Optional category to restrict the distribution to.
            Case-insensitive. Only meaningful with group_by='intent'.

    Returns:
        Mapping {label: row_count}, sorted by count descending.
    """
    df = loader.get_df()
    if scope_category is not None:
        df = df[df["category"] == scope_category.upper()]
    counts = df.loc[:, group_by].value_counts()
    return {str(k): int(v) for k, v in counts.items()}
