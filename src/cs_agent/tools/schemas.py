"""Pydantic input schemas for the data-analyst tool set.

Each schema lives next to no logic on purpose — these are the contracts the
LangGraph agent (and the FastMCP server) see. Keep field descriptions rich:
they are how the LLM learns what each parameter means.

Convention:
- ``category`` arguments are matched case-insensitively (we upper-case in the tool).
- ``intent``   arguments are matched case-insensitively (we lower-case in the tool).
- ``keyword``  arguments do a case-insensitive substring match against the user
  ``instruction`` column.

LLM-artifact tolerance:
    Open-source models (especially Llama 3.x) sometimes emit string ``"null"`` /
    ``"None"`` / ``""`` for optional fields, or JSON-encoded strings (``'["a","b"]'``)
    for list fields, instead of proper ``null`` / arrays. ``LLMToolBase`` cleans
    these artefacts in a ``model_validator`` before per-field validation kicks in,
    so we don't waste ReAct iterations on schema-rejection retries.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

NULLISH_STRINGS = {"null", "none", ""}


class LLMToolBase(BaseModel):
    """Base class for tool input schemas. Pre-cleans common LLM artefacts."""

    @model_validator(mode="before")
    @classmethod
    def _clean_llm_artifacts(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        cleaned: dict[str, Any] = {}
        for key, value in data.items():
            # Drop string sentinels that should be Python None.
            if isinstance(value, str) and value.strip().lower() in NULLISH_STRINGS:
                continue
            # Try to parse JSON-encoded structures (mostly the `columns` list).
            if isinstance(value, str):
                stripped = value.strip()
                if (stripped.startswith("[") and stripped.endswith("]")) or (
                    stripped.startswith("{") and stripped.endswith("}")
                ):
                    try:
                        value = json.loads(stripped)
                    except json.JSONDecodeError:
                        pass
            cleaned[key] = value
        return cleaned


class NoArgs(LLMToolBase):
    """Empty schema for tools that take no arguments (e.g. list_categories)."""


class ListIntentsArgs(LLMToolBase):
    """Arguments for the ``list_intents`` tool."""

    category: str | None = Field(
        default=None,
        description=(
            "Optional. High-level category to scope the listing, e.g. 'REFUND', "
            "'ACCOUNT'. Case-insensitive. If omitted, returns intents across all "
            "categories."
        ),
    )


class DistributionArgs(LLMToolBase):
    """Arguments for the ``get_distribution`` tool."""

    group_by: Literal["category", "intent"] = Field(
        ...,
        description=(
            "Which column to group rows by: 'category' (high-level) or 'intent' "
            "(specific). Use 'intent' with scope_category to get the intent "
            "distribution within a single category."
        ),
    )
    scope_category: str | None = Field(
        default=None,
        description=(
            "Optional. Restrict the distribution to a single category, e.g. "
            "'ACCOUNT'. Case-insensitive. Useful for 'distribution of intents in "
            "the ACCOUNT category'."
        ),
    )


class CountRowsArgs(LLMToolBase):
    """Arguments for the ``count_rows`` tool — composable filters all optional."""

    category: str | None = Field(
        default=None,
        description="Optional. Category filter, e.g. 'REFUND'. Case-insensitive.",
    )
    intent: str | None = Field(
        default=None,
        description=(
            "Optional. Intent filter, e.g. 'track_refund'. Case-insensitive. "
            "Combine with category to count rows matching both."
        ),
    )
    keyword: str | None = Field(
        default=None,
        description=(
            "Optional. Case-insensitive substring matched against the user "
            "'instruction' column. Use this for fuzzy semantic-ish matching like "
            "'money back' or 'refund'."
        ),
    )


class GetExamplesArgs(LLMToolBase):
    """Arguments for the ``get_examples`` tool."""

    category: str | None = Field(
        default=None,
        description="Optional. Category filter, e.g. 'REFUND'. Case-insensitive.",
    )
    intent: str | None = Field(
        default=None,
        description="Optional. Intent filter, e.g. 'track_refund'. Case-insensitive.",
    )
    keyword: str | None = Field(
        default=None,
        description=("Optional. Case-insensitive substring matched against the user 'instruction' column."),
    )
    n: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of examples to return (1-50). Default 5.",
    )
    columns: list[str] | None = Field(
        default=None,
        description=(
            "Optional. Subset of columns to return per example. Allowed: "
            "'instruction', 'response', 'category', 'intent', 'flags'. "
            "Default: ['category', 'intent', 'instruction']."
        ),
    )


class SearchByKeywordArgs(LLMToolBase):
    """Arguments for the ``search_by_keyword`` tool."""

    keyword: str = Field(
        ...,
        min_length=1,
        description=(
            "Substring to match (case-insensitive) against the user 'instruction' "
            "column. Use this for paraphrased questions like 'people wanting their "
            "money back' (keyword='money back')."
        ),
    )
    n: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Number of matching examples to return (1-50). Default 10.",
    )


class SummarizeArgs(LLMToolBase):
    """Arguments for the ``summarize`` tool — open-ended/unstructured questions."""

    category: str | None = Field(
        default=None,
        description="Optional. Category to summarize, e.g. 'FEEDBACK'. Case-insensitive.",
    )
    intent: str | None = Field(
        default=None,
        description=(
            "Optional. Intent to summarize, e.g. 'complaint'. Case-insensitive. "
            "Combine with category for tighter scoping."
        ),
    )
    role: Literal["instruction", "response"] = Field(
        default="response",
        description=(
            "Which side of the conversation to summarize: "
            "'instruction' = how users phrase their requests, "
            "'response' = how customer-service agents typically reply."
        ),
    )
    sample_size: int = Field(
        default=20,
        ge=3,
        le=50,
        description=(
            "How many rows to sample for the summary (3-50). More rows give a "
            "broader summary but cost more tokens. Default 20."
        ),
    )
