"""LLM-backed summarization tool for unstructured/open-ended questions.

This is the only tool in the toolkit that calls an LLM internally. The agent
uses it for questions like:
- "Summarize the FEEDBACK category."
- "How do customer service representatives typically respond to cancellation
  requests?"

The router is expected to classify such questions as 'unstructured', and the
agent then sends them through this tool. Returns a short natural-language
summary string the agent can quote in its final answer.
"""

from __future__ import annotations

from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from cs_agent.data import loader
from cs_agent.llm import get_agent_llm
from cs_agent.tools.filter import _apply_filters
from cs_agent.tools.schemas import SummarizeArgs

SUMMARIZE_SYSTEM = (
    "You are a helpful data analyst. The user provides up to a few dozen "
    "anonymised customer-support {role}s (separated by '---'). Produce a "
    "short summary (4-7 bullet points) of the recurring patterns, themes, "
    "tone, and structure across the samples. Be concrete and specific. "
    "Do not invent details that are not present in the samples."
)

SUMMARIZE_USER_TEMPLATE = (
    "Scope: category={category!r}, intent={intent!r}, role={role!r}, "
    "sampled_rows={n_sampled}.\n\n"
    "Samples:\n{text}"
)


@tool("summarize", args_schema=SummarizeArgs)
def summarize(
    category: str | None = None,
    intent: str | None = None,
    role: Literal["instruction", "response"] = "response",
    sample_size: int = 20,
) -> str:
    """Summarize patterns across user instructions or agent responses for a slice of the dataset.

    Use this for OPEN-ENDED questions that require synthesis over many rows
    (e.g. "Summarize the FEEDBACK category", "How do agents respond to
    complaints?"). For factual lookups (counts, distributions, exact examples)
    prefer the structured tools instead.

    Args:
        category: Optional category to scope to, e.g. 'FEEDBACK'.
        intent: Optional intent to scope to, e.g. 'complaint'.
        role: Which side to summarize. 'response' for agent replies (default),
            'instruction' for how users phrase their requests.
        sample_size: Number of rows to include in the summary input (3-50).

    Returns:
        A short natural-language summary, or an explanation if no rows match.
    """
    df = _apply_filters(loader.get_df(), category, intent, keyword=None)
    if df.empty:
        return (
            f"No rows matched the requested scope (category={category!r}, "
            f"intent={intent!r}). Try list_categories or list_intents to see "
            f"what's available."
        )

    n_sampled = min(sample_size, len(df))
    rows = df.sample(n=n_sampled, random_state=42)
    text = "\n---\n".join(rows[role].astype(str).tolist())

    llm = get_agent_llm()
    response = llm.invoke(
        [
            SystemMessage(SUMMARIZE_SYSTEM.format(role=role)),
            HumanMessage(
                SUMMARIZE_USER_TEMPLATE.format(
                    category=category,
                    intent=intent,
                    role=role,
                    n_sampled=n_sampled,
                    text=text,
                )
            ),
        ]
    )
    return str(response.content).strip()
