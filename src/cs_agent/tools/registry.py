"""Canonical list of tools the agent can use.

Importers should depend on ``DATA_TOOLS`` from this module rather than the
individual tool functions, so the agent and the FastMCP server stay in sync
when we add or remove tools.
"""

from __future__ import annotations

from langchain_core.tools import BaseTool

from cs_agent.tools.catalog import get_distribution, list_categories, list_intents
from cs_agent.tools.filter import count_rows, get_examples, search_by_keyword
from cs_agent.tools.summarize import summarize

DATA_TOOLS: list[BaseTool] = [
    list_categories,
    list_intents,
    get_distribution,
    count_rows,
    get_examples,
    search_by_keyword,
    summarize,
]
"""All seven Bitext data-analyst tools, ordered roughly by 'discovery → filter → synthesize'."""

TOOLS_BY_NAME: dict[str, BaseTool] = {t.name: t for t in DATA_TOOLS}
