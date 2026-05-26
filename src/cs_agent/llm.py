"""Cached factories for the two Nebius-hosted LLMs used by the agent.

We keep them as `lru_cache`d singletons so that:
1. The HTTP client / auth state is created once per process, not per call.
2. Calling code can `from cs_agent.llm import get_agent_llm` anywhere without
   threading the instance through function signatures.

Two roles, two models:
- Router / profile-distillation:  small, fast model (Llama 3.1 8B fast).
- Agent reasoning + summarization: larger model (Llama 3.3 70B Instruct).
"""

from __future__ import annotations

from functools import lru_cache

from langchain_openai import ChatOpenAI

from cs_agent.config import (
    AGENT_MODEL,
    AGENT_TEMPERATURE,
    NEBIUS_BASE_URL,
    ROUTER_MODEL,
    ROUTER_TEMPERATURE,
    require_api_key,
)


@lru_cache(maxsize=1)
def get_router_llm() -> ChatOpenAI:
    """Return the cached small LLM used by the router and the profile-distillation node.

    Temperature 0 because both consumers expect deterministic, structured output.
    """
    return ChatOpenAI(
        model=ROUTER_MODEL,
        api_key=require_api_key(),
        base_url=NEBIUS_BASE_URL,
        temperature=ROUTER_TEMPERATURE,
        # Router is a small, fast classifier — typical latency 0.5-3s.
        # Fail fast at 20s and retry once. Worst case wall time: ~40s.
        timeout=20,
        max_retries=1,
    )


@lru_cache(maxsize=1)
def get_agent_llm() -> ChatOpenAI:
    """Return the cached large LLM used for ReAct reasoning, tool calls, and summarization.

    Temperature is slightly above 0 to allow natural-sounding summaries while still
    being predictable for tool selection.
    """
    return ChatOpenAI(
        model=AGENT_MODEL,
        api_key=require_api_key(),
        base_url=NEBIUS_BASE_URL,
        temperature=AGENT_TEMPERATURE,
        # Agent generations (incl. summarize tool) are typically 1-15s.
        # 30s catches the long tail; retry once for transient HTTP blips.
        # Worst case per LLM call: ~60s.
        timeout=30,
        max_retries=1,
    )
