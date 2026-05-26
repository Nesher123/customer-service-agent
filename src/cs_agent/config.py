"""Central configuration: paths, model identifiers, and tunable constants.

Reads `.env` once at import time. Anything overridable via env var has a
sensible default here so the project runs out of the box.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import SecretStr

load_dotenv()

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
"""Filesystem root of the assignment package (the directory containing pyproject.toml)."""

DATA_DIR: Path = PROJECT_ROOT / "data"
PROFILES_DIR: Path = PROJECT_ROOT / "profiles"
PARQUET_PATH: Path = DATA_DIR / "bitext.parquet"
CHECKPOINT_PATH: Path = PROJECT_ROOT / "checkpoints.sqlite"

HF_CACHE_DIR: Path = DATA_DIR / ".hf_cache"
"""Project-local HuggingFace cache. Keeps downloads inside the workspace so the
project is self-contained and works in sandboxed environments without depending
on ``~/.cache/huggingface/``."""

BITEXT_HF_ID: str = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"

NEBIUS_BASE_URL: str = os.getenv(
    "CS_AGENT_NEBIUS_BASE_URL",
    "https://api.tokenfactory.nebius.com/v1/",
)
"""Nebius Token Factory OpenAI-compatible endpoint. Override with
``CS_AGENT_NEBIUS_BASE_URL`` if needed (e.g. ``https://api.studio.nebius.com/v1/``
which is the older Studio alias for the same backend)."""

ROUTER_MODEL: str = os.getenv(
    "CS_AGENT_ROUTER_MODEL",
    "Qwen/Qwen3-32B",
)
AGENT_MODEL: str = os.getenv(
    "CS_AGENT_AGENT_MODEL",
    "meta-llama/Llama-3.3-70B-Instruct",
)

MAX_ITERATIONS: int = int(os.getenv("CS_AGENT_MAX_ITERATIONS", "12"))
"""Maximum number of agent_node visits per turn before the graceful fallback."""

ROUTER_TEMPERATURE: float = 0.0
AGENT_TEMPERATURE: float = 0.0


def require_api_key() -> SecretStr:
    """Return the Nebius API key, raising a friendly error if missing."""
    key = os.getenv("NEBIUS_API_KEY")
    if not key:
        raise RuntimeError(
            "NEBIUS_API_KEY is not set. Copy .env.example to .env and paste your "
            "Nebius Token Factory key, or export NEBIUS_API_KEY in your shell."
        )
    return SecretStr(key)
