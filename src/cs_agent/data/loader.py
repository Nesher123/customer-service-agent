"""Bitext dataset loader with parquet-backed local cache.

The first call to :func:`get_df` downloads the HuggingFace dataset and writes
it to ``data/bitext.parquet``; every subsequent call (in the same process or
across restarts) reads the parquet directly. The in-process result is cached
with :func:`functools.lru_cache` so repeated tool calls within a single user
turn share the same DataFrame instance.

Normalisation we apply once at load time so tools don't have to:
- ``category`` is upper-cased ("REFUND", "ACCOUNT", ...).
- ``intent``   is lower-cased ("track_refund", "create_account", ...).
This matches the user-facing convention from the assignment brief and makes
case-insensitive filtering trivial in the tool layer.

Note on dataset versions:
    The HuggingFace README is slightly out of date relative to the live CSV.
    As of this writing, the actual schema is::

        categories = [ACCOUNT, CANCEL, CONTACT, DELIVERY, FEEDBACK, INVOICE,
                      ORDER, PAYMENT, REFUND, SHIPPING, SUBSCRIPTION]

        REFUND  intents include 'get_refund' (the README omits it)
        ACCOUNT intents include 'recover_password', 'registration_problems'

    Tools must therefore *never* hardcode categories/intents — they always
    read them from :func:`get_df` / :func:`dataset_summary`.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache

from cs_agent.config import BITEXT_HF_ID, HF_CACHE_DIR, PARQUET_PATH

# Redirect the HuggingFace cache into the project before importing `datasets`,
# so the first download lands in `data/.hf_cache/` (gitignored) instead of
# `~/.cache/huggingface/`. Honours an existing HF_HOME if the user set one.
os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))

from collections.abc import Iterator

import pandas as pd
from datasets import load_dataset

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ("flags", "instruction", "category", "intent", "response")


@lru_cache(maxsize=1)
def get_df() -> pd.DataFrame:
    """Return the Bitext customer-support DataFrame.

    On first call:
        Downloads the dataset from HuggingFace, normalises ``category``/``intent``,
        and persists it as parquet under ``data/bitext.parquet``.

    On subsequent calls (same process):
        Returns the cached DataFrame instance — O(1).

    On subsequent process starts:
        Reads the parquet directly — fast, no network, no HF auth needed.
    """
    if PARQUET_PATH.exists():
        logger.debug("Loading Bitext from parquet cache: %s", PARQUET_PATH)
        df = pd.read_parquet(PARQUET_PATH)
        _validate_columns(df)
        return df

    logger.info("Downloading Bitext from HuggingFace (%s)…", BITEXT_HF_ID)
    HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(BITEXT_HF_ID, split="train")
    df = _as_dataframe(ds.to_pandas())
    _validate_columns(df)

    df["category"] = df["category"].astype(str).str.upper()
    df["intent"] = df["intent"].astype(str).str.lower()

    PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PARQUET_PATH, index=False)
    logger.info("Cached Bitext to %s (%d rows)", PARQUET_PATH, len(df))
    return df


@lru_cache(maxsize=1)
def dataset_summary() -> dict:
    """Return a lightweight summary describing the dataset's shape.

    Used inside the agent's system prompt so the LLM knows the schema, the
    full set of categories, and which intents belong to which category — and
    therefore can pick the right filter values without guessing.
    """
    df = get_df()
    summary: dict = {
        "n_rows": int(len(df)),
        "columns": list(df.columns),
        "categories": sorted(df["category"].unique().tolist()),
        "intents_per_category": {
            cat: sorted(df.loc[df["category"] == cat, "intent"].unique().tolist())
            for cat in sorted(df["category"].unique().tolist())
        },
    }
    return summary


def clear_cache() -> None:
    """Clear in-process caches for both ``get_df`` and ``dataset_summary``.

    Useful in tests where we monkey-patch the underlying parquet or HF source.
    Does NOT delete the on-disk parquet file.
    """
    get_df.cache_clear()
    dataset_summary.cache_clear()


def _as_dataframe(df: pd.DataFrame | Iterator[pd.DataFrame]) -> pd.DataFrame:
    """Narrow pandas/HF union return types to a single DataFrame."""
    if isinstance(df, pd.DataFrame):
        return df
    raise TypeError(f"Expected a single DataFrame, got {type(df).__name__}")


def _validate_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Bitext DataFrame is missing required columns: {missing}. Got columns: {list(df.columns)}"
        )
