"""Shared configuration and client factories for the Document Analyst.

Centralising credential loading here keeps every other module free of
`os.environ` calls and makes the LLM / retriever easy to mock in tests.
All clients are created lazily so importing the package never requires
network access or credentials — only *invoking* a node does.
"""

from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()


def _require(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        raise OSError(
            f"Missing required environment variable: {name}. "
            "Set it in your .env (local) or the endpoint secret scope (deployed)."
        )
    return val


def get_settings() -> dict[str, str]:
    """Return all configuration values, validating the required ones.

    DATABRICKS_HOST/DATABRICKS_TOKEN are read only for local dev
    convenience (e.g. the deploy scripts' own MLflow/SDK calls) — they are
    NOT required here and must never be required by get_chat_llm(), since
    the agents.deploy() endpoint (Bonus B) authenticates automatically via
    declared `resources=` and never has a PAT in its environment.
    """
    return {
        "host": os.environ.get("DATABRICKS_HOST", "").rstrip("/"),
        "token": os.environ.get("DATABRICKS_TOKEN", ""),
        "model": _require("DATABRICKS_MODEL"),
        "embeddings": os.environ.get("EMBEDDINGS_ENDPOINT", "databricks-gte-large-en"),
        "vs_endpoint": os.environ.get("VECTOR_SEARCH_ENDPOINT", ""),
        "vs_index": os.environ.get("VECTOR_SEARCH_INDEX", ""),
    }


@lru_cache(maxsize=1)
def get_chat_llm(temperature: float = 0.0):
    """Configured ChatDatabricks client pointed at Databricks Model Serving.

    Uses databricks_langchain's auto-auth-aware client instead of
    langchain_openai.ChatOpenAI + an explicit PAT: it resolves credentials
    via the Databricks SDK's default auth chain, which picks up an
    explicit DATABRICKS_HOST/TOKEN when present (local dev, Part 2's
    manually-created endpoint) and otherwise falls back to the
    auto-provisioned OBO credential that agents.deploy() injects for a
    declared DatabricksServingEndpoint resource (Bonus B) — no PAT needed.
    """
    from databricks_langchain import ChatDatabricks

    s = get_settings()
    return ChatDatabricks(endpoint=s["model"], temperature=temperature)
