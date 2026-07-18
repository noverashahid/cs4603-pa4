"""MLflow models-from-code definition (Task 2.1).

TODO: Make this file self-contained so MLflow can serialise it:
  - validate DATABRICKS_HOST/TOKEN/MODEL at import time (clear error if missing),
  - rebuild the graph with production clients (LLM, Vector Search retriever,
    MCP tools),
  - end with `mlflow.models.set_model(graph)`.

Must import cleanly:  python -c "import deployment.agent_model"
"""

from __future__ import annotations
# TODO: import os, mlflow, build_graph, get_chat_llm, get_retriever, load_mcp_tools
import os

import mlflow

from agent.graph import build_graph
from config import get_chat_llm
from rag.store import get_retriever

# TODO: validate env vars
_REQUIRED_ENV_VARS = [
    "DATABRICKS_HOST",
    "DATABRICKS_TOKEN",
    "DATABRICKS_MODEL",
    "VECTOR_SEARCH_ENDPOINT",
    "VECTOR_SEARCH_INDEX",
    "EMBEDDINGS_ENDPOINT",
]

_missing = [v for v in _REQUIRED_ENV_VARS if not os.environ.get(v)]
if _missing:
    raise OSError(
        f"deployment/agent_model.py: missing required environment variable(s): "
        f"{_missing}. These must be set on the serving endpoint's "
        f"environment_vars (Task 2.3) — secrets for DATABRICKS_HOST/TOKEN/MODEL, "
        f"plaintext for the Vector Search vars."
    )


# TODO: graph = build_graph(...)
_llm = get_chat_llm()
_retriever = get_retriever()

graph = build_graph(llm=_llm, retriever=_retriever, tools=None)

# TODO: mlflow.models.set_model(graph)
mlflow.models.set_model(graph)