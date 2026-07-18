"""Vector Search retriever factory (Task 1.4 support / rag/store.py).

TODO: Implement `get_retriever(k=4)` that returns a LangChain retriever over the
Databricks Vector Search index built by `ingest.py`, using
`DatabricksVectorSearch` from `databricks_langchain`. Read endpoint/index names
from config.get_settings(). This exact retriever is reused by the deployed model.
"""

from __future__ import annotations

from config import get_settings

TEXT_COLUMN = "chunk_to_retrieve"
CITATION_COLUMNS = ["chunk_id", "source", "page"]


def get_vector_store():
    """Return a DatabricksVectorSearch handle over the Task 0.3 index.

    Reads endpoint/index names from config.get_settings() rather than
    os.environ directly, and reads DATABRICKS_HOST/DATABRICKS_TOKEN
    implicitly via the databricks-vectorsearch client's own auth
    resolution (env vars are picked up automatically).
    """
    from databricks_langchain import DatabricksVectorSearch

    settings = get_settings()
    if not settings["vs_endpoint"] or not settings["vs_index"]:
        raise OSError(
            "VECTOR_SEARCH_ENDPOINT / VECTOR_SEARCH_INDEX must be set "
            "(local .env or the endpoint's environment_vars when deployed)."
        )

    return DatabricksVectorSearch(
        endpoint=settings["vs_endpoint"],
        index_name=settings["vs_index"],
        columns=[TEXT_COLUMN] + CITATION_COLUMNS,
        # columns=CITATION_COLUMNS,
        # disable_notice=True,
    )




def get_retriever(k: int = 4):
    """Return a top-k LangChain retriever over the Vector Search index."""
    store = get_vector_store()
    return store.as_retriever(search_kwargs={"k": k})
