"""RAG agent node (Task 1.4) — retrieves from Databricks Vector Search.

TODO: Implement `make_rag_agent(retriever, llm)` returning a node that:
  - retrieves top-k chunks for the current step,
  - formats them with [source: file, p.N] citations,
  - extracts a single cited fact via the LLM (or 'not found in documents'),
  - appends the fact to step_results and increments current_step_index.
Reuse `rag/store.py::get_retriever()` so local and deployed retrieval match.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from agent.prompts import RAG_EXTRACT_PROMPT
from agent.state import AnalystState

NOT_FOUND = "not found in documents"

def format_docs(docs) -> str:
    """Format retrieved chunks with source citations for the LLM prompt."""
    if not docs:
        return ""

    formatted = []
    for doc in docs:
        text = doc.page_content.strip()
        meta = doc.metadata or {}
        
        source = meta.get("source", meta.get("file_name", "annual_report.pdf"))
        page = meta.get("page", meta.get("page_number", "?"))
        
        formatted.append(f"{text} [source: {source}, p.{page}]")

    return "\n\n".join(formatted)


def make_rag_agent(retriever, llm):
    def rag_agent(state: AnalystState) -> dict:
        idx = state["current_step_index"]
        step_query = state["plan"][idx]

        docs = retriever.invoke(step_query)
        context = format_docs(docs)

        if not context:
            result = NOT_FOUND
        else:
            response = llm.invoke(
                [
                    SystemMessage(content=RAG_EXTRACT_PROMPT),
                    HumanMessage(
                        content=f"Step to answer: {step_query}\n\n"
                        f"Retrieved context:\n{context}"
                    ),
                ]
            )
            result = response.content.strip()

            if result and result.lower() != NOT_FOUND and "[source:" not in result:
                # Extract the first citation tag from context if available
                first_doc_meta = docs[0].metadata if docs else {}
                src = first_doc_meta.get("source", "annual_report.pdf")
                pg = first_doc_meta.get("page", "?")
                result = f"{result} [source: {src}, p.{pg}]"

        return {
            "step_results": state["step_results"] + [result],
            "current_step_index": idx + 1,
        }

    return rag_agent
