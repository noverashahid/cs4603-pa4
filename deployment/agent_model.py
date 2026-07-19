"""ChatModel wrapper for Bonus B (databricks-agents SDK compatibility).

agents.deploy() requires the model's output schema to be either
ChatCompletionResponse or StringResponse. Our graph's native output is
the full AnalystState (plan, step_results, next_agent, final_answer,
messages, ...), which the Agent Framework's schema validator rejects.
This file wraps the IDENTICAL graph behind an mlflow.pyfunc.ChatModel so
it exposes the shape agents.deploy() expects, while still running the
exact same planner -> supervisor -> rag_agent/mcp_tools -> synthesizer
graph underneath. Nothing about the agent's logic changes.
"""

from __future__ import annotations

import os as _os
import sys as _sys

if not hasattr(_sys.stderr, "fileno"):
    _sys.stderr = _os.fdopen(_os.open(_os.devnull, _os.O_WRONLY), "w")
else:
    try:
        _sys.stderr.fileno()
    except Exception:
        _sys.stderr = _os.fdopen(_os.open(_os.devnull, _os.O_WRONLY), "w")

import os

import mlflow
from mlflow.pyfunc import ChatModel
from mlflow.types.llm import ChatChoice, ChatCompletionResponse, ChatMessage

from agent.graph import build_graph
from config import get_chat_llm
from rag.store import get_retriever

_REQUIRED_ENV_VARS = [
    # DATABRICKS_HOST/DATABRICKS_TOKEN are intentionally NOT required here:
    # the agents.deploy() endpoint (Bonus B) authenticates automatically via
    # declared `resources=` and never has a PAT in its environment.
    "DATABRICKS_MODEL",
    "VECTOR_SEARCH_ENDPOINT",
    "VECTOR_SEARCH_INDEX",
    "EMBEDDINGS_ENDPOINT",
]

_missing = [v for v in _REQUIRED_ENV_VARS if not os.environ.get(v)]
if _missing:
    raise OSError(
        f"agent_model.py: missing required environment variable(s): {_missing}."
    )


class AnalystChatModel(ChatModel):
    def load_context(self, context):
        llm = get_chat_llm()
        retriever = get_retriever()
        self.graph = build_graph(llm=llm, retriever=retriever, tools=None)

    def predict(self, context, messages, params=None):
        input_messages = [{"role": m.role, "content": m.content} for m in messages]
        state = self.graph.invoke({"messages": input_messages})

        answer = state.get("final_answer")
        if not answer and state.get("messages"):
            last = state["messages"][-1]
            answer = last.get("content") if isinstance(last, dict) else getattr(last, "content", "")

        return ChatCompletionResponse(
            choices=[ChatChoice(message=ChatMessage(role="assistant", content=answer or ""))],
        )


mlflow.models.set_model(AnalystChatModel())