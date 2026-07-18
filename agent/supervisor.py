"""Supervisor node + routing edge (Task 1.3).

TODO:
  - `make_supervisor(llm)`: if current_step_index >= len(plan) -> next_agent =
    'synthesizer'; else classify the current step to 'rag_agent' or 'mcp_tools'.
  - `route_from_supervisor(state)`: return state["next_agent"] for the
    conditional edge.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from agent.prompts import SUPERVISOR_PROMPT
from agent.state import AnalystState

RAG = "rag_agent"
MCP = "mcp_tools"
SYNTH = "synthesizer"

_VALID_ROUTES = {RAG, MCP}

def make_supervisor(llm):
    def supervisor(state: AnalystState) -> dict:
        plan = state["plan"]
        idx = state["current_step_index"]

        if idx >= len(plan):
            return {"next_agent": SYNTH}

        current_step = plan[idx]
        next_agent = _classify_step(llm, current_step)

        return {"next_agent": next_agent}

    return supervisor


def route_from_supervisor(state: AnalystState) -> str:
    return state["next_agent"]


def _classify_step(llm, step: str) -> str:
    """Ask the LLM to classify one step, with a keyword-based fallback
    if the structured response doesn't parse cleanly."""
    response = llm.invoke(
        [
            SystemMessage(content=SUPERVISOR_PROMPT),
            HumanMessage(content=step),
        ]
    )
    raw = response.content.strip().lower().strip('."\' ')

    if raw in _VALID_ROUTES:
        return raw

    calc_keywords = (
        "calculate", "compute", "multiply", "divide", "percentage",
        "growth", "compare", "convert", "sum", "average", "ratio",
    )
    if any(kw in step.lower() for kw in calc_keywords):
        return MCP
    return RAG 
 
