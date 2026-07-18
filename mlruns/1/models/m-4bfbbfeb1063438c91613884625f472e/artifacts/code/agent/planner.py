"""Planner node (Task 1.2).

TODO: Implement `make_planner(llm)` returning a node that:
  - reads the user question from state["messages"],
  - asks the LLM (PLANNER_PROMPT) for a JSON list of 2-5 steps,
  - parses it robustly (fallback to a single step on parse failure),
  - returns {"plan": [...], "current_step_index": 0, "step_results": []}.
"""

from __future__ import annotations

from agent.state import AnalystState
import json
import re

from langchain_core.messages import HumanMessage, SystemMessage
from agent.prompts import PLANNER_PROMPT


def _extract_last_user_query(state: AnalystState) -> str:
    """Pull the most recent human message's text content out of state["messages"]."""
    for msg in reversed(state["messages"]):
        # Works whether messages are LangChain message objects or dicts
        # (the deployed endpoint may hand us either, depending on how
        # MLflow deserializes the incoming {"messages": [...]} payload).
        role = getattr(msg, "type", None) or (msg.get("role") if isinstance(msg, dict) else None)
        if role in ("human", "user"):
            content = getattr(msg, "content", None) or (
                msg.get("content") if isinstance(msg, dict) else None
            )
            if content:
                return content
    raise ValueError("No user message found in state['messages']")


def _parse_plan(raw_text: str) -> list[str]:
    """Parse the LLM's JSON list output, tolerating markdown fences and stray text."""
    text = raw_text.strip()

    # Strip markdown code fences if the model added them despite instructions
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

    # If there's leading/trailing prose, try to isolate the first [...] block
    if not text.startswith("["):
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            text = match.group(0)

    parsed = json.loads(text)  # let this raise; caller handles the fallback

    if not isinstance(parsed, list) or not all(isinstance(s, str) for s in parsed):
        raise ValueError("Parsed plan is not a list of strings")
    if not parsed:
        raise ValueError("Parsed plan is empty")

    return parsed


def make_planner(llm):
    def planner(state: AnalystState) -> dict:
        query = _extract_last_user_query(state)

        response = llm.invoke(
            [
                SystemMessage(content=PLANNER_PROMPT),
                HumanMessage(content=query),
            ]
        )

        try:
            plan = _parse_plan(response.content)
        except (json.JSONDecodeError, ValueError) as e:
            # Fallback: treat the whole question as a single atomic step.
            # This keeps the graph moving instead of crashing the run —
            # the supervisor will still route it to whichever agent fits.
            print(f"[planner] JSON parse failed ({e}); falling back to single-step plan.")
            plan = [query]

        # Cap defensively at 5 steps even if the LLM ignored the instruction
        plan = plan[:5]

        return {
            "plan": plan,
            "current_step_index": 0,
            "step_results": [],
        }

    return planner
