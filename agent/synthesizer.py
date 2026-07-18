"""Synthesizer node (Task 1.6).

TODO: Implement `make_synthesizer(llm)` returning a node that combines
step_results into one cited answer and writes it to BOTH `final_answer` AND
the `messages` channel as an AIMessage (required for the OpenAI-compatible
serving contract — see spec Task 1.6).
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent.prompts import SYNTHESIZER_PROMPT
from agent.state import AnalystState


def make_synthesizer(llm):
    def synthesizer(state: AnalystState) -> dict:
        plan = state["plan"]
        step_results = state["step_results"]

        context = _format_step_results(plan, step_results)

        response = llm.invoke(
            [
                SystemMessage(content=SYNTHESIZER_PROMPT),
                HumanMessage(content=f"Step results:\n\n{context}"),
            ]
        )
        answer = response.content.strip()

        return {
            "final_answer": answer,
            "messages": [AIMessage(content=answer)],
        }

    return synthesizer



def _format_step_results(plan: list[str], step_results: list[str]) -> str:
    """Pair each plan step with its result so the LLM can see which step
    produced which fact (and which steps failed)."""
    lines = []
    for i, result in enumerate(step_results):
        step_text = plan[i] if i < len(plan) else f"(step {i})"
        lines.append(f"Step {i + 1}: {step_text}\nResult: {result}")
    return "\n\n".join(lines)
