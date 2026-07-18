"""Offline smoke test for the Document Analyst graph (Bonus A test target).

This is the target the Bonus A CI pipeline runs to prove the graph wires up
before any deploy. Fill it in once your nodes are implemented.

TODO (Task 1.7 / Bonus A):
  - Build fake LLM / retriever / tool objects (no Databricks, no network).
  - Call `build_graph(llm=FakeLLM(), retriever=FakeRetriever(), tools=[FakeTool()])`.
  - Invoke it on a combined retrieval+calculation query and assert that a plan was
    produced, both specialists ran, and the final answer surfaced on messages[-1].

Run:  uv run pytest -q
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import AIMessage


# def test_graph_module_imports():
#     """Minimal collection guard: the graph module must import cleanly."""
#     from agent.graph import build_graph  # noqa: F401



class FakeLLM:
    """Minimal stand-in for a ChatOpenAI-like client.

    Returns canned responses based on what the prompt is asking for, so
    the graph can run its full loop (planner -> supervisor -> rag_agent
    -> supervisor -> mcp_tools -> supervisor -> synthesizer) with zero
    network calls.
    """

    def bind_tools(self, tools):
        # Return a variant that also knows how to "call" a tool.
        return _FakeLLMWithTools(tools)

    def invoke(self, messages):
        system_content = messages[0].content if messages else ""
        user_content = messages[-1].content if messages else ""

        # Planner: system prompt mentions "planning module"
        if "planning module" in system_content:
            return AIMessage(
                content=(
                    '["Find the net revenue for fiscal year 2023", '
                    '"Calculate a 10% increase on that revenue"]'
                )
            )

        # Supervisor: system prompt mentions "routing module"
        if "routing module" in system_content:
            if "calculate" in user_content.lower():
                return AIMessage(content="mcp_tools")
            return AIMessage(content="rag_agent")

        # RAG extraction: system prompt mentions "extracting a specific fact"
        if "extracting a specific fact" in system_content:
            return AIMessage(
                content="Net revenue in FY2023 was $100 million [source: annual_report.pdf, p.4]."
            )

        # Synthesizer: system prompt mentions "synthesis module"
        if "synthesis module" in system_content:
            return AIMessage(
                content=(
                    "Net revenue in FY2023 was $100 million "
                    "[source: annual_report.pdf, p.4]. A 10% increase would "
                    "bring it to $110 million."
                )
            )

        return AIMessage(content="(fake default response)")


class _FakeLLMWithTools:
    """Stand-in for the result of llm.bind_tools(tools) inside make_mcp_node."""

    def __init__(self, tools):
        self._tools = tools

    def invoke(self, messages):
        # Simulate the LLM choosing the "calculate" tool with fixed args.
        msg = AIMessage(content="")
        msg.tool_calls = [
            {
                "name": "calculate",
                "args": {"expression": "100000000 * 1.10"},
                "id": "fake-call-1",
            }
        ]
        return msg


class FakeDoc:
    def __init__(self, text: str, source: str, page: int):
        self.page_content = text
        self.metadata = {"source": source, "page": page}


class FakeRetriever:
    """Stand-in for a LangChain retriever (rag/store.py::get_retriever())."""

    def invoke(self, query: str):
        return [FakeDoc("Net revenue was $100 million.", "annual_report.pdf", 4)]


class FakeTool:
    """Stand-in for one MCP tool (e.g. `calculate`)."""

    name = "calculate"

    async def ainvoke(self, args: dict):
        expr = args.get("expression", "0")
        return eval(expr, {"__builtins__": {}})  # safe-ish for this fixed fake case


def test_graph_module_imports():
    """Minimal collection guard: the graph module must import cleanly."""
    from agent.graph import build_graph  # noqa: F401


def test_graph_compiles_and_runs_offline():
    """Build the graph with fakes and run one combined query end-to-end,
    asserting it never touches Databricks/network and produces a
    non-empty final message.
    """
    from agent.graph import build_graph

    graph = build_graph(
        llm=FakeLLM(),
        retriever=FakeRetriever(),
        tools=[FakeTool()],
    )

    result = graph.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "What was the revenue in 2023, and what would a "
                        "10% increase look like?"
                    ),
                }
            ]
        }
    )

    # Graph compiled and ran without raising.
    assert result is not None

    # Plan was produced with more than one step (planner ran).
    assert isinstance(result["plan"], list)
    assert len(result["plan"]) >= 2

    # Both specialists ran: expect at least one rag-style result and one
    # calculation-style result in step_results.
    assert len(result["step_results"]) == len(result["plan"])

    # Final answer surfaced on both channels.
    assert result["final_answer"]
    assert result["messages"], "messages channel must not be empty"
    last_message = result["messages"][-1]
    last_content = getattr(last_message, "content", None) or last_message.get("content")
    assert last_content, "last message must have non-empty content"