"""Full Document Analyst graph (Tasks 1.5 + 1.7).

TODO:
  - `load_mcp_tools(server_path=None)`: connect the GIVEN MCP server over stdio
    (see langchain-mcp-adapters) and return its tools.
  - `make_mcp_node(tools, llm)`: execute one calculation step by letting the LLM
    call exactly one MCP tool, then append the result and increment the index.
  - `build_graph(llm=None, retriever=None, tools=None)`: assemble
    planner -> supervisor -> {rag_agent | mcp_tools} -> ... -> synthesizer.
    Inject dependencies so the graph can be unit-tested offline with fakes.
"""

from __future__ import annotations
import asyncio
import os
import sys

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from agent.planner import make_planner
from agent.rag_agent import make_rag_agent
from agent.state import AnalystState
from agent.supervisor import MCP, RAG, SYNTH, make_supervisor, route_from_supervisor
from agent.synthesizer import make_synthesizer
from langchain_mcp_adapters.client import MultiServerMCPClient

from agent.prompts import MCP_STEP_PROMPT

_DEFAULT_SERVER_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "tools", "mcp_server.py"
)

def load_mcp_tools(server_path: str | None = None):
    path = server_path or _DEFAULT_SERVER_PATH

    client = MultiServerMCPClient(
        {
            "analyst": {
                "command": sys.executable,
                "args": [path],
                "transport": "stdio",
            }
        }
    )

    async def _get_tools():
        return await client.get_tools()

    old_stderr = sys.stderr
    needs_swap = not hasattr(old_stderr, "fileno")
    if not needs_swap:
        try:
            old_stderr.fileno()
        except Exception:
            needs_swap = True

    if needs_swap:
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        sys.stderr = os.fdopen(devnull_fd, "w")

    try:
        return _safe_async_run(_get_tools())
    finally:
        if needs_swap:
            sys.stderr.close()
            sys.stderr = old_stderr

def make_mcp_node(tools, llm):
    """Return a node that executes one calculation step via exactly one
    MCP tool call, appending the result to step_results.
    """
    llm_with_tools = llm.bind_tools(tools)
    tools_by_name = {t.name: t for t in tools}

    async def _invoke_tool_call(prompt_input: str) -> str:
        response = llm_with_tools.invoke(
            [
                SystemMessage(content=MCP_STEP_PROMPT),
                HumanMessage(content=prompt_input),
            ]
        )

        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            return response.content.strip()

        call = tool_calls[0]
        tool = tools_by_name.get(call["name"])
        if tool is None:
            return f"error: unknown tool '{call['name']}' requested"

        try:
            result = await tool.ainvoke(call["args"])
        except Exception as e:  # noqa: BLE001
            return f"error executing '{call['name']}': {e}"

        return f"{call['name']}({call['args']}) = {result}"

    def mcp_tools(state: AnalystState) -> dict:
        idx = state["current_step_index"]
        step_query = state["plan"][idx]

        context = "\n".join(state["step_results"]) or "(no prior results)"
        prompt_input = f"Step: {step_query}\n\nPrior results:\n{context}"

        result = _safe_async_run(_invoke_tool_call(prompt_input))

        return {
            "step_results": state["step_results"] + [result],
            "current_step_index": idx + 1,
        }

    return mcp_tools


def build_graph(llm=None, retriever=None, tools=None):
    """Assemble planner -> supervisor -> {rag_agent | mcp_tools} -> ... -> synthesizer.

    Dependencies are injected (not hard-coded) so the graph can be unit
    tested offline with fakes — tests/test_smoke.py passes a mocked llm/
    retriever/tools and never touches Databricks or spawns the MCP subprocess.
    """
    if llm is None:
        from config import get_chat_llm
        llm = get_chat_llm()

    if retriever is None:
        from rag.store import get_retriever
        retriever = get_retriever()

    if tools is None:
        tools = load_mcp_tools()

    builder = StateGraph(AnalystState)
    builder.add_node("planner", make_planner(llm))
    builder.add_node("supervisor", make_supervisor(llm))
    builder.add_node("rag_agent", make_rag_agent(retriever, llm))
    builder.add_node("mcp_tools", make_mcp_node(tools, llm))
    builder.add_node("synthesizer", make_synthesizer(llm))

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {RAG: "rag_agent", MCP: "mcp_tools", SYNTH: "synthesizer"},
    )
    builder.add_edge("rag_agent", "supervisor")
    builder.add_edge("mcp_tools", "supervisor")
    builder.add_edge("synthesizer", END)

    return builder.compile()


def _safe_async_run(coro):
    """Safely executes a coroutine whether an asyncio event loop is already running or not."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop detected (standard terminal / CI test execution)
        return asyncio.run(coro)
    else:
        # A loop is already running (Jupyter Notebook / IPython kernel)
        import nest_asyncio
        nest_asyncio.apply()
        return loop.run_until_complete(coro)