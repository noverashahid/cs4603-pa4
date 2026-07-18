"""
CS4603 PA4 — MCP tool server  [GIVEN — do not modify]
=====================================================

Exposes deterministic math / finance tools over the Model Context Protocol
(stdio transport). Your LangGraph supervisor routes calculation steps here so
numbers are computed by real Python, not hallucinated by the LLM.

Run standalone (for a quick smoke test):

    uv run python tools/mcp_server.py

Your agent connects to it programmatically via `langchain-mcp-adapters`
(see agent/graph.py). The five tools below match PA4 Task 0.2.
"""

from __future__ import annotations

import ast
import operator

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("cs4603-analyst-tools")


# ─── Safe arithmetic evaluator ───────────────────────────────────────────────
# We evaluate `calculate` with a whitelisted AST walker instead of eval() so a
# malicious or malformed expression can never execute arbitrary code.

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError(f"Unsupported constant: {node.value!r}")
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("Expression contains an unsupported operation")


# ─── Tools ───────────────────────────────────────────────────────────────────

@mcp.tool()
def calculate(expression: str) -> str:
    """Evaluate a math expression. Supports + - * / ** % and parentheses.

    Args:
        expression: e.g. "16.91 * (1.08 ** 3)"
    """
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree)
    except (SyntaxError, ValueError, ZeroDivisionError) as exc:
        return f"Error evaluating '{expression}': {exc}"
    return f"{expression} = {result:g}"


@mcp.tool()
def percentage_change(old_value: float, new_value: float) -> str:
    """Compute the percentage change from old_value to new_value.

    Args:
        old_value: the baseline value
        new_value: the new value
    """
    if old_value == 0:
        return "Error: percentage change is undefined when old_value is 0"
    pct = (new_value - old_value) / abs(old_value) * 100
    direction = "increase" if pct >= 0 else "decrease"
    return f"{old_value} -> {new_value} is a {pct:+.2f}% {direction}"


@mcp.tool()
def growth_rate(start_value: float, rate: float, years: float) -> str:
    """Project a value under compound annual growth: start_value * (1 + rate) ** years.

    Args:
        start_value: the starting value
        rate: annual growth rate as a decimal (e.g. 0.08 for 8%)
        years: number of years to compound
    """
    try:
        projected = start_value * (1 + rate) ** years
    except (OverflowError, ValueError) as exc:
        return f"Error computing growth: {exc}"
    return (
        f"{start_value:g} at {rate * 100:g}% CAGR for {years:g} years "
        f"= {projected:g}"
    )


@mcp.tool()
def compare_values(a: float, b: float) -> str:
    """Compare two numbers and report which is larger and by how much (absolute and %).

    Args:
        a: first value
        b: second value
    """
    if a == b:
        return f"{a:g} and {b:g} are equal"
    larger, smaller = (a, b) if a > b else (b, a)
    diff = larger - smaller
    pct = (diff / abs(smaller) * 100) if smaller != 0 else float("inf")
    label = "a" if a > b else "b"
    return (
        f"{larger:g} ({label}) is larger by {diff:g} "
        f"({pct:.2f}% greater than {smaller:g})"
    )


@mcp.tool()
def unit_convert(value: float, from_unit: str, to_unit: str) -> str:
    """Convert a value between common financial-reporting scale units.

    Supported units (case-insensitive): 'ones', 'thousand', 'million',
    'billion', 'trillion', and 'percent' <-> 'ratio'.

    Args:
        value: the numeric value to convert
        from_unit: source unit
        to_unit: target unit
    """
    scale = {
        "ones": 1.0,
        "thousand": 1e3,
        "million": 1e6,
        "billion": 1e9,
        "trillion": 1e12,
    }
    f, t = from_unit.lower().strip(), to_unit.lower().strip()

    # percentage <-> ratio special case
    if {f, t} == {"percent", "ratio"}:
        result = value / 100 if f == "percent" else value * 100
        return f"{value:g} {from_unit} = {result:g} {to_unit}"

    if f not in scale or t not in scale:
        return (
            f"Unknown unit. Supported: {', '.join(scale)}, percent, ratio "
            f"(got from='{from_unit}', to='{to_unit}')"
        )
    result = value * scale[f] / scale[t]
    return f"{value:g} {from_unit} = {result:g} {to_unit}"


if __name__ == "__main__":
    # Default transport is stdio — this is what the LangGraph MCP client connects to.
    mcp.run()
