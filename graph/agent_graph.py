"""LangGraph workflow: User Query -> Planner -> Product Agent -> Comparison Agent -> Final Output."""

import sys
from pathlib import Path
from typing import TypedDict
import json
import os
import time

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from langgraph.graph import StateGraph, END

from agents.planner_agent import run_planner
from agents.product_agent import run_product_agent
from agents.comparison_agent import run_comparison_agent, rank_products
from agents.sales_insights_agent import run_sales_insights_agent


class AgentState(TypedDict, total=False):
    """State passed through the agent graph."""
    query: str
    plan: dict | None
    planner_reasoning: str
    products: list
    product_reasoning: str
    recommendation: str
    comparison_reasoning: str
    reasoning_steps: list[str]
    attempt: int
    max_attempts: int
    replan_reasoning: str
    sales_insights: dict
    sales_insights_reasoning: str
    initial_plan: dict | None
    dev_metrics: dict


def planner_node(state: AgentState) -> dict:
    """Node: analyze query and produce plan (intent, use_case, budget, tools)."""
    plan, reasoning = run_planner(state["query"])
    initial_plan = state.get("initial_plan")
    if initial_plan is None:
        initial_plan = plan
    return {
        "plan": plan,
        "initial_plan": initial_plan,
        "planner_reasoning": reasoning,
        "reasoning_steps": state.get("reasoning_steps", []) + [f"[Planner] {reasoning}"],
    }


def product_node(state: AgentState) -> dict:
    """Node: run product search and filter from plan."""
    plan = state.get("plan") or {}
    products, reasoning = run_product_agent(plan)
    return {
        "products": products,
        "product_reasoning": reasoning,
        "reasoning_steps": state.get("reasoning_steps", []) + [f"[Product Agent] {reasoning}"],
    }


def _relax_plan(plan: dict, product_reasoning: str) -> tuple[dict, str]:
    """
    Rule-based replanning when no products remain after filtering.
    Priority:
    1) If price filter removed everything -> drop price filter tool (keep budget as a preference).
    2) If still too narrow -> broaden use_case to general.
    """
    new_plan = dict(plan or {})
    tools = list(new_plan.get("tools", ["search_products", "filter_by_price", "compare_products"]))
    budget = new_plan.get("budget")
    use_case = new_plan.get("use_case", "general")

    if budget is not None and "filter_by_price" in tools:
        tools = [t for t in tools if t != "filter_by_price"]
        new_plan["tools"] = tools
        return (
            new_plan,
            f"Relaxed constraints: removed price filter after 0 results (budget ${budget} kept as preference).",
        )

    if use_case and use_case != "general":
        new_plan["use_case"] = "general"
        return (
            new_plan,
            f"Relaxed constraints: broadened use_case from '{use_case}' to 'general' after 0 results.",
        )

    return new_plan, "No further constraints to relax; returning best available alternatives."


def replanner_node(state: AgentState) -> dict:
    """Node: re-run planning when candidates don't meet criteria."""
    attempt = int(state.get("attempt", 0))
    plan = state.get("plan") or {}
    product_reasoning = state.get("product_reasoning", "")

    # If OpenAI is available, ask the Planner again with feedback. Otherwise relax deterministically.
    if os.getenv("OPENAI_API_KEY"):
        feedback_prompt = (
            "The previous tool run returned zero candidates.\n"
            f"Original user request: {state.get('query','')}\n"
            f"Previous plan: {json.dumps(plan)}\n"
            f"Tool outcome: {product_reasoning}\n\n"
            "Re-plan and choose tools. If budget is too strict, relax constraints or suggest near-budget alternatives."
        )
        new_plan, reasoning = run_planner(feedback_prompt)
        return {
            "plan": new_plan,
            "replan_reasoning": reasoning,
            "attempt": attempt + 1,
            "reasoning_steps": state.get("reasoning_steps", []) + [f"[Replanner] {reasoning}"],
        }

    new_plan, reasoning = _relax_plan(plan, product_reasoning)
    return {
        "plan": new_plan,
        "replan_reasoning": reasoning,
        "attempt": attempt + 1,
        "reasoning_steps": state.get("reasoning_steps", []) + [f"[Replanner] {reasoning}"],
    }


def needs_replan(state: AgentState) -> str:
    """Conditional edge: replan if no products and attempts remain."""
    products = state.get("products") or []
    attempt = int(state.get("attempt", 0))
    max_attempts = int(state.get("max_attempts", 2))
    if len(products) == 0 and attempt < max_attempts:
        return "replanner"
    return "comparison_agent"


def comparison_node(state: AgentState) -> dict:
    """Node: compare products and generate recommendation."""
    products = state.get("products", [])
    plan = state.get("plan") or {}
    # Sort product cards to match the ranking used in the comparison output
    try:
        products = rank_products(products, plan)
    except Exception:
        pass
    recommendation, reasoning = run_comparison_agent(
        products, state["query"], plan
    )
    return {
        "products": products,
        "recommendation": recommendation,
        "comparison_reasoning": reasoning,
        "reasoning_steps": state.get("reasoning_steps", []) + [f"[Comparison Agent] {reasoning}"],
    }


def sales_insights_node(state: AgentState) -> dict:
    """Node: summarize reviews/marketing insights for recommended products."""
    products = state.get("products", [])
    insights, reasoning = run_sales_insights_agent(products)
    return {
        "sales_insights": insights,
        "sales_insights_reasoning": reasoning,
        "reasoning_steps": state.get("reasoning_steps", []) + [f"[Sales Insights] {reasoning}"],
    }


def create_graph() -> StateGraph:
    """Build the LangGraph with looping replanning when needed."""
    workflow = StateGraph(AgentState)

    workflow.add_node("planner", planner_node)
    workflow.add_node("product_agent", product_node)
    workflow.add_node("replanner", replanner_node)
    workflow.add_node("comparison_agent", comparison_node)
    workflow.add_node("sales_insights", sales_insights_node)

    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "product_agent")
    workflow.add_conditional_edges("product_agent", needs_replan)
    workflow.add_edge("replanner", "product_agent")
    workflow.add_edge("comparison_agent", "sales_insights")
    workflow.add_edge("sales_insights", END)

    return workflow.compile()


def run_agent_graph(query: str) -> dict:
    """
    Run the full agent pipeline and return state with recommendation and reasoning.
    """
    graph = create_graph()
    initial: AgentState = {
        "query": query,
        "plan": None,
        "initial_plan": None,
        "planner_reasoning": "",
        "products": [],
        "product_reasoning": "",
        "recommendation": "",
        "comparison_reasoning": "",
        "reasoning_steps": [],
        "attempt": 0,
        "max_attempts": 2,
        "sales_insights": {},
        "sales_insights_reasoning": "",
        "dev_metrics": {},
    }
    start = time.perf_counter()
    final_state = graph.invoke(initial)
    elapsed_ms = int((time.perf_counter() - start) * 1000)

    initial_plan = final_state.get("initial_plan") or {}
    final_plan = final_state.get("plan") or {}
    tools_selected = list((final_plan.get("tools") or []))
    tools_initial = list((initial_plan.get("tools") or []))

    dev_metrics = {
        "execution_time_ms": elapsed_ms,
        "steps_taken": len(final_state.get("reasoning_steps") or []),
        "tools_selected_initial": tools_initial,
        "tools_selected_final": tools_selected,
        "loops_replans": int(final_state.get("attempt", 0)),
        "products_returned": len(final_state.get("products") or []),
        "nodes_executed": [
            "planner",
            "product_agent",
            "replanner" if int(final_state.get("attempt", 0)) > 0 else None,
            "comparison_agent",
            "sales_insights",
        ],
    }
    dev_metrics["nodes_executed"] = [n for n in dev_metrics["nodes_executed"] if n]

    final_state["dev_metrics"] = dev_metrics
    return final_state
