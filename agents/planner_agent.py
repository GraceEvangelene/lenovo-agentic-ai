"""Planner Agent: analyzes user query, determines intent, and decides which tools to use."""

import json
import re
import os
from typing import Any

# Optional LangChain
try:
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

PLANNER_SYSTEM = """You are a sales planning agent for Lenovo. Analyze the user's query and output a JSON object with:
- "intent": one of "laptop_recommendation", "compare_laptops", "product_search", "general"
- "use_case": extracted use case (e.g. "machine learning", "gaming", "students", "business") or empty string
- "budget": maximum price in dollars as a number, or null if not mentioned
- "tools": list of tools to use, from ["search_products", "filter_by_price", "compare_products"]

Only output valid JSON, no other text."""


def _parse_budget(text: str) -> float | None:
    """Extract budget from natural language (e.g. 'under $1800', 'max 1500')."""
    # $1800, $1,800, 1800 dollars, under 1800
    patterns = [
        # Prefer explicit budget cues first (prevents partial matches like "100" from "1000")
        r"(?:under|below|max|maximum|up to)\s*\$?\s*(\d+(?:,\d{3})*(?:\.\d+)?)",
        # Generic currency/number mention (handles 4+ digits without commas)
        r"\$?\s*(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:dollars?|usd)?",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.I)
        if m:
            num = m.group(1).replace(",", "")
            try:
                return float(num)
            except ValueError:
                pass
    return None


def _parse_use_case(text: str) -> str:
    """Simple keyword extraction for use case."""
    text_lower = text.lower()
    mapping = [
        ("machine learning", ["machine learning", "ml", "ai", "deep learning", "training models"]),
        ("gaming", ["gaming", "games", "gamer"]),
        ("students", ["student", "college", "school", "university"]),
        ("business", ["business", "work", "office", "enterprise"]),
        ("creators", ["creator", "creative", "design", "video editing"]),
        ("portability", ["portable", "lightweight", "travel", "on the go"]),
    ]
    for use_case, keywords in mapping:
        if any(k in text_lower for k in keywords):
            return use_case
    return ""


def _fallback_plan(query: str) -> dict[str, Any]:
    """Rule-based planner when LLM is not available."""
    intent = "laptop_recommendation"
    if any(w in query.lower() for w in ["compare", "comparison", "vs", "versus"]):
        intent = "compare_laptops"
    use_case = _parse_use_case(query)
    budget = _parse_budget(query)
    tools = ["search_products"]
    if budget is not None:
        tools.append("filter_by_price")
    tools.append("compare_products")
    return {
        "intent": intent,
        "use_case": use_case or "general",
        "budget": budget,
        "tools": tools,
    }


def run_planner(query: str) -> tuple[dict[str, Any], str]:
    """
    Run the planner agent. Returns (plan_dict, reasoning_string).
    Uses OpenAI if LANGCHAIN_AVAILABLE and OPENAI_API_KEY is set; else fallback.
    """
    reasoning_parts = []

    if LANGCHAIN_AVAILABLE and os.getenv("OPENAI_API_KEY"):
        try:
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            messages = [
                SystemMessage(content=PLANNER_SYSTEM),
                HumanMessage(content=query),
            ]
            response = llm.invoke(messages)
            content = response.content.strip()
            reasoning_parts.append(f"Planner (LLM) raw output: {content[:200]}...")
            # Extract JSON from response (handle markdown code blocks)
            json_match = re.search(r"\{[^{}]*\}", content, re.DOTALL)
            if json_match:
                plan = json.loads(json_match.group())
                plan.setdefault("tools", ["search_products", "filter_by_price", "compare_products"])
                plan.setdefault("use_case", _parse_use_case(query) or "general")
                plan.setdefault("budget", _parse_budget(query))
                reasoning_parts.append(f"Intent: {plan.get('intent')}, use_case: {plan.get('use_case')}, budget: {plan.get('budget')}")
                return plan, "\n".join(reasoning_parts)
        except Exception as e:
            reasoning_parts.append(f"LLM fallback due to: {e}")

    plan = _fallback_plan(query)
    reasoning_parts.append(f"Rule-based plan: intent={plan['intent']}, use_case={plan['use_case']}, budget={plan['budget']}, tools={plan['tools']}")
    return plan, "\n".join(reasoning_parts)
