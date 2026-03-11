"""Sales Insights Agent: summarizes review signals for recommended products."""

from __future__ import annotations

import os

try:
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

from tools.review_insights import sales_insights_for_products


SYSTEM = """You are a sales insights agent. Given structured review summaries per Lenovo product, write:
- 3-5 bullet \"selling points\" per product
- 1-2 bullet \"watch-outs\" per product
- Keep it concise and sales-ready
Do not invent information; only use the provided summaries."""


def run_sales_insights_agent(products: list[dict]) -> tuple[dict, str]:
    """
    Returns (insights_dict, reasoning_string).
    insights_dict includes:
      - per_product: structured signals from reviews.json
      - narrative (optional): LLM-written sales bullets if OPENAI_API_KEY is set
    """
    names = [p.get("name") for p in (products or []) if p.get("name")]
    per_product = sales_insights_for_products(names)
    reasoning = f"Loaded review insights for {len(names)} products."

    if LANGCHAIN_AVAILABLE and os.getenv("OPENAI_API_KEY") and names:
        try:
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
            messages = [
                SystemMessage(content=SYSTEM),
                HumanMessage(content=str(per_product)),
            ]
            resp = llm.invoke(messages).content.strip()
            return {"per_product": per_product, "narrative": resp}, reasoning
        except Exception as e:
            reasoning += f" LLM failed ({e}); using structured insights only."

    return {"per_product": per_product, "narrative": ""}, reasoning

