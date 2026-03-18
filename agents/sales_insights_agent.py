"""Sales Insights Agent: summarizes review signals for recommended products."""

from __future__ import annotations

import os

from openai import OpenAI

from tools.review_insights import sales_insights_for_products

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
client: OpenAI | None = None
if NVIDIA_API_KEY:
    client = OpenAI(api_key=NVIDIA_API_KEY, base_url="https://integrate.api.nvidia.com/v1")


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
      - narrative: LLM-written sales bullets when NVIDIA_API_KEY is set (empty otherwise)
    """
    names = [p.get("name") for p in (products or []) if p.get("name")]
    per_product = sales_insights_for_products(names)
    reasoning = f"Loaded review insights for {len(names)} products."

    if client is not None and names:
        try:
            resp = client.chat.completions.create(
                model="meta/llama-3.1-70b-instruct",
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": str(per_product)},
                ],
                temperature=0.2,
            )
            text = (resp.choices[0].message.content or "").strip()
            resp = text
            return {"per_product": per_product, "narrative": resp}, reasoning
        except Exception as e:
            reasoning += f" LLM failed ({e}); using structured insights only."

    return {"per_product": per_product, "narrative": ""}, reasoning

