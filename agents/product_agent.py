"""Product Agent: uses search and filter tools to retrieve relevant Lenovo laptops."""

import sys
from pathlib import Path

# Ensure project root is on path
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from tools.search_products import search_products, build_index
from tools.filter_products import filter_by_price


def run_product_agent(plan: dict) -> tuple[list[dict], str]:
    """
    Execute product search and optional price filter based on planner output.
    Returns (list of product dicts, reasoning string).
    """
    use_case = plan.get("use_case") or "general"
    budget = plan.get("budget")
    tools = plan.get("tools", ["search_products", "filter_by_price", "compare_products"])

    reasoning = []
    products = []
    searched = False
    filtered = False

    if "search_products" in tools:
        searched = True
        # Build index on first use (for semantic search)
        try:
            build_index()
        except Exception:
            pass
        query = use_case if use_case != "general" else "Lenovo laptop productivity"
        products = search_products(query, top_k=6)
        reasoning.append(f"Product search for '{query}' returned {len(products)} candidates.")

    if "filter_by_price" in tools and budget is not None and products:
        filtered = True
        before = len(products)
        products = filter_by_price(products, max_price=budget)
        reasoning.append(f"Filtered by max price ${budget}: {before} -> {len(products)} products.")

    if searched and filtered and not products:
        # Important: allow empty results so the Planner can re-run and relax constraints.
        reasoning.append("No candidates remained after filtering; requesting replanning/relaxation.")
        return [], "\n".join(reasoning)

    if not products:
        # Fallback only when search didn't produce anything (or search tool wasn't used)
        from tools.search_products import _load_products
        products = _load_products()
        if budget is not None and "filter_by_price" in tools:
            products = filter_by_price(products, max_price=budget)
        products = products[:6]
        reasoning.append("No results from search; used full catalog (and applied price filter if available).")

    return products, "\n".join(reasoning)
