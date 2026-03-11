"""Product filtering and comparison tools."""

from typing import Optional


def filter_by_price(products: list[dict], max_price: Optional[float] = None) -> list[dict]:
    """Filter products by maximum price. Returns products with price <= max_price."""
    if max_price is None:
        return products
    return [p for p in products if p.get("price", 0) <= max_price]


def compare_products(products: list[dict]) -> str:
    """
    Produce a structured comparison of products by GPU, RAM, price, and use case.
    Returns a text summary suitable for the comparison agent.
    """
    if not products:
        return "No products to compare."

    lines = ["## Product comparison\n"]
    for i, p in enumerate(products, 1):
        lines.append(f"### {i}. {p.get('name', 'Unknown')}")
        lines.append(f"- **Price:** ${p.get('price', 'N/A')}")
        lines.append(f"- **CPU:** {p.get('cpu', 'N/A')}")
        lines.append(f"- **GPU:** {p.get('gpu', 'N/A')}")
        lines.append(f"- **RAM:** {p.get('ram', 'N/A')}")
        lines.append(f"- **Weight:** {p.get('weight', 'N/A')}")
        lines.append(f"- **Use case:** {p.get('use_case', 'N/A')}")
        lines.append(f"- **Description:** {p.get('description', 'N/A')}")
        lines.append("")
    return "\n".join(lines)
