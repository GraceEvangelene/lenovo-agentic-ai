"""Comparison Agent: compares candidate laptops and generates a final recommendation."""

import os
import sys
from pathlib import Path
import re

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from tools.filter_products import compare_products

try:
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False


COMPARISON_SYSTEM = """You are a Lenovo sales expert. Given a comparison of Lenovo laptops, write a short recommendation (2-4 paragraphs) that:
1. Summarizes the best options for the customer's stated need
2. Highlights key differentiators (GPU, RAM, price, use case)
3. Gives a clear top recommendation with a brief justification
4. Uses a friendly, professional tone. Do not make up specs; only use the provided data."""


def _parse_ram_gb(ram: str | None) -> int:
    if not ram:
        return 0
    m = re.search(r"(\d+)", str(ram))
    return int(m.group(1)) if m else 0


def _parse_weight_lbs(weight: str | None) -> float:
    if not weight:
        return 0.0
    m = re.search(r"(\d+(?:\.\d+)?)", str(weight))
    return float(m.group(1)) if m else 0.0


def _gpu_score(gpu: str | None) -> int:
    """Rough GPU tier scoring for demo ranking."""
    if not gpu:
        return 0
    g = str(gpu).lower()
    if "rtx 4070" in g:
        return 10
    if "rtx 4060" in g:
        return 9
    if "rtx a2000" in g:
        return 8
    if "rtx 3060" in g:
        return 7
    if "rtx 3050" in g:
        return 5
    if "rx" in g or "radeon" in g:
        return 4
    if "iris" in g:
        return 3
    if "uhd" in g:
        return 1
    return 2


def _use_case_fit(product_use_case: str, target_use_case: str) -> int:
    if not target_use_case or target_use_case == "general":
        return 1
    p = (product_use_case or "").lower()
    t = target_use_case.lower()
    return 5 if t in p else 0


def _rank_products(products: list[dict], plan: dict) -> list[dict]:
    """Attach a '_score' field and return products sorted by score desc."""
    target_use_case = plan.get("use_case", "general")
    budget = plan.get("budget")

    ranked: list[dict] = []
    for p in products:
        price = float(p.get("price", 0) or 0)
        ram_gb = _parse_ram_gb(p.get("ram"))
        weight_lbs = _parse_weight_lbs(p.get("weight"))
        gpu_tier = _gpu_score(p.get("gpu"))
        fit = _use_case_fit(p.get("use_case", ""), target_use_case)

        score = 0.0
        score += fit * 2.0
        score += gpu_tier * (2.0 if target_use_case in ["machine learning", "gaming"] else 1.0)
        score += min(ram_gb, 64) / 8.0  # up to +8
        if weight_lbs:
            score += max(0.0, 5.0 - weight_lbs) * (
                1.0 if target_use_case in ["students", "portability"] else 0.3
            )
        if budget:
            b = float(budget)
            # Budget preference:
            # - If under budget: small bonus for being comfortably under.
            # - If over budget: strong penalty, so closest-to-budget wins when no exact matches exist.
            diff_ratio = abs(price - b) / max(b, 1.0)
            score -= diff_ratio * 20.0
            if price > b:
                over_ratio = (price - b) / max(b, 1.0)
                score -= over_ratio * 10.0
            else:
                under_ratio = (b - price) / max(b, 1.0)
                score += min(under_ratio, 1.0) * 2.0

        ranked.append({**p, "_score": round(score, 2)})

    ranked.sort(key=lambda x: x.get("_score", 0), reverse=True)
    return ranked


def rank_products(products: list[dict], plan: dict) -> list[dict]:
    """Public helper for UI/graph to sort product cards by the same ranking."""
    return _rank_products(products, plan)


def _pros_cons(p: dict, plan: dict) -> tuple[list[str], list[str]]:
    target_use_case = plan.get("use_case", "general")
    budget = plan.get("budget")

    pros: list[str] = []
    cons: list[str] = []

    price = float(p.get("price", 0) or 0)
    ram_gb = _parse_ram_gb(p.get("ram"))
    weight_lbs = _parse_weight_lbs(p.get("weight"))
    gpu_tier = _gpu_score(p.get("gpu"))

    if budget:
        b = float(budget)
        if price <= b:
            pros.append(f"Within budget (${int(price)} ≤ ${int(b)})")
        else:
            cons.append(f"Over budget (${int(price)} > ${int(b)})")

    if ram_gb >= 32:
        pros.append(f"High memory ({ram_gb}GB RAM)")
    elif ram_gb >= 16:
        pros.append(f"Solid memory ({ram_gb}GB RAM)")
    else:
        cons.append(f"Lower memory for heavy workloads ({ram_gb}GB RAM)")

    if target_use_case in ["machine learning", "gaming"]:
        if gpu_tier >= 8:
            pros.append(f"Strong GPU for {target_use_case} ({p.get('gpu')})")
        elif gpu_tier >= 5:
            pros.append(f"Capable GPU for light {target_use_case} ({p.get('gpu')})")
        else:
            cons.append(f"Entry/integrated GPU may limit {target_use_case} ({p.get('gpu')})")
    else:
        if gpu_tier <= 3:
            pros.append("Efficient integrated graphics (good for everyday tasks)")

    if weight_lbs:
        if weight_lbs <= 3.0:
            pros.append(f"Very portable ({weight_lbs} lbs)")
        elif weight_lbs >= 5.0:
            cons.append(f"Heavier to carry ({weight_lbs} lbs)")

    use_case_text = (p.get("use_case") or "").lower()
    if target_use_case and target_use_case != "general":
        if target_use_case.lower() in use_case_text:
            pros.append(f"Designed for {target_use_case}")
        else:
            cons.append(f"Not specifically targeted for {target_use_case}")

    return pros[:4], cons[:3]


def run_comparison_agent(products: list[dict], user_query: str, plan: dict) -> tuple[str, str]:
    """
    Compare products and generate recommendation text.
    Returns (recommendation_text, reasoning_string).
    """
    ranked = _rank_products(products, plan)
    comparison_text = compare_products(ranked)
    reasoning = f"Ranked {len(ranked)} products and generated pros/cons."

    if LANGCHAIN_AVAILABLE and os.getenv("OPENAI_API_KEY") and products:
        try:
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
            messages = [
                SystemMessage(content=COMPARISON_SYSTEM),
                HumanMessage(content=f"User request: {user_query}\n\n{comparison_text}"),
            ]
            response = llm.invoke(messages)
            return response.content.strip(), reasoning
        except Exception as e:
            reasoning += f"; LLM failed ({e}), using template."

    # Template-based ranked recommendation with pros/cons when no LLM
    if not os.getenv("OPENAI_API_KEY"):
        reasoning += " (No OPENAI_API_KEY set; using non-LLM summary.)"
    use_case = plan.get("use_case", "general")
    budget = plan.get("budget")

    if not ranked:
        return "No products to compare.", reasoning

    lines: list[str] = []
    if not os.getenv("OPENAI_API_KEY"):
        lines.append(
            "> Note: This would be a better narrative summary with an OpenAI key, but I'm currently out of tokens.\n"
        )
    if budget:
        b = float(budget)
        within = [p for p in ranked if float(p.get("price", 0) or 0) <= b]
        if not within:
            closest = min(ranked, key=lambda p: abs(float(p.get("price", 0) or 0) - b))
            lines.append(
                f"> ⚠️ No laptops match **under ${int(b)}** exactly in this demo catalog. "
                f"Here are the closest alternatives, starting with **{closest.get('name')} (${closest.get('price')})**.\n"
            )
    lines.append("## Ranked recommendations\n")
    if use_case and use_case != "general":
        lines.append(f"**Use case:** {use_case}\n")
    if budget:
        lines.append(f"**Budget:** up to ${int(float(budget))}\n")

    top = ranked[0]
    lines.append(f"### Top pick: {top.get('name')}\n")
    lines.append(
        f"- **Price:** ${top.get('price')}  \n"
        f"- **GPU:** {top.get('gpu')}  \n"
        f"- **RAM:** {top.get('ram')}  \n"
    )

    lines.append("\n### Ranked list (with pros/cons)\n")
    for i, p in enumerate(ranked, 1):
        pros, cons = _pros_cons(p, plan)
        lines.append(f"#### {i}. {p.get('name')} (score: {p.get('_score')})\n")
        lines.append(
            f"- **Price:** ${p.get('price')} | **GPU:** {p.get('gpu')} | **RAM:** {p.get('ram')}\n"
        )
        if pros:
            lines.append("- **Pros:**\n" + "\n".join([f"  - {pr}" for pr in pros]) + "\n")
        if cons:
            lines.append("- **Cons:**\n" + "\n".join([f"  - {co}" for co in cons]) + "\n")

    lines.append("\n---\n")
    lines.append("### Full comparison\n")
    lines.append(comparison_text)

    return "\n".join(lines).strip(), reasoning
