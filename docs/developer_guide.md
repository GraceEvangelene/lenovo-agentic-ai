# Lenovo AI Sales Intelligence System — Developer Guide

## Architecture overview

The system is a **multi-agent pipeline** orchestrated by **LangGraph** with **looping replanning** and an additional **Sales Insights** stage. Each step is implemented as a node; state is passed across nodes.

```
User Query
    ↓
Planner Agent  (analyze intent, budget, use case; decide tools)
    ↓
Product Agent  (search_products, filter_by_price)
    ↓
Replanner (conditional loop; relax constraints when filtering removes all results)
    ↓
Comparison Agent (compare by GPU, RAM, price, use case; generate recommendation)
    ↓
Sales Insights Agent (summarize review signals: rating/sentiment/themes/highlights/watch-outs)
    ↓
Final Response (recommendation + reasoning + product list + sales insights + dev metrics)
```

- **Frontend:** Streamlit (`frontend/streamlit_app.py`) — Chat, Docs, Sales Insights tabs + developer insights popup.
- **Orchestration:** LangGraph graph in `graph/agent_graph.py`.
- **Agents:** `agents/planner_agent.py`, `agents/product_agent.py`, `agents/comparison_agent.py`, `agents/sales_insights_agent.py`.
- **Tools:** `tools/search_products.py`, `tools/filter_products.py`, `tools/review_insights.py`.
- **Data:** `data/laptops.json` (catalog) and `data/reviews.json` (review dataset for sales insights).

## Agent workflow

### 1. Planner Agent

- **Role:** Analyze the user query and produce a structured plan.
- **Input:** Raw user message (string).
- **Output:** A plan dict with:
  - `intent`: e.g. `laptop_recommendation`, `compare_laptops`, `product_search`, `general`
  - `use_case`: e.g. `machine learning`, `gaming`, `students`, `business`
  - `budget`: max price in dollars or `null`
  - `tools`: list of tools to run, e.g. `["search_products", "filter_by_price", "compare_products"]`
- **Implementation:** Uses OpenAI (if `OPENAI_API_KEY` is set) with a system prompt that asks for JSON; otherwise a rule-based fallback parses budget and use-case keywords.

### 2. Product Agent

- **Role:** Retrieve relevant Lenovo laptops using the planner’s plan.
- **Input:** The plan from the Planner Agent.
- **Output:** A list of product dicts and a short reasoning string.
- **Behavior:**
  - Calls `search_products(use_case)` (semantic search with SentenceTransformers + FAISS, or keyword fallback).
  - If the plan includes a budget, calls `filter_by_price(products, max_price)`.
  - Returns up to a small set of candidates (e.g. 6) for the Comparison Agent.
  - If filtering removes all candidates, it can return an empty list to trigger replanning.

### 3. Comparison Agent

- **Role:** Compare the candidate laptops and produce a natural-language recommendation.
- **Input:** List of products, original user query, and plan.
- **Output:** Recommendation text and a short reasoning string.
- **Behavior:**
  - Ranks candidates and outputs a **ranked list with pros/cons** (GPU tier, RAM, price/budget closeness, weight/portability, and use-case fit).
  - If nothing matches budget exactly (after replanning), it emits a **warning** and suggests closest alternatives.
  - If OpenAI is available, an LLM can generate a more polished narrative; otherwise it produces a deterministic ranked summary.

### 4. Sales Insights Agent

- **Role:** Add sales-ready context from review signals.
- **Input:** The final ranked product list.
- **Output:** `sales_insights` dict with per-product metrics and (optional) narrative.
- **Behavior:** Summarizes `data/reviews.json` into average rating, sentiment mix, top themes, highlight quotes, and watch-outs. If OpenAI is available, it can generate concise selling points and watch-outs.

## LangGraph orchestration

- **State:** `AgentState` in `graph/agent_graph.py` holds:
  - `query`, `plan`, `initial_plan`, `products`, `recommendation`, `sales_insights`, `reasoning_steps`, `attempt`, `dev_metrics`, and supporting reasoning fields.
- **Nodes:**
  - `planner` → runs the Planner Agent, writes `plan` and `planner_reasoning`.
  - `product_agent` → runs the Product Agent, writes `products` and `product_reasoning`.
  - `replanner` → runs when `product_agent` returns 0 products; relaxes constraints and loops back to `product_agent`.
  - `comparison_agent` → ranks products and generates recommendation text.
  - `sales_insights` → summarizes review signals for recommended products.
- **Edges:** `START → planner → product_agent → (replanner ↺ product_agent)* → comparison_agent → sales_insights → END`
- **Invocation:** `run_agent_graph(query)` compiles the graph, invokes it, and returns the final state including `dev_metrics`.

## Developer analytics (Agentic Planning Metrics)

The graph runner records basic execution metrics in `dev_metrics`:

- `execution_time_ms`
- `steps_taken` (reasoning steps count)
- `tools_selected_initial` vs `tools_selected_final`
- `loops_replans`
- `products_returned`
- `nodes_executed`

The Streamlit UI exposes these in a **“View developer insights”** popup for demo/debugging.

## Tools

| Tool | Module | Purpose |
|------|--------|--------|
| `search_products(query, top_k, use_semantic)` | `tools/search_products.py` | Returns laptops relevant to the query (use case/description). Uses SentenceTransformers + FAISS when available; otherwise keyword matching. |
| `filter_by_price(products, max_price)` | `tools/filter_products.py` | Filters a list of product dicts by `price <= max_price`. |
| `compare_products(products)` | `tools/filter_products.py` | Builds a text comparison of products (GPU, RAM, price, use case, description) for the Comparison Agent. |
| `sales_insights_for_products(product_names)` | `tools/review_insights.py` | Returns per-product review summaries (avg rating, sentiment mix, themes, highlights, watch-outs). |

Product dicts follow the schema in `data/laptops.json`: `name`, `price`, `cpu`, `gpu`, `ram`, `weight`, `use_case`, `description`.

## Tech stack (summary)

- **Python 3.10+**
- **LangGraph** — graph orchestration and state.
- **LangChain** — optional; used for OpenAI chat (planner and comparison) when `OPENAI_API_KEY` is set.
- **OpenAI** — optional; improves planner and comparison quality.
- **SentenceTransformers + FAISS** — optional; improves semantic product search.
- **Streamlit** — web UI (chat + documentation tab).

## Extending the system

- **New tools:** Add functions in `tools/` and have the Planner decide when to use them; call them from the Product Agent (or a dedicated tool-execution node) and pass results in state.
- **New agents:** Add a new node in `graph/agent_graph.py`, update state and edges as needed.
- **Conditional flow:** Replace fixed edges with conditional edges based on state (e.g. additional loops, fallbacks) using LangGraph’s `add_conditional_edges`.
- **More data:** Extend `data/laptops.json` and re-run index build if you use FAISS; the rest of the pipeline stays the same.
