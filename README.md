# Lenovo Multi-Agent Sales Intelligence System

An **agentic AI** demo that helps Lenovo sales teams recommend laptops to customers. The system uses a multi-agent pipeline (Planner → Product Agent → Comparison Agent) orchestrated with **LangGraph** and exposes a **Streamlit** chat interface.

## Project description

Users ask questions in natural language (e.g. *“Find the best Lenovo laptop for machine learning under $1800”*, *“Compare Lenovo laptops for gaming”*, *“Recommend a Lenovo laptop for college students”*). The system:

1. **Understands intent** — Planner Agent analyzes the query and decides which tools to use.
2. **Retrieves products** — Product Agent uses semantic/keyword search and optional price filtering.
3. **Compares and recommends** — Comparison Agent evaluates candidates and returns a clear recommendation plus reasoning.

This mirrors enterprise agentic setups where a model plans steps and uses tools, rather than acting as a single chatbot.

## Architecture diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         USER QUERY (Streamlit)                           │
└─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  PLANNER AGENT                                                           │
│  • Analyze query → intent, use_case, budget                              │
│  • Decide tools: search_products, filter_by_price, compare_products      │
└─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  PRODUCT AGENT (Tool use)                                                │
│  • search_products(use_case)  →  candidates                              │
│  • filter_by_price(candidates, budget)  →  filtered list                 │
└─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  COMPARISON AGENT                                                        │
│  • compare_products(products)  →  structured comparison                  │
│  • LLM or template  →  final recommendation text                       │
└─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  FINAL RESPONSE                                                          │
│  • Recommendation text + reasoning steps + recommended laptops           │
└─────────────────────────────────────────────────────────────────────────┘
```

## Project structure

```
lenovo-agentic-ai/
├── data/
│   └── laptops.json           # Lenovo laptop catalog
├── agents/
│   ├── planner_agent.py       # Intent + tool selection
│   ├── product_agent.py       # Search + filter
│   └── comparison_agent.py    # Compare + recommend
├── tools/
│   ├── search_products.py      # Semantic/keyword search (FAISS optional)
│   └── filter_products.py     # filter_by_price, compare_products
├── graph/
│   └── agent_graph.py         # LangGraph workflow
├── frontend/
│   └── streamlit_app.py       # Chat UI + Documentation tab
├── docs/
│   ├── user_guide.md
│   └── developer_guide.md
├── requirements.txt
└── README.md
```

## Setup instructions

### 1. Clone and enter project

```bash
cd lenovo-agentic-ai
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. (Optional) OpenAI for better answers

Set your API key for LLM-powered planner and comparison:

```bash
export OPENAI_API_KEY=sk-...
```

Without it, the app still runs using rule-based planning and template recommendations.

### 5. Run the app

```bash
streamlit run frontend/streamlit_app.py
```

Open the URL shown (e.g. `http://localhost:8501`). Use the **Chat** tab to ask questions and the **Documentation** tab for a short reference and the GitHub link.

## Example queries

- *“Find the best Lenovo laptop for machine learning under $1800.”*
- *“Compare Lenovo laptops for gaming.”*
- *“Recommend a Lenovo laptop for college students.”*
- *“Business laptop under $1500.”*
- *“Lightweight laptop for travel.”*

## Tech stack

| Component | Technology |
|-----------|------------|
| Orchestration | LangGraph |
| LLM (optional) | LangChain + OpenAI |
| Semantic search | SentenceTransformers + FAISS |
| Frontend | Streamlit |
| Language | Python 3.10+ |

All of these can be used with free tiers or local/open-source options.

## Future improvements

- **Conditional branching** — e.g. skip comparison when no products match; retry with relaxed filters.
- **More tools** — availability check, store locator, spec-by-spec comparison.
- **Larger catalog** — scale FAISS index and add filters (brand line, screen size).
- **Evaluation** — benchmark queries and compare rule-based vs LLM planner/comparison.
- **Auth and deployment** — optional login for sales reps; deploy Streamlit to cloud (Streamlit Community Cloud, AWS, etc.).
- **Open-source LLM fallback** — e.g. Ollama/Llamafile for fully local runs without OpenAI.

## License

Use and modify as needed for your Lenovo demo or internal use.
