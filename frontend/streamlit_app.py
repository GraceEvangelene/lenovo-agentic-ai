"""
Lenovo AI Sales Intelligence Agent — Streamlit frontend.
Run with: streamlit run frontend/streamlit_app.py
"""

import sys
from pathlib import Path
import base64
import html
import os

# Add project root for imports
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import streamlit.components.v1 as components
from graph.agent_graph import run_agent_graph
import pandas as pd
import altair as alt

# Markdown renderer for agent output (so comparisons render nicely)
try:
    from markdown_it import MarkdownIt

    _md = MarkdownIt("commonmark", {"html": False}).enable("strikethrough").enable("table")
except Exception:
    _md = None

# Wire NVIDIA API key from Streamlit secrets into environment for NIM clients
if "NVIDIA_API_KEY" in st.secrets:
    os.environ.setdefault("NVIDIA_API_KEY", st.secrets["NVIDIA_API_KEY"])

# Logo paths (tab: LS_logo or L_logo, navbar: L_logo)
DATA_DIR = ROOT / "data"
LOGO_NAVBAR = DATA_DIR / "L_logo.png"
LOGO_TAB = DATA_DIR / "LS_logo.png" if (DATA_DIR / "LS_logo.png").exists() else DATA_DIR / "L_logo.png"
AVATAR_AGENT = DATA_DIR / "avatar_agent.svg"
AVATAR_USER = DATA_DIR / "avatar_user.svg"

# Page config: use logo on tab if available
page_icon = str(LOGO_TAB) if LOGO_TAB.exists() else "💻"
st.set_page_config(
    page_title="Lenovo AI Sales Intelligence Agent",
    page_icon=page_icon,
    layout="wide",
    initial_sidebar_state="collapsed",
)

# White background and navbar styling
st.markdown("""
<style>
    /* White background for entire app */
    .stApp, [data-testid="stAppViewContainer"], main {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    section[data-testid="stSidebar"] {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    .main-header {
        font-size: 1.8rem;
        font-weight: 700;
        color: #E2231A;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        color: #555;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }
    .reasoning-box {
        background: #f8f9fa;
        border-left: 4px solid #E2231A;
        padding: 1rem 1.25rem;
        margin: 0.75rem 0;
        border-radius: 0 8px 8px 0;
        font-size: 0.9rem;
    }
    .product-card {
        background: #fff;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        margin: 0.5rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    .product-name { font-weight: 600; color: #1a1a1a; }
    .product-spec { color: #555; font-size: 0.85rem; }
    .stChatMessage { border-radius: 8px; }

    /* Force chat output text to be visible on white */
    [data-testid="stChatMessageContent"] * {
        color: #000000 !important;
    }

    /* Chat bubble styling (best-effort selectors across Streamlit versions) */
    div[data-testid="stChatMessage"] {
        border-radius: 14px !important;
        padding: 0.35rem 0.2rem !important;
    }

    /* Avatar (tiny icon) styling */
    div[data-testid="stChatMessage"][aria-label="assistant"] [data-testid="stChatMessageAvatar"],
    div[data-testid="stChatMessage"].assistant [data-testid="stChatMessageAvatar"],
    div[data-testid="stChatMessage"].stChatMessage--assistant [data-testid="stChatMessageAvatar"] {
        background: #E2231A !important;
        color: #ffffff !important;
        border: 1px solid rgba(0,0,0,0.06) !important;
    }
    div[data-testid="stChatMessage"][aria-label="assistant"] [data-testid="stChatMessageAvatar"] svg,
    div[data-testid="stChatMessage"][aria-label="assistant"] [data-testid="stChatMessageAvatar"] svg * {
        fill: #ffffff !important;
        color: #ffffff !important;
    }

    div[data-testid="stChatMessage"][aria-label="user"] [data-testid="stChatMessageAvatar"],
    div[data-testid="stChatMessage"].user [data-testid="stChatMessageAvatar"],
    div[data-testid="stChatMessage"].stChatMessage--user [data-testid="stChatMessageAvatar"] {
        background: linear-gradient(135deg, #ffffff 0%, #eef0f3 100%) !important;
        color: #111111 !important;
        border: 1px solid rgba(0,0,0,0.08) !important;
    }
    div[data-testid="stChatMessage"][aria-label="user"] [data-testid="stChatMessageAvatar"] svg,
    div[data-testid="stChatMessage"][aria-label="user"] [data-testid="stChatMessageAvatar"] svg * {
        fill: #111111 !important;
        color: #111111 !important;
    }

    /* Assistant: white bubble + Lenovo red accent line */
    div[data-testid="stChatMessage"][aria-label="assistant"],
    div[data-testid="stChatMessage"].assistant,
    div[data-testid="stChatMessage"].stChatMessage--assistant {
        background: #ffffff !important;
        border: 1px solid #f0f0f0 !important;
        border-left: 6px solid #E2231A !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04) !important;
    }

    /* User: white bubble with subtle gray gradient */
    div[data-testid="stChatMessage"][aria-label="user"],
    div[data-testid="stChatMessage"].user,
    div[data-testid="stChatMessage"].stChatMessage--user {
        background: linear-gradient(135deg, #ffffff 0%, #f3f4f6 100%) !important;
        border: 1px solid #e7e7e7 !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04) !important;
    }
    /* Navbar (Streamlit markdown styles can override p tags, so use high specificity) */
    p.tagline,
    .stMarkdown p.tagline,
    [data-testid="stMarkdownContainer"] p.tagline {
        color: #E2231A !important;
        font-size: clamp(1.5rem, 1.6vw, 2.27rem) !important;
        font-weight: 750 !important;
        line-height: 1.1 !important;
    }

    .navbar {
        display: flex;
        align-items: center;
        gap: 1rem;
        padding: 0.75rem 0 1.25rem 0;
        border-bottom: 1px solid #eee;
        margin-bottom: 1rem;
    }
    .navbar .logo {
        height: 44px;
        width: auto;
        flex: 0 0 auto;
    }

    /* Tabs: make labels visible on white background */
    [data-testid="stTabs"] [data-baseweb="tab"] {
        color: #000000 !important;
    }
    [data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"] {
        color: #E2231A !important;
        font-weight: 800 !important;
    }

    /* Buttons: visible on white background */
    div.stButton > button {
        background: #ffffff !important;
        color: #000000 !important;
        border: 2px solid #E2231A !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
    }
    div.stButton > button:hover {
        background: #E2231A !important;
        color: #ffffff !important;
        border: 2px solid #E2231A !important;
    }

    /* Keep chat input visible (fixed bottom) */
    [data-testid="stChatInput"] {
        position: fixed !important;
        left: 1rem !important;
        right: 1rem !important;
        bottom: 0.85rem !important;
        background: #ffffff !important;
        padding: 0.35rem 0.35rem 0.15rem 0.35rem !important;
        border: 1px solid #e7e7e7 !important;
        border-radius: 14px !important;
        box-shadow: 0 8px 24px rgba(0,0,0,0.10) !important;
        z-index: 2000 !important;
    }
    /* Prevent content from being hidden behind fixed input */
    [data-testid="stAppViewContainer"] main {
        padding-bottom: 8.5rem !important;
    }
</style>
""", unsafe_allow_html=True)

# Navbar: logo + vertically centered tagline (single flex row)
logo_html = ""
if LOGO_NAVBAR.exists():
    encoded = base64.b64encode(LOGO_NAVBAR.read_bytes()).decode("utf-8")
    logo_html = f"<img class='logo' src='data:image/png;base64,{encoded}' alt='Lenovo logo' />"

st.markdown(
    f"""
<div class="navbar">
  {logo_html}
  <p class="tagline" style="margin:0;">Ask. Compare. Choose.  AI-Powered Lenovo Laptop Recommendations.</p>
</div>
""",
    unsafe_allow_html=True,
)

# Tabs: Chat | Sales Insights | Docs
tab_chat, tab_insights, tab_docs = st.tabs(["Chat", "Sales Insights", "Docs"])

with tab_chat:
    st.markdown('<p class="main-header">Lenovo AI Sales Intelligence Agent</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Ask for laptop recommendations (e.g. "Best Lenovo for ML under $1800", "Compare gaming laptops", "Laptop for college students").</p>',
        unsafe_allow_html=True,
    )

    # Chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Developer insights state
    if "show_dev_insights" not in st.session_state:
        st.session_state.show_dev_insights = False

    def _render_dev_insights(metrics: dict):
        if not metrics:
            st.info("No developer insights available yet. Run a recommendation first.")
            return
        st.markdown("### Agentic Planning Metrics")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Execution time (ms)", metrics.get("execution_time_ms"))
        with c2:
            st.metric("Steps taken", metrics.get("steps_taken"))
        with c3:
            st.metric("Loops / replans", metrics.get("loops_replans"))
        with c4:
            st.metric("Products returned", metrics.get("products_returned"))

        st.markdown("### Tools")
        st.write(
            {
                "initial": metrics.get("tools_selected_initial", []),
                "final": metrics.get("tools_selected_final", []),
            }
        )
        st.markdown("### Nodes executed")
        st.write(metrics.get("nodes_executed", []))

    # Chat controls (developer insights / clear)
    ctl0, ctl2, _ = st.columns([0.26, 0.16, 0.58])
    with ctl0:
        if st.button("View developer insights", use_container_width=True):
            st.session_state.show_dev_insights = True
    with ctl2:
        if st.button("Clear chat", use_container_width=True, type="secondary"):
            st.session_state.messages = []
            st.rerun()

    # Modal/popup: developer insights (closable)
    if st.session_state.show_dev_insights:
        latest_metrics = None
        for m in reversed(st.session_state.get("messages", [])):
            if m.get("role") == "assistant" and m.get("dev_metrics"):
                latest_metrics = m.get("dev_metrics")
                break

        if hasattr(st, "dialog"):
            @st.dialog("Developer insights")
            def _dev_dialog():
                _render_dev_insights(latest_metrics or {})
                if st.button("Close"):
                    st.session_state.show_dev_insights = False
                    st.rerun()

            _dev_dialog()
        else:
            with st.expander("Developer insights", expanded=True):
                _render_dev_insights(latest_metrics or {})
                if st.button("Close developer insights"):
                    st.session_state.show_dev_insights = False
                    st.rerun()

    # Scrollable chat window (custom HTML so we control scroll precisely)
    msgs = st.session_state.messages

    def _data_uri(path: Path, mime: str) -> str:
        if not path.exists():
            return ""
        b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
        return f"data:{mime};base64,{b64}"

    agent_avatar_uri = _data_uri(AVATAR_AGENT, "image/svg+xml")
    user_avatar_uri = _data_uri(AVATAR_USER, "image/svg+xml")

    # Find newest user message index (scroll anchor goes right above it)
    last_user_idx = None
    for j in range(len(msgs) - 1, -1, -1):
        if msgs[j].get("role") == "user":
            last_user_idx = j
            break

    blocks: list[str] = []
    for i, msg in enumerate(msgs):
        role = msg.get("role", "assistant")
        is_user = role == "user"
        avatar = user_avatar_uri if is_user else agent_avatar_uri

        if last_user_idx is not None and i == last_user_idx:
            blocks.append("<div id='scroll-target'></div>")

        content = (msg.get("content") or "").strip()
        if content:
            if _md is not None:
                # Render markdown to HTML for nicer formatting
                content_html = _md.render(content)
            else:
                content_html = "<br/>".join(html.escape(content).splitlines())
        else:
            content_html = ""

        reasoning_steps = msg.get("reasoning_steps") or []
        reasoning_html = ""
        if reasoning_steps:
            items = "".join(f"<li>{html.escape(str(s))}</li>" for s in reasoning_steps)
            reasoning_html = f"""
<details class="reasoning">
  <summary>Agent reasoning steps</summary>
  <ul>{items}</ul>
</details>
"""

        products = msg.get("products") or []
        products_html = ""
        if products:
            cards = []
            for p in products:
                name = html.escape(str(p.get("name", "N/A")))
                price = html.escape(str(p.get("price", "N/A")))
                gpu = html.escape(str(p.get("gpu", "N/A")))
                ram = html.escape(str(p.get("ram", "N/A")))
                cards.append(
                    f"""<div class="chat-card">
  <div class="chat-card-title">{name}</div>
  <div class="chat-card-sub">${price} • {gpu} • {ram}</div>
</div>"""
                )
            products_html = (
                "<div class='chat-section-title'>Recommended laptops</div>"
                + "".join(cards)
            )

        blocks.append(
            f"""<div class="msg {'user' if is_user else 'assistant'}">
  <img class="avatar" src="{avatar}" alt="{role} avatar"/>
  <div class="bubble">
    {products_html}
    {reasoning_html}
    {f"<div class='chat-text'>{content_html}</div>" if content_html else ""}
  </div>
</div>"""
        )

    components.html(
        f"""
<div id="chat-scroll" class="chat-scroll">
  {''.join(blocks) if blocks else "<div class='empty'>Ask a question to get started.</div>"}
</div>

<style>
  .chat-scroll {{
    height: 520px;
    overflow-y: auto;
    border: 1px solid #efefef;
    border-radius: 14px;
    padding: 14px 12px;
    background: #ffffff;
  }}
  .msg {{
    display: flex;
    gap: 12px;
    align-items: flex-start;
    margin: 10px 0;
  }}
  .avatar {{
    width: 44px;
    height: 44px;
    border-radius: 14px;
    flex: 0 0 auto;
  }}
  .bubble {{
    width: 100%;
    border-radius: 16px;
    padding: 12px 14px;
    border: 1px solid #ededed;
    background: #ffffff;
    box-shadow: 0 2px 10px rgba(0,0,0,0.04);
  }}
  .msg.assistant .bubble {{
    border-left: 6px solid #E2231A;
  }}
  .msg.user .bubble {{
    background: linear-gradient(135deg, #ffffff 0%, #f3f4f6 100%);
  }}
  .chat-section-title {{
    font-weight: 800;
    margin: 2px 0 10px 0;
    color: #000;
  }}
  .chat-card {{
    border: 1px solid #e6e6e6;
    border-radius: 12px;
    padding: 10px 12px;
    margin: 8px 0;
    background: #fff;
  }}
  .chat-card-title {{ font-weight: 800; }}
  .chat-card-sub {{ color: #444; font-size: 0.92rem; }}
  .chat-text {{ margin-top: 10px; color: #000; }}
  .chat-text h1, .chat-text h2, .chat-text h3 {{ margin: 10px 0 6px 0; }}
  .chat-text h2 {{ font-size: 1.15rem; }}
  .chat-text h3 {{ font-size: 1.05rem; }}
  .chat-text ul {{ margin: 8px 0 8px 20px; }}
  .chat-text li {{ margin: 2px 0; }}
  .chat-text strong {{ font-weight: 800; }}
  .chat-text code {{ background: #f3f4f6; padding: 1px 6px; border-radius: 6px; }}
  .chat-text pre {{ background: #0f172a; color: #e5e7eb; padding: 12px; border-radius: 12px; overflow-x: auto; }}
  .chat-text pre code {{ background: transparent; padding: 0; }}
  details.reasoning {{ margin-top: 6px; }}
  details.reasoning > summary {{ cursor: pointer; font-weight: 800; }}
  details.reasoning ul {{ margin: 8px 0 0 18px; }}
  .empty {{ color: #555; padding: 10px 2px; }}
</style>

<script>
  const scroller = document.getElementById('chat-scroll');
  const target = document.getElementById('scroll-target');
  if (scroller) {{
    if (target) {{
      scroller.scrollTop = Math.max(0, target.offsetTop - 12);
    }} else {{
      scroller.scrollTop = scroller.scrollHeight;
    }}
  }}
</script>
""",
        height=560,
    )

    # Chat input
    if prompt := st.chat_input("Ask for a laptop recommendation..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Run agent pipeline and append assistant message, then rerun so it appears in the scrollable chat window
        with st.spinner("Running agents..."):
            try:
                result = run_agent_graph(prompt)
            except Exception as e:
                result = {
                    "recommendation": f"Something went wrong: {e}",
                    "reasoning_steps": [],
                    "products": [],
                }

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result.get("recommendation", ""),
                "reasoning_steps": result.get("reasoning_steps") or [],
                "products": result.get("products") or [],
                "sales_insights": result.get("sales_insights") or {},
                "dev_metrics": result.get("dev_metrics") or {},
            }
        )
        st.rerun()

with tab_docs:
    st.markdown("## Docs")
    doc_user, doc_dev = st.tabs(["User guide", "Developer documentation"])

    # Load and display docs from files
    user_guide_path = ROOT / "docs" / "user_guide.md"
    dev_guide_path = ROOT / "docs" / "developer_guide.md"

    with doc_user:
        if user_guide_path.exists():
            user_guide_text = user_guide_path.read_text(encoding="utf-8")
            st.markdown(user_guide_text)
        else:
            st.info("User guide file not found at `docs/user_guide.md`.")

    with doc_dev:
        if dev_guide_path.exists():
            dev_guide_text = dev_guide_path.read_text(encoding="utf-8")
            st.markdown(dev_guide_text)
        else:
            st.info("Developer guide file not found at `docs/developer_guide.md`.")

    st.markdown("---")
    st.markdown("### 🔗 GitHub")
    st.markdown(
        "Source code and full documentation:  \n"
        "[**Lenovo Multi-Agent Sales Intelligence System**](https://github.com/GraceEvangelene/lenovo-agentic-ai)"
    )

with tab_insights:
    st.markdown("## Sales Insights")
    # Developer insights button for this tab too
    if st.button("View developer insights", key="dev_insights_sales", use_container_width=False):
        st.session_state.show_dev_insights = True
    # Pull latest assistant message with insights
    latest = None
    for m in reversed(st.session_state.get("messages", [])):
        if m.get("role") == "assistant" and m.get("sales_insights"):
            latest = m
            break

    if not latest:
        st.info("Run a recommendation first to see Sales Insights based on reviews.")
    else:
        insights = latest.get("sales_insights", {})
        per_product = (insights.get("per_product") or {}) if isinstance(insights, dict) else {}
        narrative = (insights.get("narrative") or "") if isinstance(insights, dict) else ""

        product_names = list(per_product.keys())
        if product_names:
            # Visual comparison across all products
            rows = []
            for name, s in per_product.items():
                if not isinstance(s, dict) or not s.get("available"):
                    continue
                sc = s.get("sentiment_counts", {}) or {}
                rows.append(
                    {
                        "product": name,
                        "avg_rating": s.get("avg_rating"),
                        "review_count": s.get("review_count", 0),
                        "positive": sc.get("positive", 0),
                        "neutral": sc.get("neutral", 0),
                        "negative": sc.get("negative", 0),
                        "top_themes": ", ".join(s.get("top_themes", []) or []),
                    }
                )

            if rows:
                df = pd.DataFrame(rows)
                st.markdown("### Visual comparison (all recommended laptops)")

                c1, c2 = st.columns([0.5, 0.5])
                with c1:
                    st.markdown("**Average rating**")
                    chart_rating = (
                        alt.Chart(df)
                        .mark_bar(color="#E2231A")
                        .encode(
                            x=alt.X("avg_rating:Q", title="Avg rating", scale=alt.Scale(domain=[0, 5])),
                            y=alt.Y("product:N", title=None, sort="-x"),
                            tooltip=["product", "avg_rating", "review_count"],
                        )
                        .properties(height=220)
                    )
                    st.altair_chart(chart_rating, use_container_width=True)

                with c2:
                    st.markdown("**Sentiment mix**")
                    df_long = df.melt(
                        id_vars=["product"],
                        value_vars=["positive", "neutral", "negative"],
                        var_name="sentiment",
                        value_name="count",
                    )
                    color_scale = alt.Scale(
                        domain=["positive", "neutral", "negative"],
                        range=["#16a34a", "#6b7280", "#dc2626"],
                    )
                    chart_sent = (
                        alt.Chart(df_long)
                        .mark_bar()
                        .encode(
                            x=alt.X("count:Q", title="Count"),
                            y=alt.Y("product:N", title=None, sort=alt.SortField(field="product", order="ascending")),
                            color=alt.Color("sentiment:N", scale=color_scale, title=None),
                            tooltip=["product", "sentiment", "count"],
                        )
                        .properties(height=220)
                    )
                    st.altair_chart(chart_sent, use_container_width=True)

                st.markdown("**Themes snapshot**")
                st.dataframe(
                    df[["product", "top_themes", "review_count"]],
                    use_container_width=True,
                    hide_index=True,
                )

                st.markdown("---")

            selected = st.selectbox("Product", product_names)
            s = per_product.get(selected, {})
            if not s or not s.get("available"):
                st.warning("No review insights available for this product.")
            else:
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Avg rating", s.get("avg_rating"))
                with c2:
                    st.metric("Reviews", s.get("review_count"))
                with c3:
                    sc = s.get("sentiment_counts", {})
                    st.metric(
                        "Sentiment (+/=/−)",
                        f"{sc.get('positive', 0)}/{sc.get('neutral', 0)}/{sc.get('negative', 0)}",
                    )

                st.markdown("### Top themes")
                st.write(", ".join(s.get("top_themes", []) or []))

                st.markdown("### Highlights")
                for h in s.get("highlights", []) or []:
                    st.markdown(f"- {h}")

                st.markdown("### Watch-outs")
                for c in s.get("cautions", []) or []:
                    st.markdown(f"- {c}")

        if narrative:
            st.markdown("---")
            st.markdown("### Sales-ready summary")
            st.markdown(narrative)
