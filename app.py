from __future__ import annotations

import base64
import html
import os
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from rag_engine import LegalRetriever, load_corpus, run_agent, source_summary
from trusted_sources import ingest_trusted_url
from pdf_ingestor import DuplicatePDFError, ingest_pdf, preview_pdf
from pdf_parser import PDFIngestionError
load_dotenv()

st.set_page_config(
    page_title="NyayaRoute | Institutional Intelligence",
    page_icon="⚖",
    layout="wide",
)

# Function to load and base64 encode the user's uploaded legal image
def get_image_b64() -> str:
    paths = [
        Path(r"C:\Users\Vybha\.gemini\antigravity-ide\brain\26493043-ee59-4102-beff-3c715ce5832a\media__1784889718562.png"),
        Path("legal_hero.png"),
    ]
    for p in paths:
        if p.exists():
            try:
                with open(p, "rb") as f:
                    return base64.b64encode(f.read()).decode("utf-8")
            except Exception:
                pass
    return ""

img_b64 = get_image_b64()
img_src = f"data:image/png;base64,{img_b64}" if img_b64 else "https://images.unsplash.com/photo-1589829545856-d10d557cf95f?auto=format&fit=crop&w=1200&q=80"

# ── Pure CSS Design Integration & Image Animations (NO JS SCRIPTS) ────────────
st.markdown(f"""
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;0,700;0,800;1,400&family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@24,300,0,0" rel="stylesheet">

<style>
  /* Base background and typography */
  html, body, .stApp {{
    background-color: #faf9f7 !important;
    color: #1a1c1b !important;
    font-family: 'Inter', sans-serif !important;
  }}

  /* Hide default Streamlit chrome */
  #MainMenu, footer, header {{ visibility: hidden; }}
  .block-container {{
    padding-top: 1rem !important;
    padding-bottom: 2rem !important;
    max-width: 1240px !important;
  }}

  /* Fonts for headings */
  h1, h2, h3, h4, h5, h6,
  [data-testid="stMarkdownContainer"] h1,
  [data-testid="stMarkdownContainer"] h2,
  [data-testid="stMarkdownContainer"] h3 {{
    font-family: 'Playfair Display', serif !important;
    color: #1a1c1b !important;
    letter-spacing: -0.01em;
  }}

  /* Body text color enforcement */
  p, span, label, li, div, [data-testid="stMarkdownContainer"] p {{
    color: #1a1c1b;
    font-family: 'Inter', sans-serif;
  }}

  /* Sidebar styling */
  [data-testid="stSidebar"] {{
    background-color: #f4f3f1 !important;
    border-right: 1px solid #c5c6ce !important;
  }}
  [data-testid="stSidebar"] * {{
    color: #1a1c1b !important;
  }}

  /* Tabs styling */
  button[data-baseweb="tab"] p {{
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    color: #44474d !important;
  }}
  button[aria-selected="true"][data-baseweb="tab"] p {{
    color: #000000 !important;
  }}
  button[aria-selected="true"][data-baseweb="tab"] {{
    border-bottom: 2px solid #735a3a !important;
  }}

  /* Text area styling */
  .stTextArea textarea {{
    background-color: #ffffff !important;
    border: 1px solid #c5c6ce !important;
    border-radius: 4px !important;
    font-family: 'Playfair Display', serif !important;
    font-style: italic !important;
    font-size: 1.05rem !important;
    color: #1a1c1b !important;
    padding: 0.85rem !important;
  }}
  .stTextArea textarea:focus {{
    border-color: #735a3a !important;
    box-shadow: 0 0 0 1px #735a3a !important;
  }}

  /* Buttons styling */
  .stButton button {{
    background-color: #ffffff !important;
    color: #1a1c1b !important;
    border: 1px solid #c5c6ce !important;
    border-radius: 4px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
    padding: 0.6rem 1rem !important;
    transition: all 0.2s ease !important;
  }}
  .stButton button:hover {{
    background-color: #efeeec !important;
    border-color: #735a3a !important;
  }}
  .stButton button[kind="primary"] {{
    background-color: #000000 !important;
    color: #ffffff !important;
    border: none !important;
  }}
  .stButton button[kind="primary"] p {{
    color: #ffffff !important;
  }}

  /* Download Button */
  .stDownloadButton button {{
    background-color: transparent !important;
    color: #b9c7e6 !important;
    border: 1px solid #b9c7e6 !important;
    border-radius: 4px !important;
    width: 100% !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    font-weight: 600 !important;
    padding: 0.7rem !important;
  }}
  .stDownloadButton button p {{
    color: #b9c7e6 !important;
  }}

  /* Expanders */
  [data-testid="stExpander"] {{
    background-color: #ffffff !important;
    border: 1px solid #c5c6ce !important;
    border-radius: 4px !important;
    margin-bottom: 0.75rem !important;
  }}
  [data-testid="stExpander"] summary span p {{
    font-family: 'Playfair Display', serif !important;
    font-weight: 600 !important;
    color: #1a1c1b !important;
  }}

  /* Material Symbols styling */
  .material-symbols-outlined {{
    font-family: 'Material Symbols Outlined' !important;
    font-weight: normal;
    font-style: normal;
    display: inline-block;
    line-height: 1;
    text-transform: none;
    letter-spacing: normal;
    word-wrap: normal;
    white-space: nowrap;
    direction: ltr;
  }}

  /* ── IMAGE ANIMATIONS ── */
  @keyframes heroFloat {{
    0% {{
      transform: translateY(0px);
      box-shadow: 0 15px 35px rgba(12, 27, 51, 0.16), 0 0 0 1px rgba(115, 90, 58, 0.3);
    }}
    50% {{
      transform: translateY(-12px);
      box-shadow: 0 28px 50px rgba(115, 90, 58, 0.28), 0 0 0 1px rgba(115, 90, 58, 0.7);
    }}
    100% {{
      transform: translateY(0px);
      box-shadow: 0 15px 35px rgba(12, 27, 51, 0.16), 0 0 0 1px rgba(115, 90, 58, 0.3);
    }}
  }}

  @keyframes kenBurnsZoom {{
    0% {{ transform: scale(1.0); }}
    50% {{ transform: scale(1.07); }}
    100% {{ transform: scale(1.0); }}
  }}

  @keyframes shimmerGlow {{
    0% {{ opacity: 0.4; }}
    50% {{ opacity: 0.95; }}
    100% {{ opacity: 0.4; }}
  }}

  .animated-hero-card {{
    animation: heroFloat 7s ease-in-out infinite;
    border-radius: 8px;
    overflow: hidden;
    position: relative;
    background: #0c1b33;
    transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1);
  }}

  .animated-hero-card:hover {{
    transform: translateY(-16px) scale(1.01);
  }}

  .animated-hero-img {{
    animation: kenBurnsZoom 14s ease-in-out infinite;
    width: 100%;
    height: 330px;
    object-fit: cover;
    display: block;
  }}

  .hero-img-badge {{
    animation: shimmerGlow 3s ease-in-out infinite;
  }}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Preparing legal retrieval workspace…")
def get_retriever() -> LegalRetriever:
    return LegalRetriever(load_corpus())


def reset_retriever() -> None:
    get_retriever.clear()


def render_source_links(actions: list[dict]) -> None:
    for action in actions:
        st.markdown(f"- [{action['label']}]({action['url']}) — {action['why']}")


def memo(question: str, result: dict) -> str:
    sources = "\n".join(
        f"- [S{i}] {item['title']} — {item['url']}"
        for i, item in enumerate(result["evidence"], 1)
    ) or "- No adequate source material was retrieved."
    outcome = result["answer"] or f"Insufficient evidence: {result['evidence_reason']}"
    return f"""# NyayaRoute Legal Research Memo

**Question:** {question}

**Research outcome**

{outcome}

## Sources consulted
{sources}

## Research audit
- Query quality: {result['quality_reason']}
- Search query: {result['query_used']}
- Retrieval route: {result['route']} — {result['route_reason']}
- Evidence decision: {result['evidence_reason']}
- Retry used: {'Yes' if result['retried'] else 'No'}

*This is legal research information, not legal advice. Verify the current primary text and seek qualified legal counsel for advice.*
"""


# ── Top Navigation Bar ────────────────────────────────────────────────────────
st.markdown("""
<div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid rgba(197,198,206,0.5); padding:0.6rem 0 1rem; margin-bottom:1.5rem;">
  <div style="display:flex; align-items:center; gap:2rem;">
    <span style="font-family:'Playfair Display',serif; font-size:1.35rem; font-weight:700; text-transform:uppercase; letter-spacing:0.15em; color:#000;">NYAYAROUTE</span>
    <div style="display:flex; gap:1.5rem; align-items:center;">
      <span style="font-family:'Inter',sans-serif; font-size:0.82rem; font-weight:600; color:#000; border-bottom:1.5px solid #000; padding-bottom:2px;">Research</span>
      <span style="font-family:'Inter',sans-serif; font-size:0.82rem; color:#44474d;">Precedents</span>
      <span style="font-family:'Inter',sans-serif; font-size:0.82rem; color:#44474d;">Library</span>
      <span style="font-family:'Inter',sans-serif; font-size:0.82rem; color:#44474d;">Archive</span>
    </div>
  </div>
  <div style="display:flex; align-items:center; gap:1.25rem;">
    <span class="material-symbols-outlined" style="color:#44474d; font-size:22px; cursor:pointer;">account_balance</span>
    <span class="material-symbols-outlined" style="color:#44474d; font-size:22px; cursor:pointer;">settings</span>
    <div style="background:#0c1b33; color:#ffffff; font-family:'Inter',sans-serif; font-size:0.72rem; font-weight:600; text-transform:uppercase; letter-spacing:0.1em; padding:0.5rem 1.25rem; border-radius:2px;">
      Institutional Access
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ── Hero Section with Animated Legal Picture Showcase ─────────────────────────
st.markdown(f"""
<div style="padding:1rem 0 2rem; border-bottom:1px solid rgba(197,198,206,0.3); margin-bottom:2rem; display:grid; grid-template-columns: 1.1fr 0.9fr; gap:2.5rem; align-items:center;">
  <div>
    <span style="font-family:'Inter',sans-serif; font-size:0.72rem; font-weight:700; text-transform:uppercase; letter-spacing:0.3em; color:#735a3a; display:block; margin-bottom:0.75rem;">
      Legacy of Intelligence
    </span>
    <h1 style="font-family:'Playfair Display',serif; font-size:2.8rem; font-weight:700; line-height:1.1; color:#000; margin:0 0 1rem; letter-spacing:-0.02em;">
      Institutional <span style="font-style:italic;">Authority</span> in Legal Jurisprudence.
    </h1>
    <p style="font-family:'Inter',sans-serif; font-size:0.95rem; line-height:1.75; color:#44474d; margin:0 0 1.5rem;">
      NyayaRoute searches the configured legal corpus, checks source relevance, and routes current-information questions to live web search.
    </p>
    <div style="display:flex; gap:1rem;">
      <span style="background:#000000; color:#ffffff; font-family:'Inter',sans-serif; font-size:0.75rem; font-weight:600; text-transform:uppercase; letter-spacing:0.1em; padding:0.75rem 1.75rem; border-radius:3px; display:inline-block; cursor:pointer;">Begin Inquiry</span>
      <span style="border:1px solid #75777e; color:#1a1c1b; font-family:'Inter',sans-serif; font-size:0.75rem; font-weight:600; text-transform:uppercase; letter-spacing:0.1em; padding:0.75rem 1.75rem; border-radius:3px; display:inline-block; cursor:pointer;">White Papers</span>
    </div>
  </div>

  <!-- Animated Legal Picture Card -->
  <div class="animated-hero-card">
    <div style="position:relative; overflow:hidden;">
      <img src="{img_src}" alt="Scales of Justice and Gavel" class="animated-hero-img" />
      <div style="position:absolute; inset:0; background:linear-gradient(to top, rgba(12,27,51,0.85) 0%, transparent 60%); pointer-events:none;"></div>
      <div style="position:absolute; bottom:1.25rem; left:1.25rem; right:1.25rem; z-index:2; display:flex; justify-content:space-between; align-items:flex-end;">
        <div>
          <span class="hero-img-badge" style="background:#735a3a; color:#ffffff; font-family:'Inter',sans-serif; font-size:0.6rem; font-weight:700; text-transform:uppercase; letter-spacing:0.12em; padding:3px 8px; border-radius:2px; display:inline-block; margin-bottom:0.4rem;">
            Sovereign Jurisprudence
          </span>
          <h4 style="font-family:'Playfair Display',serif; font-size:1.15rem; font-weight:600; color:#ffffff; margin:0;">
            Scales of Justice &amp; Precedent Archive
          </h4>
        </div>
        <span class="material-symbols-outlined" style="color:#d7e3ff; font-size:24px;">gavel</span>
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


workspace_tab, library_tab, kb_tab, about_tab = st.tabs(
    ["Research workspace", "Source library", "Knowledge Base", "How to use this"]
)
# ──────────────────────────────────────────────────────────────────────────────
with workspace_tab:
    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None

    corpus = load_corpus()
    source_options = sorted({item["source"] for item in corpus})

    # ── Sidebar ────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("<h3 style='font-family:\"Playfair Display\",serif; font-size:1.2rem; font-weight:700; margin-bottom:0.75rem;'>Research Settings</h3>", unsafe_allow_html=True)
        selected_sources = st.multiselect(
            "Search these sources",
            source_options,
            default=source_options,
        )
        st.caption("Use the source filter to focus statutory research or case-law research.")
        st.divider()
        st.metric("Local Legal Chunks", len(corpus))
        st.caption(f"{len(source_summary(corpus))} source page(s) indexed")
        if os.getenv("GROQ_API_KEY"):
            st.success(f"Groq Ready · {os.getenv('GROQ_MODEL', 'llama-3.3-70b-versatile')}")
        else:
            st.warning("Add GROQ_API_KEY to .env to generate grounded answers.")
        st.info("Not legal advice. Verify citations in their original source.")

    # ── Institutional Access Search Banner ───────────────────────────────────
    st.markdown("""
    <div style="background:#0c1b33; border-radius:6px; padding:2rem; margin-bottom:1.5rem; color:#ffffff;">
      <div style="display:flex; justify-content:space-between; align-items:flex-end; margin-bottom:1.25rem; flex-wrap:wrap; gap:1rem;">
        <div>
          <h2 style="font-family:'Playfair Display',serif; font-size:1.6rem; font-weight:600; color:#ffffff; margin:0 0 0.3rem;">Institutional Access</h2>
          <p style="font-family:'Inter',sans-serif; font-size:0.85rem; color:#7684a1; margin:0; max-width:520px;">
            Execute complex thematic searches across the unified codebase of federal and state statutes.
          </p>
        </div>
        <span style="font-family:'Inter',sans-serif; font-size:0.68rem; text-transform:uppercase; letter-spacing:0.12em; color:#7684a1;">
          Search modes: local corpus and live web
        </span>
      </div>
      <div style="display:flex; flex-wrap:wrap; gap:1rem; align-items:center; margin-top:0.75rem; border-top:1px solid rgba(255,255,255,0.1); padding-top:1rem;">
        <span style="font-family:'Inter',sans-serif; font-size:0.7rem; text-transform:uppercase; color:#7684a1;">Query types:</span>
        <span style="font-family:'Inter',sans-serif; font-size:0.7rem; color:#ffffff;">Statutes and provisions</span>
        <span style="font-family:'Inter',sans-serif; font-size:0.7rem; color:#ffffff;">Cases and citations</span>
        <span style="font-family:'Inter',sans-serif; font-size:0.7rem; color:#ffffff;">Recent developments</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Question text area
    default_val = st.session_state.get("pending_question") or ""
    question = st.text_area(
        "Your legal research question",
        value=default_val,
        placeholder="Query the Code: fundamental rights under environmental distress...",
        height=110,
        label_visibility="collapsed",
    )

    # Sample query buttons
    samples = st.columns(3)
    sample_items = [
        ("Fundamental Rights (Art 21)", "What does Article 21 of the Constitution protect?"),
        ("IT Act Amendments", "What is the latest amendment to the IT Act?"),
        ("Writ & Liberty Cases", "Which Supreme Court cases discuss personal liberty?"),
    ]
    for col, (label, val) in zip(samples, sample_items):
        if col.button(label, use_container_width=True):
            st.session_state.pending_question = val
            st.rerun()

    investigate = st.button("Research this question", type="primary")

    if investigate:
        if not question.strip():
            st.warning("Please enter a legal research question first.")
        elif not selected_sources:
            st.warning("Select at least one source to search.")
        else:
            with st.status("Analyzing legal query…", expanded=True) as status:
                result = run_agent(
                    question.strip(),
                    get_retriever(),
                    set(selected_sources),
                    progress_callback=lambda message: status.update(
                        label=message, state="running", expanded=True
                    ),
                )
                st.write(result)
                status.update(label="Research decision complete", state="complete")
            st.session_state["research_result"] = result
            st.session_state["research_question"] = question.strip()

    result = st.session_state.get("research_result")
    research_question = st.session_state.get("research_question")

    if result and research_question:
        # Active Workspace Header & Metrics
        st.markdown(f"""
        <div style="display:flex; align-items:center; justify-content:space-between; border-bottom:2px solid #000; padding-bottom:0.6rem; margin:2.5rem 0 1.5rem;">
          <div style="display:flex; align-items:center; gap:0.75rem;">
            <h3 style="font-family:'Playfair Display',serif; font-size:1.4rem; font-weight:700; color:#000; margin:0;">Active Workspace</h3>
            <span style="background:#fddab2; color:#785e3e; font-family:'Inter',sans-serif; font-size:0.65rem; font-weight:700; text-transform:uppercase; padding:3px 10px; letter-spacing:0.05em;">Live Analysis</span>
          </div>
          <div style="display:flex; gap:0.75rem;">
            <div style="background:#fff; border:1px solid #c5c6ce; padding:0.5rem 0.9rem; text-align:center; border-radius:4px;">
              <span style="font-family:'Playfair Display',serif; font-size:0.95rem; font-weight:700; color:#000; display:block;">{len(corpus):,}</span>
              <span style="font-family:'Inter',sans-serif; font-size:0.6rem; text-transform:uppercase; color:#44474d; letter-spacing:0.05em;">Records</span>
            </div>
            <div style="background:#fff; border:1px solid #c5c6ce; padding:0.5rem 0.9rem; text-align:center; border-radius:4px;">
              <span style="font-family:'Playfair Display',serif; font-size:0.95rem; font-weight:700; color:#000; display:block;">{result['route']}</span>
              <span style="font-family:'Inter',sans-serif; font-size:0.6rem; text-transform:uppercase; color:#44474d; letter-spacing:0.05em;">Retrieval route</span>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        left_col, right_col = st.columns([2, 1], gap="large")

        with left_col:
            # Core Authority Card
            if result["enough"] and result["answer"]:
                st.markdown(f"""
                <div style="background:#ffffff; border:1px solid #c5c6ce; border-radius:6px; padding:1.75rem; margin-bottom:1.5rem;">
                  <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:0.75rem;">
                    <div>
                      <span style="font-family:'Inter',sans-serif; font-size:0.68rem; font-weight:700; text-transform:uppercase; letter-spacing:0.1em; color:#735a3a;">
                        Core Authority &amp; Synthesis
                      </span>
                      <h4 style="font-family:'Playfair Display',serif; font-size:1.3rem; font-weight:600; color:#1a1c1b; margin:0.3rem 0 0;">
                        {html.escape(research_question)}
                      </h4>
                    </div>
                    <span class="material-symbols-outlined" style="color:#735a3a; font-size:22px;">bookmark</span>
                  </div>
                  <div style="height:1px; background:#c5c6ce; opacity:0.5; margin-bottom:1.25rem;"></div>
                  <div style="display:grid; grid-template-columns:repeat(3,1fr); gap:0.75rem; margin-top:1.25rem;">
                    <div style="background:#f4f3f1; border-radius:4px; padding:0.85rem;">
                      <p style="font-size:0.62rem; text-transform:uppercase; letter-spacing:0.05em; color:#44474d; margin-bottom:0.2rem;">Retrieval Route</p>
                      <p style="font-weight:700; font-size:0.88rem; color:#1a1c1b; margin:0;">{result['route']}</p>
                    </div>
                    <div style="background:#f4f3f1; border-radius:4px; padding:0.85rem;">
                      <p style="font-size:0.62rem; text-transform:uppercase; letter-spacing:0.05em; color:#44474d; margin-bottom:0.2rem;">Citations</p>
                      <p style="font-weight:700; font-size:0.88rem; color:#1a1c1b; margin:0;">{len(result['evidence'])} Authorities</p>
                    </div>
                    <div style="background:#f4f3f1; border-radius:4px; padding:0.85rem;">
                      <p style="font-size:0.62rem; text-transform:uppercase; letter-spacing:0.05em; color:#44474d; margin-bottom:0.2rem;">Gate Status</p>
                      <p style="font-weight:700; font-size:0.88rem; color:#047857; margin:0;">Evidence Passed</p>
                    </div>
                  </div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown(result["answer"])

            elif result["enough"]:
                if result.get("generation_error"):
                    st.error(f"Evidence passed, but answer generation failed: {result['generation_error']}")
                else:
                    st.info("Evidence gate passed. Add **GROQ_API_KEY** to `.env` to generate a cited answer from this evidence.")

            else:
                st.markdown(f"""
                <div style="background:#fffbf1; border:1px solid #fddab2; border-left:4px solid #735a3a; border-radius:6px; padding:1.75rem; margin-bottom:1.5rem;">
                  <span style="font-family:'Inter',sans-serif; font-size:0.68rem; font-weight:700; text-transform:uppercase; letter-spacing:0.1em; color:#b45309;">
                    Insufficient Evidence
                  </span>
                  <h4 style="font-family:'Playfair Display',serif; font-size:1.25rem; font-weight:600; color:#1a1c1b; margin:0.3rem 0 0.5rem;">
                    Library Material Insufficient
                  </h4>
                    <p style="color:#44474d; font-size:0.92rem; margin:0;">{html.escape(result["evidence_reason"])}</p>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("**Useful next steps**")
                render_source_links(result["actions"])
                st.caption("Open a relevant primary document, then add its specific URL in the Source library tab.")

            # Authorities & Citations
            st.markdown("<h4 style='font-family:\"Playfair Display\",serif; font-size:1.25rem; font-weight:700; font-style:italic; margin-bottom:1rem;'>Authorities and citations</h4>", unsafe_allow_html=True)
            if not result["evidence"]:
                st.caption("No relevant chunks were retrieved from the selected local source library.")

            for index, item in enumerate(result["evidence"], 1):
                relevance = f"{item['relevance']:.0%} query-anchor match"
                with st.expander(f"[S{index}] {item['title']} · {item['source']} · {relevance}"):
                    if item["url"].startswith("http"):
                        st.markdown(f"**Open source:** [{item['url']}]({item['url']})")
                    else:
                        st.caption(f"Local source reference: {item['url']}")
                    st.caption(f"Source status: {item.get('authority_level', 'Unverified')}. Verify the full text, date, court/bench, and later treatment before relying on it.")
                    st.write(item["text"])

            with st.expander("Research audit trail"):
                analysis = result.get("analysis", {})
                cols = st.columns(4)
                quality = "Ready" if result["good_query"] else "Clarified"
                values = [
                    ("Query quality", quality, result["quality_reason"]),
                    ("Retrieval route", result["route"], result["route_reason"]),
                    ("Retry", "Used" if result["retried"] else "—",
                     f"{len(result['evidence'])} cited chunks retained"),
                    ("Evidence gate", "Passed" if result["enough"] else "Not passed", result["evidence_reason"]),
                ]
                for col, (title, value, detail) in zip(cols, values):
                    col.markdown(f"""
                    <div style="border:1px solid #c5c6ce; border-radius:4px; padding:0.85rem; background:#fff;">
                      <small style="color:#735a3a; font-weight:700; text-transform:uppercase; font-size:0.65rem;">{title}</small><br>
                      <strong style="color:#1a1c1b; font-size:0.9rem;">{value}</strong><br>
                      <small style="color:#44474d; font-size:0.72rem;">{detail}</small>
                    </div>
                    """, unsafe_allow_html=True)
                st.caption(f"Search query used: **{result['query_used']}**")
                st.write("**Detected legal terms:** " + (", ".join(analysis.get("legal_terms", [])) or "None detected"))
                st.write("**Entities:** " + (", ".join(analysis.get("entities", [])) or "None detected"))
                st.write("**Case references:** " + (", ".join(analysis.get("case_references", [])) or "None detected"))
                st.write("**Citations:** " + (", ".join(analysis.get("citations", [])) or "None detected"))
                st.write("**Information type:** " + analysis.get("information_type", "unknown"))
                if analysis.get("ambiguous"):
                    st.warning("Ambiguity check: " + analysis.get("ambiguity_reason", "Query clarified."))
                if result.get("retried"):
                    st.write("**Retry query:** " + result.get("retry_query", ""))

        with right_col:
            st.subheader("Research decision")
            st.write(f"**Query quality:** {'Clear' if result['good_query'] else 'Reformulated'}")
            st.write(f"**Information type:** {result.get('analysis', {}).get('information_type', 'unknown')}")
            st.write(f"**Route:** {result['route']}")
            st.write(f"**Evidence gate:** {'Passed' if result['enough'] else 'Insufficient evidence'}")

            st.download_button(
                "Export Brief (.md)",
                memo(research_question, result),
                file_name=f"nyayaroute-{datetime.now():%Y%m%d-%H%M}.md",
                mime="text/markdown",
                use_container_width=True,
            )

# ──────────────────────────────────────────────────────────────────────────────
with library_tab:
    st.markdown("<h3 style='font-family:\"Playfair Display\",serif; font-size:1.35rem; font-weight:700; margin-bottom:0.5rem;'>Source Library</h3>", unsafe_allow_html=True)
    st.write(
        "Established-law questions search the local collection; current-information questions search the live web. "
        "Add specific legal document pages, not broad landing pages or bulk crawls."
    )
    current_sources = source_summary(load_corpus())
    if current_sources:
        for item in current_sources:
            st.markdown(
                f"**{item['title']}**  \n"
                f"{item['source']} · {item['chunks']} chunk(s) · [Open source]({item['url']})"
            )
    else:
        st.info("No sources are indexed yet. Use the form below to add a trusted source.")

    st.divider()
    st.markdown("<h3 style='font-family:\"Playfair Display\",serif; font-size:1.35rem; font-weight:700; margin-bottom:0.5rem;'>Add a Trusted Source</h3>", unsafe_allow_html=True)
    st.caption(
        "Allowed: India Code, Supreme Court of India, Indian Kanoon, and official High Court sites. "
        "The tool is deliberately page-by-page and does not bulk crawl."
    )

    with st.form("ingest_source", clear_on_submit=True):
        source_url = st.text_input(
            "Specific document URL",
            placeholder="https://www.indiacode.nic.in/…",
        )
        display_title = st.text_input(
            "Optional label",
            placeholder="e.g., Information Technology Act, 2000",
        )
        submit_source = st.form_submit_button("Add to source library")

    if submit_source:
        if not source_url.strip():
            st.warning("Enter a source URL.")
        else:
            try:
                with st.spinner("Fetching and indexing the trusted source page…"):
                    count, title = ingest_trusted_url(
                        source_url.strip(), display_title.strip() or None
                    )
                reset_retriever()
                if count:
                    st.success(
                        f"Added {count} chunks from: {title}. "
                        "Return to Research workspace and ask your question again."
                    )
                else:
                    st.info("That page is already indexed; no new chunks were added.")
            except Exception as exc:
                st.error(f"Could not add this source: {exc}")

# ──────────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────
# KNOWLEDGE BASE — PDF INGESTION
with kb_tab:
    st.markdown(
        "<h2 style='font-family:\"Playfair Display\",serif;'>Knowledge Base</h2>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "Upload official legal PDFs to expand NyayaRoute's searchable knowledge base."
    )

    uploaded_files = st.file_uploader(
        "Upload legal PDF(s)",
        type=["pdf"],
        accept_multiple_files=True,
        help="Acts, Constitution, judgments, Gazette notifications, Law Commission reports, etc.",
    )

    if uploaded_files:

        for uploaded_file in uploaded_files:

            st.markdown("---")
            st.markdown(f"### {uploaded_file.name}")

            file_bytes = uploaded_file.getvalue()

            # ── Preview ──────────────────────────────────────────────────────
            try:
                parsed, profile, structure_kind, chunks = preview_pdf(
                    file_bytes,
                    uploaded_file.name,
                )

                st.success("PDF successfully parsed.")

                # Document metadata
                st.markdown("#### Detected metadata")

                metadata_col1, metadata_col2 = st.columns(2)

                with metadata_col1:
                    st.write(f"**Document kind:** {profile.kind}")
                    st.write(f"**Act:** {profile.act}")
                    st.write(f"**Year:** {profile.year}")

                with metadata_col2:
                    st.write(f"**Source:** {profile.source}")
                    st.write(f"**Court:** {profile.court}")
                    st.write(f"**Structure:** {structure_kind}")

                # PDF statistics
                stat1, stat2, stat3 = st.columns(3)

                with stat1:
                    st.metric("Pages", parsed.num_pages_total)

                with stat2:
                    st.metric("Pages used", parsed.num_pages_used)

                with stat3:
                    st.metric("Chunks", len(chunks))

                # ── Metadata preview ─────────────────────────────────────────
                with st.expander("Preview extracted chunks"):

                    for i, chunk in enumerate(chunks[:5], start=1):

                        st.markdown(f"**Chunk {i}**")

                        st.write(
                            f"Structure: {chunk.structure_kind}"
                        )

                        if chunk.part:
                            st.write(f"Part: {chunk.part}")

                        if chunk.chapter:
                            st.write(f"Chapter: {chunk.chapter}")

                        if chunk.article:
                            st.write(f"Article: {chunk.article}")

                        if chunk.section:
                            st.write(f"Section: {chunk.section}")

                        if chunk.heading:
                            st.write(f"Heading: {chunk.heading}")

                        st.text_area(
                            "Text",
                            chunk.text,
                            height=150,
                            key=f"preview_{uploaded_file.name}_{i}",
                        )

                        st.markdown("---")

                # ── Classification override ─────────────────────────────────
                with st.expander("Override detected classification (optional)"):

                    source_override = st.text_input(
                        "Source",
                        value=profile.source,
                        key=f"source_{uploaded_file.name}",
                    )

                    act_override = st.text_input(
                        "Act / document name",
                        value=profile.act,
                        key=f"act_{uploaded_file.name}",
                    )

                # ── Index button ────────────────────────────────────────────
                if st.button(
                    f"Index {uploaded_file.name}",
                    type="primary",
                    key=f"index_{uploaded_file.name}",
                ):

                    try:

                        progress = st.progress(0)

                        progress.progress(20)

                        result = ingest_pdf(
                            file_bytes,
                            uploaded_file.name,
                            source_override=source_override or None,
                            act_override=act_override or None,
                        )

                        progress.progress(100)
                        reset_retriever()

                        st.success(
                            f"Successfully indexed {result.filename}: "
                            f"{result.chunks_created} chunks created."
                        )

                        # Results
                        result_col1, result_col2, result_col3 = st.columns(3)

                        with result_col1:
                            st.metric(
                                "Chunks created",
                                result.chunks_created,
                            )

                        with result_col2:
                            st.metric(
                                "Duplicates skipped",
                                result.chunks_skipped_duplicate,
                            )

                        with result_col3:
                            st.metric(
                                "Pages used",
                                result.pages_used,
                            )

                        if result.chunks_created == 0:

                            st.info(
                                "That PDF was already indexed; no new chunks were added."
                            )

                    except DuplicatePDFError:

                        st.warning(
                            "This PDF has already been indexed."
                        )

                    except PDFIngestionError as exc:

                        st.error(
                            f"Could not process this PDF: {exc}"
                        )

                    except Exception as exc:

                        st.error(
                            f"Unexpected indexing error: {exc}"
                        )

            except PDFIngestionError as exc:

                st.error(
                    f"Could not read {uploaded_file.name}: {exc}"
                )

            except Exception as exc:

                st.error(
                    f"Could not preview {uploaded_file.name}: {exc}"
                )

# ──────────────────────────────────────────────────────────────────────────────
with about_tab:
    st.markdown("<h3 style='font-family:\"Playfair Display\",serif; font-size:1.35rem; font-weight:700; margin-bottom:0.5rem;'>A Safer Way to Use Legal RAG</h3>", unsafe_allow_html=True)
    st.markdown("""
    1. Ask a focused legal research question.
    2. **NyayaRoute** checks whether the query needs clarification and selects BM25, vector, or hybrid retrieval.
    3. It reranks candidate chunks and tests whether a sufficiently relevant authority was retrieved.
    4. It either produces a cited, evidence-constrained research answer — or says what is missing and directs you to trusted sources.

    For current amendments, always add the current statutory text or notification first.  
    **A language model being available does not make it a source of law.**
    """)

# ── Institutional Footer ──────────────────────────────────────────────────────
st.markdown("""
<div style="border-top:1px solid #c5c6ce; margin-top:3rem; padding:1.5rem 0; display:flex; flex-wrap:wrap; justify-content:space-between; align-items:center; gap:1rem;">
  <span style="font-family:'Inter',sans-serif; font-size:0.68rem; text-transform:uppercase; letter-spacing:0.1em; color:#44474d;">
    © 2026 NYAYAROUTE INTERNATIONAL. ALL RIGHTS RESERVED.
  </span>
  <div style="display:flex; gap:1.5rem; flex-wrap:wrap;">
    <a href="#" style="font-family:'Inter',sans-serif; font-size:0.68rem; text-transform:uppercase; letter-spacing:0.08em; color:#44474d; text-decoration:underline; text-underline-offset:4px;">
      Terms of Service
    </a>
    <a href="#" style="font-family:'Inter',sans-serif; font-size:0.68rem; text-transform:uppercase; letter-spacing:0.08em; color:#44474d; text-decoration:underline; text-underline-offset:4px;">
      Privacy Protocol
    </a>
    <a href="#" style="font-family:'Inter',sans-serif; font-size:0.68rem; text-transform:uppercase; letter-spacing:0.08em; color:#44474d; text-decoration:underline; text-underline-offset:4px;">
      Compliance
    </a>
    <a href="#" style="font-family:'Inter',sans-serif; font-size:0.68rem; text-transform:uppercase; letter-spacing:0.08em; color:#44474d; text-decoration:underline; text-underline-offset:4px;">
      Institutional Contact
    </a>
  </div>
</div>
""", unsafe_allow_html=True)
