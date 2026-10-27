from __future__ import annotations

import base64
import os
from datetime import datetime
from html import escape as html_escape
from pathlib import Path

import importlib
import streamlit as st
from dotenv import load_dotenv


try:
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
except Exception:
    pass

import rag_engine
import live_web_searcher
importlib.reload(rag_engine)
importlib.reload(live_web_searcher)

from rag_engine import LegalRetriever, load_corpus, run_agent, source_summary
from trusted_sources import ingest_trusted_url
from pdf_ingestor import DuplicatePDFError, ingest_pdf, preview_pdf
from pdf_parser import PDFIngestionError
from vector_store import reset_vector_store_cache
load_dotenv()

st.set_page_config(
    page_title="NyayaRoute | Institutional Intelligence",
    page_icon="⚖",
    layout="wide",
    initial_sidebar_state="expanded",
)

def get_image_b64() -> str:
    paths = [
        Path(__file__).parent / "legal_hero.png",
        Path("legal_hero.png"),
    ]
    for image_path in paths:
        if image_path.exists():
            try:
                with image_path.open("rb") as image_file:
                    return base64.b64encode(image_file.read()).decode("utf-8")
            except OSError:
                continue
    return ""


image_b64 = get_image_b64()
img_src = (
    f"data:image/png;base64,{image_b64}"
    if image_b64
    else "https://images.unsplash.com/photo-1589829545856-d10d557cf95f?auto=format&fit=crop&w=1200&q=80"
)

def get_courthouse_b64() -> str:
    for filename in ["courthouse_clean.png", "courthouse_sketch.png"]:
        path = Path(filename)
        if path.exists():
            try:
                with path.open("rb") as f:
                    return base64.b64encode(f.read()).decode("utf-8")
            except OSError:
                pass
    return ""

courthouse_b64 = get_courthouse_b64()
courthouse_src = (
    f"data:image/png;base64,{courthouse_b64}"
    if courthouse_b64
    else ""
)

def source_sort_key(item: dict) -> int:
    title = item.get("title", "")
    if "Article 21" in title and "21A" not in title:
        return 0
    if "Judgments service" in title:
        return 1
    if "Article 12" in title:
        return 2
    if "Article 13" in title:
        return 3
    if "Article 14" in title:
        return 4
    if "Article 21A" in title:
        return 5
    if "Article 22" in title:
        return 6
    if "Indian Kanoon" in title:
        return 7
    return 10


def render_page_hero(eyebrow: str, title: str, subtitle: str, col_ratio=[1.55, 1.45]) -> None:
    hero_col_left, hero_col_right = st.columns(col_ratio, gap="large")
    with hero_col_left:
        st.markdown(
            f"""
            <div style="margin-top:0.3rem; margin-bottom:1.15rem;">
                <div class="source-lib-eyebrow">{html_escape(eyebrow)}</div>
                <h1 class="source-lib-title">{html_escape(title)}</h1>
                <div style="display:flex; align-items:center; gap:0.55rem; width:5.5rem; margin:0.35rem 0 0.55rem;">
                    <div style="flex:1; height:1px; background:#A8752D;"></div>
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#A8752D" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" style="flex-shrink:0;">
                        <path d="m16 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z"/>
                        <path d="m2 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z"/>
                        <path d="M7 21h10"/>
                        <path d="M12 3v18"/>
                        <path d="M3 7h2c2 0 5-1 7-2 2 1 5 2 7 2h2"/>
                    </svg>
                    <div style="flex:1; height:1px; background:#A8752D;"></div>
                </div>
                <p class="source-lib-subtitle">{html_escape(subtitle)}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with hero_col_right:
        if courthouse_src:
            st.markdown(
                f"""
                <div style="display:flex; justify-content:flex-end; align-items:flex-end; height:100%; min-height:190px; padding-bottom:0.2rem;">
                    <img src="{courthouse_src}" alt="Supreme Court of India" style="max-height:215px; width:100%; max-width:620px; object-fit:contain; object-position:right bottom; opacity:0.98; pointer-events:none; user-select:none; mix-blend-mode:multiply; filter:contrast(1.08) brightness(0.98);">
                </div>
                """,
                unsafe_allow_html=True,
            )


def get_source_metadata(item: dict) -> tuple[str, str, str, str]:
    title = item.get("title", "")
    source = item.get("source", "")
    url = item.get("url", "")
    t_lower = title.lower()
    s_lower = source.lower()

    if "judgment" in t_lower or "precedent" in t_lower or "kanoon" in t_lower or "supreme court" in s_lower:
        category = "JUDGMENTS"
        doc_type = "Judgment"
        icon = "gavel"
    elif "article 21" in t_lower or "article 14" in t_lower:
        category = "CONSTITUTIONAL LAW"
        doc_type = "Constitutional provision"
        icon = "account_balance" if "article 14" in t_lower else "balance"
    elif "article 12" in t_lower or "article 13" in t_lower:
        category = "STATUTE"
        doc_type = "Constitutional provision"
        icon = "description"
    elif "constitution" in t_lower:
        category = "CONSTITUTIONAL LAW"
        doc_type = "Constitutional provision"
        icon = "account_balance"
    else:
        category = "STATUTE"
        doc_type = "Primary statute"
        icon = "article"

    if url.startswith("http://") or url.startswith("https://"):
        open_url = url
    else:
        open_url = "https://www.indiacode.nic.in/handle/123456789/1362/simple-search?query=constitution"

    return category, doc_type, icon, open_url


# ── Workspace Styling ─────────────────────────────────────────────────────────
st.markdown(f"""
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,500;0,600;0,700;0,800;0,900;1,400;1,500;1,600;1,700;1,800&family=Inter:wght@300;400;500;600;700;800&family=Cinzel:wght@500;600;700&display=swap" rel="stylesheet">
<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@24,300,0,0" rel="stylesheet">

<style>
  @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,500;0,600;0,700;0,800;0,900;1,400;1,500;1,600;1,700;1,800&family=Inter:wght@300;400;500;600;700;800&family=Cinzel:wght@500;600;700&display=swap');
  @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@24,300,0,0');

  /* Base background and typography */
  html, body, .stApp {{
    background-color: #FAF7F0 !important;
    color: #1a1c1b !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
  }}

  /* Hide default Streamlit menu and decoration while preserving sidebar controls */
  #MainMenu {{ visibility: hidden; }}
  footer {{ visibility: hidden; }}
  [data-testid="stToolbar"] {{ display: none !important; }}
  [data-testid="stDecoration"] {{ display: none !important; }}

  /* Ensure header and all sidebar collapse/expand controls are 100% clickable and visible */
  header[data-testid="stHeader"] {{
    background: transparent !important;
        pointer-events: none !important;
    z-index: 99999 !important;
  }}

  /* Make sure the sidebar collapse and expand toggle is always accessible, prominent, and clickable */
  [data-testid="collapsedControl"],
  [data-testid="stSidebarCollapsedControl"],
  [data-testid="stSidebarCollapseButton"],
  button[data-testid="stSidebarCollapseButton"],
  button[aria-label="Expand sidebar"],
  button[aria-label="Close sidebar"] {{
    visibility: visible !important;
    display: flex !important;
    pointer-events: auto !important;
    opacity: 1 !important;
  }}

  [data-testid="collapsedControl"],
  [data-testid="stSidebarCollapsedControl"] {{
    background-color: #0c1b33 !important;
    color: #ffffff !important;
    border: 2px solid #735a3a !important;
    border-radius: 6px !important;
    top: 0.6rem !important;
    left: 0.6rem !important;
    z-index: 9999999 !important;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.35) !important;
    padding: 4px 8px !important;
  }}
  [data-testid="collapsedControl"] button,
  [data-testid="stSidebarCollapsedControl"] button,
  [data-testid="stSidebarCollapseButton"] button {{
    color: #ffffff !important;
  }}
  [data-testid="collapsedControl"] svg,
  [data-testid="stSidebarCollapsedControl"] svg,
  [data-testid="stSidebarCollapseButton"] svg {{
    fill: #ffffff !important;
    stroke: #ffffff !important;
  }}

  .block-container {{
    padding-top: 1.2rem !important;
    padding-bottom: 3.5rem !important;
    padding-left: 3.5rem !important;
    padding-right: 3.5rem !important;
    max-width: 100% !important;
    width: 100% !important;
    margin: 0 !important;
  }}

  /* Fonts for headings */
  h1, h2, h3, h4, h5, h6,
  [data-testid="stMarkdownContainer"] h1,
  [data-testid="stMarkdownContainer"] h2,
  [data-testid="stMarkdownContainer"] h3,
  .playfair,
  .home-title,
  .home-title *,
  h1.home-title,
  h1.home-title * {{
    font-family: 'Playfair Display', 'Baskerville', 'Georgia', 'Cambria', 'Times New Roman', serif !important;
    letter-spacing: -0.01em;
  }}

  .home-title em,
  .home-title span.title-accent,
  h1.home-title span.title-accent {{
    font-style: italic !important;
    color: #735a3a !important;
  }}

  /* Body text color enforcement - do not force Inter on spans inside headings */
  p, label, li, [data-testid="stMarkdownContainer"] p {{
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
        text-transform: none !important;
        letter-spacing: 0 !important;
    color: #44474d !important;
  }}
  button[aria-selected="true"][data-baseweb="tab"] p {{
    color: #000000 !important;
  }}
  button[aria-selected="true"][data-baseweb="tab"] {{
    border-bottom: 2px solid #735a3a !important;
  }}

    /* ── Hide ONLY the Radio Circle Indicator ── */
    div[data-testid="stRadio"] [role="radiogroup"] input[type="radio"],
    div[data-testid="stRadio"] [role="radiogroup"] svg,
    div[data-testid="stRadio"] [role="radiogroup"] label > div:first-child:has(input),
    div[data-testid="stRadio"] [role="radiogroup"] label > div:first-child [role="radio"],
    div[data-testid="stRadio"] [role="radiogroup"] [class*="e1dm8mpf4"] {{
        display: none !important;
        visibility: hidden !important;
        width: 0 !important;
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
        opacity: 0 !important;
        position: absolute !important;
        pointer-events: none !important;
    }}

    div[data-testid="stRadio"] [role="radiogroup"] label *::before,
    div[data-testid="stRadio"] [role="radiogroup"] label *::after {{
        display: none !important;
        content: none !important;
    }}

    /* ── Clean Horizontal Navbar Matching Image 1 ── */
    div[data-testid="stRadio"] {{
        width: 100% !important;
        margin: 0 !important;
        padding: 0 !important;
        display: flex !important;
        justify-content: flex-start !important;
    }}
    div[data-testid="stRadio"] [data-testid="stRadioGroup"],
    div[data-testid="stRadio"] [role="radiogroup"] {{
        display: flex !important;
        flex-direction: row !important;
        flex-wrap: nowrap !important;
        align-items: center !important;
        justify-content: flex-start !important;
        gap: 2.1rem !important;
        border: none !important;
        margin: 0 !important;
        padding: 0 0 0 2rem !important;
        width: fit-content !important;
    }}
    div[data-testid="stRadio"] [role="radiogroup"] > div,
    div[data-testid="stRadio"] [data-testid="stRadioGroup"] > div {{
        flex: 0 0 auto !important;
        width: auto !important;
        margin: 0 !important;
        padding: 0 !important;
    }}
    div[data-testid="stRadio"] [role="radiogroup"] label {{
        display: inline-flex !important;
        align-items: center !important;
        background: transparent !important;
        border: none !important;
        border-radius: 0 !important;
        padding: 0.35rem 0.2rem 0.35rem !important;
        margin: 0 !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
        white-space: nowrap !important;
        border-bottom: 2px solid transparent !important;
    }}
    div[data-testid="stRadio"] [role="radiogroup"] label [data-testid="stRadioOption"],
    div[data-testid="stRadio"] [role="radiogroup"] label > div:last-child,
    div[data-testid="stRadio"] [role="radiogroup"] label [data-testid="stMarkdownContainer"] {{
        display: inline-flex !important;
        align-items: center !important;
        visibility: visible !important;
        opacity: 1 !important;
        position: static !important;
        width: auto !important;
        height: auto !important;
    }}
    div[data-testid="stRadio"] [role="radiogroup"] label p,
    div[data-testid="stRadio"] [role="radiogroup"] label span {{
        display: inline-block !important;
        visibility: visible !important;
        opacity: 1 !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.90rem !important;
        font-weight: 500 !important;
        color: #44474d !important;
        letter-spacing: 0.01em !important;
        white-space: nowrap !important;
        margin: 0 !important;
        padding-bottom: 3px !important;
        transition: color 0.15s ease !important;
    }}
    div[data-testid="stRadio"] [role="radiogroup"] label:hover p,
    div[data-testid="stRadio"] [role="radiogroup"] label:hover span {{
        color: #102744 !important;
    }}
    div[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked),
    div[data-testid="stRadio"] [role="radiogroup"] label[aria-checked="true"],
    div[data-testid="stRadio"] [role="radiogroup"] label:has([aria-checked="true"]) {{
        border-bottom: 2px solid #A47B3C !important;
    }}
    div[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) p,
    div[data-testid="stRadio"] [role="radiogroup"] label[aria-checked="true"] p,
    div[data-testid="stRadio"] [role="radiogroup"] label:has([aria-checked="true"]) p {{
        color: #102744 !important;
        font-weight: 700 !important;
    }}

    /* ── Source Library Premium Royal Styling ── */
    .source-lib-eyebrow {{
        font-family: 'Inter', sans-serif !important;
        font-size: 0.68rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.18em !important;
        color: #A47B3C !important;
        text-transform: uppercase !important;
        margin-bottom: 0.25rem !important;
    }}
    .source-lib-title {{
        font-family: 'Playfair Display', 'Baskerville', Georgia, serif !important;
        font-size: 2.75rem !important;
        font-weight: 700 !important;
        line-height: 1.15 !important;
        color: #102744 !important;
        margin: 0 !important;
    }}
    .source-lib-subtitle {{
        font-family: 'Inter', sans-serif !important;
        font-size: 0.90rem !important;
        color: #68778A !important;
        margin: 0 !important;
    }}
    .source-filter-toolbar {{
        margin-bottom: 1.1rem !important;
    }}
    .source-filter-toolbar [data-testid="stTextInput"] input {{
        background: #FFFFFF !important;
        border: 1px solid #D8C39E !important;
        border-radius: 7px !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.84rem !important;
        color: #102744 !important;
        padding: 0.45rem 0.75rem !important;
        height: 38px !important;
    }}
    .source-filter-toolbar [data-baseweb="select"] {{
        background: #FFFFFF !important;
        border: 1px solid #D8C39E !important;
        border-radius: 7px !important;
        height: 38px !important;
    }}
    .source-filter-toolbar [data-baseweb="select"] > div {{
        background: #FFFFFF !important;
        border: none !important;
        border-radius: 7px !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.82rem !important;
        color: #102744 !important;
        min-height: 36px !important;
        height: 36px !important;
        padding: 0 0.5rem !important;
    }}
    .source-filter-toolbar button {{
        background: #FAF7F0 !important;
        border: 1px solid #D8C39E !important;
        border-radius: 7px !important;
        color: #68778A !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.80rem !important;
        font-weight: 500 !important;
        height: 38px !important;
        padding: 0 0.6rem !important;
        transition: all 0.2s ease !important;
    }}
    .source-filter-toolbar button:hover {{
        color: #102744 !important;
        border-color: #A47B3C !important;
        background: #FFFFFF !important;
    }}
    .legal-source-card {{
        background: #FFFDFA;
        border: 1px solid #E7DED0;
        border-radius: 8px;
        padding: 0.95rem 1.25rem;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1.15rem;
        transition: all 0.2s ease-in-out;
        box-shadow: 0 1px 4px rgba(16, 39, 68, 0.02);
    }}
    .legal-source-card.active-card {{
        border-left: 4px solid #A47B3C !important;
    }}
    .legal-source-card:hover {{
        border-color: #D8C39E;
        box-shadow: 0 4px 14px rgba(16, 39, 68, 0.06);
        transform: translateY(-1px);
    }}
    .source-card-left {{
        display: flex;
        align-items: center;
        gap: 1.1rem;
        min-width: 0;
        flex: 1;
    }}
    .source-card-icon-box {{
        width: 44px;
        height: 44px;
        min-width: 44px;
        background: #FAF7F0;
        border: 1px solid #E7DED0;
        border-radius: 6px;
        display: flex;
        align-items: center;
        justify-content: center;
    }}
    .source-card-icon-box .material-symbols-outlined {{
        font-size: 22px;
        color: #A47B3C;
    }}
    .source-card-details {{
        display: flex;
        flex-direction: column;
        gap: 0.15rem;
        min-width: 0;
    }}
    .source-badge {{
        display: inline-block;
        width: fit-content;
        background: #FAF7F0;
        border: 1px solid #D8C39E;
        border-radius: 3px;
        padding: 1px 6px;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.58rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.08em !important;
        color: #A47B3C !important;
        text-transform: uppercase !important;
        line-height: 1.4 !important;
        margin-bottom: 0.1rem;
    }}
    .source-card-title {{
        font-family: 'Playfair Display', 'Baskerville', Georgia, serif !important;
        font-size: 1.08rem !important;
        font-weight: 600 !important;
        color: #102744 !important;
        line-height: 1.35 !important;
        margin: 0.1rem 0 !important;
        word-break: break-word !important;
    }}
    .source-card-meta {{
        font-family: 'Inter', sans-serif !important;
        font-size: 0.74rem !important;
        color: #68778A !important;
        display: flex;
        align-items: center;
        gap: 0.45rem;
        margin: 0 !important;
        line-height: 1.3 !important;
    }}
    .source-card-meta .meta-dot {{
        color: #D8C39E;
        font-size: 0.8rem;
    }}
    .source-card-btn {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        gap: 0.35rem;
        background: #FFFFFF;
        border: 1px solid #D8C39E;
        border-radius: 999px;
        padding: 0.42rem 1.05rem;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.76rem !important;
        font-weight: 500 !important;
        color: #102744 !important;
        text-decoration: none !important;
        white-space: nowrap;
        transition: all 0.2s ease;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02);
    }}
    .source-card-btn:hover {{
        background: #FAF7F0 !important;
        border-color: #A47B3C !important;
        color: #0c1b33 !important;
        box-shadow: 0 2px 6px rgba(16, 39, 68, 0.08);
    }}
    .trusted-source-container [data-testid="stForm"] {{
        background: #FAF7F0 !important;
        border: 1px solid #D8C39E !important;
        border-radius: 8px !important;
        padding: 1.4rem 1.4rem 1.5rem 1.4rem !important;
        box-shadow: inset 0 0 0 3px #FAF7F0, inset 0 0 0 4px rgba(216, 195, 158, 0.45), 0 4px 16px rgba(16, 39, 68, 0.03) !important;
    }}
    .trusted-source-container [data-testid="stForm"] label p {{
        font-family: 'Inter', sans-serif !important;
        font-size: 0.80rem !important;
        font-weight: 600 !important;
        color: #102744 !important;
        margin-bottom: 0.15rem !important;
    }}
    .trusted-source-container [data-testid="stForm"] input {{
        background: #FFFFFF !important;
        border: 1px solid #D8C39E !important;
        border-radius: 6px !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.82rem !important;
        color: #102744 !important;
        padding: 0.5rem 0.75rem !important;
    }}
    .trusted-source-container [data-testid="stForm"] button[kind="secondaryFormSubmit"],
    .trusted-source-container [data-testid="stForm"] button {{
        background: #102744 !important;
        color: #FFFFFF !important;
        border: 1px solid #102744 !important;
        border-radius: 6px !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.88rem !important;
        font-weight: 600 !important;
        padding: 0.65rem 1rem !important;
        width: 100% !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 2px 8px rgba(16, 39, 68, 0.2) !important;
        margin-top: 0.5rem !important;
        margin-bottom: 0.4rem !important;
    }}
    .trusted-source-container [data-testid="stForm"] button:hover {{
        background: #1b385e !important;
        border-color: #A47B3C !important;
        box-shadow: 0 4px 12px rgba(16, 39, 68, 0.3) !important;
    }}

    /* ── Knowledge Base Upload Card (Section 5) ── */
    .kb-outer-container {{
        max-width: 1200px;
        margin: 0 auto;
    }}
    .st-key-kb_upload_card,
    .kb-card {{
        background: #FFFDF9 !important;
        border: 1px solid #D8BD91 !important;
        border-left: 3.5px solid #A8752D !important;
        border-radius: 8px !important;
        padding: 1.8rem 2.2rem 2.2rem 2.2rem !important;
        box-shadow: 0 4px 18px rgba(16, 36, 66, 0.04) !important;
        margin-bottom: 2rem !important;
        max-width: 1200px !important;
        margin-left: auto !important;
        margin-right: auto !important;
    }}
    .kb-header {{
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        margin-bottom: 1.35rem;
    }}
    .kb-icon-title-wrap {{
        display: flex;
        align-items: center;
        gap: 1.1rem;
    }}
    .kb-icon-box {{
        width: 46px;
        height: 46px;
        min-width: 46px;
        border-radius: 8px;
        background: #F5EFE3;
        border: 1px solid #D8BD91;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
    }}
    .kb-title {{
        font-family: 'Playfair Display', 'Baskerville', Georgia, serif !important;
        font-size: 1.45rem !important;
        font-weight: 700 !important;
        color: #102442 !important;
        margin: 0 0 0.2rem !important;
        line-height: 1.2 !important;
    }}
    .kb-subtitle {{
        font-family: 'Inter', sans-serif !important;
        font-size: 0.85rem !important;
        color: #50627C !important;
        margin: 0 !important;
    }}
    .kb-help-btn {{
        width: 22px;
        height: 22px;
        border-radius: 50%;
        border: 1.5px solid #50627C;
        color: #50627C;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-family: 'Inter', sans-serif;
        font-size: 0.76rem;
        font-weight: 700;
        cursor: help;
        margin-top: 0.2rem;
        opacity: 0.85;
        transition: all 0.2s ease;
    }}
    .kb-help-btn:hover {{
        border-color: #A8752D;
        color: #A8752D;
        opacity: 1;
    }}
    .st-key-kb_upload_card [data-testid="stFileUploader"],
    .kb-card [data-testid="stFileUploader"] {{
        width: 100% !important;
        margin: 0 !important;
        padding: 0 !important;
    }}
    .st-key-kb_upload_card [data-testid="stFileUploader"] section,
    .kb-card [data-testid="stFileUploader"] section {{
        background: #FBF8F1 !important;
        border: 1.5px dashed #D8BD91 !important;
        border-radius: 8px !important;
        padding: 2.8rem 2rem 2.2rem 2rem !important;
        text-align: center !important;
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: center !important;
        cursor: pointer !important;
        transition: all 0.2s ease-in-out !important;
        box-sizing: border-box !important;
    }}
    .st-key-kb_upload_card [data-testid="stFileUploader"] section:hover,
    .kb-card [data-testid="stFileUploader"] section:hover {{
        border-color: #A8752D !important;
        background: #F5EFE3 !important;
    }}

    /* 1. Centered circular document-upload icon badge (Top) */
    .st-key-kb_upload_card [data-testid="stFileUploader"] section::before,
    .kb-card [data-testid="stFileUploader"] section::before {{
        content: "" !important;
        display: block !important;
        width: 60px !important;
        height: 60px !important;
        border-radius: 50% !important;
        background-color: #FAF7F0 !important;
        border: 1px solid #D8BD91 !important;
        margin: 0 auto 1.15rem auto !important;
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23A8752D' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z'/%3E%3Cpolyline points='14 2 14 8 20 8'/%3E%3Cline x1='12' y1='18' x2='12' y2='11'/%3E%3Cpolyline points='9 14 12 11 15 14'/%3E%3C/svg%3E") !important;
        background-repeat: no-repeat !important;
        background-position: center !important;
        background-size: 26px 26px !important;
        order: 1 !important;
    }}

    /* 2. Drag & Drop Heading + Subtitle (Middle-Top) */
    .st-key-kb_upload_card [data-testid="stFileUploaderDropzoneInstructions"],
    .kb-card [data-testid="stFileUploaderDropzoneInstructions"] {{
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: center !important;
        margin: 0 !important;
        padding: 0 !important;
        order: 2 !important;
    }}
    /* Hide ALL native Streamlit text, limit messages, and spans inside instructions */
    .st-key-kb_upload_card [data-testid="stFileUploaderDropzoneInstructions"] *,
    .kb-card [data-testid="stFileUploaderDropzoneInstructions"] * {{
        display: none !important;
    }}
    .st-key-kb_upload_card [data-testid="stFileUploaderDropzoneInstructions"]::before,
    .kb-card [data-testid="stFileUploaderDropzoneInstructions"]::before {{
        content: "Drag and drop your legal PDFs here." !important;
        display: block !important;
        font-family: 'Playfair Display', Georgia, serif !important;
        font-size: 1.35rem !important;
        font-weight: 700 !important;
        color: #102442 !important;
        margin-bottom: 0.35rem !important;
        text-align: center !important;
        line-height: 1.3 !important;
    }}
    .st-key-kb_upload_card [data-testid="stFileUploaderDropzoneInstructions"]::after,
    .kb-card [data-testid="stFileUploaderDropzoneInstructions"]::after {{
        content: "or browse files from your device." !important;
        display: block !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.88rem !important;
        color: #50627C !important;
        text-align: center !important;
        line-height: 1.3 !important;
        margin-bottom: 0.2rem !important;
    }}

    /* 3. Primary Navy "Choose PDF Files" Button (Middle-Bottom) */
    .st-key-kb_upload_card [data-testid="stFileUploaderDropzone"] button:not([aria-label]),
    .kb-card [data-testid="stFileUploaderDropzone"] button:not([aria-label]) {{
        background: #102442 !important;
        border: 1px solid #102442 !important;
        border-radius: 6px !important;
        padding: 0.65rem 1.95rem !important;
        min-height: 42px !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        order: 3 !important;
        margin: 0.95rem auto 0.95rem auto !important;
        box-shadow: 0 2px 8px rgba(16, 36, 66, 0.22) !important;
        transition: all 0.2s ease !important;
        cursor: pointer !important;
        position: relative !important;
    }}
    .st-key-kb_upload_card [data-testid="stFileUploaderDropzone"] button:not([aria-label]):hover,
    .kb-card [data-testid="stFileUploaderDropzone"] button:not([aria-label]):hover {{
        background: #18355c !important;
        border-color: #A8752D !important;
        box-shadow: 0 4px 14px rgba(16, 36, 66, 0.32) !important;
        transform: translateY(-1px);
    }}
    /* Hide all child elements inside ONLY the browse button so NO text leaks or duplicates */
    .st-key-kb_upload_card [data-testid="stFileUploaderDropzone"] button:not([aria-label]) *,
    .kb-card [data-testid="stFileUploaderDropzone"] button:not([aria-label]) * {{
        display: none !important;
    }}
    /* Render exactly ONE single clean label */
    .st-key-kb_upload_card [data-testid="stFileUploaderDropzone"] button:not([aria-label])::after,
    .kb-card [data-testid="stFileUploaderDropzone"] button:not([aria-label])::after {{
        content: "Choose PDF Files" !important;
        display: block !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.88rem !important;
        font-weight: 600 !important;
        color: #FFFFFF !important;
        letter-spacing: 0.01em !important;
        white-space: nowrap !important;
        line-height: 1 !important;
    }}

    /* Keep all secondary icon buttons (delete, add more files) completely unstyled from Choose PDF Files */
    .st-key-kb_upload_card [data-testid="stFileUploaderDropzone"] button[aria-label],
    .kb-card [data-testid="stFileUploaderDropzone"] button[aria-label],
    .st-key-kb_upload_card [data-testid="stFileUploader"] button[aria-label],
    .kb-card [data-testid="stFileUploader"] button[aria-label] {{
        all: unset !important;
        cursor: pointer !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        color: #718096 !important;
        padding: 4px 8px !important;
        border-radius: 4px !important;
    }}
    .st-key-kb_upload_card [data-testid="stFileUploaderDropzone"] button[aria-label]:hover,
    .kb-card [data-testid="stFileUploaderDropzone"] button[aria-label]:hover,
    .st-key-kb_upload_card [data-testid="stFileUploader"] button[aria-label]:hover,
    .kb-card [data-testid="stFileUploader"] button[aria-label]:hover {{
        background: #EDF2F7 !important;
        color: #C53030 !important;
    }}
    .st-key-kb_upload_card [data-testid="stFileUploaderDropzone"] button[aria-label]::after,
    .kb-card [data-testid="stFileUploaderDropzone"] button[aria-label]::after,
    .st-key-kb_upload_card [data-testid="stFileUploader"] button[aria-label]::after,
    .kb-card [data-testid="stFileUploader"] button[aria-label]::after {{
        display: none !important;
        content: "" !important;
    }}
    .st-key-kb_upload_card [data-testid="stFileUploaderDropzone"] button[aria-label] *,
    .kb-card [data-testid="stFileUploaderDropzone"] button[aria-label] *,
    .st-key-kb_upload_card [data-testid="stFileUploader"] button[aria-label] *,
    .kb-card [data-testid="stFileUploader"] button[aria-label] * {{
        display: inline-block !important;
    }}

    /* 4. Small File Restriction Label (Bottom) */
    .st-key-kb_upload_card [data-testid="stFileUploader"] section::after,
    .kb-card [data-testid="stFileUploader"] section::after {{
        content: "PDF format · Maximum 200 MB per file" !important;
        display: block !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.76rem !important;
        color: #50627C !important;
        letter-spacing: 0.02em !important;
        order: 4 !important;
        margin-top: 0.15rem !important;
        text-align: center !important;
    }}

    /* 5. Selected / Uploaded File Chips */
    .st-key-kb_upload_card [data-testid="stFileUploaderFile"],
    .kb-card [data-testid="stFileUploaderFile"] {{
        order: 5 !important;
        background: #FFFFFF !important;
        border: 1px solid #D8BD91 !important;
        border-radius: 6px !important;
        padding: 0.5rem 0.8rem !important;
        margin-top: 0.85rem !important;
        color: #102442 !important;
        max-width: 500px !important;
        width: 100% !important;
    }}

    /* ── How to Use Guide Cards ── */
    .guide-container {{
        max-width: 1320px;
        margin: 0 auto;
    }}
    .guide-grid {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1.35rem;
        margin-bottom: 1.35rem;
    }}
    .guide-card {{
        background: #FFFDF9;
        border: 1px solid #D8BC91;
        border-radius: 8px;
        padding: 1.45rem 1.65rem;
        display: flex;
        align-items: center;
        box-shadow: 0 1px 4px rgba(16, 36, 66, 0.03);
        transition: all 0.2s ease;
    }}
    .guide-card:hover {{
        border-color: #A8752C;
        box-shadow: 0 4px 14px rgba(16, 36, 66, 0.06);
        transform: translateY(-1px);
    }}
    .guide-icon-box {{
        width: 52px;
        height: 52px;
        min-width: 52px;
        border-radius: 50%;
        background: #F5EFE3;
        border: 1px solid #D8BC91;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
    }}
    .guide-v-divider {{
        width: 1px;
        height: 52px;
        background: #E7DED0;
        margin: 0 1.25rem 0 1.1rem;
        flex-shrink: 0;
    }}
    .guide-step-num {{
        font-family: 'Inter', sans-serif !important;
        font-size: 0.72rem !important;
        font-weight: 700 !important;
        color: #A8752C !important;
        letter-spacing: 0.06em !important;
        margin-bottom: 0.15rem !important;
    }}
    .guide-step-title {{
        font-family: 'Playfair Display', 'Baskerville', Georgia, serif !important;
        font-size: 1.2rem !important;
        font-weight: 700 !important;
        color: #102442 !important;
        margin: 0 0 0.35rem !important;
        line-height: 1.25 !important;
    }}
    .guide-step-desc {{
        font-family: 'Inter', sans-serif !important;
        font-size: 0.82rem !important;
        line-height: 1.5 !important;
        color: #50627C !important;
        margin: 0 !important;
    }}
    .safeguard-card {{
        background: #FFFDF9 !important;
        border: 1px solid #D8BC91 !important;
        border-left: 4px solid #A8752C !important;
        border-radius: 8px !important;
        padding: 1.45rem 1.85rem !important;
        display: flex !important;
        align-items: center !important;
        box-shadow: 0 2px 10px rgba(16, 36, 66, 0.04) !important;
        margin-top: 1.45rem !important;
    }}
    .safeguard-v-divider {{
        width: 1px !important;
        height: 64px !important;
        background: #E7DED0 !important;
        margin: 0 1.35rem 0 1.2rem !important;
        flex-shrink: 0 !important;
    }}

    .home-image-frame {{
        position: relative;
        overflow: hidden;
        aspect-ratio: 1.29 / 1;
        border: 1px solid #735a3a;
        border-radius: 6px;
        box-shadow: 0 18px 42px rgba(12, 27, 51, 0.18);
    }}
    .home-hero-image {{
        display: block;
        width: 100%;
        height: 100%;
        object-fit: cover;
        transform: scale(1.38);
    }}
    .home-image-caption {{
        position: absolute;
        right: 0;
        bottom: 0;
        left: 0;
        padding: 0.5rem 0.85rem;
        background: linear-gradient(to top, rgba(12, 27, 51, 0.96), rgba(12, 27, 51, 0.72));
    }}
    .home-image-label {{
        color: #ffffff !important;
        font-size: 0.58rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }}
    .home-image-caption h2.home-image-title {{
        color: #ffffff !important;
        font-family: 'Playfair Display', serif !important;
        font-size: 0.95rem !important;
        font-weight: 600;
        line-height: 1.25 !important;
        margin: 0.2rem 0 0 !important;
    }}
    .home-kicker {{
        color: #735a3a !important;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.14em;
        text-transform: uppercase;
    }}
    .home-title {{
        color: #0c1b33 !important;
        font-family: 'Playfair Display', serif !important;
        font-size: 2.8rem !important;
        font-weight: 700;
        line-height: 1.1 !important;
        margin: 0.7rem 0 0.8rem !important;
    }}
    .home-title em {{
        color: #735a3a !important;
    }}
    .home-description {{
        color: #44474d !important;
        font-size: 0.9rem !important;
        line-height: 1.5 !important;
        max-width: 34rem;
    }}
    .home-feature {{
        color: #44474d !important;
        font-size: 0.78rem !important;
        font-weight: 600;
        letter-spacing: 0;
    }}
    .home-feature-row {{
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        border-top: 1px solid #c5c6ce;
        margin: 0.75rem 0 0;
    }}
    .home-feature-item {{
        display: flex;
        align-items: center;
        gap: 0.75rem;
        min-height: 3.5rem;
        padding: 0.55rem 0.75rem;
    }}
    .home-feature-item + .home-feature-item {{
        border-left: 1px solid #c5c6ce;
    }}
    .home-feature-icon {{
        color: #735a3a !important;
        font-size: 1.45rem;
    }}
    .home-section-heading {{
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin: 0 0 0.35rem;
    }}
    .home-section-heading::before, .home-section-heading::after {{
        content: "";
        height: 1px;
        background: #735a3a;
    }}
    .home-section-heading::before {{
        width: 1rem;
    }}
    .home-section-heading::after {{
        flex: 1;
    }}
    .home-section-heading span {{
        color: #0c1b33 !important;
        font-family: 'Playfair Display', serif !important;
        font-size: 1.4rem;
        font-weight: 600;
    }}
    .home-foundation-card {{
        min-height: 9.25rem;
        padding: 0.85rem 1rem;
        background: #ffffff;
        border: 1px solid #c5c6ce;
        border-left: 3px solid var(--foundation-accent);
        border-radius: 4px;
    }}
    .home-foundation-icon {{
        color: var(--foundation-accent) !important;
        font-size: 1.3rem;
    }}
    .home-foundation-title {{
        color: #0c1b33 !important;
        font-family: 'Playfair Display', serif !important;
        font-size: 1.1rem !important;
        font-style: italic;
        line-height: 1.25 !important;
        margin: 0.25rem 0 !important;
    }}
    .home-foundation-copy {{
        color: #44474d !important;
        font-size: 0.72rem !important;
        line-height: 1.4 !important;
        margin: 0 !important;
    }}
    .home-header-note {{
        color: #735a3a !important;
        font-size: 0.6rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-align: right;
        white-space: nowrap;
    }}
    :root {{
        --legal-navy: #0c1b33;
        --legal-gold: #735a3a;
        --legal-ivory: #faf9f7;
        --legal-surface: #ffffff;
        --legal-text: #1a1c1b;
        --legal-muted: #44474d;
        --legal-border: #c5c6ce;
        --legal-green: #047857;
    }}
    .st-key-research_query_panel {{
        background: #FFFEFB !important;
        border: 1px solid #D8BD91 !important;
        border-left: 3.5px solid #A8752D !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 18px rgba(16, 36, 66, 0.05) !important;
        padding: 1.1rem 1.35rem 1.35rem !important;
        margin-bottom: 1.25rem !important;
    }}
    .st-key-research_query_panel [data-testid="stTextArea"] label p {{
        color: #A8752D !important;
        font-size: 0.72rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.1em !important;
        text-transform: uppercase;
        font-family: 'Inter', sans-serif !important;
    }}
    .st-key-research_query_panel .stTextArea textarea {{
        min-height: 6.5rem;
        border-color: #D8BD91 !important;
        border-radius: 6px !important;
        background: #ffffff !important;
        box-shadow: none !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.92rem !important;
        color: #102442 !important;
    }}
    .st-key-research_query_panel .stTextArea textarea:focus {{
        border-color: #A8752D !important;
        box-shadow: 0 0 0 2px rgba(168, 117, 45, 0.18) !important;
    }}
    .st-key-research_query_panel .stButton button:not([kind="primary"]) {{
        min-height: 2.75rem;
        background: #ffffff !important;
        border-color: #D8BD91 !important;
        color: #102442 !important;
        font-size: 0.78rem !important;
        font-weight: 600 !important;
        text-align: left;
        border-radius: 6px !important;
        transition: all 0.2s ease !important;
    }}
    .st-key-research_query_panel .stButton button:not([kind="primary"]):hover {{
        background: #FAF7F0 !important;
        border-color: #A8752D !important;
        color: #0c1b33 !important;
    }}
    .research-panel-heading {{
        color: #0c1b33 !important;
        font-family: 'Playfair Display', serif !important;
        font-size: 1.3rem;
        font-weight: 600;
        margin: 0 0 0.2rem;
    }}
    .research-panel-subtitle {{
        color: #44474d !important;
        font-size: 0.84rem;
        margin: 0 0 0.8rem;
    }}
    .research-section-kicker {{
        color: #735a3a !important;
        font-size: 0.64rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }}
    .core-authority-heading {{
        color: #0c1b33 !important;
        font-family: 'Playfair Display', serif !important;
        font-size: 1.2rem;
        font-weight: 600;
        margin: 0;
    }}
    .core-authority-header {{
        margin-bottom: 0.9rem;
        padding-bottom: 0.8rem;
        border-bottom: 1px solid #e2ddd3;
    }}
    .core-authority-header h3 {{
        color: #252a32 !important;
        font-family: 'Playfair Display', serif !important;
        font-size: 1.35rem;
        line-height: 1.3;
        overflow-wrap: anywhere;
        margin: 0.25rem 0 0;
    }}
    .query-transformation-panel {{
        margin-top: 0.8rem;
        padding: 0.75rem 0.85rem;
        background: #faf9f7;
        border: 1px solid #e2ddd3;
        border-left: 3px solid #b08d57;
        border-radius: 4px;
        color: #252a32 !important;
        font-size: 0.78rem;
        line-height: 1.5;
        overflow-wrap: anywhere;
    }}
    .core-authority-header .query-transformation-panel code {{
        color: #0c1b33;
        white-space: pre-wrap;
        overflow-wrap: anywhere;
    }}
    .st-key-core_authority_card {{
        margin-bottom: 0.8rem;
        padding: 1rem 1.1rem;
        background: #ffffff !important;
        border-color: #b08d57 !important;
        border-radius: 8px !important;
        box-shadow: 0 5px 16px rgba(12, 27, 51, 0.045);
    }}
    .st-key-core_authority_card [data-testid="stMarkdownContainer"] p {{
        color: #252a32;
        font-size: 0.94rem;
        line-height: 1.65;
        max-width: 76ch;
    }}
    .st-key-core_authority_card [data-testid="stMarkdownContainer"] h1,
    .st-key-core_authority_card [data-testid="stMarkdownContainer"] h2,
    .st-key-core_authority_card [data-testid="stMarkdownContainer"] h3,
    .st-key-core_authority_card [data-testid="stMarkdownContainer"] h4 {{
        color: #0c1b33 !important;
        font-family: 'Playfair Display', serif !important;
    }}
    .score-note {{
        color: #44474d !important;
        font-size: 0.72rem;
        line-height: 1.4;
    }}
    .pipeline-overview {{
        display: block;
        margin: 0.75rem 0 1.15rem;
        padding: 0.85rem;
        background: rgba(255, 255, 255, 0.92);
        border: 1px solid #e2ddd3;
        border-radius: 8px;
    }}
    .pipeline-overview-heading {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 1rem;
        margin-bottom: 0.7rem;
        color: #0c1b33 !important;
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }}
    .pipeline-result {{
        padding: 0.25rem 0.55rem;
        border: 1px solid #c5c6ce;
        border-radius: 999px;
        font-size: 0.64rem;
        letter-spacing: 0;
        text-transform: none;
    }}
    .pipeline-result.success {{
        color: #047857 !important;
        border-color: rgba(4, 120, 87, 0.35);
        background: rgba(4, 120, 87, 0.06);
    }}
    .pipeline-result.review {{
        color: #735a3a !important;
        border-color: rgba(115, 90, 58, 0.35);
        background: rgba(115, 90, 58, 0.06);
    }}
    .pipeline-grid {{
        position: relative;
        display: grid;
        grid-template-columns: repeat(7, minmax(0, 1fr));
        gap: 0.45rem;
    }}
    .pipeline-grid::before {{
        position: absolute;
        top: 1.48rem;
        right: 2rem;
        left: 2rem;
        height: 1px;
        background: #c5c6ce;
        content: "";
    }}
    .pipeline-stage {{
        position: relative;
        z-index: 1;
        min-width: 0;
        text-align: center;
    }}
    .pipeline-stage-number {{
        display: grid;
        width: 1.35rem;
        height: 1.35rem;
        margin: 0 auto 0.45rem;
        place-items: center;
        border: 1px solid #c5c6ce;
        border-radius: 50%;
        background: #ffffff;
        color: #44474d !important;
        font-size: 0.68rem;
        font-weight: 700;
    }}
    .pipeline-stage.success .pipeline-stage-number {{
        border-color: #047857;
        color: #047857 !important;
    }}
    .pipeline-stage.review .pipeline-stage-number {{
        border-color: #735a3a;
        color: #735a3a !important;
    }}
    .pipeline-stage-name {{
        color: #1a1c1b !important;
        font-size: 0.68rem;
        font-weight: 700;
        line-height: 1.25;
    }}
    .pipeline-stage-state {{
        display: block;
        margin-top: 0.2rem;
        color: #667085 !important;
        font-size: 0.62rem;
        line-height: 1.25;
        overflow-wrap: anywhere;
    }}
    .pipeline-stage.success .pipeline-stage-state {{ color: #047857 !important; }}
    .pipeline-stage.review .pipeline-stage-state {{ color: #735a3a !important; }}
    .result-section-label {{
        color: #735a3a !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.68rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.1em !important;
        text-transform: uppercase;
    }}
    .workspace-result-header {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        margin: 0.25rem 0 0.9rem;
        padding-bottom: 0.7rem;
        border-bottom: 1px solid #b08d57;
    }}
    .workspace-result-header h2 {{
        color: #0c1b33 !important;
        font-family: 'Playfair Display', serif !important;
        font-size: 1.25rem;
        margin: 0.15rem 0 0.3rem;
    }}
    .result-status {{
        display: inline-flex;
        padding: 0.2rem 0.5rem;
        border: 1px solid #c5c6ce;
        border-radius: 999px;
        font-size: 0.66rem;
        font-weight: 700;
    }}
    .result-status.success {{
        border-color: rgba(4, 120, 87, 0.35);
        color: #047857 !important;
        background: rgba(4, 120, 87, 0.06);
    }}
    .result-status.review {{
        border-color: rgba(115, 90, 58, 0.4);
        color: #735a3a !important;
        background: rgba(115, 90, 58, 0.06);
    }}
    .workspace-result-metrics {{
        display: flex;
        flex-wrap: wrap;
        justify-content: flex-end;
        gap: 0.5rem;
    }}
    .workspace-result-metric {{
        min-width: 5.5rem;
        padding: 0.4rem 0.6rem;
        border: 1px solid #e2ddd3;
        border-radius: 6px;
        background: #ffffff;
        text-align: center;
    }}
    .workspace-result-metric strong {{
        display: block;
        color: #0c1b33 !important;
        font-family: 'Playfair Display', serif !important;
        font-size: 0.95rem;
    }}
    .workspace-result-metric span {{
        display: block;
        color: #667085 !important;
        font-size: 0.58rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }}
    .evidence-card {{
        background: #ffffff;
        border: 1px solid #e2ddd3;
        border-left: 3px solid var(--evidence-accent, #735a3a);
        border-radius: 6px;
        padding: 0.85rem 1rem;
        height: 100%;
    }}
    .evidence-card-title {{
        color: #735a3a !important;
        font-size: 0.64rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }}
    .evidence-card-copy {{
        color: #44474d !important;
        font-size: 0.76rem;
        line-height: 1.45;
        margin: 0.4rem 0;
    }}
    .authority-meta {{
        display: flex;
        flex-wrap: wrap;
        gap: 0.35rem 0.75rem;
        color: #667085 !important;
        font-size: 0.72rem;
    }}
    .authority-excerpt {{
        padding: 0.75rem 0.85rem;
        background: #faf9f7;
        border-left: 2px solid #b08d57;
        border-radius: 0 4px 4px 0;
        color: #252a32 !important;
        line-height: 1.55;
    }}
    .intelligence-panel {{
        background: #0c1b33;
        border: 1px solid #735a3a;
        border-radius: 8px;
        padding: 1.15rem;
        color: #faf9f7;
    }}
    .intelligence-panel h3 {{
        color: #faf9f7 !important;
        font-family: 'Playfair Display', serif !important;
        font-size: 1.15rem;
        margin: 0 0 0.8rem;
    }}
    .intelligence-row {{
        display: flex;
        justify-content: space-between;
        gap: 0.8rem;
        padding: 0.55rem 0;
        border-bottom: 1px solid rgba(250, 249, 247, 0.16);
        font-size: 0.76rem;
    }}
    .intelligence-row:last-child {{ border-bottom: 0; }}
    .intelligence-label {{ color: #d5d7dc !important; }}
    .intelligence-value {{ color: #ffffff !important; font-weight: 600; text-align: right; overflow-wrap: anywhere; }}
    .intelligence-value.success {{ color: #67c7a4 !important; }}
    .intelligence-value.review {{ color: #d0b277 !important; }}
    .intelligence-value.neutral {{ color: #e4e5e8 !important; }}
    [data-testid="stExpander"] {{
        border-color: #e2ddd3 !important;
        border-radius: 6px !important;
        background: #ffffff !important;
    }}
    [data-testid="stExpander"] summary {{ overflow-wrap: anywhere; }}
    a {{
        color: #0c1b33;
        text-decoration-color: #735a3a;
        text-underline-offset: 4px;
    }}
    a:hover {{
        color: #735a3a;
    }}
    a:focus-visible, .stButton button:focus-visible {{
        outline: 2px solid #735a3a !important;
        outline-offset: 3px;
    }}

  /* Text area styling */
  .stTextArea textarea {{
    background-color: #ffffff !important;
        border: 1px solid #c5c6ce !important;
    border-radius: 4px !important;
    font-family: 'Inter', sans-serif !important;
    font-style: normal !important;
    font-size: 1rem !important;
    color: #1a1c1b !important;
    padding: 0.9rem !important;
    line-height: 1.55 !important;
  }}
  .stTextArea textarea:focus {{
        border-color: #735a3a !important;
        box-shadow: 0 0 0 2px rgba(115, 90, 58, 0.16) !important;
  }}

  /* Buttons styling */
  .stButton button {{
    background-color: #ffffff !important;
    color: #1a1c1b !important;
    border: 1px solid #c5c6ce !important;
    border-radius: 4px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    letter-spacing: 0 !important;
    padding: 0.6rem 1rem !important;
    transition: all 0.2s ease !important;
  }}
  .stButton button:hover {{
    background-color: #efeeec !important;
    border-color: #0c1b33 !important;
  }}
  .stButton button[kind="primary"] {{
        background-color: #0c1b33 !important;
    color: #ffffff !important;
        border: 1px solid #735a3a !important;
        padding: 0.75rem 1.35rem !important;
        font-size: 0.9rem !important;
    }}
    .stButton button[kind="primary"]:hover {{
        background-color: #0c1b33 !important;
        border-color: #735a3a !important;
        filter: brightness(0.92);
  }}
  .stButton button[kind="primary"] p {{
    color: #ffffff !important;
  }}
    .stButton button:disabled {{
        opacity: 0.58 !important;
        cursor: not-allowed !important;
    }}

  /* Download Button */
  .stDownloadButton button {{
        background-color: #ffffff !important;
        color: #0c1b33 !important;
        border: 1px solid #0c1b33 !important;
    border-radius: 4px !important;
    width: 100% !important;
        text-transform: none !important;
        letter-spacing: 0 !important;
    font-weight: 600 !important;
    padding: 0.7rem !important;
  }}
  .stDownloadButton button p {{
        color: #0c1b33 !important;
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

  @media (max-width: 700px) {{
    .block-container {{
      padding: 0.75rem 1rem 1.5rem !important;
    }}
        .home-title {{
            font-size: 1.55rem !important;
        }}
        .home-image-frame {{
            aspect-ratio: 1.2 / 1;
        }}
        .home-feature-row {{
            grid-template-columns: 1fr;
            margin: 0.75rem 0 1.1rem;
        }}
        .home-feature-item {{
            min-height: 2.75rem;
            padding: 0.4rem 0.25rem;
        }}
        .home-feature-item + .home-feature-item {{
            border-top: 1px solid #c5c6ce;
            border-left: 0;
        }}
        .home-header-note {{
            display: none;
        }}
        [data-testid="stRadio"] [role="radiogroup"] {{
            flex-wrap: wrap;
            justify-content: flex-start;
            overflow-x: visible;
        }}
        [data-testid="stRadio"] [role="radiogroup"] label,
        [data-testid="stRadio"] [role="radiogroup"] label p {{
            padding: 0.55rem 0.35rem !important;
            font-size: 0.68rem !important;
        }}
        .pipeline-overview {{
            padding: 0.7rem;
        }}
        .pipeline-overview-heading {{
            align-items: flex-start;
            flex-direction: column;
        }}
        .workspace-result-header {{
            align-items: flex-start;
            flex-direction: column;
        }}
        .workspace-result-metrics {{
            justify-content: flex-start;
            width: 100%;
        }}
        .pipeline-grid {{
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.65rem;
        }}
        .pipeline-grid::before {{ display: none; }}
        .pipeline-stage {{
            display: grid;
            grid-template-columns: 1.4rem minmax(0, 1fr);
            column-gap: 0.45rem;
            text-align: left;
        }}
        .pipeline-stage-number {{ grid-row: span 2; margin: 0; }}
        .pipeline-stage-name {{ align-self: end; }}
        .pipeline-stage-state {{ margin-top: 0.1rem; }}
  }}

    @media (max-width: 1250px) {{
        [data-testid="stRadio"] [role="radiogroup"]::after {{
            display: none;
        }}
    }}

    @media (min-width: 701px) and (max-width: 1100px) {{
        .home-title {{
            font-size: 2.3rem !important;
        }}
        [data-testid="stRadio"] [role="radiogroup"] {{
            overflow-x: hidden;
            gap: 0.85rem !important;
        }}
        [data-testid="stRadio"] [role="radiogroup"]::after {{
            display: none;
        }}
        [data-testid="stRadio"] [role="radiogroup"] label {{
            padding: 0.3rem 0.1rem !important;
        }}
        [data-testid="stRadio"] [role="radiogroup"] label p {{
            font-size: 0.82rem !important;
        }}
    }}

    @media (min-width: 1101px) and (max-width: 1250px) {{
        .home-title {{
            font-size: 2.6rem !important;
        }}
        .home-header-note {{
            display: none;
        }}
    }}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Preparing legal retrieval workspace…")
def _init_legal_retriever(_corpus: list[dict], cache_token: str = "v4_live") -> LegalRetriever:
    return LegalRetriever(_corpus)


def get_retriever() -> LegalRetriever:
    corpus = load_corpus()
    retriever = _init_legal_retriever(corpus, cache_token="v4_live")
    if not hasattr(retriever, "live_web_search"):
        _init_legal_retriever.clear()
        retriever = LegalRetriever(corpus)
    return retriever


def reset_retriever() -> None:
    _init_legal_retriever.clear()


def render_source_links(actions: list[dict]) -> None:
    for action in actions:
        st.markdown(f"- [{action['label']}]({action['url']}) — {action['why']}")


def memo(question: str, result: dict) -> str:
    sources = "\n".join(
        f"- [S{i}] {item['title']} — {item['url']} (Cross-Encoder: {item.get('cross_encoder_prob', 0):.1%})"
        for i, item in enumerate(result["evidence"], 1)
    ) or "- No adequate source material was retrieved."
    outcome = result["answer"] or f"Insufficient evidence: {result['evidence_reason']}"
    reformulate_note = f" (Reformulated from original inquiry: '{result.get('reformulate_reason', '')}')" if result.get('was_reformulated') else " (Original query verified specific)"
    retry_note = f"\n- Stage 6 Retry Query: {result.get('retry_query')}" if result.get('retried') else ""
    return f"""# NyayaRoute Legal Research Memo

**Original Question:** {question}
**Reformulated Search Query:** {result['query_used']}{reformulate_note}

**Research outcome**

{outcome}

## Sources consulted
{sources}

## 7-Stage Pipeline Audit Trail
- Stage 1 Query Analysis: {result.get('quality_reason', '')}
- Stage 2 Reformulated Query: {result['query_used']} (Was Reformulated: {'Yes' if result.get('was_reformulated') else 'No'})
- Stage 3 Retrieval Route: {result['route']} — {result['route_reason']}
- Stage 4 Neural Reranker: ms-marco-MiniLM evaluated {result.get('candidate_count', 0)} passages ({result.get('live_web_count', 0)} live judicial precedents)
- Stage 5 Evidence Sufficiency Gate: {'Passed' if result['enough'] else 'Not Passed'} — {result['evidence_reason']}
- Stage 6 Retry Executed: {'Yes' if result['retried'] else 'No'}{retry_note}
- Stage 7 Guardrail Status: {'Authoritative Answer' if not result.get('is_insufficient_evidence_response') else 'Explicit Refusal to Hallucinate'}
- Live Web Retrieval: {'Active (Real-time Judicial Precedents)' if result.get('use_live_web') else 'Disabled'}

*This is legal research information, not legal advice. Verify the current primary text and seek qualified legal counsel for advice.*
"""


def render_pipeline_overview(result: dict) -> None:
    enough = bool(result.get("enough"))
    retried = bool(result.get("retried"))
    candidate_count = result.get("candidate_count", len(result.get("evidence", [])))
    stages = [
        ("Query Analysis", "Terms and citations reviewed", "success"),
        (
            "Reformulation",
            "Refined" if result.get("was_reformulated") else "Original query retained",
            "success" if result.get("was_reformulated") else "neutral",
        ),
        ("Retrieval", result.get("route") or "Route selected", "success"),
        (
            "Neural Reranking",
            f"{candidate_count} candidates",
            "success" if candidate_count else "review",
        ),
        (
            "Evidence Gate",
            "Passed" if enough else "Insufficient",
            "success" if enough else "review",
        ),
        (
            "Retry Assessment",
            result.get("retry_route") or "Not required" if retried else "Not required",
            "review" if retried else "neutral",
        ),
        (
            "Answer Synthesis",
            "Grounded answer" if enough else "Safe refusal",
            "success" if enough else "review",
        ),
    ]
    overall_status = "Evidence gate passed" if enough else "Insufficient evidence"
    overall_class = "success" if enough else "review"
    stage_html = "".join(
        f'<div class="pipeline-stage {status}"><span class="pipeline-stage-number">{index}</span><span class="pipeline-stage-name">{html_escape(name)}</span><span class="pipeline-stage-state">{html_escape(str(detail))}</span></div>'
        for index, (name, detail, status) in enumerate(stages, 1)
    )
    st.markdown(
        f'<section class="pipeline-overview" aria-label="Seven-stage legal research pipeline"><div class="pipeline-overview-heading"><span>Seven-stage legal research pipeline</span><span class="pipeline-result {overall_class}">{overall_status}</span></div><div class="pipeline-grid">{stage_html}</div></section>',
        unsafe_allow_html=True,
    )


# ── Global Sidebar ─────────────────────────────────────────────────────────────
corpus = load_corpus()
source_options = sorted({item["source"] for item in corpus})

if "use_live_web" not in st.session_state:
    st.session_state["use_live_web"] = True
if "selected_sources" not in st.session_state:
    st.session_state["selected_sources"] = source_options

with st.sidebar:
    st.markdown("<h3 style='font-family:\"Playfair Display\",serif; font-size:1.3rem; font-weight:700; margin-bottom:0.75rem; color:#0c1b33;'>⚖ Research Settings</h3>", unsafe_allow_html=True)
    
    use_live_web_sb = st.toggle(
        "🌐 Real-Time Precedent Search",
        value=st.session_state["use_live_web"],
        key="sb_toggle_live_web",
        help="When enabled, retrieves live Supreme Court and High Court judgments and precedents in real-time."
    )
    st.session_state["use_live_web"] = use_live_web_sb
    if use_live_web_sb:
        st.caption("🟢 Live Retrieval Active: Real-time judicial precedents will be retrieved, neural re-ranked, and cited.")
    else:
        st.caption("⚪ Local Mode: Limited to indexed local corpus chunks.")

    selected_sources_sb = st.multiselect(
        "Search these sources",
        source_options,
        default=st.session_state["selected_sources"],
        key="sb_multiselect_sources",
    )
    st.session_state["selected_sources"] = selected_sources_sb
    st.caption("Use the source filter to focus statutory research or case-law research.")
    
    # st.divider()
    # st.metric("Local Legal Chunks", len(corpus))
    # st.caption(f"{len(source_summary(corpus))} source page(s) indexed")
    
    # with st.expander("AI Synthesis Mode", expanded=not bool(os.getenv("GROQ_API_KEY"))):
    #     with st.form("groq_api_key_form"):
    #         api_key_input = st.text_input(
    #             "Groq API Key (Optional)",
    #             value=os.getenv("GROQ_API_KEY", ""),
    #             type="password",
    #             help="Enter your Groq key for generative synthesis with Groq LLMs. If omitted, local extractive legal synthesis is used automatically.",
    #         )
    #         apply_api_key = st.form_submit_button("Apply API key")
    #     if apply_api_key and api_key_input and api_key_input != os.getenv("GROQ_API_KEY"):
    #         os.environ["GROQ_API_KEY"] = api_key_input
    #         st.rerun()

    if os.getenv("GROQ_API_KEY"):
        st.success(f"Generative LLM Active · {os.getenv('GROQ_MODEL', 'qwen/qwen3.8-27b')}")
    else:
        st.info("Local Extractive Synthesis Active. (Add Groq key for generative reasoning).")
    st.info("Not legal advice. Verify citations in their original source.")


def go_to_research() -> None:
    st.session_state["main_navigation"] = "Research workspace"


def set_research_question(value: str) -> None:
    """Set the sample inquiry before Streamlit recreates its text-area widget."""
    st.session_state.pending_question = value
    st.session_state.research_question_input = value


navigation_items = [
    "Home",
    "Research workspace",
    "Source Library",
    "Knowledge Base",
    "How to use this",
]
brand_column, navigation_column, badge_column = st.columns(
    [2.2, 5.8, 2.0], gap="medium", vertical_alignment="center"
)
with brand_column:
    st.markdown(
        """
        <div style="display:flex; align-items:center; gap:0.65rem; padding:0.2rem 0; white-space:nowrap;">
            <span class="material-symbols-outlined" style="color:#A47B3C; font-size:2rem; line-height:1;">balance</span>
            <div>
                <span style="font-family:'Playfair Display', Georgia, serif; font-size:1.45rem; font-weight:700; letter-spacing:0.02em; color:#102744; line-height:1; display:block;">NyayaRoute</span>
                <span style="font-family:'Inter', sans-serif; font-size:0.54rem; font-weight:700; letter-spacing:0.18em; color:#A47B3C; text-transform:uppercase; display:block; margin-top:3px;">Legal Research Workspace</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with navigation_column:
    selected_page = st.radio(
        "Main navigation",
        navigation_items,
        horizontal=True,
        label_visibility="collapsed",
        key="main_navigation",
    )
with badge_column:
    st.markdown(
        """
        <div style="display:flex; align-items:center; justify-content:flex-end; border-left:1px solid #D8C39E; padding-left:1.2rem; height:2.2rem; width:100%;">
            <span style="font-family:'Inter',sans-serif; font-size:0.65rem; font-weight:700; letter-spacing:0.14em; color:#A47B3C; text-transform:uppercase; white-space:nowrap;">EVIDENCE · LAW · CLARITY</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown('<div style="border-bottom:1px solid #E7DED0; margin-top:0.35rem; margin-bottom:1.4rem;"></div>', unsafe_allow_html=True)

if selected_page == "Home":
    st.markdown(
        """
        <style>
          [data-testid="stSidebar"] { display: none !important; }
          .block-container {
            max-width: 100% !important;
            width: 100% !important;
            padding: 1.2rem 3.5rem 3.5rem !important;
            margin: 0 !important;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )
    home_copy, home_image = st.columns([1.15, 0.95], gap="large")
    with home_copy:
        st.markdown(
            """
            <div style="display:flex; align-items:center; gap:0.75rem; margin-top:1rem; margin-bottom:1.4rem;">
                <div style="width:2.2rem; height:1px; background:#735a3a;"></div>
                <span style="font-family:'Inter',sans-serif; font-size:0.72rem; font-weight:700; letter-spacing:0.18em; color:#735a3a; text-transform:uppercase;">EVIDENCE-FIRST LEGAL RESEARCH</span>
                <div style="width:2.2rem; height:1px; background:#735a3a;"></div>
            </div>
            <h1 class="home-title" style="font-family:'Playfair Display', 'Baskerville', Georgia, 'Times New Roman', serif !important; font-size:3.2rem; font-weight:700; line-height:1.15; color:#0c1b33 !important; margin:0 0 1.25rem;">
                Institutional <span class="title-accent" style="font-family:'Playfair Display', 'Baskerville', Georgia, 'Times New Roman', serif !important; font-style:italic !important; color:#735a3a !important;">Authority</span> in<br>Legal Jurisprudence
            </h1>
            <p style="font-family:'Inter',sans-serif; font-size:1.05rem; line-height:1.65; color:#44474d; margin:0 0 2rem; max-width:38rem;">
                Research Indian law with source-linked answers grounded in statutes and judicial decisions.
            </p>
            """,
            unsafe_allow_html=True,
        )
        st.button(
            "Begin inquiry  →",
            type="primary",
            key="begin_inquiry",
            on_click=go_to_research,
        )
        st.markdown(
            """
            <div style="display:grid; grid-template-columns:1fr 1fr 1fr; border-top:1px solid #c5c6ce; margin-top:2.6rem; padding-top:1.4rem; max-width:44rem;">
              <div style="display:flex; align-items:center; gap:0.65rem; padding-right:0.75rem; border-right:1px solid #c5c6ce;">
                <span class="material-symbols-outlined" style="color:#735a3a; font-size:1.5rem;">description</span>
                <span style="font-family:'Inter',sans-serif; font-size:0.82rem; font-weight:600; color:#1a1c1b;">Statutes and<br>judgments</span>
              </div>
              <div style="display:flex; align-items:center; gap:0.65rem; padding:0 0.75rem; border-right:1px solid #c5c6ce;">
                <span class="material-symbols-outlined" style="color:#735a3a; font-size:1.5rem;">balance</span>
                <span style="font-family:'Inter',sans-serif; font-size:0.82rem; font-weight:600; color:#1a1c1b;">Cited research<br>answers</span>
              </div>
              <div style="display:flex; align-items:center; gap:0.65rem; padding-left:0.75rem;">
                <span class="material-symbols-outlined" style="color:#735a3a; font-size:1.5rem;">article</span>
                <span style="font-family:'Inter',sans-serif; font-size:0.82rem; font-weight:600; color:#1a1c1b;">Exportable<br>legal briefs</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with home_image:
        st.markdown(
            f"""
            <div style="position:relative; border-radius:14px; overflow:hidden; border:1px solid #735a3a; box-shadow:0 14px 36px rgba(12,27,51,0.16); aspect-ratio:386/298; max-width:620px; width:100%; margin-left:auto; margin-top:1rem;">
                <img src="{img_src}" alt="Scales of Justice & Precedent Archive" style="display:block; width:100%; height:100%; object-fit:cover;">
                <div style="position:absolute; bottom:0; left:0; right:0; padding:1.25rem 1.4rem; background:linear-gradient(to top, rgba(12,27,51,0.92) 0%, rgba(12,27,51,0.45) 75%, transparent 100%);">
                    <div style="font-family:'Inter',sans-serif; font-size:0.62rem; font-weight:700; letter-spacing:0.16em; color:#d7e3ff; text-transform:uppercase; margin-bottom:0.25rem;">SOVEREIGN JURISPRUDENCE</div>
                    <div style="font-family:'Playfair Display',serif; font-size:1.15rem; font-style:italic; font-weight:600; color:#ffffff;">Scales of Justice &amp; Precedent Archive</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ──────────────────────────────────────────────────────────────────────────────
elif selected_page == "Research workspace":
    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None

    render_page_hero(
        eyebrow="EVIDENCE-FIRST LEGAL RETRIEVAL",
        title="Research Workspace",
        subtitle="Search indexed legal sources and review the authorities behind each answer.",
        col_ratio=[1.55, 1.45],
    )

    with st.container(border=True, key="research_query_panel"):
        default_val = st.session_state.get("pending_question") or ""
        question = st.text_area(
            "Legal question or statutory inquiry",
            value=default_val,
            placeholder="e.g. What protections does Article 21 provide?",
            height=105,
            key="research_question_input",
        )

        samples = st.columns(3, gap="small")
        sample_items = [
            ("Fundamental Rights · Article 21", "What does Article 21 of the Constitution protect?", ":material/balance:", "sample_article_21"),
            ("IT Act Amendments", "What is the latest amendment to the IT Act?", ":material/gavel:", "sample_it_act"),
            ("Writ & Liberty Cases", "Which Supreme Court cases discuss personal liberty?", ":material/account_balance:", "sample_writ_liberty"),
        ]
        for column, (label, value, icon, key) in zip(samples, sample_items):
            column.button(
                label,
                icon=icon,
                use_container_width=True,
                key=key,
                on_click=set_research_question,
                args=(value,),
            )

        investigate = st.button(
            "Research this question",
            type="primary",
            icon=":material/search:",
            use_container_width=True,
            key="research_question_submit",
        )

    if investigate:
        selected_raw = st.session_state.get("selected_sources")
        active_sources = source_options if selected_raw is None else selected_raw
        active_live_web = st.session_state.get("use_live_web", True)
        if not question.strip():
            st.warning("Please enter a legal research question first.")
        elif not active_sources:
            st.warning("Select at least one source to search.")
        else:
            with st.status("Executing seven-stage legal research pipeline…", expanded=False) as status:
                result = run_agent(question.strip(), get_retriever(), set(active_sources), use_live_web=active_live_web)
                
                # Dynamic visual reporting of the 7 stages from the PPT
                analysis = result.get("analysis", {})
                terms_display = ", ".join(result.get("legal_terms", [])) if result.get("legal_terms") else "None detected"
                cits_display = ", ".join(result.get("citations", [])) if result.get("citations") else "None detected"
                st.write(f"✓ **Stage 1 (Query Analysis):** Legal terms: [{terms_display}] · Citations: [{cits_display}] · Ambiguity: {'Flagged' if analysis.get('is_ambiguous') else 'Clean/Specific'}.")
                
                if result.get("was_reformulated"):
                    st.write(f"✓ **Stage 2 (Reformulation):** Reformulated weak/ambiguous query to: *\"{result['query_used']}\"* ({result['reformulate_reason']})")
                else:
                    st.write("✓ **Stage 2 (Query Validation):** Query sufficiently grounded; proceeding with original query.")
                
                st.write(f"✓ **Stage 3 (Routing):** Routed to **{result['route']}** ({result.get('candidate_count', 0)} candidates).")
                
                if result.get("use_live_web"):
                    st.write(f"✓ **Stage 4 (Neural Cross-Encoder):** Evaluated {result.get('candidate_count', 0)} candidate authorities ({result.get('live_web_count', 0)} live judicial precedents) using ms-marco-MiniLM.")
                else:
                    st.write(f"✓ **Stage 4 (Cross-Encoder Re-Ranking):** Scored candidate passages using ms-marco-MiniLM neural relevance model.")
                
                if result.get("enough"):
                    st.write(f"✓ **Stage 5 (Evidence Sufficiency Gate):** **PASSED** ({result['evidence_reason']})")
                    st.write("✓ **Stage 6 (Retry Assessment):** Bypassed (first-pass evidence passed sufficiency gate).")
                    st.write("✓ **Stage 7 (Answer Synthesis):** Grounded legal brief synthesized with source citations.")
                    status.update(label="Seven-stage pipeline complete: Evidence gate passed", state="complete")
                else:
                    st.write(f"⚠ **Stage 5 (Evidence Sufficiency Gate):** **INSUFFICIENT** ({result['evidence_reason']})")
                    if result.get("retried"):
                        st.write(f"✓ **Stage 6 (Targeted Retry):** Reformulated query retry escalated to {result.get('retry_route', 'Hybrid Retrieval')}.")
                    st.write("✓ **Stage 7 (Safe Refusal):** Explicit Insufficient-Evidence Declaration returned (hallucination prevented).")
                    status.update(label="Seven-stage pipeline complete: Insufficient evidence", state="complete")

            st.session_state["research_result"] = result
            st.session_state["research_question"] = question.strip()

    result = st.session_state.get("research_result")
    research_question = st.session_state.get("research_question")

    if result and research_question:
        render_pipeline_overview(result)

        status_text = "Evidence gate passed" if result.get("enough") else "Insufficient evidence"
        status_class = "success" if result.get("enough") else "review"
        st.markdown(f"""
                <div class="workspace-result-header">
                    <div>
                        <span class="result-section-label">Active workspace</span>
                        <h2>Core Authority &amp; Synthesis</h2>
                        <span class="result-status {status_class}">{status_text}</span>
                    </div>
                    <div class="workspace-result-metrics">
                        <div class="workspace-result-metric"><strong>{len(corpus)}</strong><span>Legal chunks</span></div>
                        <div class="workspace-result-metric"><strong>{len(result.get('evidence', []))}</strong><span>Authorities</span></div>
                        <div class="workspace-result-metric"><strong>{result.get('latency_seconds', 0):.2f}s</strong><span>Latency</span></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        left_col, right_col = st.columns([2, 1], gap="large")

        with left_col:
            query_used = result.get('query_used', research_question)
            was_reformulated = result.get('was_reformulated', False)
            reformulate_reason = result.get('reformulate_reason', '')
            retried = result.get('retried', False)
            retry_query = result.get('retry_query', '')

            # Core Authority Card (or Explicit Insufficient-Evidence Declaration)
            if result["enough"] and result.get("answer"):
                badge_bg = '#d1fae5' if was_reformulated else '#e0f2fe'
                badge_color = '#065f46' if was_reformulated else '#0369a1'
                badge_text = '✓ Reformulated for Precision' if was_reformulated else '✓ Specific Query Retained'

                safe_question = html_escape(research_question)
                safe_query_used = html_escape(query_used)
                safe_retry_query = html_escape(retry_query) if retry_query else ''
                safe_reason = html_escape(reformulate_reason) if reformulate_reason else ''

                retry_badge_html = f'<div style="margin-top:6px; padding-top:6px; border-top:1px dashed #cbd5e1;"><span style="font-weight:700; color:#b45309; font-size:0.75rem; text-transform:uppercase;">Stage 6 Retry Query: </span><code style="background:#fffbeb; color:#b45309; padding:2px 6px; border-radius:3px; font-weight:600; border:1px solid #fde68a;">{safe_retry_query}</code></div>' if retried and retry_query else ''

                rationale_html = f'<div style="margin-top:4px; font-size:0.75rem; color:#475569;"><em>Rationale:</em> {safe_reason}</div>' if reformulate_reason and was_reformulated else ''

                header_html = f"""<div style="background:#ffffff; border:1px solid #c5c6ce; border-radius:6px; padding:1.5rem 1.75rem 1.25rem; margin-bottom:1.25rem;">
<div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:0.75rem;">
<div>
<span style="font-family:'Inter',sans-serif; font-size:0.68rem; font-weight:700; text-transform:uppercase; letter-spacing:0.1em; color:#735a3a;">
Core Authority &amp; Synthesis (7-Stage Pipeline Output)
</span>
<h4 style="font-family:'Playfair Display',serif; font-size:1.3rem; font-weight:600; color:#1a1c1b; margin:0.3rem 0 0;">
{safe_question}
</h4>
</div>
<span class="material-symbols-outlined" style="color:#735a3a; font-size:22px;">verified</span>
</div>
<div style="height:1px; background:#c5c6ce; opacity:0.5; margin:0.75rem 0 1rem;"></div>
<div style="background:#f8f9fa; border:1px solid #e2e8f0; border-left:4px solid #735a3a; border-radius:4px; padding:0.85rem 1rem; margin-bottom:0.25rem;">
<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.4rem;">
<span style="font-family:'Inter',sans-serif; font-size:0.68rem; font-weight:700; text-transform:uppercase; letter-spacing:0.08em; color:#735a3a;">Stage 2 Search Query Transformation</span>
<span style="font-family:'Inter',sans-serif; font-size:0.65rem; font-weight:700; background:{badge_bg}; color:{badge_color}; padding:2px 8px; border-radius:3px;">{badge_text}</span>
</div>
<div style="font-family:'Inter',sans-serif; font-size:0.85rem; line-height:1.5; color:#1a1c1b;">
<div style="margin-bottom:0.35rem;"><span style="font-weight:600; color:#64748b; font-size:0.75rem; text-transform:uppercase;">Original Inquiry: </span><span style="color:#334155; font-style:italic;">"{safe_question}"</span></div>
<div><span style="font-weight:700; color:#0c1b33; font-size:0.75rem; text-transform:uppercase;">Reformulated Search Query: </span><code style="background:#ffffff; border:1px solid #cbd5e1; color:#0c1b33; font-weight:600; font-size:0.83rem; padding:2px 8px; border-radius:3px;">{safe_query_used}</code></div>
{rationale_html}
{retry_badge_html}
</div>
</div>
</div>"""
                st.markdown(header_html, unsafe_allow_html=True)
                st.markdown(result["answer"])

                metrics_html = f"""<div style="display:grid; grid-template-columns:repeat(3,1fr); gap:0.75rem; margin:1.25rem 0 1.5rem;">
<div style="background:#f4f3f1; border-radius:4px; padding:0.85rem; border:1px solid #e2e8f0;">
<p style="font-size:0.62rem; text-transform:uppercase; letter-spacing:0.05em; color:#44474d; margin-bottom:0.2rem;">Retrieval Route</p>
<p style="font-weight:700; font-size:0.85rem; color:#1a1c1b; margin:0;">{result['route']}</p>
</div>
<div style="background:#f4f3f1; border-radius:4px; padding:0.85rem; border:1px solid #e2e8f0;">
<p style="font-size:0.62rem; text-transform:uppercase; letter-spacing:0.05em; color:#44474d; margin-bottom:0.2rem;">Citations</p>
<p style="font-weight:700; font-size:0.85rem; color:#1a1c1b; margin:0;">{len(result['evidence'])} Authorities</p>
</div>
<div style="background:#f4f3f1; border-radius:4px; padding:0.85rem; border:1px solid #e2e8f0;">
<p style="font-size:0.62rem; text-transform:uppercase; letter-spacing:0.05em; color:#44474d; margin-bottom:0.2rem;">Gate Status</p>
<p style="font-weight:700; font-size:0.85rem; color:#047857; margin:0;">Evidence Passed</p>
</div>
</div>"""
                st.markdown(metrics_html, unsafe_allow_html=True)

            else:
                badge_color = '#735a3a' if was_reformulated else '#44474d'
                badge_text = 'Reformulated' if was_reformulated else 'Evaluated query'

                retry_badge_html = f'<div style="margin-top:6px; padding-top:6px; border-top:1px dashed #c5c6ce;"><span style="font-weight:700; color:#735a3a; font-size:0.72rem; text-transform:uppercase;">Retry query: </span><code style="background:#ffffff; color:#0c1b33; padding:2px 6px; border-radius:3px; font-weight:600; border:1px solid #e2ddd3;">{html_escape(retry_query)}</code></div>' if retried and retry_query else ''

                rationale_html = f'<div style="margin-top:4px; color:#667085; font-size:0.74rem;"><em>Rationale:</em> {html_escape(reformulate_reason)}</div>' if reformulate_reason and was_reformulated else ''

                refusal_header_html = f"""<div class="core-authority-header">
<div class="result-section-label">Core Authority &amp; Synthesis · Insufficient evidence</div>
<h3>{html_escape(research_question)}</h3>
<div class="query-transformation-panel">
<div style="display:flex; justify-content:space-between; gap:0.75rem; flex-wrap:wrap; margin-bottom:0.35rem;">
<span class="result-section-label">Search query transformation</span>
<span style="color:{badge_color}; font-size:0.68rem; font-weight:700;">{badge_text}</span>
</div>
<div><strong>Original inquiry:</strong> {html_escape(research_question)}</div>
<div><strong>Evaluated query:</strong> <code>{html_escape(query_used)}</code></div>
{rationale_html}
{retry_badge_html}
</div>
</div>"""
                with st.container(border=True, key="core_authority_refusal_card"):
                    st.markdown(refusal_header_html, unsafe_allow_html=True)
                    st.markdown(result.get('answer', result.get('evidence_reason')))
                    st.markdown("**Recommended Primary Legal Portals**")
                    render_source_links(result["actions"])
                    st.caption("Verify against official primary databases or ingest the relevant Bare Act / Judgment PDF into the Source Library.")

            snapshot_html = f"""<div style="display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:0.65rem; margin:0.75rem 0 1rem;">
<div class="evidence-card" style="--evidence-accent:{'#047857' if result['enough'] else '#735a3a'};">
<span class="evidence-card-title">Evidence gate</span>
<p class="evidence-card-copy">{html_escape(str(result.get('evidence_reason', 'No evidence-gate detail was returned.')))}</p>
<span style="color:{'#047857' if result['enough'] else '#735a3a'}; font-size:0.68rem; font-weight:700;">{ 'Passed' if result['enough'] else 'Insufficient evidence' }</span>
</div>
<div class="evidence-card" style="--evidence-accent:#0c1b33;">
<span class="evidence-card-title">Routing decision</span>
<p class="evidence-card-copy"><strong>{html_escape(str(result.get('route', 'Not reported')))}</strong> · {html_escape(str(result.get('route_reason', '')))}</p>
<span class="score-note">{int(result.get('candidate_count', 0))} candidates · Cross-encoder reranking</span>
</div>
<div class="evidence-card" style="--evidence-accent:#047857;">
<span class="evidence-card-title">Corpus coverage</span>
<p class="evidence-card-copy">FAISS vector index and BM25 are available across {len(corpus)} local legal chunks.</p>
<span class="score-note">Indexed pages: {len(source_summary(corpus))}</span>
</div>
</div>"""
            st.markdown(snapshot_html, unsafe_allow_html=True)

            st.markdown('<div class="research-section-kicker">Supporting authorities</div><h3 class="core-authority-heading">Authorities &amp; Citations</h3>', unsafe_allow_html=True)
            if not result["evidence"]:
                st.caption("No relevant chunks were retrieved from the selected local source library.")

            for index, item in enumerate(result["evidence"], 1):
                ce_prob = item.get("cross_encoder_prob", item.get("relevance", 0))
                ce_score = item.get("cross_encoder_score", item.get("score", 0))
                anchor_match = item.get("relevance", 0)
                is_live = item.get("source") == "Indian Kanoon" and str(item.get("id", "")).startswith("live_")
                source_tag = "Judicial precedent" if is_live else str(item.get("source", "Legal source"))
                title = str(item.get("title", "Untitled authority"))
                with st.expander(f"[S{index}] {title}"):
                    st.markdown(f"**Source institution:** {source_tag}")
                    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
                    optional_fields = [
                        ("Court", item.get("court") or metadata.get("court")),
                        ("Date", item.get("judgment_date") or item.get("date") or metadata.get("judgment_date") or metadata.get("date")),
                        ("Authority type", item.get("authority_type") or item.get("kind") or metadata.get("authority_type") or metadata.get("kind")),
                    ]
                    available_metadata = [f"**{label}:** {value}" for label, value in optional_fields if value]
                    if available_metadata:
                        st.markdown(" · ".join(available_metadata))
                    st.caption(
                        f"Retrieval score: {item.get('score', 0):.2f} · "
                        f"Cross-encoder normalized score: {ce_prob:.1%} · "
                        f"Raw reranker score: {ce_score:.2f} · "
                        f"Query-term overlap: {anchor_match:.0%}"
                    )
                    st.caption("Relevance signals describe ranking and query overlap; they do not independently authenticate a source.")
                    if is_live:
                        st.markdown(f"**Live Judicial Precedent:** [{item['url']}]({item['url']}) · [View Full Judgment ↗]({item['url']})")
                    else:
                        st.markdown(f"**Open original source:** [{item['url']}]({item['url']})")
                    st.markdown("**Relevant excerpt**")
                    st.write(item["text"])

            # Comprehensive 7-Stage Pipeline Audit Trail
            with st.expander("Detailed seven-stage pipeline diagnostics", expanded=False):
                st.markdown("##### Stage-by-Stage Verification Breakdown")
                
                s_cols1 = st.columns(3)
                with s_cols1[0]:
                    st.markdown("""<div style="border:1px solid #c5c6ce; border-radius:4px; padding:0.85rem; background:#fff; height:100%;">
<small style="color:#735a3a; font-weight:700; text-transform:uppercase; font-size:0.65rem;">Stage 1: Query Analysis</small><br>
<strong style="color:#1a1c1b; font-size:0.88rem;">Terms, Entities &amp; Citations</strong><br>
</div>""", unsafe_allow_html=True)
                    st.caption(f"**Legal terms:** {', '.join(result.get('legal_terms', [])) or 'None'}")
                    st.caption(f"**Citations:** {', '.join(result.get('citations', [])) or 'None'}")
                    st.caption(f"**Entities:** {', '.join(result.get('entities', [])) or 'None'}")
                    st.caption(f"**Ambiguity:** {'Flagged' if result.get('analysis', {}).get('is_ambiguous') else 'Clean/Specific'}")

                with s_cols1[1]:
                    st.markdown("""<div style="border:1px solid #c5c6ce; border-radius:4px; padding:0.85rem; background:#fff; height:100%;">
<small style="color:#735a3a; font-weight:700; text-transform:uppercase; font-size:0.65rem;">Stage 2: Reformulation</small><br>
<strong style="color:#1a1c1b; font-size:0.88rem;">Weak/Ambiguous Resolution</strong><br>
</div>""", unsafe_allow_html=True)
                    st.caption(f"**Was Reformulated:** {'Yes' if result.get('was_reformulated') else 'No'}")
                    st.caption(f"**Effective Query:** {result.get('query_used', '')}")
                    st.caption(f"**Rationale:** {result.get('reformulate_reason', 'Query already specific.')}")

                with s_cols1[2]:
                    st.markdown("""<div style="border:1px solid #c5c6ce; border-radius:4px; padding:0.85rem; background:#fff; height:100%;">
<small style="color:#735a3a; font-weight:700; text-transform:uppercase; font-size:0.65rem;">Stage 3: Retrieval Route</small><br>
<strong style="color:#1a1c1b; font-size:0.88rem;">BM25 / Dense / Hybrid</strong><br>
</div>""", unsafe_allow_html=True)
                    st.caption(f"**Selected Route:** {result.get('route', '')}")
                    st.caption(f"**Rationale:** {result.get('route_reason', '')}")
                    st.caption(f"**Candidates Retrieved:** {result.get('candidate_count', len(result.get('evidence', [])))}")

                top_match_display = f"{result['evidence'][0].get('cross_encoder_prob', 0):.1%}" if result.get("evidence") else "N/A"
                s_cols2 = st.columns(4)
                with s_cols2[0]:
                    st.markdown(f"""<div style="border:1px solid #c5c6ce; border-radius:4px; padding:0.85rem; background:#fff; margin-top:0.75rem;">
<small style="color:#735a3a; font-weight:700; text-transform:uppercase; font-size:0.65rem;">Stage 4: Re-Ranking</small><br>
<strong style="color:#1a1c1b; font-size:0.85rem;">Cross-Encoder</strong><br>
<small style="color:#44474d; font-size:0.72rem;">Model: ms-marco-MiniLM-L-6-v2<br>Top Neural Match: {top_match_display}</small>
</div>""", unsafe_allow_html=True)

                with s_cols2[1]:
                    st.markdown(f"""<div style="border:1px solid #c5c6ce; border-radius:4px; padding:0.85rem; background:#fff; margin-top:0.75rem;">
<small style="color:#735a3a; font-weight:700; text-transform:uppercase; font-size:0.65rem;">Stage 5: Sufficiency Gate</small><br>
<strong style="color:{'#047857' if result['enough'] else '#b45309'}; font-size:0.85rem;">{'Passed' if result['enough'] else 'Insufficient'}</strong><br>
<small style="color:#44474d; font-size:0.72rem;">{result.get('evidence_reason', '')[:65]}...</small>
</div>""", unsafe_allow_html=True)

                with s_cols2[2]:
                    st.markdown(f"""<div style="border:1px solid #c5c6ce; border-radius:4px; padding:0.85rem; background:#fff; margin-top:0.75rem;">
<small style="color:#735a3a; font-weight:700; text-transform:uppercase; font-size:0.65rem;">Stage 6: Retry</small><br>
<strong style="color:#1a1c1b; font-size:0.85rem;">{'Invoked' if result['retried'] else 'Bypassed'}</strong><br>
<small style="color:#44474d; font-size:0.72rem;">{'Escalated to Hybrid search' if result['retried'] else 'First-pass passed'}</small>
</div>""", unsafe_allow_html=True)

                with s_cols2[3]:
                    st.markdown(f"""<div style="border:1px solid #c5c6ce; border-radius:4px; padding:0.85rem; background:#fff; margin-top:0.75rem;">
<small style="color:#735a3a; font-weight:700; text-transform:uppercase; font-size:0.65rem;">Stage 7: Output Guardrail</small><br>
<strong style="color:#1a1c1b; font-size:0.85rem;">{'Synthesis' if not result.get('is_insufficient_evidence_response') else 'Refusal'}</strong><br>
<small style="color:#44474d; font-size:0.72rem;">{'Grounded with citations' if not result.get('is_insufficient_evidence_response') else 'Hallucination prevented'}</small>
</div>""", unsafe_allow_html=True)

        with right_col:
            summary_items = [
                ("Query analysis", "Clear" if result.get("good_query") else "Needs refinement", "success" if result.get("good_query") else "review"),
                ("Effective query", result.get("query_used", research_question), "neutral"),
                ("Retrieval route", result.get("route", "Not reported"), "neutral"),
                ("Neural reranker", "Cross-encoder", "neutral"),
                ("Live precedents", "Enabled" if result.get("use_live_web") else "Disabled", "success" if result.get("use_live_web") else "neutral"),
                ("Retry assessment", f"Executed · {result.get('retry_route') or 'Hybrid retrieval'}" if result.get("retried") else "Not required", "review" if result.get("retried") else "neutral"),
                ("Evidence gate", "Passed" if result.get("enough") else "Insufficient", "success" if result.get("enough") else "review"),
            ]
            summary_rows = "".join(
                f'<div class="intelligence-row"><span class="intelligence-label">{html_escape(label)}</span><span class="intelligence-value {state}">{html_escape(str(value))}</span></div>'
                for label, value, state in summary_items
            )
            st.markdown(
                f'<div class="intelligence-panel"><h3>Intelligence Summary</h3>{summary_rows}</div>',
                unsafe_allow_html=True,
            )

            st.download_button(
                "Export Brief (.md)",
                memo(research_question, result),
                file_name=f"nyayaroute-{datetime.now():%Y%m%d-%H%M}.md",
                mime="text/markdown",
                use_container_width=True,
            )

            # Precedent Heatmap dynamically rendered
            evidence_scores = [ev.get("relevance", 0.0) for ev in result.get("evidence", [])]
            score_bars = ""
            for score in evidence_scores[:5]:
                score_bars += f'<div style="flex:{score+0.2:.2f}; height:35px; background:rgba(115,90,58,{max(0.25, score):.2f}); border-radius:2px;" title="{score:.0%} relevance"></div>'
            if not score_bars:
                score_bars = '<div style="flex:1; height:35px; background:rgba(115,90,58,0.2); border-radius:2px;"></div>'

            st.markdown(f"""<div style="background:#ffffff; border:1px solid #c5c6ce; border-radius:6px; overflow:hidden; margin-top:1.25rem;">
<div style="background:#f4f3f1; border-bottom:1px solid #c5c6ce; padding:0.75rem 1rem;">
<span style="font-family:'Inter',sans-serif; font-size:0.68rem; font-weight:700; text-transform:uppercase; letter-spacing:0.1em; color:#000000;">Precedent Heatmap</span>
</div>
<div style="padding:1rem; display:flex; flex-direction:column; gap:0.5rem;">
<div style="display:flex; gap:0.35rem; align-items:center;">
{score_bars}
</div>
<p style="font-family:'Inter',sans-serif; font-size:0.65rem; color:#44474d; margin:0;">
Relevance density across top {len(evidence_scores)} retrieved legal authorities.
</p>
</div>
</div>""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
elif selected_page == "Source Library":
    st.markdown(
        """
        <style>
          [data-testid="stSidebar"] { display: none !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    render_page_hero(
        eyebrow="LEGAL KNOWLEDGE REPOSITORY",
        title="Source Library",
        subtitle="A curated collection of trusted primary legal authorities.",
        col_ratio=[1.55, 1.45],
    )

    current_sources = source_summary(load_corpus())

    # Two-column layout: Left (Cards ~69%) and Right (Sidebar ~31%)
    content_col, sidebar_col = st.columns([2.15, 0.95], gap="large")

    with content_col:
        # Search and Filter Toolbar
        st.markdown('<div class="source-filter-toolbar">', unsafe_allow_html=True)
        t_col1, t_col2, t_col3, t_col4 = st.columns([5.0, 2.3, 2.3, 1.2], gap="small", vertical_alignment="center")
        with t_col1:
            search_filter = st.text_input(
                "Search sources",
                placeholder="Search statutes, constitutional articles, judgments...",
                label_visibility="collapsed",
                key="source_search_input",
            )
        with t_col2:
            cat_filter = st.selectbox(
                "Category",
                ["All Categories", "Constitutional Law", "Judgments", "Statute"],
                label_visibility="collapsed",
                key="source_category_select",
            )
        with t_col3:
            all_auths = ["All Authorities"] + sorted(list({s.get("source", "Legal Source") for s in current_sources}))
            auth_filter = st.selectbox(
                "Authority",
                all_auths,
                label_visibility="collapsed",
                key="source_authority_select",
            )
        with t_col4:
            if st.button("↺ Clear", key="clear_source_filters", use_container_width=True):
                st.session_state.source_search_input = ""
                st.session_state.source_category_select = "All Categories"
                st.session_state.source_authority_select = "All Authorities"
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        # Filtered list of sources
        filtered_items = []
        for item in sorted(current_sources, key=source_sort_key):
            category, doc_type, icon, open_url = get_source_metadata(item)
            if search_filter.strip():
                q = search_filter.strip().lower()
                if (
                    q not in item.get("title", "").lower()
                    and q not in item.get("source", "").lower()
                    and q not in category.lower()
                    and q not in doc_type.lower()
                ):
                    continue
            if cat_filter != "All Categories":
                if category.lower() != cat_filter.lower():
                    continue
            if auth_filter != "All Authorities":
                if item.get("source", "").lower() != auth_filter.lower():
                    continue
            filtered_items.append((item, category, doc_type, icon, open_url))

        if filtered_items:
            for idx, (item, category, doc_type, icon, open_url) in enumerate(filtered_items):
                is_active = (idx == 0)
                chunk_text = "1 indexed chunk" if item.get("chunks") == 1 else f"{item.get('chunks', 0)} indexed chunks"
                card_html = f"""
                <div class="legal-source-card {'active-card' if is_active else ''}">
                    <div class="source-card-left">
                        <div class="source-card-icon-box">
                            <span class="material-symbols-outlined">{icon}</span>
                        </div>
                        <div class="source-card-details">
                            <span class="source-badge">{category}</span>
                            <div class="source-card-title">{html_escape(item['title'])}</div>
                            <div class="source-card-meta">
                                <span>{html_escape(item['source'])}</span>
                                <span class="meta-dot">·</span>
                                <span>{doc_type}</span>
                                <span class="meta-dot">·</span>
                                <span>{chunk_text}</span>
                            </div>
                        </div>
                    </div>
                    <a href="{open_url}" target="_blank" rel="noopener noreferrer" class="source-card-btn" title="Open source documentation">
                        Open source ↗
                    </a>
                </div>
                """
                st.markdown(card_html, unsafe_allow_html=True)
        else:
            st.markdown(
                """
                <div style="background:#FFFDFA; border:1px dashed #D8C39E; border-radius:8px; padding:2.5rem 1.5rem; text-align:center; color:#68778A; margin-top:0.5rem;">
                    <span class="material-symbols-outlined" style="font-size:2.4rem; color:#A47B3C; display:block; margin-bottom:0.5rem;">search_off</span>
                    <div style="font-family:'Playfair Display', Georgia, serif; font-size:1.2rem; font-weight:600; color:#102744; margin-bottom:0.35rem;">No legal authorities match your query</div>
                    <p style="font-family:'Inter', sans-serif; font-size:0.82rem; margin:0;">Try adjusting your search terms or clearing the category and authority filters.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with sidebar_col:
        st.markdown('<div class="trusted-source-container">', unsafe_allow_html=True)
        with st.form("ingest_source", clear_on_submit=True):
            st.markdown(
                """
                <span class="material-symbols-outlined" style="color:#A47B3C; font-size:1.85rem; line-height:1; display:block; margin-bottom:0.35rem;">post_add</span>
                <h3 style="font-family:'Playfair Display', Georgia, serif; font-size:1.35rem; font-weight:700; color:#102744; margin:0 0 0.35rem; line-height:1.25;">Add a Trusted Source</h3>
                <p style="font-family:'Inter', sans-serif; font-size:0.78rem; line-height:1.45; color:#68778A; margin:0 0 1rem;">
                    Extend the collection with specific documents from approved legal authorities.
                </p>
                """,
                unsafe_allow_html=True,
            )
            source_url = st.text_input(
                "Specific document URL *",
                placeholder="https://www.example.gov.in/…",
                key="ingest_url_input",
            )
            display_title = st.text_input(
                "Optional label / Source title",
                placeholder="e.g., Supreme Court Judgments, Article 370...",
                key="ingest_title_input",
            )
            submit_source = st.form_submit_button("Add to Source Library →", use_container_width=True)

            st.markdown(
                """
                <div style="font-family:'Inter', sans-serif; font-size:0.68rem; line-height:1.45; color:#68778A; margin-top:0.75rem; padding-top:0.75rem; border-top:1px solid #E7DED0;">
                    <strong style="color:#102744;">Allowed:</strong> India Code, Supreme Court of India, and official High Court portals. The tool is deliberately page-by-page and does not bulk crawl.
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

        if submit_source:
            if not source_url.strip():
                st.warning("Enter a specific document URL.")
            else:
                try:
                    with st.spinner("Fetching and indexing the trusted source page…"):
                        count, title = ingest_trusted_url(
                            source_url.strip(), display_title.strip() or None
                        )
                    reset_retriever()
                    reset_vector_store_cache()
                    if count:
                        st.success(
                            f"Added {count} chunks from: {title}. "
                            "The source is now indexed in the library."
                        )
                        st.rerun()
                    else:
                        st.info("That page is already indexed in the local library; no new chunks were added.")
                except Exception as exc:
                    st.error(f"Could not add this source: {exc}")

# ──────────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────
# KNOWLEDGE BASE — PDF INGESTION
elif selected_page == "Knowledge Base":
    st.markdown(
        """
        <style>
          [data-testid="stSidebar"] { display: none !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    render_page_hero(
        eyebrow="LEGAL KNOWLEDGE REPOSITORY",
        title="Knowledge Base",
        subtitle="Upload official legal PDFs to expand NyayaRoute's searchable legal knowledge base.",
        col_ratio=[1.55, 1.45],
    )

    with st.container(key="kb_upload_card"):
        st.markdown(
            """
            <div class="kb-header">
                <div class="kb-icon-title-wrap">
                    <div class="kb-icon-box">
                        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#A8752D" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
                            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                            <polyline points="14 2 14 8 20 8"/>
                            <line x1="16" y1="13" x2="8" y2="13"/>
                            <line x1="16" y1="17" x2="8" y2="17"/>
                            <line x1="10" y1="9" x2="8" y2="9"/>
                        </svg>
                    </div>
                    <div>
                        <div class="kb-title">Add Legal Documents</div>
                        <div class="kb-subtitle">Upload official legal PDFs to expand your searchable knowledge base.</div>
                    </div>
                </div>
                <span class="kb-help-btn" title="Acts, Constitution, judgments, Gazette notifications, Law Commission reports, etc.">?</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        uploaded_files = st.file_uploader(
            "Upload legal PDF(s)",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            help="Acts, Constitution, judgments, Gazette notifications, Law Commission reports, etc.",
        )

    if uploaded_files:
        st.markdown('<div class="kb-outer-container" style="margin-top: 1.5rem;">', unsafe_allow_html=True)

        for uploaded_file in uploaded_files:
            file_bytes = uploaded_file.getvalue()
            import hashlib
            file_hash = hashlib.sha256(file_bytes).hexdigest()[:16]

            # ── Processing Loader (No buttons shown during or before ingestion) ──
            already_indexed = False

            if f"indexed_result_{file_hash}" not in st.session_state:
                with st.status(f"⚡ Integrating '{uploaded_file.name}' into RAG Knowledge Base…", expanded=True) as status_box:
                    st.write("📄 **1. Extracting Document Pages:** Reading and cleaning raw text via PyPDF…")
                    try:
                        st.write("⚖️ **2. Structural Legal Chunking:** Detecting Parts, Chapters, Sections, and Articles…")
                        ingest_res = ingest_pdf(file_bytes, uploaded_file.name)

                        st.write("🧠 **3. Neural Embedding & FAISS:** Indexing 384-dimensional dense vectors…")
                        reset_retriever()
                        reset_vector_store_cache()

                        st.write("🔄 **4. Retriever Synchronization:** Refreshing BM25 lexical index and RAG cache…")
                        if "selected_sources" in st.session_state:
                            st.session_state["selected_sources"] = sorted(
                                list(set(st.session_state["selected_sources"]) | {ingest_res.doc_profile.source})
                            )

                        st.session_state[f"indexed_result_{file_hash}"] = {
                            "filename": ingest_res.filename,
                            "chunks_created": ingest_res.chunks_created,
                            "pages_used": ingest_res.pages_used,
                            "pages_total": ingest_res.pages_total,
                            "is_duplicate": False,
                        }
                        status_box.update(label=f"✓ '{uploaded_file.name}' Successfully Integrated into Knowledge Base!", state="complete", expanded=False)
                    except DuplicatePDFError:
                        already_indexed = True
                        st.session_state[f"indexed_result_{file_hash}"] = {
                            "filename": uploaded_file.name,
                            "is_duplicate": True,
                        }
                        status_box.update(label=f"ℹ️ '{uploaded_file.name}' is already active in Knowledge Base", state="complete", expanded=False)
                    except Exception as exc:
                        status_box.update(label=f"❌ Error indexing '{uploaded_file.name}'", state="error", expanded=True)
                        st.error(f"Could not process PDF: {exc}")
                        continue
            else:
                cached_info = st.session_state[f"indexed_result_{file_hash}"]
                already_indexed = cached_info.get("is_duplicate", False)

            # ── Once Done: Display Contribution Banner ──────────────────────────
            if not already_indexed:
                st.markdown(
                    f"""
                    <div style="background: linear-gradient(135deg, #FFFDF9 0%, #FBF8F1 100%); border: 1.5px solid #D8BD91; border-left: 4.5px solid #A8752D; border-radius: 8px; padding: 1.3rem 1.6rem; margin: 1.2rem 0; box-shadow: 0 3px 12px rgba(168, 117, 45, 0.09);">
                        <div style="display:flex; align-items:flex-start; gap: 1rem;">
                            <span style="font-size: 2rem; line-height: 1;">🎉</span>
                            <div>
                                <div style="font-family:'Playfair Display', Georgia, serif; font-size:1.25rem; font-weight:700; color:#102442; margin-bottom:0.35rem;">
                                    Thank You for Contributing to the NyayaRoute Knowledge Base!
                                </div>
                                <div style="font-family:'Inter', sans-serif; font-size:0.9rem; color:#4A5568; line-height:1.5;">
                                    <strong>{uploaded_file.name}</strong> has been successfully parsed, structurally chunked, and embedded into the permanent RAG vector store. The AI research agent will now actively cite and ground legal reasoning in this knowledge source.
                                </div>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div style="background: #FBF8F1; border: 1.5px solid #D8BD91; border-left: 4.5px solid #102442; border-radius: 8px; padding: 1.2rem 1.5rem; margin: 1.2rem 0;">
                        <div style="display:flex; align-items:flex-start; gap: 0.9rem;">
                            <span style="font-size: 1.8rem; line-height: 1;">📚</span>
                            <div>
                                <div style="font-family:'Playfair Display', Georgia, serif; font-size:1.2rem; font-weight:700; color:#102442; margin-bottom:0.3rem;">
                                    Knowledge Source Active & Grounded
                                </div>
                                <div style="font-family:'Inter', sans-serif; font-size:0.88rem; color:#4A5568; line-height:1.5;">
                                    <strong>{uploaded_file.name}</strong> is already permanently preserved in the RAG vector corpus. All of its provisions are available for immediate research inquiries.
                                </div>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # ── Extracted Document Information & Structural Chunks ───────────────
            try:
                parsed, profile, structure_kind, chunks = preview_pdf(
                    file_bytes,
                    uploaded_file.name,
                )

                st.markdown(f"#### 📋 Extracted Document Information — `{uploaded_file.name}`")

                metadata_col1, metadata_col2 = st.columns(2)
                with metadata_col1:
                    st.write(f"**Document Kind:** {profile.kind.title()}")
                    st.write(f"**Identified Act / Title:** {profile.act or uploaded_file.name}")
                    st.write(f"**Year:** {profile.year or 'Not specified'}")

                with metadata_col2:
                    st.write(f"**Source Category:** {profile.source}")
                    st.write(f"**Jurisdiction / Court:** {profile.court or 'National / Central'}")
                    st.write(f"**Structure Hierarchy:** {structure_kind.title()}")

                stat1, stat2, stat3 = st.columns(3)
                with stat1:
                    st.metric("Total Pages", parsed.num_pages_total)
                with stat2:
                    st.metric("Pages Utilized", parsed.num_pages_used)
                with stat3:
                    st.metric("Chunks Generated", len(chunks))

                with st.expander("🔍 Preview Extracted Structural Chunks"):
                    for i, chunk in enumerate(chunks[:8], start=1):
                        st.markdown(f"**Chunk {i}** — *{chunk.structure_kind.title()}*")
                        if chunk.part:
                            st.caption(f"Part: {chunk.part}")
                        if chunk.chapter:
                            st.caption(f"Chapter: {chunk.chapter}")
                        if chunk.article:
                            st.caption(f"Article: {chunk.article}")
                        if chunk.section:
                            st.caption(f"Section: {chunk.section}")
                        if chunk.heading:
                            st.caption(f"Heading: {chunk.heading}")
                        st.text_area(
                            "Content",
                            chunk.text,
                            height=120,
                            key=f"chunk_preview_{uploaded_file.name}_{i}",
                            disabled=True,
                        )
                        st.markdown("---")

                # ── Action Navigation: Direct Jump to Research Workspace ────────
                col_btn1, col_btn2 = st.columns([2, 1], gap="medium")
                with col_btn1:
                    if st.button("⚖️ Query this Document in Research Workspace", type="primary", key=f"nav_to_research_{uploaded_file.name}_{file_hash}"):
                        st.session_state["main_navigation"] = "Research workspace"
                        st.rerun()
                with col_btn2:
                    if st.button("🔄 Refresh Local Corpus View", key=f"refresh_view_{uploaded_file.name}_{file_hash}"):
                        st.rerun()

            except Exception as exc:
                st.error(f"Could not load details for {uploaded_file.name}: {exc}")

        st.markdown('</div>', unsafe_allow_html=True)

    # ── Display Persisted Knowledge Base Documents ────────────────────────────
    st.markdown('<div class="kb-outer-container" style="margin-top: 2rem;">', unsafe_allow_html=True)
    st.markdown("### 📚 Indexed Documents in Knowledge Base")
    st.caption("All documents below are persistently saved in the local JSONL & FAISS vector corpus and actively queried by the legal RAG agent.")

    from pathlib import Path
    import json
    pdf_corpus_file = Path(__file__).resolve().parent / "data" / "pdf_legal_corpus.jsonl"
    if pdf_corpus_file.exists():
        pdf_docs = {}
        for line in pdf_corpus_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    r = json.loads(line)
                    fn = r.get("filename") or "Unknown Document"
                    if fn not in pdf_docs:
                        pdf_docs[fn] = {
                            "filename": fn,
                            "title": r.get("act") or r.get("title") or fn,
                            "source": r.get("source") or "Uploaded PDF",
                            "kind": r.get("structure_kind") or r.get("kind") or "Document",
                            "chunks": 0,
                            "ingested_at": r.get("ingested_at", "")[:10] if r.get("ingested_at") else "N/A",
                        }
                    pdf_docs[fn]["chunks"] += 1
                except Exception:
                    continue
        if pdf_docs:
            doc_rows = []
            for doc in pdf_docs.values():
                doc_rows.append({
                    "Document / File": doc["filename"],
                    "Detected Title / Act": doc["title"],
                    "Source Category": doc["source"],
                    "Structure": doc["kind"].title(),
                    "Indexed Chunks": doc["chunks"],
                    "Date Ingested": doc["ingested_at"],
                })
            try:
                st.dataframe(doc_rows, width="stretch", hide_index=True)
            except Exception:
                st.dataframe(doc_rows, use_container_width=True, hide_index=True)
        else:
            st.info("No PDF documents have been indexed yet. Upload a PDF above to expand the knowledge base.")
    else:
        st.info("No PDF documents have been indexed yet. Upload a PDF above to expand the knowledge base.")
    st.markdown('</div>', unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
else:
    st.markdown(
        """
        <style>
          [data-testid="stSidebar"] { display: none !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    render_page_hero(
        eyebrow="GUIDE TO RESPONSIBLE LEGAL RESEARCH",
        title="How to Use NyayaRoute",
        subtitle="Research with trusted legal authorities, transparent evidence, and responsible AI-assisted retrieval.",
        col_ratio=[1.55, 1.45],
    )

    st.markdown(
        """<div class="guide-container">
<div class="guide-grid">
<div class="guide-card">
<div class="guide-icon-box">
<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#A8752C" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h6"/>
<path d="M14 2v4a2 2 0 0 0 2 2h4"/>
<circle cx="16.5" cy="16.5" r="3.5"/>
<path d="m19 19 3 3"/>
<line x1="8" y1="11" x2="12" y2="11"/>
<line x1="8" y1="15" x2="11" y2="15"/>
</svg>
</div>
<div class="guide-v-divider"></div>
<div>
<div class="guide-step-num">01</div>
<div class="guide-step-title">Ask a Focused Legal Question</div>
<p class="guide-step-desc">Begin with a clear legal research question that identifies the issue, relevant facts, and applicable jurisdiction.</p>
</div>
</div>
<div class="guide-card">
<div class="guide-icon-box">
<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#A8752C" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
<polygon points="12 2 2 7 12 12 22 7 12 2"/>
<polyline points="2 12 12 17 22 12"/>
<polyline points="2 17 12 22 22 17"/>
</svg>
</div>
<div class="guide-v-divider"></div>
<div>
<div class="guide-step-num">02</div>
<div class="guide-step-title">Retrieve Relevant Authorities</div>
<p class="guide-step-desc">NyayaRoute checks whether the query needs clarification and selects BM25, vector, or hybrid retrieval.</p>
</div>
</div>
<div class="guide-card">
<div class="guide-icon-box">
<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#A8752C" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
<path d="m16 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z"/>
<path d="m2 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z"/>
<path d="M7 21h10"/>
<path d="M12 3v18"/>
<path d="M3 7h18"/>
</svg>
</div>
<div class="guide-v-divider"></div>
<div>
<div class="guide-step-num">03</div>
<div class="guide-step-title">Verify the Evidence</div>
<p class="guide-step-desc">It reranks candidate chunks and tests whether a sufficiently relevant authority was retrieved.</p>
</div>
</div>
<div class="guide-card">
<div class="guide-icon-box">
<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#A8752C" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
<path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/>
<path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/>
</svg>
</div>
<div class="guide-v-divider"></div>
<div>
<div class="guide-step-num">04</div>
<div class="guide-step-title">Review the Cited Answer</div>
<p class="guide-step-desc">It either produces a cited, evidence-constrained research answer — or says what is missing and directs you to trusted sources.</p>
</div>
</div>
</div>
<div class="safeguard-card">
<div class="guide-icon-box">
<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#A8752C" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
<polyline points="14 2 14 8 20 8"/>
<line x1="12" y1="12" x2="12" y2="15"/>
<line x1="12" y1="18" x2="12.01" y2="18"/>
</svg>
</div>
<div class="safeguard-v-divider"></div>
<div style="flex:1;">
<div style="font-family:'Inter',sans-serif; font-size:0.68rem; font-weight:700; letter-spacing:0.16em; color:#A8752C; text-transform:uppercase; margin-bottom:0.25rem;">IMPORTANT LEGAL SAFEGUARD</div>
<p style="font-family:'Inter',sans-serif; font-size:0.86rem; color:#50627C; margin:0 0 0.45rem; line-height:1.45;">For current amendments, always add the current statutory text or notification first.</p>
<div style="height:1px; background:#E7DED0; width:100%; margin:0.45rem 0 0.55rem;"></div>
<p style="font-family:'Playfair Display','Baskerville',Georgia,serif; font-size:1.05rem; font-weight:700; color:#102442; margin:0; line-height:1.35;">A language model being available does not make it a source of law.</p>
</div>
</div>
</div>""",
        unsafe_allow_html=True,
    )

# ── Institutional Footer ──────────────────────────────────────────────────────
st.markdown("""
<div style="border-top:1px solid #D8C39E; margin-top:3.5rem; padding:1.4rem 0; display:flex; flex-wrap:wrap; justify-content:space-between; align-items:center; gap:1rem;">
  <span style="font-family:'Inter',sans-serif; font-size:0.68rem; text-transform:uppercase; letter-spacing:0.12em; color:#68778A;">
    © 2026 NYAYAROUTE INTERNATIONAL. ALL RIGHTS RESERVED.
  </span>
  <div style="display:flex; gap:1.2rem; align-items:center; flex-wrap:wrap;">
    <a href="#" style="font-family:'Inter',sans-serif; font-size:0.68rem; text-transform:uppercase; letter-spacing:0.08em; color:#68778A; text-decoration:underline; text-underline-offset:4px;">
      Terms of Service
    </a>
    <span style="color:#D8C39E;">|</span>
    <a href="#" style="font-family:'Inter',sans-serif; font-size:0.68rem; text-transform:uppercase; letter-spacing:0.08em; color:#68778A; text-decoration:underline; text-underline-offset:4px;">
      Privacy Protocol
    </a>
    <span style="color:#D8C39E;">|</span>
    <a href="#" style="font-family:'Inter',sans-serif; font-size:0.68rem; text-transform:uppercase; letter-spacing:0.08em; color:#68778A; text-decoration:underline; text-underline-offset:4px;">
      Compliance
    </a>
    <span style="color:#D8C39E;">|</span>
    <a href="#" style="font-family:'Inter',sans-serif; font-size:0.68rem; text-transform:uppercase; letter-spacing:0.08em; color:#68778A; text-decoration:underline; text-underline-offset:4px;">
      Institutional Contact
    </a>
  </div>
</div>
""", unsafe_allow_html=True)
