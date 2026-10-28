from pathlib import Path

p = Path(r"C:\Users\gnana\OneDrive\Desktop\5thyear\MINI PROJECT\Project\final\Agentic-RAG\app.py")
text = p.read_text(encoding="utf-8")

# --- Step 1: Simplify Source Library (remove sidebar_col, make sources full width) ---
old_source_lib_block = """    # Two-column layout: Left (Cards ~69%) and Right (Sidebar ~31%)
    content_col, sidebar_col = st.columns([2.15, 0.95], gap="large")

    with content_col:
        # Search and Filter Toolbar"""

new_source_lib_block = """    # Clean, simplified full-width display of all available sources
    with st.container():
        # Search and Filter Toolbar"""

text = text.replace(old_source_lib_block, new_source_lib_block)

# Remove the sidebar_col block from Source Library
old_sidebar_col_block = """    with sidebar_col:
        st.markdown('<div class="trusted-source-container">', unsafe_allow_html=True)
        with st.form("ingest_source", clear_on_submit=True):
            st.markdown(
                \"\"\"
                <span class=\"material-symbols-outlined\" style=\"color:#A47B3C; font-size:1.85rem; line-height:1; display:block; margin-bottom:0.35rem;\">post_add</span>
                <h3 style=\"font-family:'Playfair Display', Georgia, serif; font-size:1.35rem; font-weight:700; color:#102744; margin:0 0 0.35rem; line-height:1.25;\">Add a Trusted Source</h3>
                <p style=\"font-family:'Inter', sans-serif; font-size:0.78rem; line-height:1.45; color:#68778A; margin:0 0 1rem;\">
                    Extend the collection with specific documents from approved legal authorities.
                </p>
                \"\"\",
                unsafe_allow_html=True,
            )
            source_url = st.text_input(
                \"Specific document URL *\",
                placeholder=\"https://www.example.gov.in/…\",
                key=\"ingest_url_input\",
            )
            display_title = st.text_input(
                \"Optional label / Source title\",
                placeholder=\"e.g., Supreme Court Judgments, Article 370...\",
                key=\"ingest_title_input\",
            )
            submit_source = st.form_submit_button(\"Add to Source Library →\", width=\"stretch\")

            st.markdown(
                \"\"\"
                <div style=\"font-family:'Inter', sans-serif; font-size:0.68rem; line-height:1.45; color:#68778A; margin-top:0.75rem; padding-top:0.75rem; border-top:1px solid #E7DED0;\">
                    <strong style=\"color:#102744;\">Allowed:</strong> India Code, Supreme Court of India, and official High Court portals. The tool is deliberately page-by-page and does not bulk crawl.
                </div>
                \"\"\",
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

        if submit_source:
            if not source_url.strip():
                st.warning(\"Enter a specific document URL.\")
            else:
                try:
                    with st.spinner(\"Fetching and indexing the trusted source page…\"):
                        count, title = ingest_trusted_url(
                            source_url.strip(), display_title.strip() or None
                        )
                    reset_retriever()
                    reset_vector_store_cache()
                    if count:
                        st.success(
                            f\"Added {count} chunks from: {title}. \"
                            \"The source is now indexed in the library.\"
                        )
                        st.rerun()
                    else:
                        st.info(\"That page is already indexed in the local library; no new chunks were added.\")
                except Exception as exc:
                    st.error(f\"Could not add this source: {exc}\")"""

text = text.replace(old_sidebar_col_block, "")

# --- Step 2: Knowledge Base Tabs (PDF upload + Trusted URL Ingestion) ---
old_kb_start = """    with st.container(key="kb_upload_card"):
        st.markdown(
            \"\"\"
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
            \"\"\",
            unsafe_allow_html=True,
        )

        uploaded_files = st.file_uploader(
            "Upload legal PDF(s)",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            help="Acts, Constitution, judgments, Gazette notifications, Law Commission reports, etc.",
        )"""

new_kb_start = """    kb_tab1, kb_tab2 = st.tabs(["📁 Add Legal PDF Document", "🌐 Ingest Trusted Web Source (URL)"])

    with kb_tab1:
        with st.container(key="kb_upload_card"):
            st.markdown(
                \"\"\"
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
                \"\"\",
                unsafe_allow_html=True,
            )

            uploaded_files = st.file_uploader(
                "Upload legal PDF(s)",
                type=["pdf"],
                accept_multiple_files=True,
                label_visibility="collapsed",
                help="Acts, Constitution, judgments, Gazette notifications, Law Commission reports, etc.",
            )

    with kb_tab2:
        st.markdown('<div class="nr-card" style="margin-top:1rem; border-top: 3px solid #A8752D;">', unsafe_allow_html=True)
        with st.form("ingest_kb_url_form", clear_on_submit=True):
            st.markdown(
                \"\"\"
                <h3 style="font-family:'Playfair Display', Georgia, serif; font-size:1.25rem; font-weight:700; color:#102442; margin:0 0 0.35rem;">
                    Ingest Official Legal Website
                </h3>
                <p style="font-family:'Inter', sans-serif; font-size:0.84rem; color:#4A5568; margin:0 0 1rem;">
                    Fetch, chunk, and embed authoritative statutes or reported judicial rulings directly into the RAG vector store.
                </p>
                \"\"\",
                unsafe_allow_html=True,
            )
            kb_source_url = st.text_input(
                "Official Document URL *",
                placeholder="https://www.indiacode.nic.in/... or https://indiankanoon.org/doc/...",
                key="kb_url_input",
            )
            kb_display_title = st.text_input(
                "Optional Title / Enactment Name",
                placeholder="e.g. Information Technology Act, 2000, Section 66A",
                key="kb_title_input",
            )
            kb_submit_url = st.form_submit_button("⚡ Ingest Web Authority →", width="stretch")

            st.markdown(
                \"\"\"
                <div style="font-size:0.75rem; color:#718096; line-height:1.45; margin-top:0.75rem; padding-top:0.75rem; border-top:1px solid #E2E8F0;">
                    <strong>Allowed Domains:</strong> India Code (<code>indiacode.nic.in</code>), Supreme Court of India (<code>sci.gov.in</code>), Indian Kanoon (<code>indiankanoon.org</code>), and official High Court portals (<code>*.hc.gov.in</code>).
                </div>
                \"\"\",
                unsafe_allow_html=True,
            )

        if kb_submit_url:
            if not kb_source_url.strip():
                st.warning("Please enter a valid document URL.")
            elif not is_trusted_legal_url(kb_source_url.strip()):
                st.error("Only verified government and court domains are permitted.")
            else:
                try:
                    with st.spinner("Fetching, parsing, and embedding web authority into knowledge base…"):
                        count, title = ingest_trusted_url(kb_source_url.strip(), kb_display_title.strip() or None)
                        reset_retriever()
                        reset_vector_store_cache()
                        if count:
                            st.success(f"✓ Successfully ingested {count} chunks for: **{title}**! The document is now active in the knowledge base.")
                            st.rerun()
                        else:
                            st.info("That document is already indexed in the local knowledge base.")
                except Exception as exc:
                    st.error(f"Could not ingest web authority: {exc}")
        st.markdown('</div>', unsafe_allow_html=True)"""

text = text.replace(old_kb_start, new_kb_start)

# Include legal_corpus.jsonl in the persisted documents table
old_catalog_read = """    from pathlib import Path
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
                    continue"""

new_catalog_read = """    from pathlib import Path
    import json
    data_dir = Path(__file__).resolve().parent / "data"
    pdf_docs = {}
    for corpus_filename in ("pdf_legal_corpus.jsonl", "legal_corpus.jsonl"):
        c_file = data_dir / corpus_filename
        if c_file.exists():
            for line in c_file.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    try:
                        r = json.loads(line)
                        fn = r.get("filename") or r.get("title") or "Document"
                        if fn not in pdf_docs:
                            pdf_docs[fn] = {
                                "filename": fn,
                                "title": r.get("act") or r.get("title") or fn,
                                "source": r.get("source") or ("Uploaded PDF" if "pdf" in corpus_filename else "Ingested Web URL"),
                                "kind": r.get("structure_kind") or r.get("kind") or "Document",
                                "chunks": 0,
                                "ingested_at": r.get("ingested_at", "")[:10] if r.get("ingested_at") else "N/A",
                            }
                        pdf_docs[fn]["chunks"] += 1
                    except Exception:
                        continue"""

text = text.replace(old_catalog_read, new_catalog_read)

p.write_text(text, encoding="utf-8")
print("Part 2: Full UI layout successfully updated!")
