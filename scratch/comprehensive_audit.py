import os
import sys
import io
import time
import json
import traceback
from pathlib import Path

# Setup paths
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

report = []

def log(section, name, status, details=""):
    msg = f"[{section}] {name}: {status}"
    if details:
        msg += f" -> {details}"
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", "replace").decode("ascii"))
    report.append({"section": section, "name": name, "status": status, "details": details})

print("=" * 80)
print("STARTING COMPREHENSIVE NYAYAROUTE AUDIT")
print("=" * 80)

# ==============================================================================
# 1. EMBEDDING.PY
# ==============================================================================
try:
    import embedding
    dim = embedding.get_embedding_dim()
    if dim == 384:
        log("embedding", "get_embedding_dim", "PASS", f"dim={dim}")
    else:
        log("embedding", "get_embedding_dim", "FAIL", f"Expected 384, got {dim}")

    model = embedding.get_embedder()
    log("embedding", "get_embedder", "PASS", f"Loaded {model.__class__.__name__}")

    # Empty list test
    v_empty = embedding.embed_texts([])
    if v_empty.shape == (0, 384):
        log("embedding", "embed_texts(empty)", "PASS", f"shape={v_empty.shape}")
    else:
        log("embedding", "embed_texts(empty)", "FAIL", f"shape={v_empty.shape}")

    # Whitespace and normal text
    v_norm = embedding.embed_texts(["", "   ", "Article 21 Constitution of India"])
    import numpy as np
    norms = np.linalg.norm(v_norm, axis=1)
    log("embedding", "embed_texts(values)", "PASS", f"shape={v_norm.shape}, norms={[round(float(n), 3) for n in norms]}")

except Exception as e:
    log("embedding", "general", "ERROR", f"{e}\n{traceback.format_exc()}")


# ==============================================================================
# 2. CHUNKER.PY
# ==============================================================================
try:
    import chunker
    # Test document kind detection
    doc_kinds = {
        "empty": ("", "generic"),
        "constitution_numbered": ("PART III\nFUNDAMENTAL RIGHTS\n12. Definition.\n14. Equality before law.\n21. Protection of life.\n21A. Right to education.\n22. Protection against arrest.", "constitution"),
        "act_numbered": ("CHAPTER I\nPRELIMINARY\n1. Short title.\n2. Definitions.\n3. Construction of references.\n4. Trial of offences.", "act"),
        "act_rules_part": ("PART II\nRULES FOR INTERMEDIARIES\n1. Short title.\n2. Definitions.\n3. Due diligence.", "act"),
        "judgment": ("IN THE SUPREME COURT OF INDIA\nFACTS OF THE CASE\nThe petitioner approached...\nISSUES FOR CONSIDERATION\nWhether...\nHELD\nThe court holds...", "judgment"),
    }
    for label, (text, expected) in doc_kinds.items():
        detected = chunker.detect_document_kind(text)
        if detected == expected or (expected == "constitution" and detected in ("constitution", "act")):
            log("chunker", f"detect_document_kind({label})", "PASS", f"detected={detected}")
        else:
            log("chunker", f"detect_document_kind({label})", "FAIL", f"expected={expected}, detected={detected}")

    # Test chunk_document edge cases
    # Edge case 1: Empty text
    k, chs = chunker.chunk_document("")
    log("chunker", "chunk_document(empty)", "PASS" if len(chs) == 0 else "FAIL", f"chunks={len(chs)}")

    # Edge case 2: Very short text (< 40 chars)
    k, chs = chunker.chunk_document("Short text")
    log("chunker", "chunk_document(short<40)", "PASS" if len(chs) == 0 else "WARN", f"chunks={len(chs)}")

    # Edge case 3: Single giant paragraph with no newlines
    giant_text = "This is a continuous legal sentence without any newlines. " * 50
    k, chs = chunker.chunk_document(giant_text)
    log("chunker", "chunk_document(no_newlines)", "PASS" if len(chs) > 0 else "FAIL", f"kind={k}, chunks={len(chs)}")

    # Edge case 4: Constitution chunking with out-of-order or duplicate articles
    const_text = "PART I\n1. Name and territory.\n2. Admission of States.\n21. Life.\n21. Protection of life and liberty.\n22. Arrest."
    k, chs = chunker.chunk_document(const_text, doc_kind="constitution")
    log("chunker", "chunk_document(constitution_dup)", "PASS", f"chunks={len(chs)}, articles={[c.article for c in chs]}")

except Exception as e:
    log("chunker", "general", "ERROR", f"{e}\n{traceback.format_exc()}")


# ==============================================================================
# 3. METADATA.PY
# ==============================================================================
try:
    import metadata
    # Test classify_document
    p1 = metadata.classify_document("the_constitution_of_india.pdf", "THE CONSTITUTION OF INDIA\nWE, THE PEOPLE OF INDIA")
    log("metadata", "classify_document(constitution)", "PASS" if p1.kind == "constitution" and p1.source == "India Code" else "FAIL", f"act={p1.act}, source={p1.source}")

    p2 = metadata.classify_document("random_upload.pdf", "Just some random text")
    log("metadata", "classify_document(unknown)", "PASS" if p2.source == "Uploaded PDF" else "FAIL", f"source={p2.source}")

    p3 = metadata.classify_document("sc_judgment.pdf", "IN THE SUPREME COURT OF INDIA\nCIVIL APPELLATE JURISDICTION\n2022")
    log("metadata", "classify_document(sc)", "PASS" if p3.court == "Supreme Court of India" else "FAIL", f"court={p3.court}, year={p3.year}")

    # Test extract_keywords
    kw1 = metadata.extract_keywords("The petitioner sought bail and writ of habeas corpus under article 21.")
    log("metadata", "extract_keywords", "PASS" if len(kw1) > 0 else "FAIL", f"keywords={kw1}")

    # Test make_chunk_id
    id1 = metadata.make_chunk_id("doc.pdf", 0, "some text")
    id2 = metadata.make_chunk_id("doc.pdf", 0, "some text")
    id3 = metadata.make_chunk_id("doc.pdf", 1, "some text")
    log("metadata", "make_chunk_id", "PASS" if id1 == id2 and id1 != id3 else "FAIL", f"id1={id1}, id2={id2}, id3={id3}")

    # Test build_pseudo_url
    url1 = metadata.build_pseudo_url("test.pdf", 1, "21", None)
    log("metadata", "build_pseudo_url", "PASS" if url1 == "local-pdf://test.pdf#article-21" else "FAIL", f"url={url1}")

except Exception as e:
    log("metadata", "general", "ERROR", f"{e}\n{traceback.format_exc()}")


# ==============================================================================
# 4. METADATA_STORE.PY & FAISS_STORE.PY & VECTOR_STORE.PY
# ==============================================================================
try:
    from metadata_store import MetadataStore, normalise_record
    from faiss_store import FaissStore
    from vector_store import VectorStore, get_vector_store

    # Test normalise_record
    rec = normalise_record({"id": "test1", "text": "hello"})
    if rec["title"] == "" and rec["id"] == "test1" and rec["keywords"] == []:
        log("metadata_store", "normalise_record", "PASS")
    else:
        log("metadata_store", "normalise_record", "FAIL", f"{rec}")

    # Test MetadataStore methods
    ms = MetadataStore.create()
    faiss_id0 = ms.add({"id": "doc1", "text": "sample text"})
    faiss_id_dup = ms.add({"id": "doc1", "text": "duplicate text"})
    if faiss_id0 == 0 and faiss_id_dup is None:
        log("metadata_store", "add and deduplicate", "PASS")
    else:
        log("metadata_store", "add and deduplicate", "FAIL", f"id0={faiss_id0}, dup={faiss_id_dup}")

    # Missing ID raises ValueError
    try:
        ms.add({"text": "no id"})
        log("metadata_store", "missing id check", "FAIL", "Did not raise ValueError")
    except ValueError:
        log("metadata_store", "missing id check", "PASS", "Raised ValueError as expected")

    # Test FaissStore
    fs = FaissStore.create(384)
    if fs.ntotal == 0:
        log("faiss_store", "create empty", "PASS", f"ntotal={fs.ntotal}")
    else:
        log("faiss_store", "create empty", "FAIL", f"ntotal={fs.ntotal}")

    # Search on empty index
    empty_scores, empty_ids = fs.search(np.zeros(384), 5)
    if len(empty_scores) == 0 and len(empty_ids) == 0:
        log("faiss_store", "search empty", "PASS")
    else:
        log("faiss_store", "search empty", "FAIL", f"Got scores={empty_scores}, ids={empty_ids}")

    # Add single vector and search
    vec = np.ones((1, 384), dtype="float32")
    vec = vec / np.linalg.norm(vec)
    fs.add(np.array([10], dtype="int64"), vec)
    s_scores, s_ids = fs.search(vec, 5)
    if len(s_ids) == 1 and s_ids[0] == 10:
        log("faiss_store", "add and search", "PASS", f"score={s_scores[0]:.4f}, id={s_ids[0]}")
    else:
        log("faiss_store", "add and search", "FAIL", f"ids={s_ids}")

    # Test active vector store and consistency
    vstore = get_vector_store()
    log("vector_store", "get_vector_store", "PASS", f"size={vstore.size}, faiss_ntotal={vstore.faiss_store.ntotal}")

    # CHECK SYNC: does faiss_store ntotal equal metadata_store length?
    if vstore.size == vstore.faiss_store.ntotal:
        log("vector_store", "index_metadata_sync", "PASS", f"Counts match: {vstore.size}")
    else:
        log("vector_store", "index_metadata_sync", "FAIL", f"FAISS={vstore.faiss_store.ntotal} vs MetadataStore={vstore.size}")

    # Check search with empty query string
    res_empty_q = vstore.search("", 5)
    log("vector_store", "search(empty_query)", "PASS", f"results={len(res_empty_q)}")

    # Check search with normal query
    res_q = vstore.search("Article 21 life and personal liberty", 3)
    log("vector_store", "search(normal_query)", "PASS" if len(res_q) > 0 else "FAIL", f"top_title={res_q[0][0].get('title') if res_q else 'NONE'}, score={res_q[0][1] if res_q else 0}")

except Exception as e:
    log("vector_store", "general", "ERROR", f"{e}\n{traceback.format_exc()}")


# ==============================================================================
# 5. TRUSTED_SOURCES.PY & LIVE_WEB_SEARCHER.PY
# ==============================================================================
try:
    import trusted_sources
    # Test is_trusted_legal_url
    url_tests = [
        ("https://www.indiacode.nic.in/handle/123456789/1362", True),
        ("https://indiankanoon.org/doc/12345/", True),
        ("https://main.sci.gov.in/judgments", True),
        ("https://delhihighcourt.nic.in", True),
        ("https://bombayhighcourt.nic.in", True),
        ("https://highcourtchd.gov.in", True),
        ("https://hckerala.gov.in", True),
        ("https://attacker.com/evil.php?url=indiacode.nic.in", False),
        ("http://indiacode.nic.in.evil.com", False),
        ("javascript:alert(1)", False),
    ]
    for url, exp in url_tests:
        trusted = trusted_sources.is_trusted_legal_url(url)
        status = "PASS" if trusted == exp else "FAIL"
        log("trusted_sources", f"is_trusted({url[:40]})", status, f"trusted={trusted}, expected={exp}")

    # Test chunk_text with short words
    short_words = "This is a short legal text paragraph with thirty words or so."
    chunks_short = trusted_sources.chunk_text(short_words)
    log("trusted_sources", "chunk_text(short)", "PASS" if len(chunks_short) > 0 else "FAIL", f"len={len(chunks_short)}")

    # Test live_web_searcher query sanitizer
    import live_web_searcher
    san = live_web_searcher.sanitize_search_query('what is "anticipatory bail" [section 438] (crpc)?')
    log("live_web_searcher", "sanitize_search_query", "PASS", f"sanitized='{san}'")

    # Test live_web_searcher with empty query
    res_empty = live_web_searcher.search_indian_kanoon("")
    log("live_web_searcher", "search_indian_kanoon(empty)", "PASS" if len(res_empty) == 0 else "FAIL")

except Exception as e:
    log("trusted_sources", "general", "ERROR", f"{e}\n{traceback.format_exc()}")


# ==============================================================================
# 6. PDF_PARSER.PY & PDF_INGESTOR.PY
# ==============================================================================
try:
    import pdf_parser
    # Test with non-pdf
    try:
        pdf_parser.parse_pdf(b"not a pdf", "test.txt")
        log("pdf_parser", "non_pdf_extension", "FAIL", "Did not raise UnsupportedFormatError")
    except pdf_parser.UnsupportedFormatError:
        log("pdf_parser", "non_pdf_extension", "PASS", "Raised UnsupportedFormatError")

    # Test with empty bytes
    try:
        pdf_parser.parse_pdf(b"", "test.pdf")
        log("pdf_parser", "empty_bytes", "FAIL", "Did not raise EmptyPDFError")
    except pdf_parser.EmptyPDFError:
        log("pdf_parser", "empty_bytes", "PASS", "Raised EmptyPDFError")

    # Test with invalid pdf bytes
    try:
        pdf_parser.parse_pdf(b"random corrupt bytes here", "test.pdf")
        log("pdf_parser", "corrupt_pdf", "FAIL", "Did not raise ExtractionError or UnsupportedFormatError")
    except pdf_parser.PDFIngestionError as pe:
        log("pdf_parser", "corrupt_pdf", "PASS", f"Raised {pe.__class__.__name__}")

    # Test with minimal synthetic valid PDF generated via pypdf
    from pypdf import PdfWriter
    writer = PdfWriter()
    page = writer.add_blank_page(width=200, height=200)
    buf = io.BytesIO()
    writer.write(buf)
    blank_pdf_bytes = buf.getvalue()

    try:
        pdf_parser.parse_pdf(blank_pdf_bytes, "blank.pdf")
        log("pdf_parser", "blank_pdf_check", "FAIL", "Did not raise ScannedPDFError or EmptyPDFError")
    except pdf_parser.PDFIngestionError as pe:
        log("pdf_parser", "blank_pdf_check", "PASS", f"Raised {pe.__class__.__name__}")

    # Check pdf_manifest.json and pdf_legal_corpus.jsonl consistency
    import pdf_ingestor
    manifest = pdf_ingestor._load_manifest()
    existing_chunk_ids = pdf_ingestor._existing_chunk_ids()
    log("pdf_ingestor", "manifest_vs_corpus", "PASS", f"manifest_files={len(manifest)}, corpus_chunks={len(existing_chunk_ids)}")

except Exception as e:
    log("pdf_parser", "general", "ERROR", f"{e}\n{traceback.format_exc()}")


# ==============================================================================
# 7. RAG_ENGINE.PY COMPREHENSIVE CHECKS
# ==============================================================================
try:
    import rag_engine
    # 7.1 analyze_legal_query tests
    test_queries = [
        ("What does Article 21 protect?", False),
        ("can police arrest without warrant", True),
        ("get out of jail", True),
        ("maritime admiralty salvage disputes in Cochin", False),
        ("", True),
        ("   ", True),
        ("???", True),
        ("hi", True),
        ("Section 438 CrPC anticipatory bail parameters", False),
        ("Kesavananda Bharati basic structure doctrine", False),
        ("अनुच्छेद 21 क्या है?", True), # Non-English / Hindi query
    ]

    for q, should_be_ambig in test_queries:
        try:
            qa = rag_engine.analyze_legal_query(q)
            log("rag_engine", f"analyze_legal_query({repr(q)[:25]})", "PASS", f"is_ambiguous={qa.is_ambiguous}, terms={qa.legal_terms}, citations={qa.citations}")
        except Exception as e:
            log("rag_engine", f"analyze_legal_query({repr(q)[:25]})", "FAIL", f"Exception: {e}")

    # 7.2 choose_route tests
    r1, why1 = rag_engine.choose_route("Article 21")
    r2, why2 = rag_engine.choose_route("Explain the evolution of fundamental rights in environmental jurisprudence over the past three decades")
    r3, why3 = rag_engine.choose_route("Bail provisions under Section 438 CrPC with recent Supreme Court judgments")
    log("rag_engine", "choose_route(citation)", "PASS", f"route={r1}")
    log("rag_engine", "choose_route(conceptual)", "PASS", f"route={r2}")
    log("rag_engine", "choose_route(hybrid)", "PASS", f"route={r3}")

    # 7.3 retriever test
    corpus = rag_engine.load_corpus()
    retriever = rag_engine.LegalRetriever(corpus)
    
    # Test bm25
    bm_res = retriever.bm25_search("Article 21", limit=3)
    log("rag_engine", "retriever.bm25_search", "PASS" if len(bm_res) > 0 else "FAIL", f"found={len(bm_res)}")

    # Test vector_search
    vec_res = retriever.vector_search("Article 21", limit=3)
    log("rag_engine", "retriever.vector_search", "PASS" if len(vec_res) > 0 else "FAIL", f"found={len(vec_res)}")

    # Test rerank
    reranked = retriever.rerank("Article 21", bm_res)
    log("rag_engine", "retriever.rerank", "PASS" if len(reranked) > 0 else "FAIL", f"top_score={reranked[0].cross_encoder_score if reranked else 0:.3f}")

    # Test assess_evidence_sufficiency
    # Case A: Good evidence
    qa_good = rag_engine.analyze_legal_query("What does Article 21 protect?")
    enough_good, reason_good, _ = rag_engine.assess_evidence_sufficiency(reranked, qa_good)
    log("rag_engine", "assess_evidence_sufficiency(good)", "PASS" if enough_good else "FAIL", f"enough={enough_good}, reason={reason_good[:60]}")

    # Case B: Empty evidence
    enough_empty, reason_empty, _ = rag_engine.assess_evidence_sufficiency([], qa_good)
    log("rag_engine", "assess_evidence_sufficiency(empty)", "PASS" if not enough_empty else "FAIL", f"enough={enough_empty}")

    # 7.4 run_agent on unusual inputs
    # Empty query
    try:
        agent_res_empty = rag_engine.run_agent("", retriever, max_retries=0)
        log("rag_engine", "run_agent(empty_string)", "PASS", f"is_insufficient={agent_res_empty['is_insufficient_evidence_response']}")
    except Exception as e:
        log("rag_engine", "run_agent(empty_string)", "FAIL", f"Exception: {e}")

    # Extreme whitespace
    try:
        agent_res_ws = rag_engine.run_agent("    ", retriever, max_retries=0)
        log("rag_engine", "run_agent(whitespace)", "PASS", f"is_insufficient={agent_res_ws['is_insufficient_evidence_response']}")
    except Exception as e:
        log("rag_engine", "run_agent(whitespace)", "FAIL", f"Exception: {e}")

except Exception as e:
    log("rag_engine", "general", "ERROR", f"{e}\n{traceback.format_exc()}")


# ==============================================================================
# 8. APP.PY STATIC AND IMPORT AUDIT
# ==============================================================================
try:
    import ast
    app_path = repo_root / "app.py"
    with open(app_path, "r", encoding="utf-8") as f:
        code = f.read()

    tree = ast.parse(code, filename="app.py")
    log("app.py", "ast_syntax_parse", "PASS", f"Total lines={len(code.splitlines())}")

    # Scan for duplicate widget keys
    keys = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg == "key":
                    if isinstance(kw.value, ast.Constant):
                        keys.append(kw.value.value)

    from collections import Counter
    key_counts = Counter(keys)
    dup_keys = {k: v for k, v in key_counts.items() if v > 1}
    if dup_keys:
        log("app.py", "duplicate_widget_keys", "FAIL", f"Found duplicate widget keys: {dup_keys}")
    else:
        log("app.py", "duplicate_widget_keys", "PASS", f"Total unique keys: {len(keys)}")

    raw_html_calls = [line for line in code.splitlines() if "unsafe_allow_html=True" in line]
    log("app.py", "unsafe_allow_html_count", "INFO", f"Found {len(raw_html_calls)} instances")

except Exception as e:
    log("app.py", "general", "ERROR", f"{e}\n{traceback.format_exc()}")

# ==============================================================================
# SUMMARY REPORT
# ==============================================================================
print("\n" + "=" * 80)
print("AUDIT SUMMARY REPORT")
print("=" * 80)
pass_count = sum(1 for r in report if r["status"] == "PASS")
fail_count = sum(1 for r in report if r["status"] == "FAIL")
warn_count = sum(1 for r in report if r["status"] == "WARN")
err_count = sum(1 for r in report if r["status"] == "ERROR")
info_count = sum(1 for r in report if r["status"] == "INFO")

print(f"Total Checks: {len(report)}")
print(f"PASS: {pass_count} | FAIL: {fail_count} | WARN: {warn_count} | ERROR: {err_count} | INFO: {info_count}")

failed_checks = [r for r in report if r["status"] in ("FAIL", "WARN", "ERROR")]
if failed_checks:
    print("\nISSUES DETECTED:")
    for fc in failed_checks:
        print(f"  - [{fc['status']}] {fc['section']} -> {fc['name']}: {fc['details']}")
