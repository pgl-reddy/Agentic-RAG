# NyayaRoute

NyayaRoute is a Streamlit prototype for Indian legal research. It analyzes a question, selects a retrieval route, retrieves and reranks legal passages, checks whether the evidence is adequate, and produces a cited research response when Groq is configured.

It is research assistance, not legal advice. Verify every proposition against the current primary source and consult a qualified advocate for legal advice.

## Research flow

```text
Question
  → legal terms, entities, case references, citations, ambiguity and freshness analysis
  → query clarification/reformulation when needed
  → BM25, dense semantic, hybrid, or live web search
  → cross-encoder reranking
  → evidence sufficiency decision
  → broadened retry when evidence is weak
  → cited answer from retrieved passages, or an insufficient-evidence result
```

Established questions search the local corpus. Exact statute/article/section questions use BM25, conceptual questions use dense semantic search, and case/citation questions use hybrid search. Queries containing freshness signals such as “latest”, “today”, “recent”, “passed”, “signed”, or “notified” go to a DuckDuckGo live web search. Web results are discovery evidence; a result must point to a relevant official government or court domain to pass the dynamic evidence gate.

If `GROQ_API_KEY` is set, Groq reformulates ambiguous questions and generates the final answer using the retrieved context. Clear questions use the deterministic analyzer to avoid an unnecessary LLM round-trip. Without a key, retrieval and evidence checks still work, and the app shows sources or an insufficient-evidence decision without generating an answer. The cross-encoder loads on first use. The 12k-record vector index is built only for dense or hybrid searches; exact Article/Section queries and their retries stay on BM25.

## Data and source status

`data/static_legal_corpus.jsonl` contains the supplied 12,163 records. NyayaRoute loads it along with the demo records, web-ingested records, and PDF-ingested records. The static corpus is identified as a research dataset; it is not promoted to primary authority merely because its text contains legislation. A very strong exact-text match, or multiple distinct strongly matching passages from that dataset, may support a carefully labelled research answer. The answer must call the material unverified and direct the reader to the current primary text. Illustrative demo entries and locally uploaded PDFs without independently verified provenance are not accepted as primary evidence.

The vector index is built and updated automatically from all loaded JSONL records the first time a dense or hybrid search runs. It is persisted in `data/faiss.index` and `data/metadata.pkl`. It uses FAISS if installed; otherwise it uses exact NumPy cosine search, avoiding a platform-specific FAISS install requirement. BM25 uses the same JSONL records directly. New web pages and PDF uploads are included on the next retrieval after ingestion.

Live web search uses DuckDuckGo HTML results. Search-result snippets can be incomplete or stale; open the linked official source and verify its contents before relying on it. The web route requires a relevant official government/court result to pass the evidence gate.

## Setup on Windows

Run these commands from the NyayaRoute directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Put the Groq key in `.env`:

```dotenv
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
```

Then run:

```powershell
streamlit run app.py
```

`.env` is excluded from Git. Never share or commit the file containing your real key. `.env.example` shows the expected variable names.

## Add legal material

The **Source library** tab accepts one specific URL from India Code, the Supreme Court of India, Indian Kanoon, or recognized official High Court domains. URL ingestion is page-by-page and does not crawl whole sites. Respect source terms, rate limits, and copyright.

The **Knowledge Base** tab accepts text-based legal PDFs. It previews extracted text, detects likely scans/encryption/empty files, splits text by legal structure where possible, and saves chunks to the PDF corpus. Scanned PDFs need OCR before upload. Uploaded documents are not treated as verified primary sources simply because they are labelled as court or government documents.

The command-line URL ingestor is:

```powershell
python ingest.py "https://www.indiacode.nic.in/handle/123456789/15442?locale=en"
```

The one-time index migration is available when desired:

```powershell
python migrate_to_faiss.py
```

## Project files

- `app.py` — Streamlit research, source library, PDF upload, audit trail, and memo export.
- `rag_engine.py` — query analysis, static/dynamic routing, BM25/dense/hybrid retrieval, reranking, evidence gates, retries, web search, and Groq answer generation.
- `vector_store.py`, `faiss_store.py`, `metadata_store.py`, `embedding.py` — persistent dense search index.
- `pdf_parser.py`, `chunker.py`, `metadata.py`, `pdf_ingestor.py` — PDF text extraction, legal-aware chunking, metadata, and ingestion.
- `trusted_sources.py`, `ingest.py` — allowlisted single-page web ingestion.
- `data/static_legal_corpus.jsonl` — supplied 12,163-record corpus.
- `migrate_to_faiss.py` — optional explicit corpus-to-index command.
- `design.html` — visual mockup, clearly marked as not live data; `app.py` is the actual application.

The interface displays the real loaded record count, query analysis, retrieval route, source status, retry query, and evidence decision. Decorative mock metrics and the hard-coded precedent heatmap have been removed.
