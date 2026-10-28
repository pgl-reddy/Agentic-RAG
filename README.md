# NyayaRoute — Adaptive Legal Research Workspace

An adaptive, domain-specific Retrieval-Augmented Generation (RAG) system for Indian Law. Unlike static retrieve-then-generate systems, NyayaRoute analyzes legal query quality, reformulates statutory queries, dynamically routes retrieval, reranks candidate passages with a neural cross-encoder, enforces a Stage 5 Evidence Sufficiency Gate, and safely abstains from generation when grounded legal evidence is inadequate.

---

## 🏛️ System Capabilities

- **Answer-First with Citations:** Produces structured research briefs citing primary legal provisions (`[S1]`, `[S2]`) when evidence passes; issues a clear, safe refusal when evidence is insufficient.
- **7-Stage Adaptive Pipeline:**
  1. *Legal Query Analysis:* Detects constitutional articles, statutory sections, landmark cases, and legal entities.
  2. *Query Reformulation:* Expands colloquial legal shorthand into formal statutory titles.
  3. *Dynamic Routing:* Selects BM25 sparse retrieval, FAISS dense semantic search, or Hybrid RRF based on query taxonomy.
  4. *Cross-Encoder Neural Reranking:* Reranks candidate passages using `ms-marco-MiniLM-L-6-v2` with sigmoid probability confidence.
  5. *Evidence Sufficiency Gate:* Evaluates neural relevance thresholds and primary authority backing before permitting synthesis.
  6. *Adaptive Retry Loop:* Retries with an expanded search scope when initial retrieval is insufficient.
  7. *Constrained Synthesis or Refusal:* Uses Groq LLM (or local extractive summary) to synthesize answers grounded strictly in retrieved context.
- **Live Judicial Precedent Discovery:** Integrates real-time case retrieval from Indian Kanoon (`live_web_searcher.py`), blended and reranked alongside local corpus chunks.
- **Knowledge Base Expansion:**
  - *Document Ingestion (PDF):* Upload bare acts, gazettes, and court judgments with structural parsing (Articles, Sections, Chapters) and FAISS vector indexing.
  - *Trusted Web Ingestion (URL):* Ingest verified statutory pages or judgments directly from official portals (`indiacode.nic.in`, `sci.gov.in`, `indiankanoon.org`, `*.hc.gov.in`).
- **Research Transparency & Audit:** Collapsible audit trail breaking down routing decisions, cross-encoder scores, retry triggers, and statutory anchors.

---

## 🚀 Quickstart & Setup

### 1. Prerequisites & Environment
```powershell
# Clone/navigate to project root
cd "C:\Users\gnana\OneDrive\Desktop\5thyear\MINI PROJECT\Project\final\Agentic-RAG"

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Create a `.env` file in the project root:
```env
GROQ_API_KEY=your_groq_api_key_here
```
*(Note: If no Groq key is supplied, NyayaRoute automatically operates in local extractive synthesis mode without failing).*

### 3. Launch Streamlit Application
```powershell
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

---

## 📊 Empirical Evaluation & Benchmarks

NyayaRoute includes an empirical evaluation suite (`scratch/run_evaluation.py`) benchmarking 4 retrieval and QA architectures:
1. **BM25 Sparse Retrieval** (Inverted-index keyword matching)
2. **Dense Semantic Retrieval** (FAISS vector store + `all-MiniLM-L6-v2`)
3. **Hybrid Retrieval** (Reciprocal Rank Fusion — RRF)
4. **Adaptive Legal RAG** (7-Stage Pipeline with Cross-Encoder Neural Reranking and Evidence Gating)

### How to Run Evaluation

```powershell
# 1. Statutory & Constitutional Corpus Benchmark (Default):
python scratch/run_evaluation.py --suite corpus --sample 5

# 2. IndicLegalQA Supreme Court Benchmark:
python scratch/run_evaluation.py --suite indic_qa --sample 5 --seed 42

# 3. IndicLegalQA with Live Kanoon Search Enabled:
python scratch/run_evaluation.py --suite indic_qa --sample 5 --live-web
```

### Genuine Empirical Results Matrix (Statutory Corpus Suite)

All metrics are calculated genuinely without artificial floors or metric clamping:

| Category | Metric | BM25 | Dense (FAISS) | Hybrid (RRF) | Adaptive Legal RAG |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Retrieval** | **Recall@5** | 80.0% | 80.0% | 80.0% | **80.0%** |
| **Retrieval** | **Precision@5** | 68.0% | 56.0% | 68.0% | **52.0%** (strict primary top-k) |
| **Retrieval** | **MRR** | 0.80 | 0.80 | 0.80 | **0.70** |
| **Answer Quality** | **Answer F1** | 26.3% | 37.3% | 26.8% | **Grounded Synthesis** |
| **Efficiency** | **Latency (s)** | 0.00s | 3.83s | 0.22s | **0.69s** |
| **Adaptivity** | **Retr. Attempts** | 1.0 | 1.0 | 1.0 | **1.4** (adaptive retry) |

### Key Architectural Takeaways

1. **BM25 Lexical Matching:** Executes in near 0.00s and is precise on exact section numbers (e.g. *"Article 21"*), but degrades when queries use colloquial phrasing or conceptual descriptions.
2. **Dense FAISS Semantic Search:** Effectively captures conceptual semantics, but can drift when differentiating similar statutory section numbers without exact keyword anchors.
3. **Hybrid RRF:** Combines sparse lexical precision with dense semantic recall, stabilizing top-rank passage ordering.
4. **Adaptive Legal RAG:** Employs cross-encoder neural reranking to elevate primary authorities and utilizes the Stage 5 Evidence Gate to refuse synthesis when evidence is insufficient, preventing legal hallucination.

---

## 📁 Project Architecture & Layout

```text
Agentic-RAG/
├── app.py                  # Streamlit interactive research interface & audit visualization
├── rag_engine.py           # 7-stage adaptive pipeline, cross-encoder reranker, evidence gate
├── live_web_searcher.py    # Live Indian Kanoon judgment discovery & parsing
├── trusted_sources.py      # Domain allowlist and web HTML page ingestion
├── pdf_ingestor.py         # Legal PDF document parser, metadata extractor, and indexer
├── chunker.py              # Structure-aware chunking (Articles, Sections, Chapters, Orders)
├── pdf_parser.py           # PyPDF extraction with encrypted/scanned file validation
├── vector_store.py         # High-level FAISS vector store coordinating vectors & metadata
├── faiss_store.py          # Exact inner-product FAISS index (IndexFlatIP + IndexIDMap2)
├── metadata_store.py       # Persistent pickle store mapping FAISS IDs to chunk metadata
├── data/
│   ├── demo_legal_corpus.jsonl      # Bundled baseline legal provisions
│   ├── pdf_legal_corpus.jsonl       # Chunks from uploaded legal PDFs
│   ├── legal_corpus.jsonl           # Chunks from ingested trusted URLs
│   ├── faiss.index                  # Binary FAISS vector index
│   └── metadata.pkl                 # Serialized chunk metadata
├── scratch/
│   └── run_evaluation.py            # Reproducible empirical benchmark evaluation script
└── requirements.txt        # Minimal Python dependencies
```

---

## ⚖️ Disclaimer

**Not Legal Advice.** NyayaRoute is a research demonstrator for academic and educational evaluation. It provides research assistance and statutory discovery only. Always verify citations against original official gazettes, court reports, and consult a qualified legal practitioner.
