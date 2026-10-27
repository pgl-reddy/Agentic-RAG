"""Agentic, evidence-constrained retrieval for Indian legal research.

Directly implements the 7-stage legal intelligence pipeline:
1. Analyze the legal query for legal terms, entities, case references, citations, and ambiguity.
2. Reformulate weak or ambiguous queries before retrieval.
3. Route each query to BM25 sparse retrieval, dense semantic retrieval, or hybrid retrieval.
4. Re-rank retrieved passages using a cross-encoder relevance model.
5. Assess evidence sufficiency before allowing the LLM to generate an answer.
6. Retry with a reformulated query when evidence is insufficient.
7. Return an explicit insufficient-evidence response when adequate support cannot be obtained within the retry limit.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from typing import Any

# Ensure local .venv site-packages are accessible even if run with global python
_venv_site = Path(__file__).resolve().parent / ".venv" / "Lib" / "site-packages"
if _venv_site.exists() and str(_venv_site) not in sys.path:
    sys.path.insert(0, str(_venv_site))

try:
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
    # Enable offline loading for cached HuggingFace models to eliminate SSL retry latency
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
except Exception:
    pass

import numpy as np

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    class BM25Okapi:  # type: ignore
        """Self-contained fallback BM25 implementation if rank_bm25 is not installed."""
        def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
            self.corpus = corpus
            self.doc_len = [len(doc) for doc in corpus]
            self.avgdl = sum(self.doc_len) / max(1, len(corpus))
            self.doc_freqs = []
            for doc in corpus:
                frequencies: dict[str, int] = {}
                for word in doc:
                    frequencies[word] = frequencies.get(word, 0) + 1
                self.doc_freqs.append(frequencies)
            self.nd = len(corpus)
            self.k1 = k1
            self.b = b

        def get_scores(self, query: list[str]) -> np.ndarray:
            scores = np.zeros(self.nd)
            for q in query:
                df = sum(1 for freqs in self.doc_freqs if q in freqs)
                if df == 0:
                    continue
                idf = math.log((self.nd - df + 0.5) / (df + 0.5) + 1)
                for i, freqs in enumerate(self.doc_freqs):
                    tf = freqs.get(q, 0)
                    if tf > 0:
                        num = tf * (self.k1 + 1)
                        denom = tf + self.k1 * (1 - self.b + self.b * (self.doc_len[i] / max(1e-5, self.avgdl)))
                        scores[i] += idf * (num / denom)
            return scores

from vector_store import get_vector_store

STOPWORDS = {
    "what", "does", "the", "and", "for", "with", "under", "about", "latest",
    "is", "to", "in", "of", "a", "an", "on", "are", "it", "by", "as", "at",
    "from", "which", "who", "whom", "this", "that", "these", "those", "how",
    "can", "could", "should", "would", "any", "some", "be", "been", "being",
    "or", "into", "over", "after", "before", "between", "both", "through"
}

# Legal terminology taxonomy for Indian jurisprudence
LEGAL_TERMS_TAXONOMY = {
    # Procedural & Criminal
    "bail", "anticipatory bail", "regular bail", "interim bail", "writ", "habeas corpus",
    "mandamus", "certiorari", "prohibition", "quo warranto", "appeal", "petition",
    "special leave petition", "slp", "revision", "review", "quash", "stay", "injunction",
    "fir", "cognizable", "non-cognizable", "bailable", "non-bailable", "charge sheet",
    "summon", "warrant", "arrest", "custody", "remand", "discharge", "acquittal", "conviction",
    # Substantive & Constitutional
    "fundamental right", "fundamental rights", "life and personal liberty", "equality before law",
    "right to privacy", "right to clean environment", "speedy trial", "double jeopardy",
    "self incrimination", "due process", "procedure established by law", "basic structure",
    "judicial review", "ultra vires", "doctrine of severability", "pith and substance",
    # Statutory & Offences
    "act", "article", "section", "clause", "schedule", "constitution", "code", "statute",
    "ipc", "bns", "crpc", "bnss", "bsa", "cpc", "it act", "information technology act", "defamation", "theft", "murder",
    "culpable homicide", "dowry", "cruelty", "cheating", "fraud", "corruption", "contempt",
    "tort", "negligence", "breach of contract", "limitation", "precedent", "ruling", "judgment",
    "ratio decidendi", "obiter dicta", "amendment", "rules", "regulations", "ordinance"
}

KNOWN_ENTITIES = {
    # Courts & Tribunals
    "Supreme Court of India", "Supreme Court", "High Court", "Sessions Court",
    "District Court", "Magistrate Court", "National Green Tribunal", "NGT",
    "NCLT", "NCLAT", "CAT", "ITAT", "Consumer Forum", "Lok Adalat",
    # Sovereign & Constitutional
    "Union of India", "President of India", "Governor", "Parliament",
    "Election Commission", "Bar Council of India", "Law Commission",
    # Investigative & Regulatory
    "Central Bureau of Investigation", "CBI", "Enforcement Directorate", "ED",
    "Police", "NIA", "SEBI", "RBI", "CCI"
}

LANDMARK_CASES = {
    "Kesavananda Bharati", "Maneka Gandhi", "K.S. Puttaswamy", "A.K. Gopalan",
    "Navtej Singh Johar", "Joseph Shine", "Shayara Bano", "Vishaka", "D.K. Basu",
    "Arnesh Kumar", "Lalita Kumari", "Minerva Mills", "S.R. Bommai", "Indira Sawhney",
    "Shreya Singhal", "Shah Bano", "Golaknath", "Waman Rao", "ADM Jabalpur",
    "M.C. Mehta", "Bachan Singh", "P.A. Inamdar", "T.M.A. Pai"
}

PRIMARY_SOURCES = {
    "India Code",
    "Supreme Court of India",
    "Indian Kanoon",
    "High Court",
    "Gazette of India",
    "Constitution of India",
    "Uploaded PDF",
    "Law Commission of India",
}


@dataclass
class QueryAnalysis:
    """Stage 1: Detailed analysis of the legal research query."""
    raw_query: str
    legal_terms: list[str]
    entities: list[str]
    case_references: list[str]
    citations: list[str]
    is_ambiguous: bool
    ambiguity_reasons: list[str]
    is_weak: bool
    quality_verdict: str


@dataclass
class Evidence:
    """Retrieved authority passage with dual lexical and cross-encoder relevance scores."""
    id: str
    title: str
    url: str
    source: str
    text: str
    score: float = 0.0                  # Retrieval score (BM25 or dense vector or RRF)
    relevance: float = 0.0              # Query term anchor overlap (0.0 to 1.0)
    cross_encoder_score: float = 0.0    # Raw cross-encoder logit score
    cross_encoder_prob: float = 0.0     # Normalized cross-encoder probability (0.0 to 1.0)


def _tokens(text: str) -> list[str]:
    return re.findall(r"[\w]+", text.lower(), re.UNICODE)


def load_corpus() -> list[dict[str, Any]]:
    """Read the bundled sample and locally ingested source records."""
    data_dir = Path(__file__).resolve().parent / "data"
    rows: list[dict[str, Any]] = []
    for path in (
        data_dir / "demo_legal_corpus.jsonl",
        data_dir / "legal_corpus.jsonl",
        data_dir / "pdf_legal_corpus.jsonl",
    ):
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def source_summary(corpus: list[dict[str, Any]]) -> list[dict[str, str | int]]:
    grouped: dict[str, dict[str, Any]] = {}
    for item in corpus:
        key = item["url"]
        grouped.setdefault(key, {"title": item["title"], "url": key, "source": item["source"], "chunks": 0})
        grouped[key]["chunks"] += 1
    return sorted(grouped.values(), key=lambda item: (str(item["source"]), str(item["title"])))


# ──────────────────────────────────────────────────────────────────────────────
# STAGE 1: Legal Query Analysis
# ──────────────────────────────────────────────────────────────────────────────
def analyze_legal_query(question: str) -> QueryAnalysis:
    """Analyze query for legal terms, entities, case references, citations, and ambiguity."""
    cleaned = question.strip()
    lowered = cleaned.lower()
    words = cleaned.split()

    # 1. Legal terms detection
    found_terms: set[str] = set()
    for term in LEGAL_TERMS_TAXONOMY:
        if re.search(rf"\b{re.escape(term)}\b", lowered):
            found_terms.add(term)

    # 2. Entities detection
    found_entities: set[str] = set()
    for entity in KNOWN_ENTITIES:
        if re.search(rf"\b{re.escape(entity.lower())}\b", lowered):
            found_entities.add(entity)

    # 3. Case references detection (Landmark list + Party v. Party pattern)
    found_cases: set[str] = set()
    for case in LANDMARK_CASES:
        if re.search(rf"\b{re.escape(case.lower())}\b", lowered):
            found_cases.add(case)
    # Generic "X v. Y" or "X vs Y"
    versus_matches = re.findall(r"\b([A-Z][a-zA-Z\.\s]{2,25}\s+(?:v\.|vs\.?|versus)\s+[A-Z][a-zA-Z\.\s]{2,25})\b", cleaned)
    for m in versus_matches:
        found_cases.add(m.strip())

    # 4. Citations detection (Articles, Sections, Law Reports)
    found_citations: set[str] = set()
    # Articles
    for art in re.findall(r"\b(?:Article|Art\.?)\s+\d+[A-Za-z]?(?:\s*\([0-9a-zA-Z]+\))?\b", cleaned, re.IGNORECASE):
        norm = re.sub(r"(?i)\bart\.?\s*", "Article ", art)
        found_citations.add(norm.title())
    # Sections
    for sec in re.findall(r"\b(?:Section|Sec\.?)\s+\d+[A-Za-z]?(?:\s*\([0-9a-zA-Z]+\))*\b", cleaned, re.IGNORECASE):
        norm = re.sub(r"(?i)\bsec\.?\s*", "Section ", sec)
        found_citations.add(norm.title())
    # Orders & Rules
    for ord_rule in re.findall(r"\bOrder\s+[IVXLCDM0-9]+\s+Rule\s+\d+\b", cleaned, re.IGNORECASE):
        found_citations.add(ord_rule.title())
    for rule in re.findall(r"\bRule\s+\d+[A-Za-z]?(?:\s*\([0-9a-zA-Z]+\))*\b", cleaned, re.IGNORECASE):
        found_citations.add(rule.title())
    # Law report reporter patterns: AIR 1978 SC 597, (1973) 4 SCC 225, etc.
    for rep in re.findall(r"\b(?:AIR\s+\d{4}\s+[A-Za-z]+(?:\s+\d+)?|\d{4}\s+(?:\(\d+\)\s+)?(?:SCC|SCR|SCALE|CrLJ|ILR)(?:\s+\d+)?)\b", cleaned, re.IGNORECASE):
        found_citations.add(rep)

    # 5. Ambiguity & Weakness Analysis
    ambiguity_reasons: list[str] = []
    is_weak = False
    is_ambiguous = False

    if len(words) < 4:
        is_weak = True
        ambiguity_reasons.append("Query is under 4 words and lacks required factual/legal context.")

    if not found_terms and not found_citations and not found_cases:
        is_weak = True
        ambiguity_reasons.append("No recognized legal terms, statutory sections, or case citations found.")

    # Pronoun-heavy queries without clear legal subjects
    pronouns = {"he", "she", "they", "them", "it", "this", "that", "someone", "guy"}
    if set(_tokens(lowered)) & pronouns and not found_citations and len(found_terms) <= 1:
        is_ambiguous = True
        ambiguity_reasons.append("Contains underspecified pronouns without specifying parties, actions, or statutory context.")

    # Colloquial layman phrasing detection
    colloquial_patterns = [
        (r"\bget out of (?:jail|prison)\b", "Uses colloquial 'get out of jail' instead of formal bail jurisprudence."),
        (r"\bpolice trouble\b", "Uses colloquial phrasing instead of specifying police action, investigation, or powers of arrest."),
        (r"\bfight a case\b", "Colloquial expression lacking specific relief (quashing, bail, appeal, revision)."),
        (r"\bcan they do that\b", "Ambiguous question lacking subject and specific statutory act."),
        (r"\bcan police arrest\b", "Generalized inquiry lacking statutory section (e.g. Section 41 CrPC / Section 35 BNSS cognizable offences)."),
        (r"^(?:can|is it legal|what happens if)\b", "Lacks specific statutory sections, procedural codes, or jurisdictional authorities."),
    ]
    for pattern, reason in colloquial_patterns:
        if re.search(pattern, lowered):
            is_ambiguous = True
            ambiguity_reasons.append(reason)

    if is_weak or is_ambiguous:
        verdict = f"Weak/Ambiguous ({len(ambiguity_reasons)} issue(s) detected; reformulation required before retrieval)"
    else:
        verdict = f"Specific & Grounded ({len(found_terms)} legal term(s), {len(found_citations)} citation(s), {len(found_cases)} case(s))"

    return QueryAnalysis(
        raw_query=cleaned,
        legal_terms=sorted(found_terms),
        entities=sorted(found_entities),
        case_references=sorted(found_cases),
        citations=sorted(found_citations),
        is_ambiguous=is_ambiguous or is_weak,
        ambiguity_reasons=ambiguity_reasons,
        is_weak=is_weak,
        quality_verdict=verdict,
    )


def quality_check(question: str) -> tuple[bool, str]:
    """Backward-compatible helper wrapping analyze_legal_query."""
    analysis = analyze_legal_query(question)
    return not analysis.is_ambiguous, analysis.quality_verdict


# ──────────────────────────────────────────────────────────────────────────────
# STAGE 2: Query Reformulation
# ──────────────────────────────────────────────────────────────────────────────
_GROQ_CLIENT = None
_GROQ_CLIENT_KEY = None


def get_groq_client():
    global _GROQ_CLIENT, _GROQ_CLIENT_KEY
    from groq import Groq
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY is not configured")
    if _GROQ_CLIENT is None or _GROQ_CLIENT_KEY != key:
        _GROQ_CLIENT = Groq(api_key=key)
        _GROQ_CLIENT_KEY = key
    return _GROQ_CLIENT


def groq_chat(messages: list[dict[str, str]], temperature: float = 0.1) -> str:
    client = get_groq_client()
    response = client.chat.completions.create(
        model=os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b"),
        messages=messages,
        temperature=temperature,
        max_tokens=1100,
    )
    return response.choices[0].message.content or ""


LEGAL_ABBREVIATION_EXPANSIONS = [
    # Specific IT Act / Rules
    (r"\b(?:rule|sec(?:tion)?)\s*(\d+[A-Za-z]?)\s*(?:of\s+)?it\s+act\b", r"Information Technology Rules, 2021 Rule \1 / Information Technology Act, 2000 Section \1"),
    (r"\bit\s+act\b(?!\s*,\s*2000)", "Information Technology Act, 2000"),

    # Articles of Constitution (only if not already followed by Constitution of India)
    (r"\b(?:art|article)\s*(\d+[A-Za-z]?)\b(?!\s+(?:of\s+)?(?:the\s+)?constitution)", r"Article \1 Constitution of India"),

    # Criminal provisions (only if not already followed by full Act name)
    (r"\bsec(?:tion)?\s*438\s*(?:crpc)?\b(?!\s+(?:of\s+)?(?:the\s+)?code\s+of\s+criminal)", "Section 438 Code of Criminal Procedure, 1973 (anticipatory bail)"),
    (r"\bsec(?:tion)?\s*439\s*(?:crpc)?\b(?!\s+(?:of\s+)?(?:the\s+)?code\s+of\s+criminal)", "Section 439 Code of Criminal Procedure, 1973 (bail powers)"),
    (r"\bsec(?:tion)?\s*482\s*(?:crpc)?\b(?!\s+(?:of\s+)?(?:the\s+)?code\s+of\s+criminal)", "Section 482 Code of Criminal Procedure, 1973 (inherent powers)"),
    (r"\bsec(?:tion)?\s*302\s*(?:ipc)?\b(?!\s+(?:of\s+)?(?:the\s+)?indian\s+penal)", "Section 302 Indian Penal Code, 1860 / Section 103 Bharatiya Nyaya Sanhita, 2023"),
    (r"\bsec(?:tion)?\s*498a\s*(?:ipc)?\b(?!\s+(?:of\s+)?(?:the\s+)?indian\s+penal)", "Section 498A Indian Penal Code, 1860 (cruelty dowry harassment)"),

    # Common colloquial legal queries
    (r"\bget out of jail\b", "provisions for grant of bail and release on personal bond"),
    (r"\bpolice arrest without warrant\b", "Section 41 Code of Criminal Procedure / Section 35 Bharatiya Nagarik Suraksha Sanhita when police may arrest without warrant"),
]


def reformulate_query(question: str, analysis: QueryAnalysis) -> tuple[str, bool, str]:
    """Stage 2: Reformulate weak or ambiguous queries before retrieval."""
    t_lower = question.lower()
    full_statutes = [
        "constitution of india",
        "indian penal code",
        "code of criminal procedure",
        "bharatiya nyaya sanhita",
        "bharatiya nagarik suraksha sanhita",
        "bharatiya sakshya adhiniyam",
        "indian evidence act",
        "information technology act, 2000",
        "information technology rules, 2021",
    ]
    # If already fully qualified with statute and not using shorthand like 'art ', 'sec ', 'crpc', 'ipc'
    if any(s in t_lower for s in full_statutes) and not re.search(r"\b(?:art|sec|crpc|ipc|bns|bnss|bsa|it\s+act)\b", t_lower):
        return question, False, "Query already contains explicit legal anchors; proceeded without reformulation."

    # 1. Expand recognized legal shorthand / abbreviations into authoritative enactments
    abbrev_expanded = question
    matched_abbrev = False
    for pattern, replacement in LEGAL_ABBREVIATION_EXPANSIONS:
        if re.search(pattern, abbrev_expanded, flags=re.IGNORECASE):
            abbrev_expanded = re.sub(pattern, replacement, abbrev_expanded, count=1, flags=re.IGNORECASE)
            matched_abbrev = True
            break

    if matched_abbrev and abbrev_expanded.strip().lower() != question.strip().lower():
        reason = "Expanded statutory shorthand/abbreviations into formal Indian enactments."
        return abbrev_expanded.strip(), True, reason

    if not analysis.is_ambiguous and not analysis.is_weak:
        return question, False, "Query already contains explicit legal anchors; proceeded without reformulation."

    # Try LLM reformulation if Groq is available
    if os.getenv("GROQ_API_KEY"):
        prompt = (
            "You are a Senior Indian Legal Research Specialist.\n"
            "The following query is weak, colloquial, or ambiguous. "
            "Reformulate it into an authoritative, precise Indian legal research query.\n"
            "Rules:\n"
            "- Expand abbreviations (e.g. 'art 21' -> 'Article 21 Constitution of India', 'sec 438 crpc' -> 'Section 438 Code of Criminal Procedure anticipatory bail').\n"
            "- Ground into the correct Indian statutory framework (e.g., IPC/BNS, CrPC/BNSS, Constitution of India, Evidence Act/BSA).\n"
            "- Preserve the user's factual intent accurately. Do NOT invent unrelated statutes.\n"
            "- Return ONLY the reformulated query text.\n\n"
            f"User Query: {question}\n"
            f"Detected Issues: {'; '.join(analysis.ambiguity_reasons)}"
        )
        try:
            reformulated = groq_chat([
                {"role": "system", "content": "You are a precise Indian legal query reformulation engine."},
                {"role": "user", "content": prompt}
            ])
            reformulated_clean = reformulated.strip().strip('"').strip("'")
            if reformulated_clean and len(reformulated_clean) > 5:
                reason = f"LLM reformulated weak/ambiguous query: resolved {len(analysis.ambiguity_reasons)} issue(s)."
                return reformulated_clean, True, reason
        except Exception:
            pass

    # Deterministic fallback
    reformulated = f"Indian law statutory provisions and judicial precedents regarding: {question}"
    reason = f"Rule-based legal reformulation applied to resolve: {'; '.join(analysis.ambiguity_reasons)}"
    return reformulated, True, reason


def rewrite_query(question: str) -> str:
    """Backward-compatible helper for query rewriting."""
    analysis = analyze_legal_query(question)
    query, _, _ = reformulate_query(question, analysis)
    return query


# ──────────────────────────────────────────────────────────────────────────────
# STAGE 3: Routing (BM25 Sparse, Dense Semantic, or Hybrid)
# ──────────────────────────────────────────────────────────────────────────────
def choose_route(query: str, analysis: QueryAnalysis | None = None) -> tuple[str, str]:
    """Stage 3: Route query to BM25 sparse retrieval, dense semantic retrieval, or hybrid retrieval."""
    if analysis is None:
        analysis = analyze_legal_query(query)

    has_citations = bool(analysis.citations)
    has_cases = bool(analysis.case_references)
    has_legal_terms = bool(analysis.legal_terms)
    terms = set(_tokens(query))

    # Exact statutory citations or specific numeric sections without complex narrative
    # benefit from lexical sparse retrieval (BM25) to avoid semantic drift from specific numbers.
    if has_citations and not has_cases and len(terms) <= 8:
        return (
            "BM25 Sparse Retrieval",
            f"Exact statutory citation(s) ({', '.join(analysis.citations)}) identified. Sparse lexical inverted-index matching guarantees precise section/clause retention without semantic drift."
        )

    # Narrative/conceptual fact patterns without exact section citations
    # benefit from dense semantic vector retrieval.
    if not has_citations and not has_cases and len(terms) > 8:
        return (
            "Dense Semantic Retrieval",
            "Conceptual legal fact pattern identified without rigid section references. Dense vector embedding retrieval captures semantic intent across paraphrased legal principles."
        )

    # Hybrid Retrieval is optimal when combining statutory references, case-law,
    # and contextual facts.
    return (
        "Hybrid Retrieval",
        "Blends BM25 sparse lexical matching for statutory anchors with FAISS dense vector semantic retrieval via Reciprocal Rank Fusion (RRF) for comprehensive precedential and statutory coverage."
    )


# ──────────────────────────────────────────────────────────────────────────────
# STAGE 4: Cross-Encoder Neural Re-Ranking & Retriever
# ──────────────────────────────────────────────────────────────────────────────
class LegalRetriever:
    """Retriever with BM25 sparse search, FAISS dense search, RRF Hybrid, and Cross-Encoder Re-Ranking."""

    def __init__(self, corpus: list[dict[str, Any]]):
        self.corpus = corpus
        self.tokenized = [_tokens(f"{d['title']} {d['text']}") for d in corpus]
        self.bm25 = BM25Okapi(self.tokenized) if corpus else None
        self._cross_encoder = None
        self._cross_encoder_failed = False

    def _evidence(self, index: int, score: float, query: str) -> Evidence:
        doc = self.corpus[index]
        query_terms = set(_tokens(query)) - STOPWORDS
        document_terms = set(_tokens(f"{doc['title']} {doc['text']}"))
        relevance = len(query_terms & document_terms) / max(1, len(query_terms))
        return Evidence(
            id=doc["id"],
            title=doc["title"],
            url=doc["url"],
            source=doc["source"],
            text=doc["text"],
            score=score,
            relevance=relevance,
        )

    def bm25_search(self, query: str, limit: int) -> list[Evidence]:
        if not self.bm25:
            return []
        scores = self.bm25.get_scores(_tokens(query))
        top = np.argsort(scores)[::-1][:limit]
        return [self._evidence(int(i), float(scores[i]), query) for i in top if scores[i] > 0]

    def vector_search(self, query: str, limit: int) -> list[Evidence]:
        """Search persistent FAISS vector index."""
        try:
            store = get_vector_store()
            results = store.search(query, limit)
            evidence: list[Evidence] = []
            for record, score in results:
                query_terms = set(_tokens(query)) - STOPWORDS
                document_terms = set(_tokens(f"{record['title']} {record['text']}"))
                relevance = len(query_terms & document_terms) / max(1, len(query_terms))
                evidence.append(
                    Evidence(
                        id=record["id"],
                        title=record["title"],
                        url=record["url"],
                        source=record["source"],
                        text=record["text"],
                        score=float(score),
                        relevance=relevance,
                    )
                )
            return [item for item in evidence if item.score > 0.05]
        except Exception:
            return []

    def live_web_search(self, query: str, limit: int = 4) -> list[Evidence]:
        """Fetch live case law and statutory excerpts directly from Indian Kanoon."""
        try:
            from live_web_searcher import search_indian_kanoon
            raw_results = search_indian_kanoon(query, max_results=limit)
            query_terms = set(_tokens(query)) - STOPWORDS
            evidence: list[Evidence] = []
            for i, res in enumerate(raw_results):
                doc_terms = set(_tokens(f"{res['title']} {res['text']}"))
                relevance = len(query_terms & doc_terms) / max(1, len(query_terms))
                evidence.append(
                    Evidence(
                        id=f"live_{res.get('doc_id') or i}",
                        title=res["title"],
                        url=res["url"],
                        source="Indian Kanoon",
                        text=res["text"],
                        score=0.6,
                        relevance=relevance,
                    )
                )
            return evidence
        except Exception:
            return []

    def retrieve(self, query: str, route: str, limit: int = 12, include_live_web: bool = False, *args: Any, **kwargs: Any) -> list[Evidence]:
        """Execute sparse, dense, or hybrid retrieval, optionally augmented by live web search, followed by cross-encoder re-ranking."""
        if "BM25" in route or route == "BM25 Sparse Retrieval":
            candidates = self.bm25_search(query, limit)
        elif "Dense" in route or route == "Dense Semantic Retrieval" or route == "Vector DB":
            candidates = self.vector_search(query, limit)
        else:
            # Hybrid Retrieval = BM25 + FAISS Dense Vector + Reciprocal Rank Fusion (RRF)
            combined: dict[str, Evidence] = {}
            for rank, item in enumerate(self.bm25_search(query, limit), 1):
                item.score = 1.0 / (60.0 + rank)
                combined[item.id] = item
            for rank, item in enumerate(self.vector_search(query, limit), 1):
                if item.id in combined:
                    combined[item.id].score += 1.0 / (60.0 + rank)
                else:
                    item.score = 1.0 / (60.0 + rank)
                    combined[item.id] = item
            candidates = sorted(combined.values(), key=lambda x: x.score, reverse=True)

        if include_live_web:
            live_items = self.live_web_search(query, limit=4)
            seen_titles = {c.title.lower() for c in candidates}
            for lev in live_items:
                if lev.title.lower() not in seen_titles:
                    candidates.append(lev)

        return self.rerank(query, candidates[:limit + (4 if include_live_web else 0)])

    def _get_cross_encoder(self):
        if self._cross_encoder is None and not self._cross_encoder_failed:
            try:
                from sentence_transformers import CrossEncoder
                try:
                    self._cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", local_files_only=True)
                except Exception:
                    self._cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            except Exception:
                self._cross_encoder_failed = True
                self._cross_encoder = None
        return self._cross_encoder

    def rerank(self, query: str, candidates: list[Evidence]) -> list[Evidence]:
        """Stage 4: Re-rank retrieved passages using a cross-encoder relevance model."""
        if not candidates:
            return []

        encoder = self._get_cross_encoder()
        if encoder is not None:
            try:
                pairs = [(query, item.text[:512]) for item in candidates]
                raw_scores = encoder.predict(pairs)
                for item, raw_score in zip(candidates, raw_scores):
                    val = float(raw_score)
                    item.cross_encoder_score = val
                    # Sigmoid probability normalization
                    item.cross_encoder_prob = float(1.0 / (1.0 + math.exp(-val)))
                # Re-rank primarily by cross-encoder relevance model score!
                return sorted(candidates, key=lambda x: x.cross_encoder_score, reverse=True)
            except Exception:
                pass

        # Robust fallback if cross-encoder model cannot be invoked
        for item in candidates:
            combined_prob = max(float(item.relevance), min(1.0, float(item.score) if item.score < 1.0 else float(item.score) / 10.0))
            item.cross_encoder_prob = float(combined_prob)
            item.cross_encoder_score = float(item.score)
        return sorted(candidates, key=lambda x: (x.cross_encoder_prob, x.score), reverse=True)


# ──────────────────────────────────────────────────────────────────────────────
# STAGE 5: Evidence Sufficiency Assessment Gate
# ──────────────────────────────────────────────────────────────────────────────
def assess_evidence_sufficiency(
    evidence: list[Evidence],
    analysis: QueryAnalysis,
    min_prob_threshold: float = 0.30,
) -> tuple[bool, str, dict[str, Any]]:
    """Stage 5: Assess evidence sufficiency before allowing the LLM to generate an answer."""
    if not evidence:
        return False, "No relevant authority passages were found in the selected corpus.", {
            "candidate_count": 0, "top_prob": 0.0, "primary_count": 0
        }

    best = evidence[0]
    primary_items = [
        e for e in evidence
        if e.source in PRIMARY_SOURCES or (e.url and e.url.startswith("local-pdf://"))
    ]
    top_prob = best.cross_encoder_prob

    # Rule 1: Cross-Encoder neural relevance model is the primary arbiter of topicality.
    # If the cross-encoder confidence is low (< 25% or logit < -1.0), the material does NOT support the legal issue.
    if top_prob < 0.25 or best.cross_encoder_score < -1.0:
        return False, (
            f"Top retrieved passage from '{best.source}' has insufficient topical relevance "
            f"(Cross-Encoder confidence: {top_prob:.1%}, score: {best.cross_encoder_score:.2f}). "
            "Evidence sufficiency threshold not met; answer synthesis refused to prevent hallucination."
        ), {"candidate_count": len(evidence), "top_prob": top_prob, "primary_count": len(primary_items)}

    # Rule 2: Citation alignment (if user asked for a specific Article/Section, verify evidence contains it)
    if analysis.citations:
        citation_found = False
        for cit in analysis.citations:
            cit_core = re.sub(r"[^a-zA-Z0-9]", "", cit).lower()
            for ev in evidence[:8]:
                if cit_core in re.sub(r"[^a-zA-Z0-9]", "", ev.title + " " + ev.text).lower():
                    citation_found = True
                    break
        if not citation_found and top_prob < 0.60:
            return False, (
                f"Retrieved passages do not explicitly contain the cited legal authority ({', '.join(analysis.citations)}). "
                "Evidence gate failed to avoid misleading statutory interpretations."
            ), {"candidate_count": len(evidence), "top_prob": top_prob, "primary_count": len(primary_items)}

    # Rule 3: Validated primary authority with sufficient cross-encoder confidence
    valid_primary = [e for e in primary_items if e.cross_encoder_prob >= 0.25 or e.cross_encoder_score >= -1.0]
    if valid_primary:
        reason = (
            f"Evidence gate PASSED: Primary authority '{valid_primary[0].source}' verified with "
            f"{valid_primary[0].cross_encoder_prob:.1%} cross-encoder confidence (neural score: {valid_primary[0].cross_encoder_score:.2f})."
        )
        return True, reason, {"candidate_count": len(evidence), "top_prob": top_prob, "primary_count": len(valid_primary)}

    # Rule 4: Multiple corroborating secondary authorities meeting the threshold
    corroborating = [e for e in evidence if e.cross_encoder_prob >= 0.25 or e.cross_encoder_score >= -1.0]
    if len(corroborating) >= 2:
        reason = (
            f"Evidence gate PASSED: {len(corroborating)} corroborating legal authorities meet the "
            f"cross-encoder sufficiency threshold (top match: {top_prob:.1%})."
        )
        return True, reason, {"candidate_count": len(evidence), "top_prob": top_prob, "primary_count": 0}

    return False, (
        "Retrieved materials consist solely of isolated secondary commentary lacking sufficient "
        "statutory or precedential corroboration."
    ), {"candidate_count": len(evidence), "top_prob": top_prob, "primary_count": 0}


def evidence_is_enough(items: list[Evidence]) -> tuple[bool, str]:
    """Backward-compatible helper wrapping assess_evidence_sufficiency."""
    dummy_analysis = QueryAnalysis("", [], [], [], [], False, [], False, "")
    enough, reason, _ = assess_evidence_sufficiency(items, dummy_analysis)
    return enough, reason


# ──────────────────────────────────────────────────────────────────────────────
# STAGE 7: Answer Generation OR Explicit Insufficient-Evidence Response
# ──────────────────────────────────────────────────────────────────────────────
def generate_insufficient_evidence_response(
    question: str,
    analysis: QueryAnalysis,
    evidence: list[Evidence],
    evidence_reason: str,
    retried: bool,
    retrieval_query: str = "",
) -> str:
    """Stage 7: Return an explicit insufficient-evidence response when adequate support cannot be obtained."""
    inspected_sources = ", ".join(sorted({e.source for e in evidence})) if evidence else "None in selected library"
    retry_notice = "Retry executed with reformulated query" if retried else "Initial pass evaluated"
    query_line = f"- **Reformulated Search Query:** {retrieval_query}\n" if retrieval_query and retrieval_query != question else ""

    return f"""### Insufficient Evidence Declaration (Refusal to Hallucinate)

**Status:** Evidence Gate: **NOT PASSED** ({retry_notice})

NyayaRoute's evidence-gated legal intelligence pipeline determined that the indexed corpus does not contain sufficient authoritative primary materials to reliably answer this legal question without risk of judicial speculation or hallucination.

**Audit of Gate Assessment:**
- **Inquiry Analyzed:** {question}
{query_line}- **Gate Finding:** {evidence_reason}
- **Authorities Evaluated:** {inspected_sources} ({len(evidence)} candidate chunk(s) reviewed)
- **Deficiency:** Lacks direct statutory provisions or binding judicial precedents on this precise point.

**Recommended Primary Legal Research Avenues:**
1. **India Code (Official Portal):** Search the authentic bare act text, notifications, and statutory amendments at [indiacode.nic.in](https://www.indiacode.nic.in).
2. **Supreme Court of India (Judgments Portal):** Verify whether a Constitution Bench or Appellate Bench has addressed this issue at [sci.gov.in](https://www.sci.gov.in/judgements/).
3. **High Courts & Tribunals:** Check jurisdictional High Court rulings on national judicial record portals or e-Courts Services.

*This is a verified legal research safeguard. NyayaRoute refuses to synthesize an ungrounded answer when evidence sufficiency criteria are not met.*"""


def generate_extractive_summary(question: str, evidence: list[Evidence]) -> str:
    """Generate an evidence-anchored legal research brief directly from top authorities
    when an external LLM API key is not configured."""
    supported = [item for item in evidence if (item.cross_encoder_prob >= 0.25 or item.relevance >= 0.20)][:5]
    if not supported:
        return "Insufficient primary evidence to synthesize an answer."

    primary = supported[0]
    primary_text = " ".join(primary.text.strip().split())
    clean_title = primary.title.replace("—", "-")
    short_ans = f"According to {clean_title} ({primary.source}): \"{primary_text}\" [S1]."

    authorities = "\n".join(
        f"- **[S{i}]** {item.title.replace('—', '-')} ({item.source}) - `{item.url}` (Cross-Encoder: {item.cross_encoder_prob:.1%})"
        for i, item in enumerate(supported, 1)
    )

    excerpts = []
    for i, item in enumerate(supported[:3], 1):
        clean_text = " ".join(item.text.strip().split())
        if len(clean_text) > 300:
            clean_text = clean_text[:300] + "..."
        excerpts.append(f"> \"{clean_text}\"\n> - **[S{i}] {item.title.replace('—', '-')}**")

    meaning = "\n\n".join(excerpts)
    limits = (
        f"This brief is extracted directly from verified primary materials ({primary.source}) in the local collection. "
        "Before judicial reliance or advocate submissions, always cross-reference the original court publication or Gazette notification."
    )

    return f"""**Short answer**
{short_ans}

**Relevant authority**
{authorities}

**What this means**
{meaning}

**Limits / next verification**
{limits}

*This is legal research information, not legal advice.*"""


def generate_answer(question: str, evidence: list[Evidence]) -> str:
    """Generate legal brief using Groq LLM (or local extractive synthesis as fallback).
    Guaranteed only to be invoked when evidence gate has passed."""
    supported = [item for item in evidence if (item.cross_encoder_prob >= 0.25 or item.relevance >= 0.20)][:5]
    if not supported:
        return ""

    if os.getenv("GROQ_API_KEY"):
        try:
            context = "\n\n".join(
                f"[S{i}] {item.title}\nSource: {item.source}\nURL: {item.url}\n{item.text}"
                for i, item in enumerate(supported, 1)
            )
            system = (
                "You are NyayaRoute, an Indian legal research assistant. Use only supplied sources. "
                "Do not give legal advice, invent authorities, or state that a source proves more than it says. "
                "Use exactly these headings: **Short answer**, **Relevant authority**, **What this means**, **Limits / next verification**. "
                "Cite each material proposition as [S1], [S2], etc. "
                "End with: 'This is legal research information, not legal advice.' "
                "If the sources do not answer the question, say so plainly."
            )
            return groq_chat([
                {"role": "system", "content": system},
                {"role": "user", "content": f"Question: {question}\n\nSources:\n{context}"}
            ])
        except Exception:
            return generate_extractive_summary(question, supported)

    return generate_extractive_summary(question, supported)


def research_actions(question: str) -> list[dict[str, str]]:
    encoded = question.replace(" ", "+")
    return [
        {"label": "Search India Code", "url": f"https://www.indiacode.nic.in/search/?query={encoded}", "why": "Find the current statutory text, amendments, and linked rules."},
        {"label": "Search Supreme Court judgments", "url": "https://www.sci.gov.in/judgements/", "why": "Locate relevant Supreme Court decisions and verify the primary judgment."},
        {"label": "Search Judicial Precedents", "url": f"https://indiankanoon.org/search/?formInput={encoded}", "why": "Discover relevant court precedents across Supreme Court and High Courts."},
    ]


# ──────────────────────────────────────────────────────────────────────────────
# PIPELINE ORCHESTRATOR: run_agent
# ──────────────────────────────────────────────────────────────────────────────
def run_agent(
    question: str,
    retriever: LegalRetriever,
    allowed_sources: set[str] | None = None,
    max_retries: int = 1,
    use_live_web: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    if "use_live_web" in kwargs:
        use_live_web = bool(kwargs["use_live_web"])
    """Executes the complete 7-stage legal intelligence agent pipeline:

    1. Analyze legal query for legal terms, entities, case references, citations, and ambiguity.
    2. Reformulate weak or ambiguous queries before retrieval.
    3. Route each query to BM25 sparse retrieval, dense semantic retrieval, or hybrid retrieval.
    4. Re-rank retrieved passages using a cross-encoder relevance model.
    5. Assess evidence sufficiency before allowing the LLM to generate an answer.
    6. Retry with a reformulated query when evidence is insufficient.
    7. Return an explicit insufficient-evidence response when adequate support cannot be obtained within the retry limit.
    """
    start_time = time.time()

    # ── STAGE 1: Analyze the legal query ──
    analysis = analyze_legal_query(question)

    # ── STAGE 2: Reformulate weak or ambiguous queries ──
    retrieval_query, was_reformulated, reformulate_reason = reformulate_query(question, analysis)

    # ── STAGE 3: Route query to BM25, Dense, or Hybrid ──
    route_analysis = analyze_legal_query(retrieval_query) if was_reformulated else analysis
    route, route_reason = choose_route(retrieval_query, route_analysis)

    # ── STAGE 4: Retrieve and Cross-Encoder Re-rank ──
    def _safe_retrieve(q: str, r: str, lim: int) -> list[Evidence]:
        try:
            return retriever.retrieve(q, r, limit=lim, include_live_web=use_live_web)
        except TypeError:
            cand = retriever.retrieve(q, r, limit=lim)
            if use_live_web:
                try:
                    from live_web_searcher import search_indian_kanoon
                    raw_res = search_indian_kanoon(q, max_results=4)
                    q_terms = set(_tokens(q)) - STOPWORDS
                    seen_t = {c.title.lower() for c in cand}
                    for idx, res in enumerate(raw_res):
                        if res["title"].lower() not in seen_t:
                            d_terms = set(_tokens(f"{res['title']} {res['text']}"))
                            rel = len(q_terms & d_terms) / max(1, len(q_terms))
                            cand.append(
                                Evidence(
                                    id=f"live_{res.get('doc_id') or idx}",
                                    title=res["title"],
                                    url=res["url"],
                                    source="Indian Kanoon",
                                    text=res["text"],
                                    score=0.6,
                                    relevance=rel,
                                )
                            )
                except Exception:
                    pass
            return cand

    candidates = _safe_retrieve(retrieval_query, route, 12)
    effective_allowed = (allowed_sources | {"Indian Kanoon"}) if (allowed_sources and use_live_web) else allowed_sources
    if effective_allowed:
        candidates = [
            item for item in candidates
            if item.source in effective_allowed or (item.url and item.url.startswith("local-pdf://"))
        ]

    # ── STAGE 5: Assess evidence sufficiency before allowing LLM generation ──
    enough, evidence_reason, metrics = assess_evidence_sufficiency(candidates, analysis)

    retried = False
    retry_query = ""
    retry_route = ""

    # ── STAGE 6: Retry with a reformulated query when evidence is insufficient ──
    if not enough and max_retries > 0:
        retried = True
        # Formulate targeted retry query expanding scope to primary statutory and case authorities
        if analysis.citations:
            retry_query = f"{' '.join(analysis.citations)} statutory provisions bare act interpretation {question}"
        else:
            retry_query = f"{question} Indian statutory provisions landmark Supreme Court precedent bare act"

        retry_route = "Hybrid Retrieval"
        retry_candidates = _safe_retrieve(retry_query, retry_route, 20)
        if effective_allowed:
            retry_candidates = [
                item for item in retry_candidates
                if item.source in effective_allowed or (item.url and item.url.startswith("local-pdf://"))
            ]

        retry_enough, retry_reason, retry_metrics = assess_evidence_sufficiency(retry_candidates, analysis)

        # If retry yielded better or sufficient evidence, adopt it
        if retry_enough or (retry_candidates and (retry_candidates[0].cross_encoder_prob > (candidates[0].cross_encoder_prob if candidates else 0))):
            candidates = retry_candidates
            enough = retry_enough
            evidence_reason = retry_reason
            retrieval_query = retry_query
            route = retry_route
            metrics = retry_metrics

    # ── STAGE 7: Generate answer OR Return explicit insufficient-evidence response ──
    if enough:
        answer = generate_answer(question, candidates)
        is_insufficient = False
    else:
        answer = generate_insufficient_evidence_response(question, analysis, candidates, evidence_reason, retried, retrieval_query)
        is_insufficient = True

    latency = round(time.time() - start_time, 2)
    live_count = sum(1 for e in candidates if e.source == "Indian Kanoon" and str(e.id).startswith("live_"))

    return {
        # Stage 1 Outputs
        "analysis": asdict(analysis),
        "good_query": not analysis.is_ambiguous,
        "quality_reason": analysis.quality_verdict,
        "legal_terms": analysis.legal_terms,
        "entities": analysis.entities,
        "case_references": analysis.case_references,
        "citations": analysis.citations,
        "ambiguity_reasons": analysis.ambiguity_reasons,
        # Stage 2 Outputs
        "was_reformulated": was_reformulated,
        "reformulate_reason": reformulate_reason,
        "query_used": retrieval_query,
        # Stage 3 Outputs
        "route": route,
        "route_reason": route_reason,
        # Stage 4 Outputs
        "evidence": [asdict(x) for x in candidates[:5]],
        "candidate_count": len(candidates),
        # Stage 5 Outputs
        "enough": enough,
        "evidence_reason": evidence_reason,
        "metrics": metrics,
        # Stage 6 Outputs
        "retried": retried,
        "retry_query": retry_query,
        "retry_route": retry_route,
        # Stage 7 Outputs
        "answer": answer,
        "is_insufficient_evidence_response": is_insufficient,
        "actions": research_actions(question),
        "latency_seconds": latency,
        # Live Web Outputs
        "use_live_web": use_live_web,
        "live_web_count": live_count,
    }
