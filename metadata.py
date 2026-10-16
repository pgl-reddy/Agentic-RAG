"""Metadata extraction for PDF-ingested legal chunks.

Turns a filename + a bit of the document's own text into the metadata
schema the ingestion pipeline stores per chunk, and classifies the
document into the same `source` categories the existing retrieval /
evidence-gate logic in rag_engine.py already understands (so PDF-ingested
primary legislation is treated as a primary source exactly like
URL-ingested India Code pages are).
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from datetime import datetime, timezone

LEGAL_KEYWORDS = {
    "act", "article", "section", "constitution", "code", "ipc", "bns", "bnss", "bsa",
    "judgment", "case", "court", "writ", "bail", "appeal", "petition", "precedent",
    "ruling", "amendment", "rules", "notification", "gazette", "ordinance", "schedule",
    "fundamental", "rights", "liberty", "penalty", "offence", "punishment", "definitions",
}

_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")

# (match pattern on filename/title, canonical act name, source category)
KNOWN_ACTS: list[tuple[re.Pattern, str, str]] = [
    (re.compile(r"constitution", re.I), "Constitution of India", "India Code"),
    (re.compile(r"bharatiya\s*nyaya\s*sanhita|\bbns\b", re.I), "Bharatiya Nyaya Sanhita, 2023", "India Code"),
    (re.compile(r"bharatiya\s*nagarik\s*suraksha|\bbnss\b", re.I), "Bharatiya Nagarik Suraksha Sanhita, 2023", "India Code"),
    (re.compile(r"bharatiya\s*sakshya|\bbsa\b", re.I), "Bharatiya Sakshya Adhiniyam, 2023", "India Code"),
    (re.compile(r"motor\s*vehicles", re.I), "Motor Vehicles Act", "India Code"),
    (re.compile(r"information\s*technology\s*act|\bit\s*act\b", re.I), "Information Technology Act, 2000", "India Code"),
    (re.compile(r"digital\s*personal\s*data\s*protection|\bdpdp\b", re.I), "Digital Personal Data Protection Act, 2023", "India Code"),
    (re.compile(r"indian\s*penal\s*code|\bipc\b", re.I), "Indian Penal Code", "India Code"),
    (re.compile(r"code\s*of\s*criminal\s*procedure|\bcrpc\b", re.I), "Code of Criminal Procedure", "India Code"),
    (re.compile(r"law\s*commission", re.I), "Law Commission Report", "Law Commission of India"),
]

_SUPREME_COURT_RE = re.compile(r"supreme\s*court|\bsc\b(?!\w)", re.I)
_HIGH_COURT_RE = re.compile(r"high\s*court|\bhc\b(?!\w)", re.I)


@dataclass
class DocumentProfile:
    act: str
    year: str
    source: str
    court: str
    kind: str  # "constitution" | "act" | "judgment" | "gazette" | "report" | "material"


def classify_document(
    filename: str,
    sample_text: str,
    source_override: str | None = None,
    act_override: str | None = None,
) -> DocumentProfile:
    """Best-effort classification from filename + the document's own title
    banner. Deliberately narrow: official PDFs name themselves ("THE
    CONSTITUTION OF INDIA", "IN THE SUPREME COURT OF INDIA") in their first
    few lines, so we only look there. Scanning deep into the body text
    causes false positives — e.g. a judgment that merely *cites* "the
    Constitution" or mentions "notification" in an ordinary clause must not
    be reclassified as that kind of document."""
    haystack = f"{filename} {sample_text[:400]}"

    if source_override:
        year_match = _YEAR_RE.search(act_override or "") or _YEAR_RE.search(filename)
        kind = "constitution" if act_override and "constitution" in act_override.lower() else (
            "judgment" if source_override in {"Supreme Court of India", "High Court"} else "act"
        )
        return DocumentProfile(
            act=act_override or "",
            year=year_match.group(0) if year_match else "",
            source=source_override,
            court=source_override if source_override in {"Supreme Court of India", "High Court"} else "",
            kind=kind,
        )

    for pattern, canonical_name, source in KNOWN_ACTS:
        if pattern.search(haystack):
            year_match = _YEAR_RE.search(canonical_name) or _YEAR_RE.search(filename)
            kind = "constitution" if "constitution" in canonical_name.lower() else (
                "report" if source == "Law Commission of India" else "act"
            )
            return DocumentProfile(
                act=canonical_name,
                year=year_match.group(0) if year_match else "",
                source=source,
                court="",
                kind=kind,
            )

    if _SUPREME_COURT_RE.search(haystack):
        year_match = _YEAR_RE.search(filename) or _YEAR_RE.search(sample_text[:500])
        return DocumentProfile(act="", year=year_match.group(0) if year_match else "", source="Supreme Court of India", court="Supreme Court of India", kind="judgment")

    if _HIGH_COURT_RE.search(haystack):
        year_match = _YEAR_RE.search(filename) or _YEAR_RE.search(sample_text[:500])
        return DocumentProfile(act="", year=year_match.group(0) if year_match else "", source="High Court", court="High Court", kind="judgment")

    if re.search(r"gazette", haystack, re.I):
        year_match = _YEAR_RE.search(filename) or _YEAR_RE.search(sample_text[:500])
        return DocumentProfile(act="", year=year_match.group(0) if year_match else "", source="Gazette of India", court="", kind="gazette")

    # Unrecognised source: index it, but never mark it as a primary source —
    # the evidence gate in rag_engine.py should not treat "uploaded PDF of
    # unknown provenance" the way it treats India Code / Supreme Court text.
    return DocumentProfile(act="", year="", source="Uploaded PDF", court="", kind="material")


def extract_keywords(text: str, limit: int = 8) -> list[str]:
    tokens = re.findall(r"[A-Za-z]{4,}", text.lower())
    seen: list[str] = []
    for token in tokens:
        if token in LEGAL_KEYWORDS and token not in seen:
            seen.append(token)
        if len(seen) >= limit:
            break
    # Also capture capitalised multi-word phrases (likely defined terms /
    # proper nouns) as extra signal, without duplicating.
    for phrase in re.findall(r"\b(?:[A-Z][a-z]+\s){1,3}[A-Z][a-z]+\b", text)[:20]:
        key = phrase.lower()
        if key not in seen and len(seen) < limit:
            seen.append(phrase)
    return seen


def make_chunk_id(filename: str, order: int, text: str) -> str:
    digest = hashlib.sha256(f"{filename}:{order}:{text}".encode("utf-8")).hexdigest()
    return digest[:16]


def build_pseudo_url(filename: str, page: int | None, article: str | None, section: str | None) -> str:
    """PDFs have no public URL. Build a stable local reference so the
    existing UI (which links out to `item['url']`) still has something
    meaningful to display, and so identical chunks re-ingested from the
    same file collide predictably for duplicate detection."""
    fragment = ""
    if article:
        fragment = f"#article-{article}"
    elif section:
        fragment = f"#section-{section}"
    elif page:
        fragment = f"#page-{page}"
    return f"local-pdf://{filename}{fragment}"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
