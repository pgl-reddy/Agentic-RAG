"""Legal-aware chunking.

Instead of slicing every document into fixed-size windows, this module
looks for the structural markers that Indian legal documents actually use
— Articles (Constitution), Sections (Acts), and Facts/Issues/Held/Ratio/
Order (judgments) — and turns each structural unit into one chunk. If none
of those markers are reliably present, it falls back to paragraph-aware
fixed-size packing so ingestion never fails outright.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import re

ROMAN = r"[IVXLCDM]+"

PART_RE = re.compile(rf"(?:^|\n)\s*PART[\s\-]+({ROMAN}|[A-Z]{{1,3}})\b[.\s]*([^\n]{{0,100}})?", re.MULTILINE)
CHAPTER_RE = re.compile(rf"(?:^|\n)\s*CHAPTER[\s\-]+({ROMAN}|\d{{1,3}})\b[.\s]*([^\n]{{0,100}})?", re.MULTILINE | re.IGNORECASE)

# "Article 21. Protection of life..." — the Constitution's own numbering style.
ARTICLE_RE = re.compile(
    r"(?:^|\n)\s*Article[s]?\s+(\d{1,3}[A-Z]{0,2})\.?\s*([A-Z][^\n.]{0,150})?", re.MULTILINE
)
# Indian Constitution PDFs often print article headings as:
# "21. Protection of life and personal liberty."
# rather than "Article 21."
CONSTITUTION_NUMBERED_ARTICLE_RE = re.compile(
    r"(?:^|\n)"
    r"\s*(\d{1,3}[A-Z]?)\.\s+"
    r"([A-Z][^\n]{2,200}?)"
    r"(?:[.—\-]+|$)",
    re.MULTILINE
)
# Real Acts typically number sections as "3. Short title and commencement.—"
# rather than spelling out the word "Section". Capture both styles.
NUMBERED_SECTION_RE = re.compile(
    r"(?:^|\n)\s*(\d{1,4}[A-Za-z]{0,2})\.\s+([A-Z][^\n]{2,150}?)(?:\.[—\-]|\.\s|\n)", re.MULTILINE
)
EXPLICIT_SECTION_RE = re.compile(
    r"(?:^|\n)\s*Section[s]?\s+(\d{1,4}[A-Za-z]{0,2})\.?\s*([A-Z][^\n.]{0,150})?", re.MULTILINE
)

JUDGMENT_HEADINGS = (
    "FACTS", "BRIEF FACTS", "FACTUAL BACKGROUND", "FACTS OF THE CASE",
    "ISSUES", "ISSUE", "ISSUES FOR CONSIDERATION", "POINTS FOR DETERMINATION",
    "HELD", "HOLDING", "FINDINGS",
    "RATIO", "RATIO DECIDENDI",
    "ORDER", "CONCLUSION", "DISPOSITION", "JUDGMENT",
)
_JUDGMENT_HEADING_RE = re.compile(
    r"(?:^|\n)\s*(" + "|".join(re.escape(h) for h in JUDGMENT_HEADINGS) + r")\s*[:\.\-]?\s*(?=\n|$)",
    re.IGNORECASE,
)
_HEADING_NORMALISE = {
    "FACTS": "facts", "BRIEF FACTS": "facts", "FACTUAL BACKGROUND": "facts", "FACTS OF THE CASE": "facts",
    "ISSUES": "issues", "ISSUE": "issues", "ISSUES FOR CONSIDERATION": "issues", "POINTS FOR DETERMINATION": "issues",
    "HELD": "holding", "HOLDING": "holding", "FINDINGS": "holding",
    "RATIO": "ratio", "RATIO DECIDENDI": "ratio",
    "ORDER": "order", "CONCLUSION": "order", "DISPOSITION": "order", "JUDGMENT": "judgment",
}

MIN_CHUNK_CHARS = 40


@dataclass
class StructuralChunk:
    text: str
    part: str | None = None
    chapter: str | None = None
    article: str | None = None
    section: str | None = None
    heading: str | None = None
    structure_kind: str = "generic"
    order: int = 0
    extra_title: str = field(default="")


def detect_document_kind(text: str) -> str:
    """Detect whether a document is a Constitution, judgment, or Act."""

    article_hits = len(ARTICLE_RE.findall(text))
    constitution_numbered_hits = len(
        CONSTITUTION_NUMBERED_ARTICLE_RE.findall(text)
    )

    judgment_hits = len(_JUDGMENT_HEADING_RE.findall(text))

    numbered_section_hits = len(NUMBERED_SECTION_RE.findall(text))
    explicit_section_hits = len(EXPLICIT_SECTION_RE.findall(text))
    section_hits = max(numbered_section_hits, explicit_section_hits)

    # Explicit "Article 21" style.
    if article_hits >= 3:
        return "constitution"

    # Numbered Constitution style.
    # The presence of PART markers + many numbered article headings
    # is a strong signal that this is the Constitution.
    part_hits = len(PART_RE.findall(text))

    if constitution_numbered_hits >= 10 and part_hits >= 1:
        return "constitution"

    if judgment_hits >= 3 and judgment_hits >= article_hits:
        return "judgment"

    if section_hits >= 3:
        return "act"

    return "generic"


def _preceding_marker(matches: list[re.Match], position: int) -> str | None:
    label = None
    for match in matches:
        if match.start() > position:
            break
        label = match.group(1).strip()
    return label

def _chunk_constitution(text: str) -> list[StructuralChunk]:
    """Chunk the Constitution article-by-article.

    Handles Constitution PDFs where article headings are written as:

        1. Name and territory of the Union.—
        2. Admission or establishment of new States.—
        21. Protection of life and personal liberty.—
        21A. Right to education.—
    """

    # First try explicit "Article 21" style headings.
    explicit = list(ARTICLE_RE.finditer(text))

    if len(explicit) >= 10:
        articles = explicit
    else:
        candidates = list(
            CONSTITUTION_NUMBERED_ARTICLE_RE.finditer(text)
        )

        # Group candidates by article number.
        by_number: dict[str, list[re.Match]] = {}

        for match in candidates:
            number = match.group(1).upper()

            try:
                numeric_part = int(re.match(r"\d+", number).group())
            except Exception:
                continue

            if not (1 <= numeric_part <= 395):
                continue

            by_number.setdefault(number, []).append(match)

        # Take the last occurrence of each article.
        articles = [
            matches[-1]
            for matches in by_number.values()
        ]

        # Restore document order.
        articles.sort(key=lambda m: m.start())

    if len(articles) < 5:
        return []

    parts = list(PART_RE.finditer(text))
    chapters = list(CHAPTER_RE.finditer(text))

    chunks: list[StructuralChunk] = []

    for i, match in enumerate(articles):

        start = match.start()

        end = (
            articles[i + 1].start()
            if i + 1 < len(articles)
            else len(text)
        )

        body = text[start:end].strip()

        if len(body) < MIN_CHUNK_CHARS:
            continue

        article_number = match.group(1).strip()
        title = (match.group(2) or "").strip()

        chunks.append(
            StructuralChunk(
                text=body,
                part=_preceding_marker(parts, start),
                chapter=_preceding_marker(chapters, start),
                article=article_number,
                extra_title=title,
                structure_kind="constitution",
                order=len(chunks),
            )
        )

    return chunks

def _chunk_act(text: str) -> list[StructuralChunk]:
    numbered = list(NUMBERED_SECTION_RE.finditer(text))
    explicit = list(EXPLICIT_SECTION_RE.finditer(text))
    sections = numbered if len(numbered) >= len(explicit) else explicit
    if not sections:
        return []
    chapters = list(CHAPTER_RE.finditer(text))
    chunks: list[StructuralChunk] = []
    for i, match in enumerate(sections):
        start = match.start()
        end = sections[i + 1].start() if i + 1 < len(sections) else len(text)
        body = text[start:end].strip()
        if len(body) < MIN_CHUNK_CHARS:
            continue
        chunks.append(StructuralChunk(
            text=body,
            chapter=_preceding_marker(chapters, start),
            section=match.group(1),
            extra_title=(match.group(2) or "").strip(),
            structure_kind="act",
            order=i,
        ))
    return chunks


def _chunk_judgment(text: str) -> list[StructuralChunk]:
    headings = list(_JUDGMENT_HEADING_RE.finditer(text))
    if not headings:
        return []
    chunks: list[StructuralChunk] = []
    # Anything before the first recognised heading is preamble (cause title,
    # citation, coram) — keep it as its own chunk if substantial.
    if headings[0].start() >= MIN_CHUNK_CHARS:
        preamble = text[: headings[0].start()].strip()
        if len(preamble) >= MIN_CHUNK_CHARS:
            chunks.append(StructuralChunk(text=preamble, heading="preamble", structure_kind="judgment", order=0))
    for i, match in enumerate(headings):
        start = match.start()
        end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
        body = text[start:end].strip()
        if len(body) < MIN_CHUNK_CHARS:
            continue
        normalised = _HEADING_NORMALISE.get(match.group(1).strip().upper(), match.group(1).strip().lower())
        chunks.append(StructuralChunk(text=body, heading=normalised, structure_kind="judgment", order=i + 1))
    return chunks


def _chunk_generic(text: str, size: int = 900, overlap: int = 150) -> list[StructuralChunk]:
    """Paragraph-aware fallback: pack paragraphs into ~size-character chunks
    instead of cutting mid-sentence at a fixed offset."""
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    if not paragraphs:
        paragraphs = [text.strip()]
    chunks: list[StructuralChunk] = []
    buffer: list[str] = []
    buffer_len = 0
    order = 0
    for para in paragraphs:
        if buffer_len + len(para) > size and buffer:
            body = "\n\n".join(buffer)
            if len(body) >= MIN_CHUNK_CHARS:
                chunks.append(StructuralChunk(text=body, structure_kind="generic", order=order))
                order += 1
            # keep a small tail for overlap/context continuity
            tail = body[-overlap:] if overlap else ""
            buffer = [tail] if tail else []
            buffer_len = len(tail)
        buffer.append(para)
        buffer_len += len(para)
    if buffer:
        body = "\n\n".join(buffer).strip()
        if len(body) >= MIN_CHUNK_CHARS:
            chunks.append(StructuralChunk(text=body, structure_kind="generic", order=order))
    return chunks


def chunk_document(text: str, doc_kind: str | None = None) -> tuple[str, list[StructuralChunk]]:
    """Chunk `text` using the best-matching legal structure. Returns the
    detected/used kind alongside the chunk list. Falls back to generic
    paragraph packing if the requested/detected structure yields nothing."""
    kind = doc_kind or detect_document_kind(text)

    strategies = {
        "constitution": _chunk_constitution,
        "act": _chunk_act,
        "judgment": _chunk_judgment,
    }
    if kind in strategies:
        chunks = strategies[kind](text)
        if chunks:
            return kind, chunks

    return "generic", _chunk_generic(text)
