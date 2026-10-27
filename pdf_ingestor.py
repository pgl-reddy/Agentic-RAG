"""Orchestrates PDF ingestion: parse → detect structure → chunk → attach
metadata → append to the local JSONL corpus.

Storage stays JSONL-compatible with the existing project on purpose (see
`data/pdf_legal_corpus.jsonl`, read by rag_engine.load_corpus alongside the
URL-ingested `data/legal_corpus.jsonl`). Records use the same core fields
(id, title, url, source, text) that the existing retriever already expects,
plus the additional metadata fields requested for the PDF pipeline. This
keeps a clean seam for a later swap to FAISS/ChromaDB: only this file and
embedding.py would need to change, not rag_engine.py or app.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any

from pdf_parser import ParsedPDF, PDFIngestionError, parse_pdf
from chunker import StructuralChunk, chunk_document
from metadata import DocumentProfile, build_pseudo_url, classify_document, extract_keywords, make_chunk_id, now_iso

_DATA_DIR = Path(__file__).resolve().parent / "data"
CORPUS_PATH = _DATA_DIR / "pdf_legal_corpus.jsonl"
MANIFEST_PATH = _DATA_DIR / "pdf_manifest.json"


class DuplicatePDFError(PDFIngestionError):
    pass


@dataclass
class IngestResult:
    filename: str
    doc_profile: DocumentProfile
    structure_kind: str
    chunks_created: int
    chunks_skipped_duplicate: int
    pages_total: int
    pages_used: int
    records: list[dict[str, Any]] = field(default_factory=list)


def _file_hash(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()


def _load_manifest() -> dict[str, Any]:
    if MANIFEST_PATH.exists():
        try:
            return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def _save_manifest(manifest: dict[str, Any]) -> None:
    MANIFEST_PATH.parent.mkdir(exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def _existing_chunk_ids() -> set[str]:
    if not CORPUS_PATH.exists():
        return set()
    ids: set[str] = set()
    for line in CORPUS_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                ids.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return ids


def _chunk_to_record(
    chunk: StructuralChunk,
    order: int,
    filename: str,
    profile: DocumentProfile,
    structure_kind: str,
    page_hint: int | None,
) -> dict[str, Any]:
    title_bits = [profile.act or filename]
    if chunk.article:
        title_bits.append(f"Article {chunk.article}")
    elif chunk.section:
        title_bits.append(f"Section {chunk.section}")
    elif chunk.heading:
        title_bits.append(chunk.heading.replace("_", " ").title())
    title = " — ".join(title_bits)

    record_id = make_chunk_id(filename, order, chunk.text)
    url = build_pseudo_url(filename, page_hint, chunk.article, chunk.section)

    return {
        "id": record_id,
        "title": title,
        "act": profile.act,
        "chapter": chunk.chapter or "",
        "section": chunk.section or "",
        "article": chunk.article or "",
        "year": profile.year,
        "source": profile.source,
        "court": profile.court,
        "kind": chunk.heading if structure_kind == "judgment" else structure_kind,
        "url": url,
        "keywords": extract_keywords(chunk.text),
        "text": chunk.text,
        "filename": filename,
        "structure_kind": chunk.structure_kind,
        "ingested_at": now_iso(),
    }


def preview_pdf(
    file_bytes: bytes,
    filename: str,
    source_override: str | None = None,
    act_override: str | None = None,
) -> tuple[ParsedPDF, DocumentProfile, str, list[StructuralChunk]]:
    """Parse and chunk without writing anything — used for the Streamlit
    metadata-preview step before the user confirms indexing.

    Structural chunking (constitution/act/judgment/generic) always uses its
    own marker-counting detection over the full text — it is intentionally
    NOT driven by classify_document's act/source guess. A judgment that
    merely cites "the Constitution" once must still be chunked as a
    judgment, not forced through constitution-article splitting and dropped
    to a single generic blob. See chunker.detect_document_kind.
    """
    parsed = parse_pdf(file_bytes, filename)
    profile = classify_document(filename, parsed.full_text, source_override, act_override)
    structure_kind, chunks = chunk_document(parsed.full_text)
    return parsed, profile, structure_kind, chunks


def ingest_pdf(
    file_bytes: bytes,
    filename: str,
    allow_reingest: bool = False,
    source_override: str | None = None,
    act_override: str | None = None,
) -> IngestResult:
    """Full ingestion: validate uniqueness, parse, chunk, classify, and
    append new chunks to the PDF corpus file. Raises a PDFIngestionError
    subclass (including DuplicatePDFError) on any failure the caller should
    surface to the user."""
    file_hash = _file_hash(file_bytes)
    manifest = _load_manifest()
    if file_hash in manifest and not allow_reingest:
        raise DuplicatePDFError(
            f"'{filename}' matches an already-ingested file "
            f"({manifest[file_hash]['filename']}, {manifest[file_hash]['chunk_count']} chunks). "
            "Skipping to avoid duplicate corpus entries."
        )

    parsed, profile, structure_kind, chunks = preview_pdf(file_bytes, filename, source_override, act_override)

    existing_ids = _existing_chunk_ids()
    records: list[dict[str, Any]] = []
    skipped = 0
    for order, chunk in enumerate(chunks):
        if parsed.page_numbers:
            page_idx = min(int((order / max(1, len(chunks))) * len(parsed.page_numbers)), len(parsed.page_numbers) - 1)
            page_hint = parsed.page_numbers[page_idx]
        else:
            page_hint = None
        record = _chunk_to_record(chunk, order, filename, profile, structure_kind, page_hint)
        if record["id"] in existing_ids:
            skipped += 1
            continue
        existing_ids.add(record["id"])
        records.append(record)

    CORPUS_PATH.parent.mkdir(exist_ok=True)
    with CORPUS_PATH.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    if records:
        try:
            from vector_store import get_vector_store
            get_vector_store().add_records(records, persist=True)
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning("Vector store indexing failed for %s: %s", filename, exc)

    manifest[file_hash] = {
        "filename": filename,
        "chunk_count": len(records),
        "structure_kind": structure_kind,
        "source": profile.source,
        "ingested_at": now_iso(),
    }
    _save_manifest(manifest)

    return IngestResult(
        filename=filename,
        doc_profile=profile,
        structure_kind=structure_kind,
        chunks_created=len(records),
        chunks_skipped_duplicate=skipped,
        pages_total=parsed.num_pages_total,
        pages_used=parsed.num_pages_used,
        records=records,
    )
