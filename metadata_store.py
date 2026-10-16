"""Metadata storage for FAISS-indexed chunks, kept separate from the
vectors themselves (per the project's storage split: `data/faiss.index`
holds only vectors, `data/metadata.pkl` holds everything else).

FAISS only accepts int64 ids, but every other part of this project
(rag_engine.py's Evidence dataclass, the JSONL corpora, the Streamlit UI,
duplicate detection in pdf_ingestor.py / trusted_sources.py) identifies a
chunk by its existing 16-character string `id` (a content hash). This
module owns the one-way mapping from that string id to a stable, unique
int64 FAISS id, so a chunk can be looked up either way and is never
double-embedded on repeat ingestion.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import pickle
from typing import Any

# The metadata schema every record is normalised to. Extra keys a record
# already has (e.g. "filename", "structure_kind", "ingested_at" from PDF
# ingestion) are preserved as-is, not stripped.
SCHEMA_FIELDS = (
    "id", "title", "act", "chapter", "section", "article", "year",
    "source", "court", "kind", "url", "keywords", "text",
)


def normalise_record(record: dict[str, Any]) -> dict[str, Any]:
    """Fill any missing schema field with its type-appropriate empty
    value, without dropping fields the record already has."""
    defaults = {
        "id": "", "title": "", "act": "", "chapter": "", "section": "",
        "article": "", "year": "", "source": "", "court": "", "kind": "",
        "url": "", "keywords": [], "text": "",
    }
    merged = {**defaults, **record}
    return merged


@dataclass
class MetadataStore:
    records: dict[int, dict[str, Any]] = field(default_factory=dict)
    id_to_faiss_id: dict[str, int] = field(default_factory=dict)
    next_id: int = 0

    def __len__(self) -> int:
        return len(self.records)

    def has(self, string_id: str) -> bool:
        return string_id in self.id_to_faiss_id

    def get_by_faiss_id(self, faiss_id: int) -> dict[str, Any] | None:
        return self.records.get(int(faiss_id))

    def get_by_string_id(self, string_id: str) -> dict[str, Any] | None:
        faiss_id = self.id_to_faiss_id.get(string_id)
        return self.records.get(faiss_id) if faiss_id is not None else None

    def add(self, record: dict[str, Any]) -> int | None:
        """Register a new record. Returns the assigned FAISS id, or None
        if a record with this string id is already present (caller should
        skip embedding/adding it to the vector index in that case)."""
        record = normalise_record(record)
        string_id = record["id"]
        if not string_id:
            raise ValueError("Record is missing its string 'id' field.")
        if string_id in self.id_to_faiss_id:
            return None
        faiss_id = self.next_id
        self.next_id += 1
        self.records[faiss_id] = record
        self.id_to_faiss_id[string_id] = faiss_id
        return faiss_id

    def all_records(self) -> list[dict[str, Any]]:
        return list(self.records.values())

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(
                {"records": self.records, "id_to_faiss_id": self.id_to_faiss_id, "next_id": self.next_id},
                handle,
            )

    @classmethod
    def load(cls, path: str | Path) -> "MetadataStore":
        path = Path(path)
        with path.open("rb") as handle:
            state = pickle.load(handle)
        return cls(records=state["records"], id_to_faiss_id=state["id_to_faiss_id"], next_id=state["next_id"])

    @classmethod
    def create(cls) -> "MetadataStore":
        return cls()
