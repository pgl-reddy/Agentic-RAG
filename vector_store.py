"""High-level vector store combining faiss_store.py (vectors) and
metadata_store.py (everything else) into the single object the rest of
the project talks to.

This is the seam requested by the "replace JSON retrieval with FAISS"
task: rag_engine.LegalRetriever.vector_search() calls VectorStore.search()
instead of embedding the whole corpus in memory on first use; PDF and URL
ingestion call VectorStore.add_records() instead of relying on the next
BM25 corpus reload to pick up new vector-searchable content.

Persistence: `data/faiss.index` (vectors, via faiss_store.py) and
`data/metadata.pkl` (everything else, via metadata_store.py) are always
written together, so they can never drift out of sync on disk. In memory,
one VectorStore instance is shared process-wide (see get_vector_store()
below) and mutated in place by add_records(), so incremental ingestion
never re-embeds or reloads anything that was already indexed.
"""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

# Ensure local .venv site-packages are accessible even if run with global python
_venv_site = Path(__file__).resolve().parent / ".venv" / "Lib" / "site-packages"
if _venv_site.exists() and str(_venv_site) not in sys.path:
    sys.path.insert(0, str(_venv_site))

import numpy as np

from embedding import embed_texts, get_embedding_dim
from faiss_store import FaissStore
from metadata_store import MetadataStore

DATA_DIR = Path(__file__).resolve().parent / "data"
FAISS_INDEX_PATH = DATA_DIR / "faiss.index"
METADATA_PATH = DATA_DIR / "metadata.pkl"


class VectorStore:
    def __init__(self, faiss_store: FaissStore, metadata_store: MetadataStore):
        self.faiss_store = faiss_store
        self.metadata_store = metadata_store

    @property
    def size(self) -> int:
        return len(self.metadata_store)

    def add_records(self, records: list[dict[str, Any]], persist: bool = True) -> int:
        """Embed and add only the records not already present (by their
        string 'id'). Mutates this instance's FAISS index and metadata in
        place — this is the incremental-indexing path: existing vectors
        are never recomputed or touched. Returns the number actually
        added. Safe to call with an empty or all-duplicate list."""
        new_records = []
        new_faiss_ids = []
        for record in records:
            faiss_id = self.metadata_store.add(record)
            if faiss_id is not None:
                new_records.append(self.metadata_store.get_by_faiss_id(faiss_id))
                new_faiss_ids.append(faiss_id)

        if not new_records:
            return 0

        vectors = embed_texts([r["text"] for r in new_records])
        self.faiss_store.add(np.array(new_faiss_ids, dtype="int64"), vectors)

        if persist:
            self.save()
        return len(new_records)

    def search(self, query: str, k: int) -> list[tuple[dict[str, Any], float]]:
        """Embed `query` and return up to k (record, similarity_score)
        pairs, highest similarity first."""
        if self.size == 0:
            return []
        query_vector = embed_texts([query])[0]
        scores, ids = self.faiss_store.search(query_vector, k)
        results = []
        for score, faiss_id in zip(scores, ids):
            record = self.metadata_store.get_by_faiss_id(int(faiss_id))
            if record is not None:
                results.append((record, float(score)))
        return results

    def save(self) -> None:
        self.faiss_store.save(FAISS_INDEX_PATH)
        self.metadata_store.save(METADATA_PATH)

    @classmethod
    def load_or_create(cls, dim: int | None = None) -> "VectorStore":
        dim = dim or get_embedding_dim()
        if FAISS_INDEX_PATH.exists() and METADATA_PATH.exists():
            faiss_store = FaissStore.load(FAISS_INDEX_PATH, dim)
            metadata_store = MetadataStore.load(METADATA_PATH)
            return cls(faiss_store, metadata_store)
        else:
            faiss_store = FaissStore.create(dim)
            metadata_store = MetadataStore.create()
            store = cls(faiss_store, metadata_store)
            try:
                import json
                initial_records = []
                for filename in ("demo_legal_corpus.jsonl", "legal_corpus.jsonl", "pdf_legal_corpus.jsonl"):
                    f = DATA_DIR / filename
                    if f.exists():
                        for line in f.read_text(encoding="utf-8").splitlines():
                            if line.strip():
                                initial_records.append(json.loads(line))
                if initial_records:
                    store.add_records(initial_records, persist=True)
            except Exception:
                pass
            return store


# ── Process-wide singleton ──────────────────────────────────────────────
# Loaded once (from disk if a prior index exists, otherwise empty) and
# reused for the lifetime of the process. Ingestion mutates this exact
# instance, so newly indexed chunks are searchable immediately without a
# reload — satisfying "load once, avoid rebuilding embeddings
# unnecessarily" without needing a cache-invalidation story.
_STORE: VectorStore | None = None


def get_vector_store() -> VectorStore:
    global _STORE
    if _STORE is None:
        _STORE = VectorStore.load_or_create()
    return _STORE


def reset_vector_store_cache() -> None:
    """Drop the in-memory singleton so the next get_vector_store() call
    reloads from disk. Mainly useful for tests and for picking up an index
    rebuilt by migrate_to_faiss.py in a separate process."""
    global _STORE
    _STORE = None
