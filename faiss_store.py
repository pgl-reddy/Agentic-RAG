"""Thin, low-level wrapper around a FAISS index.

Deliberately dumb: this module knows nothing about legal documents,
chunks, or metadata. It only knows how to hold vectors against int64 ids,
persist itself to disk, and search. `vector_store.py` is the layer that
gives those ids meaning.

Uses IndexFlatIP (exact inner-product search) wrapped in IndexIDMap2 so
each vector keeps a stable, caller-assigned int64 id across incremental
adds — required for `metadata_store.py` to map a FAISS hit back to its
record without relying on insertion order. Vectors are expected to already
be L2-normalised (see embedding.py), which makes inner product equivalent
to cosine similarity. IndexFlatIP is exact (no approximation/training
step), which is the right tradeoff at this corpus's scale — a future
larger deployment could swap in IndexIVFFlat behind this same interface
without touching vector_store.py.
"""
from __future__ import annotations

from pathlib import Path
import sys

# Ensure local .venv site-packages are accessible even if run with global python
_venv_site = Path(__file__).resolve().parent / ".venv" / "Lib" / "site-packages"
if _venv_site.exists() and str(_venv_site) not in sys.path:
    sys.path.insert(0, str(_venv_site))

import numpy as np


class FaissStore:
    def __init__(self, dim: int, index=None):
        import faiss
        self.dim = dim
        self.index = index if index is not None else faiss.IndexIDMap2(faiss.IndexFlatIP(dim))

    @property
    def ntotal(self) -> int:
        return int(self.index.ntotal)

    def add(self, ids: np.ndarray, vectors: np.ndarray) -> None:
        if len(ids) == 0:
            return
        if vectors.shape[1] != self.dim:
            raise ValueError(f"Vector dim {vectors.shape[1]} does not match index dim {self.dim}")
        self.index.add_with_ids(vectors.astype("float32"), np.asarray(ids, dtype="int64"))

    def search(self, query_vector: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """query_vector: shape (dim,) or (1, dim). Returns (scores, ids),
        each shape (k',) with k' <= k. ids of -1 mean 'no result' (empty
        or under-filled index) and are filtered out before returning."""
        if self.ntotal == 0:
            return np.array([], dtype="float32"), np.array([], dtype="int64")
        query = np.asarray(query_vector, dtype="float32").reshape(1, -1)
        scores, ids = self.index.search(query, k)
        scores, ids = scores[0], ids[0]
        mask = ids >= 0
        return scores[mask], ids[mask]

    def save(self, path: str | Path) -> None:
        import faiss
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path))

    @classmethod
    def load(cls, path: str | Path, dim: int) -> "FaissStore":
        import faiss
        index = faiss.read_index(str(path))
        return cls(dim=dim, index=index)

    @classmethod
    def create(cls, dim: int) -> "FaissStore":
        return cls(dim=dim)
