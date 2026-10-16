"""Persistent dense-vector index with FAISS and portable NumPy backends."""
from __future__ import annotations

from pathlib import Path

import numpy as np


class FaissStore:
    """Use FAISS when installed; otherwise use exact cosine search in NumPy.

    Embeddings are L2-normalized by embedding.py, so inner product equals
    cosine similarity. The NumPy path is exact and works well for the current
    12k-record corpus without a platform-specific FAISS wheel.
    """

    def __init__(self, dim: int, index=None, ids: np.ndarray | None = None, vectors: np.ndarray | None = None):
        self.dim = dim
        self._faiss = None
        self.index = index
        self.ids = np.asarray(ids if ids is not None else [], dtype="int64")
        self.vectors = np.asarray(vectors if vectors is not None else np.zeros((0, dim)), dtype="float32").reshape(-1, dim)
        try:
            import faiss
            self._faiss = faiss
            if index is None:
                self.index = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))
                if len(self.ids):
                    self.index.add_with_ids(self.vectors, self.ids)
        except ImportError:
            if index is not None:
                raise RuntimeError("A FAISS index was loaded but faiss-cpu is unavailable.")
            self._faiss = None

    @property
    def ntotal(self) -> int:
        return int(self.index.ntotal) if self._faiss is not None else int(len(self.ids))

    def add(self, ids: np.ndarray, vectors: np.ndarray) -> None:
        ids = np.asarray(ids, dtype="int64").reshape(-1)
        vectors = np.asarray(vectors, dtype="float32")
        if not len(ids):
            return
        if vectors.ndim != 2 or vectors.shape != (len(ids), self.dim):
            raise ValueError(f"Expected vectors with shape ({len(ids)}, {self.dim}), got {vectors.shape}")
        if self._faiss is not None:
            self.index.add_with_ids(vectors, ids)
        self.ids = np.concatenate((self.ids, ids))
        self.vectors = np.concatenate((self.vectors, vectors), axis=0)

    def search(self, query_vector: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        if self.ntotal == 0 or k <= 0:
            return np.array([], dtype="float32"), np.array([], dtype="int64")
        query = np.asarray(query_vector, dtype="float32").reshape(1, -1)
        if query.shape[1] != self.dim:
            raise ValueError(f"Expected query dimension {self.dim}, got {query.shape[1]}")
        if self._faiss is not None:
            scores, ids = self.index.search(query, k)
            scores, ids = scores[0], ids[0]
            mask = ids >= 0
            return scores[mask], ids[mask]
        scores = self.vectors @ query[0]
        count = min(k, len(scores))
        top = np.argpartition(scores, -count)[-count:]
        top = top[np.argsort(scores[top])[::-1]]
        return scores[top].astype("float32"), self.ids[top].astype("int64")

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self._faiss is not None:
            self._faiss.write_index(self.index, str(path))
        else:
            with path.open("wb") as handle:
                np.savez_compressed(handle, ids=self.ids, vectors=self.vectors, dim=np.asarray([self.dim]))

    @classmethod
    def load(cls, path: str | Path, dim: int) -> "FaissStore":
        path = Path(path)
        try:
            state = np.load(path, allow_pickle=False)
        except (OSError, ValueError, EOFError):
            state = None
        if isinstance(state, np.lib.npyio.NpzFile):
            with state:
                if {"ids", "vectors", "dim"}.issubset(state.files):
                    stored_dim = int(state["dim"][0])
                    if stored_dim != dim:
                        raise ValueError(f"Stored vector dimension {stored_dim} does not match expected {dim}")
                    return cls(dim=dim, ids=state["ids"], vectors=state["vectors"])
        try:
            import faiss
        except ImportError as exc:
            raise RuntimeError("This FAISS index was created by FAISS, but faiss-cpu is unavailable. Install faiss-cpu or rebuild the index.") from exc
        index = faiss.read_index(str(path))
        return cls(dim=dim, index=index)

    @classmethod
    def create(cls, dim: int) -> "FaissStore":
        return cls(dim=dim)
