"""Shared embedding access point for NyayaRoute."""

from __future__ import annotations

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

_model = None


def get_embedder():
    global _model

    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    return _model


def get_embedding_dim() -> int:
    """Return the dimensionality expected by the FAISS index."""
    return EMBEDDING_DIM


def embed_texts(texts: list[str]):
    """Embed texts using the shared MiniLM model.

    Returns normalized float32 vectors compatible with FAISS
    inner-product similarity.
    """
    import numpy as np

    if not texts:
        return np.zeros((0, EMBEDDING_DIM), dtype="float32")

    model = get_embedder()

    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    return np.asarray(vectors, dtype="float32")