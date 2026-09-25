"""Sentence embeddings for prompts, with a disk cache.

The routers never see raw text: they see a fixed-size vector per prompt. We use
``all-MiniLM-L6-v2`` (384 dims, ~90MB) because it runs on a laptop in seconds.
The paper used OpenAI's ``text-embedding-3-small`` for the same purpose, which
costs money per call and needs an API key.
"""

from __future__ import annotations

import hashlib
import pathlib

import numpy as np

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CACHE_DIR = pathlib.Path(__file__).resolve().parents[1] / "data" / "emb_cache"

_model = None


def get_model():
    """Load the embedding model once and reuse it (it is ~90MB on disk)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed(texts: list[str], batch_size: int = 256, cache_key: str | None = None) -> np.ndarray:
    """Embed texts and L2-normalise, so a dot product is the cosine similarity.

    ``cache_key`` stores the result under data/emb_cache so repeated runs (and
    the API server's start-up) do not re-encode the whole training set.
    """
    if cache_key is not None:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha1(f"{cache_key}:{len(texts)}".encode()).hexdigest()[:16]
        path = CACHE_DIR / f"{cache_key}-{digest}.npy"
        if path.exists():
            return np.load(path)

    vectors = get_model().encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    vectors = vectors.astype(np.float32)

    if cache_key is not None:
        np.save(path, vectors)
    return vectors
