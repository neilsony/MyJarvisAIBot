"""Local text embeddings.

`bge-small-en-v1.5` — 384 dimensions, ~130MB, Apache 2.0. Runs on CPU fast
enough on both the M4 and the Pi 5, and costs nothing per call. Embedding 25k
Canon chunks through a hosted API would be a real bill for no quality gain at
this scale.

The model loads once and is cached — first call pays ~2s of startup.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

__all__ = ["EMBEDDING_DIM", "EMBEDDING_MODEL", "embed", "embed_one"]

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384


@lru_cache(maxsize=1)
def _model() -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - optional extra
        raise RuntimeError(
            "sentence-transformers is not installed. Run: pip install -e '.[brain]'"
        ) from exc
    return SentenceTransformer(EMBEDDING_MODEL)


def embed(texts: list[str]) -> list[np.ndarray]:
    """Embed a batch. Batching matters — one call for 500 chunks, not 500 calls."""
    if not texts:
        return []
    vectors = _model().encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return [np.asarray(v, dtype=np.float32) for v in vectors]


def embed_one(text: str) -> np.ndarray:
    """Embed a single query.

    BGE models are trained with an asymmetric prefix: queries get one, documents
    don't. Skipping it measurably degrades retrieval, and the failure is silent —
    you just get slightly worse passages forever.
    """
    prefixed = f"Represent this sentence for searching relevant passages: {text}"
    return embed([prefixed])[0]
