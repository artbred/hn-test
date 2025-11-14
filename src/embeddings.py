from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer


def compute_sentence_embeddings(
    texts: Sequence[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 256,
    cache_path: Path | None = None,
    normalize: bool = False,
) -> np.ndarray:
    """Compute (or load cached) sentence embeddings for the provided texts."""
    texts_list = list(texts)
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            cached = np.load(cache_path)
            if cached.shape[0] == len(texts_list):
                return cached
            cache_path.unlink()

    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts_list,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
    )

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, embeddings)

    return embeddings

