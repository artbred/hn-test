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
    *,
    trust_remote_code: bool = False,
    device: str | None = None,
    max_seq_length: int | None = None,
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

    model = SentenceTransformer(
        model_name,
        trust_remote_code=trust_remote_code,
        device=device,
    )
    if max_seq_length is not None:
        try:
            model.max_seq_length = max_seq_length
        except AttributeError:
            pass
        tokenizer = getattr(model, "tokenizer", None)
        if tokenizer is not None:
            tokenizer.model_max_length = max_seq_length
            tokenizer.init_kwargs["model_max_length"] = max_seq_length
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

