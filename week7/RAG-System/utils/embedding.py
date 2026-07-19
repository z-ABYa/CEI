"""
utils/embedding.py
===================
Wraps a ``sentence-transformers`` model to turn chunk text into dense
vector embeddings. Handles model caching (so we only ever load the
model once per process) and batched inference for efficiency.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import List, Optional

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import Config, get_logger  # noqa: E402

logger = get_logger(__name__)

# Module-level cache so repeated calls (e.g. from Streamlit reruns)
# don't reload the model from disk every time.
_model_cache: dict = {}
_model_lock = threading.Lock()


def _resolve_device() -> str:
    """
    Pick the best available device for embedding inference.

    Falls back to CPU if torch/CUDA is unavailable, so the project
    keeps running on machines without a GPU, per the "free / offline
    models only" requirement.
    """
    try:
        import torch

        if Config.EMBEDDING_DEVICE == "cuda" and torch.cuda.is_available():
            return "cuda"
        if torch.cuda.is_available():
            logger.info("CUDA GPU detected; using it for embeddings.")
            return "cuda"
    except ImportError:  # pragma: no cover
        pass
    return "cpu"


def get_embedding_model(model_name: str = Config.EMBEDDING_MODEL):
    """
    Load (or retrieve from cache) a ``SentenceTransformer`` model.

    This function is thread-safe and idempotent: calling it multiple
    times with the same ``model_name`` returns the already-loaded
    instance instead of re-downloading/re-initializing the model.

    Parameters
    ----------
    model_name : str
        Hugging Face model identifier, e.g.
        ``"sentence-transformers/all-MiniLM-L6-v2"``.

    Returns
    -------
    SentenceTransformer
        The loaded embedding model, moved to the resolved device.
    """
    if model_name in _model_cache:
        return _model_cache[model_name]

    with _model_lock:
        # Double-checked locking: another thread may have populated the
        # cache while we were waiting for the lock.
        if model_name in _model_cache:
            return _model_cache[model_name]

        from sentence_transformers import SentenceTransformer

        device = _resolve_device()
        logger.info("Loading embedding model '%s' on device '%s'...", model_name, device)

        try:
            model = SentenceTransformer(model_name, device=device)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Failed to load embedding model '{model_name}'. Ensure you "
                f"have an internet connection for the first download, or "
                f"that the model is cached locally. Original error: {exc}"
            ) from exc

        _model_cache[model_name] = model
        logger.info("Embedding model '%s' loaded successfully.", model_name)
        return model


def embed_texts(
    texts: List[str],
    model_name: str = Config.EMBEDDING_MODEL,
    batch_size: int = Config.EMBEDDING_BATCH_SIZE,
    show_progress: bool = False,
    normalize: bool = True,
) -> np.ndarray:
    """
    Generate embeddings for a list of texts, in batches.

    Parameters
    ----------
    texts : List[str]
        Input strings to embed (typically chunk texts).
    model_name : str
        Which sentence-transformer model to use.
    batch_size : int
        Number of texts encoded per forward pass.
    show_progress : bool
        Whether to display a tqdm progress bar during encoding.
    normalize : bool
        If True, L2-normalize embeddings so that inner-product search
        in FAISS is equivalent to cosine similarity search.

    Returns
    -------
    np.ndarray
        Array of shape ``(len(texts), embedding_dim)`` and dtype
        ``float32`` (FAISS requires float32).
    """
    if not texts:
        return np.zeros((0, Config.EMBEDDING_DIM), dtype=np.float32)

    model = get_embedding_model(model_name)

    logger.info(
        "Embedding %d texts (batch_size=%d, normalize=%s)...",
        len(texts), batch_size, normalize,
    )

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
    )

    embeddings = embeddings.astype(np.float32)
    logger.info("Generated embeddings with shape %s.", embeddings.shape)
    return embeddings


def embed_single_text(
    text: str,
    model_name: str = Config.EMBEDDING_MODEL,
    normalize: bool = True,
) -> np.ndarray:
    """
    Convenience wrapper to embed a single piece of text (e.g. a user
    query) without needing to wrap it in a list manually.

    Parameters
    ----------
    text : str
        The text to embed.
    model_name : str
        Which sentence-transformer model to use.
    normalize : bool
        Whether to L2-normalize the resulting vector.

    Returns
    -------
    np.ndarray
        A 1-D array of shape ``(embedding_dim,)``.
    """
    embeddings = embed_texts([text], model_name=model_name, normalize=normalize)
    return embeddings[0]


def get_embedding_dimension(model_name: str = Config.EMBEDDING_MODEL) -> int:
    """
    Return the output vector dimensionality of the given embedding model.

    Useful for sanity-checking FAISS index dimensions match the model
    actually in use, especially if someone swaps ``EMBEDDING_MODEL`` in
    ``config.py``.
    """
    model = get_embedding_model(model_name)
    dim: Optional[int] = model.get_sentence_embedding_dimension()
    if dim is None:  # pragma: no cover - defensive fallback
        dim = Config.EMBEDDING_DIM
    return dim


if __name__ == "__main__":
    # Quick manual smoke test: `python utils/embedding.py`
    sample_texts = [
        "Retrieval-Augmented Generation combines retrieval with generation.",
        "FAISS is a library for efficient similarity search.",
    ]
    vectors = embed_texts(sample_texts, show_progress=True)
    print(f"Embedding shape: {vectors.shape}")
    print(f"Model dimension: {get_embedding_dimension()}")
