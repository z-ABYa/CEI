"""
utils/retrieval.py
===================
Handles similarity search against the persisted FAISS index: given a
user question, embed it and return the top-k most relevant chunks
along with their similarity scores and metadata.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import faiss
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import Config, get_logger  # noqa: E402
from utils.chunking import Chunk  # noqa: E402
from utils.embedding import embed_single_text  # noqa: E402

logger = get_logger(__name__)


@dataclass
class RetrievalResult:
    """
    A single retrieved chunk paired with its similarity score.

    Attributes
    ----------
    chunk : Chunk
        The retrieved chunk (text + metadata).
    score : float
        Similarity score in ``[-1, 1]`` (cosine similarity, since
        embeddings are L2-normalized and the index uses inner product).
        Higher is more similar.
    rank : int
        1-indexed rank of this result among the returned top-k.
    """

    chunk: Chunk
    score: float
    rank: int


class Retriever:
    """
    Thin wrapper around a FAISS index + aligned chunk list that
    performs top-k semantic search for a given query.

    Keeping this as a class (rather than free functions) lets the
    Streamlit app hold a single retriever instance in session state,
    avoiding repeated index loads across reruns.
    """

    def __init__(self, index: faiss.Index, chunks: List[Chunk]):
        if index.ntotal != len(chunks):
            raise ValueError(
                f"Index/metadata mismatch: FAISS index has {index.ntotal} "
                f"vectors but {len(chunks)} chunks were provided."
            )
        self.index = index
        self.chunks = chunks
        logger.info("Retriever initialized with %d chunks.", len(chunks))

    @classmethod
    def from_disk(cls) -> "Retriever":
        """
        Convenience constructor that loads the FAISS index and chunk
        metadata from the paths defined in ``Config`` and wraps them
        in a ``Retriever``.
        """
        # Imported lazily to avoid a circular import (ingest -> retrieval
        # would otherwise import each other at module load time).
        from ingest import load_index

        index, chunks = load_index()
        return cls(index, chunks)

    def search(
        self,
        query: str,
        top_k: int = Config.TOP_K,
        similarity_threshold: float = Config.SIMILARITY_THRESHOLD,
    ) -> Tuple[List[RetrievalResult], float]:
        """
        Retrieve the top-k chunks most similar to ``query``.

        Parameters
        ----------
        query : str
            The user's natural-language question.
        top_k : int
            Number of chunks to retrieve.
        similarity_threshold : float
            Minimum cosine similarity required to keep a result.
            ``0.0`` (the default) disables filtering.

        Returns
        -------
        Tuple[List[RetrievalResult], float]
            The ranked retrieval results, and the wall-clock retrieval
            latency in seconds (embedding + FAISS search combined).
        """
        if not query or not query.strip():
            raise ValueError("Query must be a non-empty string.")

        if self.index.ntotal == 0:
            logger.warning("Retriever called on an empty index.")
            return [], 0.0

        start_time = time.time()

        query_vector = embed_single_text(query)
        query_vector = np.expand_dims(query_vector, axis=0)  # FAISS expects 2-D input

        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_vector, k)

        elapsed = time.time() - start_time

        results: List[RetrievalResult] = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
            if idx < 0:
                # FAISS pads with -1 when fewer than k results exist.
                continue
            if score < similarity_threshold:
                continue
            results.append(
                RetrievalResult(chunk=self.chunks[idx], score=float(score), rank=rank)
            )

        logger.info(
            "Retrieved %d/%d chunks for query '%s...' in %.3fs.",
            len(results), k, query[:50], elapsed,
        )
        return results, elapsed


def format_context(results: List[RetrievalResult], max_chunks: int = Config.MAX_CONTEXT_CHUNKS) -> str:
    """
    Concatenate retrieved chunks into a single context string suitable
    for insertion into the LLM prompt.

    Each chunk is prefixed with a numbered source tag (e.g. ``[Source 1]``)
    so the LLM's answer -- and the UI's citation display -- can refer
    back to specific chunks unambiguously.

    Parameters
    ----------
    results : List[RetrievalResult]
        Ranked retrieval results, typically from ``Retriever.search``.
    max_chunks : int
        Maximum number of chunks to include in the context, even if
        more were retrieved (keeps the prompt within a reasonable
        token budget).

    Returns
    -------
    str
        The formatted context block.
    """
    if not results:
        return ""

    pieces = []
    for result in results[:max_chunks]:
        title = result.chunk.metadata.get("title") or result.chunk.doc_id
        pieces.append(f"[Source {result.rank}: {title}]\n{result.chunk.text}")

    return "\n\n".join(pieces)


if __name__ == "__main__":
    # Quick manual smoke test: `python utils/retrieval.py`
    # Requires that `python ingest.py` has already been run.
    retriever = Retriever.from_disk()
    demo_query = "What is retrieval-augmented generation?"
    demo_results, latency = retriever.search(demo_query, top_k=3)

    print(f"Query: {demo_query}")
    print(f"Retrieval latency: {latency:.3f}s\n")
    for r in demo_results:
        print(f"  #{r.rank} (score={r.score:.3f}) [{r.chunk.chunk_id}] {r.chunk.text[:100]}...")
