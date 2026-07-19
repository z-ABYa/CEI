"""
rag.py
======
The complete Retrieval-Augmented Generation pipeline, orchestrated
manually (no LangChain chains): retrieve relevant chunks, build a
grounded prompt, call the LLM, and package everything the UI needs
(answer, retrieved chunks, similarity scores, metadata, timings).

This is the single entry point the Streamlit app calls per question.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

sys.path.append(str(Path(__file__).resolve().parent))
from config import Config, get_logger  # noqa: E402
from utils.llm import generate_answer  # noqa: E402
from utils.retrieval import Retriever, RetrievalResult, format_context  # noqa: E402

logger = get_logger(__name__)


@dataclass
class RAGResponse:
    """
    Full result of running a question through the RAG pipeline --
    everything the Streamlit UI needs to render an answer with
    citations, scores, and performance metrics.

    Attributes
    ----------
    question : str
        The original user question.
    answer : str
        The LLM-generated, context-grounded answer.
    retrieved_chunks : List[RetrievalResult]
        The ranked chunks used to build the context, with scores.
    context_used : str
        The exact formatted context string sent to the LLM.
    model_used : str
        Name of the Ollama model that generated the answer.
    retrieval_time : float
        Seconds spent embedding the query + FAISS search.
    generation_time : float
        Seconds spent waiting on the LLM to generate the answer.
    total_time : float
        Sum of retrieval_time and generation_time.
    grounded : bool
        False if no chunks were retrieved at all (i.e. the answer
        could not possibly be grounded in any context).
    """

    question: str
    answer: str
    retrieved_chunks: List[RetrievalResult] = field(default_factory=list)
    context_used: str = ""
    model_used: str = ""
    retrieval_time: float = 0.0
    generation_time: float = 0.0
    total_time: float = 0.0
    grounded: bool = True


class RAGPipeline:
    """
    Ties together a ``Retriever`` and the LLM generation step into a
    single ``ask()`` call, mirroring how a production RAG service
    would expose one clean interface to callers (here, the Streamlit app).
    """

    def __init__(self, retriever: Retriever):
        self.retriever = retriever

    @classmethod
    def from_disk(cls) -> "RAGPipeline":
        """Build a pipeline by loading the FAISS index/chunks from disk."""
        return cls(Retriever.from_disk())

    def ask(
        self,
        question: str,
        top_k: int = Config.TOP_K,
        max_context_chunks: int = Config.MAX_CONTEXT_CHUNKS,
        temperature: float = Config.OLLAMA_TEMPERATURE,
    ) -> RAGResponse:
        """
        Run the full RAG pipeline for a single question.

        Steps
        -----
        1. Retrieve the top-k most similar chunks from FAISS.
        2. Format retrieved chunks into a single context string.
        3. Build the grounded prompt and call the local Ollama model.
        4. Package the answer alongside chunks, scores, and timings.

        Parameters
        ----------
        question : str
            The user's natural-language question.
        top_k : int
            Number of chunks to retrieve from the vector store.
        max_context_chunks : int
            Max chunks actually included in the LLM prompt (may be
            <= top_k to control prompt length).
        temperature : float
            Sampling temperature passed through to the LLM.

        Returns
        -------
        RAGResponse
            The full structured result of the pipeline run.
        """
        if not question or not question.strip():
            raise ValueError("Question must be a non-empty string.")

        # --- Step 1: Retrieve -------------------------------------------------
        retrieved_chunks, retrieval_time = self.retriever.search(question, top_k=top_k)

        if not retrieved_chunks:
            logger.warning("No chunks retrieved for question: '%s'", question)
            return RAGResponse(
                question=question,
                answer="I don't know based on the retrieved documents.",
                retrieved_chunks=[],
                context_used="",
                model_used="",
                retrieval_time=retrieval_time,
                generation_time=0.0,
                total_time=retrieval_time,
                grounded=False,
            )

        # --- Step 2: Build context -------------------------------------------
        context = format_context(retrieved_chunks, max_chunks=max_context_chunks)

        # --- Step 3: Generate ------------------------------------------------
        answer, generation_time = generate_answer(
            context=context, question=question, temperature=temperature
        )

        from utils.llm import resolve_model  # local import avoids unused import at module load

        response = RAGResponse(
            question=question,
            answer=answer,
            retrieved_chunks=retrieved_chunks,
            context_used=context,
            model_used=resolve_model(),
            retrieval_time=retrieval_time,
            generation_time=generation_time,
            total_time=retrieval_time + generation_time,
            grounded=True,
        )

        logger.info(
            "RAG pipeline complete: retrieval=%.3fs, generation=%.3fs, total=%.3fs",
            response.retrieval_time, response.generation_time, response.total_time,
        )
        return response


if __name__ == "__main__":
    # Quick manual smoke test: `python rag.py`
    # Requires: `python ingest.py` already run, and `ollama serve` running.
    pipeline = RAGPipeline.from_disk()
    demo_question = "What is retrieval-augmented generation?"

    result = pipeline.ask(demo_question)

    print(f"Question: {result.question}")
    print(f"Answer: {result.answer}\n")
    print(f"Model: {result.model_used}")
    print(f"Retrieval time: {result.retrieval_time:.3f}s | Generation time: {result.generation_time:.3f}s")
    print(f"\nRetrieved {len(result.retrieved_chunks)} chunks:")
    for r in result.retrieved_chunks:
        print(f"  #{r.rank} (score={r.score:.3f}) {r.chunk.chunk_id}: {r.chunk.text[:80]}...")
