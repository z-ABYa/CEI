"""
evaluate.py
===========
Evaluates the RAG pipeline against question/answer pairs from the
Open RAGBench dataset.

Metrics computed per sample, then averaged:
    * BLEU              - n-gram overlap between generated and reference answer
    * ROUGE-L            - longest common subsequence overlap
    * Exact Match         - strict string equality (normalized)
    * Cosine Similarity   - semantic similarity via sentence embeddings
    * Retrieval time      - seconds spent on FAISS search
    * Generation time     - seconds spent on LLM generation

Results are written to ``Config.EVAL_RESULTS_PATH`` (CSV), one row
per evaluated question plus a final "AVERAGE" summary row.

Run with:

    python evaluate.py
"""

from __future__ import annotations

import re
import string
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from config import Config, get_logger  # noqa: E402
from ingest import index_exists  # noqa: E402
from rag import RAGPipeline  # noqa: E402
from utils.embedding import embed_texts  # noqa: E402
from utils.loader import Document, load_open_ragbench  # noqa: E402

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Text normalization / metric helpers
# ---------------------------------------------------------------------------
def _normalize_text(text: str) -> str:
    """
    Lowercase, strip punctuation, and collapse whitespace -- standard
    normalization used before exact-match and token-overlap comparisons
    so that trivial formatting differences don't count as mismatches.
    """
    text = text.lower().strip()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def exact_match_score(prediction: str, reference: str) -> float:
    """Return 1.0 if normalized prediction equals normalized reference, else 0.0."""
    return 1.0 if _normalize_text(prediction) == _normalize_text(reference) else 0.0


def bleu_score(prediction: str, reference: str) -> float:
    """
    Compute sentence-level BLEU score using NLTK, with smoothing to
    avoid zero scores on short answers (common in QA settings where
    both prediction and reference may be just a few words).
    """
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

    pred_tokens = _normalize_text(prediction).split()
    ref_tokens = _normalize_text(reference).split()

    if not pred_tokens or not ref_tokens:
        return 0.0

    smoothing = SmoothingFunction().method1
    return sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothing)


def rouge_l_score(prediction: str, reference: str) -> float:
    """Compute ROUGE-L F-measure between prediction and reference."""
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = scorer.score(reference, prediction)
    return scores["rougeL"].fmeasure


def cosine_similarity_score(prediction: str, reference: str) -> float:
    """
    Compute cosine similarity between the sentence embeddings of the
    prediction and reference, capturing semantic overlap that BLEU/
    ROUGE (purely lexical) would miss.
    """
    if not prediction.strip() or not reference.strip():
        return 0.0

    embeddings = embed_texts([prediction, reference], normalize=True)
    # Embeddings are L2-normalized, so dot product == cosine similarity.
    similarity = float(np.dot(embeddings[0], embeddings[1]))
    return similarity


# ---------------------------------------------------------------------------
# Evaluation dataset preparation
# ---------------------------------------------------------------------------
def _collect_eval_samples(sample_size: int) -> List[Document]:
    """
    Load documents from Open RAGBench that have both a question and a
    reference answer attached, and return up to ``sample_size`` of them
    to use as the evaluation set.

    Parameters
    ----------
    sample_size : int
        Maximum number of QA pairs to evaluate on.

    Returns
    -------
    List[Document]
        Documents with non-empty ``question`` and ``reference_answer``
        fields, capped at ``sample_size``.
    """
    # Pull a generous pool of documents so we have a decent chance of
    # finding enough rows that actually carry a question + answer pair.
    pool_size = max(sample_size * 10, 200)
    documents = load_open_ragbench(max_documents=pool_size)

    eval_candidates = [
        doc for doc in documents if doc.question and doc.reference_answer
    ]

    if not eval_candidates:
        raise RuntimeError(
            "No documents with both a question and reference answer were "
            "found in the dataset sample. Evaluation requires QA-labeled rows."
        )

    return eval_candidates[:sample_size]


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------
def run_evaluation(sample_size: int = Config.EVAL_SAMPLE_SIZE) -> pd.DataFrame:
    """
    Run the full evaluation: for each QA-labeled sample, ask the RAG
    pipeline the dataset's own question, compare the generated answer
    against the dataset's own reference answer, and compute metrics.

    Parameters
    ----------
    sample_size : int
        Number of QA pairs to evaluate.

    Returns
    -------
    pd.DataFrame
        One row per evaluated question (plus a final "AVERAGE" row),
        with all metric columns. Also written to
        ``Config.EVAL_RESULTS_PATH``.

    Raises
    ------
    RuntimeError
        If no FAISS index exists yet (run ``python ingest.py`` first).
    """
    if not index_exists():
        raise RuntimeError(
            "No FAISS index found. Run `python ingest.py` before evaluating."
        )

    logger.info("Loading RAG pipeline for evaluation...")
    pipeline = RAGPipeline.from_disk()

    logger.info("Collecting up to %d QA samples from Open RAGBench...", sample_size)
    samples = _collect_eval_samples(sample_size)
    logger.info("Evaluating on %d QA pairs.", len(samples))

    rows: List[Dict] = []

    for i, doc in enumerate(samples, start=1):
        question = doc.question
        reference = doc.reference_answer

        logger.info("[%d/%d] Q: %s", i, len(samples), question[:80])

        try:
            response = pipeline.ask(question)
        except Exception as exc:  # noqa: BLE001
            logger.error("Skipping sample %d due to error: %s", i, exc)
            continue

        prediction = response.answer

        row = {
            "question": question,
            "reference_answer": reference,
            "predicted_answer": prediction,
            "bleu": bleu_score(prediction, reference),
            "rouge_l": rouge_l_score(prediction, reference),
            "exact_match": exact_match_score(prediction, reference),
            "cosine_similarity": cosine_similarity_score(prediction, reference),
            "retrieval_time_sec": response.retrieval_time,
            "generation_time_sec": response.generation_time,
            "num_chunks_retrieved": len(response.retrieved_chunks),
            "grounded": response.grounded,
        }
        rows.append(row)

    if not rows:
        raise RuntimeError("Evaluation produced no results; all samples failed.")

    df = pd.DataFrame(rows)

    numeric_cols = [
        "bleu", "rouge_l", "exact_match", "cosine_similarity",
        "retrieval_time_sec", "generation_time_sec",
    ]
    average_row = {
        "question": "AVERAGE",
        "reference_answer": "",
        "predicted_answer": "",
        **{col: df[col].mean() for col in numeric_cols},
        "num_chunks_retrieved": df["num_chunks_retrieved"].mean(),
        "grounded": df["grounded"].mean(),
    }
    df = pd.concat([df, pd.DataFrame([average_row])], ignore_index=True)

    Config.EVAL_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(Config.EVAL_RESULTS_PATH, index=False)
    logger.info("Evaluation results saved to '%s'.", Config.EVAL_RESULTS_PATH)

    return df


def _print_summary(df: pd.DataFrame) -> None:
    """Pretty-print the averaged metrics row to the console."""
    avg = df[df["question"] == "AVERAGE"].iloc[0]
    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY (averaged)")
    print("=" * 50)
    print(f"  BLEU                 : {avg['bleu']:.4f}")
    print(f"  ROUGE-L               : {avg['rouge_l']:.4f}")
    print(f"  Exact Match            : {avg['exact_match']:.4f}")
    print(f"  Cosine Similarity        : {avg['cosine_similarity']:.4f}")
    print(f"  Avg Retrieval Time (s)    : {avg['retrieval_time_sec']:.4f}")
    print(f"  Avg Generation Time (s)    : {avg['generation_time_sec']:.4f}")
    print("=" * 50)
    print(f"\nFull results saved to: {Config.EVAL_RESULTS_PATH}")


if __name__ == "__main__":
    start = time.time()
    try:
        results_df = run_evaluation()
    except RuntimeError as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)

    _print_summary(results_df)
    print(f"\nTotal evaluation time: {time.time() - start:.1f}s")
