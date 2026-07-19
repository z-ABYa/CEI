"""
utils/loader.py
================
Responsible for loading the Open RAGBench dataset from the Hugging
Face Hub and converting it into a clean, uniform list of ``Document``
objects that the rest of the pipeline (chunking, embedding, retrieval)
can consume without caring about the raw dataset schema.

The Open RAGBench dataset ships several possible text-bearing column
names depending on the config/subset that gets pulled down. Rather
than hard-coding a single field name, this loader auto-detects the
most plausible text column and falls back sensibly if the expected
one is missing.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

sys.path.append(str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from config import Config, get_logger  # noqa: E402

logger = get_logger(__name__)

# Candidate column names that might hold the main document text,
# ordered by preference. The first one found in the dataset schema
# is used.
_CANDIDATE_TEXT_FIELDS = [
    "document",
    "context",
    "passage",
    "text",
    "content",
    "answer_context",
    "source_text",
]

# Candidate columns used to build lightweight metadata for citations.
_CANDIDATE_ID_FIELDS = ["id", "doc_id", "document_id", "uid"]
_CANDIDATE_TITLE_FIELDS = ["title", "source", "doc_title", "url"]
_CANDIDATE_QUESTION_FIELDS = ["question", "query"]
_CANDIDATE_ANSWER_FIELDS = ["answer", "answers", "gold_answer"]


@dataclass
class Document:
    """
    A single, cleaned unit of source text pulled from the dataset.

    Attributes
    ----------
    doc_id : str
        Stable identifier for the document (used later in chunk metadata).
    text : str
        The cleaned document/passage text.
    title : Optional[str]
        Human-readable title or source label, if available.
    question : Optional[str]
        The dataset's associated question, if this row came from a QA pair.
        Kept mainly for evaluation purposes.
    reference_answer : Optional[str]
        The dataset's gold answer, if present. Used only by evaluate.py.
    metadata : Dict[str, Any]
        Catch-all for any extra fields worth keeping around.
    """

    doc_id: str
    text: str
    title: Optional[str] = None
    question: Optional[str] = None
    reference_answer: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class DatasetLoadError(RuntimeError):
    """Raised when the Open RAGBench dataset cannot be loaded or parsed."""


def _detect_field(column_names: List[str], candidates: List[str]) -> Optional[str]:
    """Return the first candidate field present in ``column_names``, if any."""
    for candidate in candidates:
        if candidate in column_names:
            return candidate
    return None


def _clean_text(raw: Any) -> str:
    """
    Normalize a raw dataset cell into clean text.

    Handles the common cases where the underlying field is a plain
    string, a list of strings (e.g. multiple answer variants), or
    something unexpected that must be stringified defensively.
    """
    if raw is None:
        return ""
    if isinstance(raw, str):
        text = raw
    elif isinstance(raw, (list, tuple)):
        text = " ".join(str(item) for item in raw if item)
    else:
        text = str(raw)

    # Collapse excessive whitespace/newlines produced by scraped sources.
    text = " ".join(text.split())
    return text.strip()


def load_open_ragbench(
    split: Optional[str] = None,
    max_documents: Optional[int] = None,
) -> List[Document]:
    """
    Load the ``vectara/open_ragbench`` dataset and convert it into a
    list of clean :class:`Document` objects.

    Parameters
    ----------
    split : Optional[str]
        Dataset split to load (e.g. "train"). Defaults to
        ``Config.DATASET_SPLIT``. If the requested split does not
        exist, the loader automatically falls back to the first
        available split.
    max_documents : Optional[int]
        Maximum number of documents to return. Defaults to
        ``Config.MAX_DOCUMENTS``. Pass ``None`` explicitly (and set
        ``Config.MAX_DOCUMENTS = None``) to load everything.

    Returns
    -------
    List[Document]
        Cleaned, de-duplicated documents ready for chunking.

    Raises
    ------
    DatasetLoadError
        If the dataset cannot be downloaded/parsed, or if no usable
        text column can be found in the schema.
    """
    split = split or Config.DATASET_SPLIT
    max_documents = Config.MAX_DOCUMENTS if max_documents is None else max_documents

    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - environment issue
        raise DatasetLoadError(
            "The 'datasets' package is required. Install it with "
            "`pip install datasets`."
        ) from exc

    logger.info("Loading dataset '%s' (split=%s)...", Config.DATASET_NAME, split)

    try:
        dataset_dict = load_dataset(Config.DATASET_NAME)
    except Exception as exc:  # noqa: BLE001 - surface as a clear, typed error
        raise DatasetLoadError(
            f"Failed to download/load dataset '{Config.DATASET_NAME}' from "
            f"the Hugging Face Hub: {exc}"
        ) from exc

    # Resolve which split to actually use.
    available_splits = list(dataset_dict.keys())
    if not available_splits:
        raise DatasetLoadError("Dataset loaded but contains no splits.")

    if split not in available_splits:
        logger.warning(
            "Requested split '%s' not found. Available splits: %s. "
            "Falling back to '%s'.",
            split,
            available_splits,
            available_splits[0],
        )
        split = available_splits[0]

    hf_dataset = dataset_dict[split]
    column_names = hf_dataset.column_names

    text_field = _detect_field(column_names, _CANDIDATE_TEXT_FIELDS)
    if text_field is None:
        raise DatasetLoadError(
            "Could not find a usable text column in the dataset. "
            f"Available columns were: {column_names}"
        )

    id_field = _detect_field(column_names, _CANDIDATE_ID_FIELDS)
    title_field = _detect_field(column_names, _CANDIDATE_TITLE_FIELDS)
    question_field = _detect_field(column_names, _CANDIDATE_QUESTION_FIELDS)
    answer_field = _detect_field(column_names, _CANDIDATE_ANSWER_FIELDS)

    logger.info(
        "Detected schema -> text: '%s', id: '%s', title: '%s', "
        "question: '%s', answer: '%s'",
        text_field, id_field, title_field, question_field, answer_field,
    )

    documents: List[Document] = []
    seen_hashes = set()
    total_rows = len(hf_dataset)
    row_limit = total_rows if max_documents is None else min(max_documents, total_rows)

    for idx in range(total_rows):
        if len(documents) >= row_limit:
            break

        row = hf_dataset[idx]
        text = _clean_text(row.get(text_field))

        if not text:
            continue  # skip empty/unusable rows

        # De-duplicate identical passages, which are common in QA datasets
        # where multiple questions share the same source context.
        text_hash = hash(text)
        if text_hash in seen_hashes:
            continue
        seen_hashes.add(text_hash)

        doc_id = str(row.get(id_field)) if id_field else f"doc_{idx}"
        title = _clean_text(row.get(title_field)) if title_field else None
        question = _clean_text(row.get(question_field)) if question_field else None
        reference_answer = (
            _clean_text(row.get(answer_field)) if answer_field else None
        )

        documents.append(
            Document(
                doc_id=doc_id,
                text=text,
                title=title or None,
                question=question or None,
                reference_answer=reference_answer or None,
                metadata={"source_index": idx, "split": split},
            )
        )

    if not documents:
        raise DatasetLoadError(
            "Dataset was loaded successfully but no valid documents "
            "could be extracted. Check the detected text column."
        )

    logger.info(
        "Loaded %d unique documents out of %d rows scanned (split='%s').",
        len(documents), total_rows, split,
    )
    return documents


if __name__ == "__main__":
    # Quick manual smoke test: `python utils/loader.py`
    docs = load_open_ragbench(max_documents=5)
    for d in docs:
        preview = d.text[:120].replace("\n", " ")
        print(f"[{d.doc_id}] {preview}...")
