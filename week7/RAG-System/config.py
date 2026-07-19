"""
config.py
=========
Central configuration for the RAG-System project.

Every tunable parameter used across the pipeline (data ingestion,
chunking, embedding, retrieval, and generation) lives here so that
the rest of the codebase never hard-codes a magic value.

Usage
-----
    from config import Config

    print(Config.EMBEDDING_MODEL)
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


# ---------------------------------------------------------------------------
# Base paths
# ---------------------------------------------------------------------------
BASE_DIR: Path = Path(__file__).resolve().parent
DATA_DIR: Path = BASE_DIR / "data"
VECTORSTORE_DIR: Path = BASE_DIR / "vectorstore"
PROMPTS_DIR: Path = BASE_DIR / "prompts"

# Ensure required directories exist at import time so downstream modules
# can assume they are present.
for _dir in (DATA_DIR, VECTORSTORE_DIR, PROMPTS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Config:
    """
    Immutable configuration object for the RAG pipeline.

    Grouped into logical sections: dataset, chunking, embedding,
    vector store, retrieval, LLM, and logging.
    """

    # -- Dataset -----------------------------------------------------------
    DATASET_NAME: str = "vectara/open_ragbench"
    # Which split(s) of the dataset to load. Some HF datasets expose
    # multiple splits (train/validation/test); we default to "train"
    # and fall back gracefully inside loader.py if it is missing.
    DATASET_SPLIT: str = "train"
    # Hard cap on number of source documents ingested, to keep the
    # demo runnable on a laptop. Set to None to ingest everything.
    MAX_DOCUMENTS: int = 2000

    # -- Chunking ------------------------------------------------------------
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64
    # Separators tried in order by RecursiveCharacterTextSplitter.
    CHUNK_SEPARATORS: List[str] = field(
        default_factory=lambda: ["\n\n", "\n", ". ", " ", ""]
    )

    # -- Embedding -----------------------------------------------------------
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_BATCH_SIZE: int = 64
    EMBEDDING_DEVICE: str = "cpu"  # switched to "cuda" automatically if available

    # -- Vector store --------------------------------------------------------
    VECTOR_DB_PATH: Path = VECTORSTORE_DIR / "faiss_index"
    METADATA_PATH: Path = VECTORSTORE_DIR / "metadata.pkl"
    EMBEDDING_DIM: int = 384  # all-MiniLM-L6-v2 output dimension

    # -- Retrieval -------------------------------------------------------------
    TOP_K: int = 5
    MAX_CONTEXT_CHUNKS: int = 5
    SIMILARITY_THRESHOLD: float = 0.0  # 0.0 disables filtering

    # -- LLM (Ollama) ----------------------------------------------------------
    OLLAMA_MODEL: str = "gemma3:4b"
    OLLAMA_FALLBACK_MODELS: List[str] = field(
        default_factory=lambda: ["llama3.2", "mistral"]
    )
    OLLAMA_HOST: str = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    OLLAMA_TEMPERATURE: float = 0.2
    OLLAMA_MAX_TOKENS: int = 512
    OLLAMA_REQUEST_TIMEOUT: int = 120  # seconds

    # -- Prompt ------------------------------------------------------------
    PROMPT_TEMPLATE_PATH: Path = PROMPTS_DIR / "rag_prompt.txt"

    # -- Evaluation ----------------------------------------------------------
    EVAL_SAMPLE_SIZE: int = 50
    EVAL_RESULTS_PATH: Path = BASE_DIR / "evaluation_results.csv"

    # -- Logging -------------------------------------------------------------
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def get_logger(name: str) -> logging.Logger:
    """
    Create (or retrieve) a module-level logger configured with the
    project-wide log level and format defined in ``Config``.

    Parameters
    ----------
    name : str
        Typically ``__name__`` of the calling module.

    Returns
    -------
    logging.Logger
        A configured logger instance. Safe to call multiple times;
        handlers are only attached once per logger name.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(Config.LOG_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(getattr(logging, Config.LOG_LEVEL.upper(), logging.INFO))
        logger.propagate = False

    return logger
