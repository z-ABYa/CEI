# RAG-System

**Retrieval-Augmented Generation (RAG) Question Answering System** built on the [Open RAGBench](https://huggingface.co/datasets/vectara/open_ragbench) dataset — fully local, fully free, no paid APIs.

---

## Overview

This project implements a complete, end-to-end RAG pipeline from scratch:

- **Document Loading** — pulls the Open RAGBench dataset directly from Hugging Face
- **Chunking** — splits documents into overlapping chunks (`RecursiveCharacterTextSplitter`)
- **Embedding** — encodes chunks with `sentence-transformers/all-MiniLM-L6-v2`
- **Vector Search** — indexes and retrieves via FAISS (cosine similarity)
- **Prompt Construction** — builds a strict, grounded prompt from retrieved context
- **Answer Generation** — generates answers locally via **Ollama** (`gemma3:4b`, with automatic fallback to `llama3.2` or `mistral`)
- **Evaluation** — scores the pipeline with BLEU, ROUGE-L, Exact Match, and semantic cosine similarity

The system is intentionally **not** built around LangChain chains — every pipeline stage (retrieval, prompting, generation, orchestration) is implemented manually so the internal mechanics of RAG are fully visible and understandable. LangChain is used only for `RecursiveCharacterTextSplitter`.

The generated answer is always grounded in retrieved context. If the retrieved documents don't contain the answer, the system responds:

> "I don't know based on the retrieved documents."

---

## Architecture

```
                              ┌────────────────────────┐
                              │   Open RAGBench (HF)    │
                              └────────────┬─────────────┘
                                           │  load_dataset()
                                           ▼
                              ┌────────────────────────┐
                              │      loader.py           │  clean documents
                              └────────────┬─────────────┘
                                           ▼
                              ┌────────────────────────┐
                              │     chunking.py           │  RecursiveCharacterTextSplitter
                              └────────────┬─────────────┘
                                           ▼
                              ┌────────────────────────┐
                              │     embedding.py           │  all-MiniLM-L6-v2
                              └────────────┬─────────────┘
                                           ▼
                              ┌────────────────────────┐
                              │   FAISS IndexFlatIP        │  ingest.py → vectorstore/
                              └────────────┬─────────────┘
                                           │
      ┌────────────────────────────────────┼────────────────────────────────────┐
      │                                     │                                     │
      ▼                                     ▼                                     ▼
┌───────────┐                     ┌──────────────────┐                  ┌───────────────┐
│  app.py    │◄───user question───│   rag.py           │──context+Q──────►│    llm.py       │
│ (Streamlit)│                     │  (orchestration)    │◄──answer─────────│   (Ollama)       │
└───────────┘                     └──────────────────┘                  └───────────────┘
      │                                     ▲
      │                                     │  top-k chunks + scores
      │                            ┌──────────────────┐
      └───────answer + citations──│   retrieval.py      │
                                    └──────────────────┘
```

---

## Folder Structure

```
RAG-System/
│
├── app.py                  # Streamlit UI
├── ingest.py                # Ingestion pipeline (load → chunk → embed → index)
├── rag.py                   # Manual RAG orchestration (retrieve → prompt → generate)
├── evaluate.py               # Evaluation harness (BLEU, ROUGE-L, EM, cosine sim)
├── config.py                  # Central configuration
├── requirements.txt
├── README.md
│
├── data/                     # (reserved for cached/local data)
├── vectorstore/                # FAISS index + chunk metadata (generated)
│
├── prompts/
│   └── rag_prompt.txt           # Grounded QA prompt template
│
└── utils/
    ├── loader.py                # Dataset loading & cleaning
    ├── chunking.py               # Text splitting
    ├── embedding.py                # Sentence-transformer embeddings
    ├── retrieval.py                # FAISS similarity search
    └── llm.py                      # Ollama client & generation
```

---

## Installation

### 1. Clone and set up a virtual environment

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Install and start Ollama

Download Ollama from [ollama.com](https://ollama.com), then:

```bash
ollama pull gemma3:4b
ollama serve
```

The system will automatically fall back to `llama3.2` or `mistral` if `gemma3:4b` isn't available — pull whichever you prefer:

```bash
ollama pull llama3.2
# or
ollama pull mistral
```

---

## Requirements

- Python 3.10+
- ~4–8 GB RAM free for the local LLM (model-dependent)
- Ollama installed and running locally
- Internet connection for the *first* dataset/model download only — everything runs offline afterward

See `requirements.txt` for the full pinned dependency list.

---

## Dataset

This project uses **only** the [Open RAGBench dataset](https://huggingface.co/datasets/vectara/open_ragbench), loaded via:

```python
from datasets import load_dataset
dataset = load_dataset("vectara/open_ragbench")
```

No PDFs, no manual uploads — the dataset's own document/context fields form the knowledge base, and (where available) its question/answer fields are used for evaluation.

---

## How to Run

### Step 1 — Build the vector index

```bash
python ingest.py
```

This loads the dataset, chunks it, generates embeddings, builds a FAISS index, and saves everything to `vectorstore/`.

### Step 2 — Launch the app

```bash
streamlit run app.py
```

Open the printed local URL, then either click **Load Index** (if you already ran `ingest.py`) or **Create Index** directly from the sidebar.

### Step 3 — (Optional) Run evaluation

```bash
python evaluate.py
```

Produces `evaluation_results.csv` with per-question and averaged metrics.

---

## Screenshots

> _Add screenshots here after running the app locally:_
>
> - `screenshots/sidebar.png` — sidebar controls
> - `screenshots/answer.png` — question + grounded answer
> - `screenshots/chunks.png` — expandable retrieved chunks with scores

---

## Example Query

**Question:**
> "What is retrieval-augmented generation?"

**Retrieved Sources:** 3 chunks (similarity scores 0.71–0.84)

**Generated Answer:**
> "According to Source 1, retrieval-augmented generation combines a retrieval step over a document collection with a text-generation step, so that the generated answer is grounded in retrieved passages rather than the model's parametric memory alone."

---

## Example Output (evaluation)

| Metric              | Score  |
|----------------------|--------|
| BLEU                  | 0.34   |
| ROUGE-L                | 0.52   |
| Exact Match              | 0.18   |
| Cosine Similarity          | 0.79   |
| Avg Retrieval Time (s)      | 0.04   |
| Avg Generation Time (s)      | 1.9    |

*(Illustrative values — actual scores depend on the LLM, sample size, and chunking parameters used.)*

---
