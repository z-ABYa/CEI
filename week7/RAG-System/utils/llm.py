"""
utils/llm.py
=============
Thin client around a local Ollama server for answer generation.

Responsible for:
    * Auto-detecting which of the project's supported models is
      actually installed locally (gemma3:4b -> llama3.2 -> mistral).
    * Loading and formatting the RAG prompt template.
    * Calling the Ollama chat/generate API and returning the answer.

No paid APIs (OpenAI, Claude, Gemini, etc.) are used anywhere in this
module -- generation runs entirely through a local Ollama instance.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import Config, get_logger  # noqa: E402

logger = get_logger(__name__)

# Cached at module level once resolved, so we don't re-query the Ollama
# server's model list on every single question asked in a session.
_resolved_model_cache: Optional[str] = None


class OllamaConnectionError(RuntimeError):
    """Raised when the local Ollama server cannot be reached at all."""


class NoModelAvailableError(RuntimeError):
    """Raised when none of the supported models are pulled locally."""


def _get_ollama_client():
    """Import and construct the Ollama client, pointed at the configured host."""
    try:
        import ollama
    except ImportError as exc:  # pragma: no cover - environment issue
        raise ImportError(
            "The 'ollama' package is required. Install it with `pip install "
            "ollama`, and make sure the Ollama server is running locally."
        ) from exc

    return ollama.Client(host=Config.OLLAMA_HOST)


def list_installed_models() -> List[str]:
    """
    Query the local Ollama server for the list of currently pulled
    model names.

    Returns
    -------
    List[str]
        Model names as reported by Ollama (e.g. ``["gemma3:4b", "mistral:latest"]``).

    Raises
    ------
    OllamaConnectionError
        If the Ollama server is not reachable at ``Config.OLLAMA_HOST``.
    """
    client = _get_ollama_client()
    try:
        response = client.list()
    except Exception as exc:  # noqa: BLE001
        raise OllamaConnectionError(
            f"Could not reach the Ollama server at '{Config.OLLAMA_HOST}'. "
            f"Is Ollama running? Start it with `ollama serve`. "
            f"Original error: {exc}"
        ) from exc

    # The ollama-python client returns objects with a `.model` attribute
    # (newer versions) or dicts with a "name" key (older versions).
    models = response.get("models", []) if isinstance(response, dict) else response.models
    names = []
    for m in models:
        name = getattr(m, "model", None) or (m.get("name") if isinstance(m, dict) else None)
        if name:
            names.append(name)
    return names


def _matches(installed_name: str, target: str) -> bool:
    """
    Check whether an installed model name matches a target model
    identifier, tolerating Ollama's ``:tag`` suffix (e.g. treating
    ``"gemma3:4b"`` as a match for target ``"gemma3:4b"`` and
    ``"mistral:latest"`` as a match for target ``"mistral"``).
    """
    installed_base = installed_name.split(":")[0]
    target_base = target.split(":")[0]
    if installed_name == target:
        return True
    return installed_base == target_base


def resolve_model(force_refresh: bool = False) -> str:
    """
    Determine which model to use for generation, preferring
    ``Config.OLLAMA_MODEL`` and falling back through
    ``Config.OLLAMA_FALLBACK_MODELS`` in order.

    Parameters
    ----------
    force_refresh : bool
        If True, bypass the cached result and re-query Ollama.

    Returns
    -------
    str
        The exact installed model name to pass to the Ollama API.

    Raises
    ------
    NoModelAvailableError
        If none of the preferred/fallback models are installed.
    """
    global _resolved_model_cache

    if _resolved_model_cache is not None and not force_refresh:
        return _resolved_model_cache

    installed = list_installed_models()
    if not installed:
        raise NoModelAvailableError(
            "No models are installed in Ollama. Run `ollama pull "
            f"{Config.OLLAMA_MODEL}` to get started."
        )

    candidates = [Config.OLLAMA_MODEL, *Config.OLLAMA_FALLBACK_MODELS]

    for candidate in candidates:
        for installed_name in installed:
            if _matches(installed_name, candidate):
                logger.info("Resolved LLM model: '%s' (matched '%s').", installed_name, candidate)
                _resolved_model_cache = installed_name
                return installed_name

    raise NoModelAvailableError(
        f"None of the supported models {candidates} are installed. "
        f"Installed models: {installed}. Run `ollama pull {Config.OLLAMA_MODEL}`."
    )


def load_prompt_template() -> str:
    """
    Read the RAG prompt template from ``Config.PROMPT_TEMPLATE_PATH``.

    Returns
    -------
    str
        The raw template text, containing ``{context}`` and
        ``{question}`` placeholders.

    Raises
    ------
    FileNotFoundError
        If the prompt template file does not exist.
    """
    if not Config.PROMPT_TEMPLATE_PATH.exists():
        raise FileNotFoundError(
            f"Prompt template not found at '{Config.PROMPT_TEMPLATE_PATH}'."
        )
    return Config.PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")


def build_prompt(context: str, question: str) -> str:
    """
    Fill the RAG prompt template with the given context and question.

    Parameters
    ----------
    context : str
        Retrieved context (typically from ``utils.retrieval.format_context``).
    question : str
        The user's question.

    Returns
    -------
    str
        The fully-formatted prompt ready to send to the LLM.
    """
    template = load_prompt_template()

    if not context.strip():
        # No relevant chunks were retrieved at all; make this explicit
        # in the context block rather than sending an empty string,
        # which helps smaller models follow the "don't know" rule.
        context = "(No relevant context was retrieved for this question.)"

    return template.format(context=context, question=question)


def generate_answer(
    context: str,
    question: str,
    model: Optional[str] = None,
    temperature: float = Config.OLLAMA_TEMPERATURE,
) -> Tuple[str, float]:
    """
    Generate a grounded answer to ``question`` using ``context`` via
    a local Ollama model.

    Parameters
    ----------
    context : str
        Retrieved context to ground the answer in.
    question : str
        The user's natural-language question.
    model : Optional[str]
        Specific Ollama model name to use. If ``None``, the best
        available model is auto-resolved via ``resolve_model()``.
    temperature : float
        Sampling temperature; kept low by default for factual,
        deterministic-leaning answers.

    Returns
    -------
    Tuple[str, float]
        The generated answer text, and the generation latency in seconds.

    Raises
    ------
    OllamaConnectionError
        If the Ollama server cannot be reached.
    NoModelAvailableError
        If no supported model is installed.
    """
    model_name = model or resolve_model()
    prompt = build_prompt(context, question)

    client = _get_ollama_client()

    logger.info("Generating answer with model '%s' (temperature=%.2f)...", model_name, temperature)
    start_time = time.time()

    try:
        response = client.generate(
            model=model_name,
            prompt=prompt,
            options={
                "temperature": temperature,
                "num_predict": Config.OLLAMA_MAX_TOKENS,
            },
        )
    except Exception as exc:  # noqa: BLE001
        raise OllamaConnectionError(
            f"Generation request to Ollama failed for model '{model_name}': {exc}"
        ) from exc

    elapsed = time.time() - start_time

    answer_text = response.get("response", "") if isinstance(response, dict) else response.response
    answer_text = answer_text.strip()

    if not answer_text:
        answer_text = "I don't know based on the retrieved documents."

    logger.info("Generated answer in %.2fs (%d chars).", elapsed, len(answer_text))
    return answer_text, elapsed


if __name__ == "__main__":
    # Quick manual smoke test: `python utils/llm.py`
    # Requires a running `ollama serve` with at least one supported model pulled.
    demo_context = (
        "[Source 1: Example]\nRetrieval-Augmented Generation (RAG) is a technique "
        "that combines information retrieval with text generation to produce "
        "answers grounded in external documents."
    )
    demo_question = "What is RAG?"

    try:
        answer, latency = generate_answer(demo_context, demo_question)
        print(f"Model used: {resolve_model()}")
        print(f"Latency: {latency:.2f}s")
        print(f"Answer: {answer}")
    except (OllamaConnectionError, NoModelAvailableError) as exc:
        print(f"[ERROR] {exc}")
