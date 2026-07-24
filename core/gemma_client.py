"""
gemma_client.py
───────────────
Standalone client for a local Ollama instance running gemma4:latest.
Tuned for 4-core CPU inference (num_thread=4, small context window).

Usage:
    from core.gemma_client import ask_gemma
    answer = ask_gemma("Summarise this paragraph …", max_tokens=50)
"""

import json
import logging
import os

import requests

logger = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma4:latest"
KEEP_ALIVE = "60m"  # keep model loaded for 60 minutes


def ask_gemma(prompt: str, max_tokens: int = 25) -> str:
    """
    Send *prompt* to local Ollama (gemma4:latest) and return the generated text.

    Parameters
    ----------
    prompt : str
        The input text / instruction.
    max_tokens : int, optional
        Maximum number of tokens to generate (default 25).

    Returns
    -------
    str
        The concatenated model response.

    Raises
    ------
    ConnectionError
        If Ollama is unreachable.
    RuntimeError
        If Ollama returns a non-200 status or the response body is malformed.
    """
    # Detect available CPU threads dynamically
    cpu_threads = os.cpu_count() or 4

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "think": False,
        "stream": True,
        "keep_alive": KEEP_ALIVE,
        "options": {
            "num_thread": cpu_threads,
            "num_ctx": 768,
            "num_predict": max_tokens,
            "temperature": 0.1,
            "use_mmap": True,
        },
    }

    try:
        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=120,
            stream=True,
        )
    except requests.ConnectionError as exc:
        logger.error("Ollama is not reachable at %s", OLLAMA_URL)
        raise ConnectionError(
            f"Cannot connect to Ollama at {OLLAMA_URL}. "
            "Is the Ollama server running?"
        ) from exc
    except requests.Timeout as exc:
        logger.error("Request to Ollama timed out")
        raise RuntimeError("Ollama request timed out after 120 s.") from exc

    if response.status_code != 200:
        logger.error(
            "Ollama returned HTTP %s: %s",
            response.status_code,
            response.text[:500],
        )
        raise RuntimeError(
            f"Ollama error (HTTP {response.status_code}): {response.text[:500]}"
        )

    # ── Stream the response token-by-token ──────────────────────────────
    fragments: list[str] = []
    try:
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            chunk = json.loads(line)
            token = chunk.get("response", "")
            if token:
                fragments.append(token)
            if chunk.get("done", False):
                break
    except (json.JSONDecodeError, KeyError) as exc:
        logger.error("Malformed Ollama streaming response: %s", exc)
        raise RuntimeError("Failed to parse Ollama streaming response.") from exc

    result = "".join(fragments).strip()
    logger.debug("Gemma response (%d tokens requested): %s", max_tokens, result[:200])
    return result
