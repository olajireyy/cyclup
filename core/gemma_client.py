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
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"
DEFAULT_MODEL_NAME = "gemma4:e4b"
KEEP_ALIVE = "60m"  # keep# ── Cloud API Recording Fallback Settings ───────────────────────────────────
def get_env_gemini_key() -> str:
    return os.getenv("GEMINI_API_KEY", "").strip()


def get_env_openai_key() -> str:
    return os.getenv("OPENAI_API_KEY", "").strip()


def is_use_cloud_api() -> bool:
    return os.getenv("USE_CLOUD_API", "false").lower() in ("true", "1", "yes")


def get_available_model() -> str:
    """Query Ollama tags endpoint and return the best available Gemma 4 model tag."""
    gemini_key = get_env_gemini_key()
    openai_key = get_env_openai_key()

    if is_use_cloud_api() or gemini_key or openai_key:
        if gemini_key:
            return "gemini-1.5-flash (Cloud Demo Mode)"
        if openai_key:
            return "gpt-4o-mini (Cloud Demo Mode)"
        return "Cloud API (Demo Mode)"

    try:
        resp = requests.get(OLLAMA_TAGS_URL, timeout=3)
        if resp.status_code == 200:
            models = resp.json().get("models", [])
            names = [m.get("name", "") for m in models]
            # Prefer exact gemma4:e4b or gemma4:latest or any gemma4 model
            for target in ["gemma4:e4b", "gemma4:latest", "gemma4:e2b"]:
                if target in names:
                    return target
            for name in names:
                if "gemma" in name.lower():
                    return name
    except Exception as exc:
        logger.warning("Could not dynamically resolve model tag: %s", exc)
    return DEFAULT_MODEL_NAME


def _ask_gemini_cloud(prompt: str, api_key: str) -> str:
    """Call Google Gemini API endpoint for instant cloud inference during demos."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    headers = {"Content-Type": "application/json"}
    resp = requests.post(url, json=payload, headers=headers, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    candidates = data.get("candidates", [])
    if candidates:
        parts = candidates[0].get("content", {}).get("parts", [])
        if parts:
            return parts[0].get("text", "").strip()
    return ""


def _ask_openai_cloud(prompt: str, api_key: str, max_tokens: int) -> str:
    """Call OpenAI API endpoint for instant cloud inference during demos."""
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }
    resp = requests.post(url, json=payload, headers=headers, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    choices = data.get("choices", [])
    if choices:
        return choices[0].get("message", {}).get("content", "").strip()
    return ""


def ask_gemma(prompt: str, max_tokens: int = 150, think: bool = False, provider: str = "auto") -> str:
    """
    Send *prompt* to local Ollama or Cloud API demo fallback.
    """
    gemini_key = get_env_gemini_key()
    openai_key = get_env_openai_key()

    use_cloud = False
    if provider == "cloud":
        use_cloud = True
    elif provider == "local":
        use_cloud = False
    else:  # "auto"
        use_cloud = is_use_cloud_api() or bool(gemini_key or openai_key)

    if use_cloud:
        try:
            if gemini_key:
                return _ask_gemini_cloud(prompt, gemini_key)
            if openai_key:
                return _ask_openai_cloud(prompt, openai_key, max_tokens)
            if provider == "cloud":
                raise RuntimeError("Cloud API requested, but neither GEMINI_API_KEY nor OPENAI_API_KEY is configured in .env.")
        except Exception as exc:
            if provider == "cloud":
                raise
            logger.error("Cloud API failed, falling back to Ollama: %s", exc)

    # Detect available CPU threads dynamically
    cpu_threads = os.cpu_count() or 4
    model_name = get_available_model()

    payload = {
        "model": model_name,
        "prompt": prompt,
        "think": think,
        "stream": True,
        "keep_alive": KEEP_ALIVE,
        "options": {
            "num_thread": cpu_threads,
            "num_ctx": 1024,
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


def ask_gemma_generator(prompt: str, max_tokens: int = 500, think: bool = False, provider: str = "auto"):
    """
    Generator streaming tokens from local Ollama model or Cloud API fallback.
    Yields dicts: {"type": "thinking" | "answer" | "error", "content": str}
    """
    gemini_key = get_env_gemini_key()
    openai_key = get_env_openai_key()

    use_cloud = False
    if provider == "cloud":
        use_cloud = True
    elif provider == "local":
        use_cloud = False
    else:  # "auto"
        use_cloud = is_use_cloud_api() or bool(gemini_key or openai_key)

    if use_cloud:
        if not gemini_key and not openai_key and provider == "cloud":
            yield {"type": "error", "content": "Cloud API requested, but neither GEMINI_API_KEY nor OPENAI_API_KEY is configured in .env."}
            return

        try:
            if think:
                yield {"type": "thinking", "content": "Analyzing campus dumps & synthesizing factual answer via Cloud API...\n"}

            if gemini_key:
                text = _ask_gemini_cloud(prompt, gemini_key)
                yield {"type": "answer", "content": text}
                return
            if openai_key:
                text = _ask_openai_cloud(prompt, openai_key, max_tokens)
                yield {"type": "answer", "content": text}
                return
        except Exception as exc:
            logger.error("Cloud API generator error: %s", exc)
            yield {"type": "error", "content": f"Cloud API error: {exc}"}
    cpu_threads = os.cpu_count() or 4
    model_name = get_available_model()

    payload = {
        "model": model_name,
        "prompt": prompt,
        "think": think,
        "stream": True,
        "keep_alive": KEEP_ALIVE,
        "options": {
            "num_thread": cpu_threads,
            "num_ctx": 2048,
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
        yield {"type": "error", "content": f"Cannot connect to Ollama at {OLLAMA_URL}. Is the Ollama server running?"}
        return
    except requests.Timeout as exc:
        logger.error("Request to Ollama timed out")
        yield {"type": "error", "content": "Ollama request timed out."}
        return
    except Exception as exc:
        logger.error("Ollama request failed: %s", exc)
        yield {"type": "error", "content": str(exc)}
        return

    if response.status_code != 200:
        logger.error("Ollama returned HTTP %s: %s", response.status_code, response.text[:500])
        yield {"type": "error", "content": f"Ollama error (HTTP {response.status_code}): {response.text[:500]}"}
        return

    is_thinking_state = False
    try:
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            chunk = json.loads(line)

            thinking_token = chunk.get("thinking", "")
            response_token = chunk.get("response", "")

            if thinking_token:
                yield {"type": "thinking", "content": thinking_token}

            if response_token:
                token = response_token
                if "<think>" in token:
                    is_thinking_state = True
                    token = token.replace("<think>", "")
                if "</think>" in token:
                    parts = token.split("</think>")
                    if parts[0]:
                        yield {"type": "thinking", "content": parts[0]}
                    is_thinking_state = False
                    if len(parts) > 1 and parts[1]:
                        yield {"type": "answer", "content": parts[1]}
                    continue

                if is_thinking_state:
                    yield {"type": "thinking", "content": token}
                else:
                    yield {"type": "answer", "content": token}

            if chunk.get("done", False):
                break
    except Exception as exc:
        logger.error("Error during streaming from Ollama: %s", exc)
        yield {"type": "error", "content": f"Streaming error: {str(exc)}"}

