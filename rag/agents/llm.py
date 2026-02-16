"""Shared LLM client and completion helper (OpenAI or Ollama)."""

import json
import re
from typing import Any, Optional

from rag.config import get_settings


def _chat_openai(
    messages: list[dict[str, str]],
    model: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> str:
    from openai import OpenAI
    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    return (resp.choices[0].message.content or "").strip()


def _chat_ollama(
    messages: list[dict[str, str]],
    model: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> str:
    import ollama
    settings = get_settings()
    client = ollama.Client(host=settings.ollama_base_url)
    try:
        resp = client.chat(
            model=model,
            messages=messages,
            options={
                "temperature": temperature,
                "top_p": top_p,
                "num_predict": max_tokens,
            },
        )
    except Exception as e:
        err_msg = str(e).lower()
        if "not found" in err_msg or "404" in err_msg or "try pulling" in err_msg:
            raise RuntimeError(
                f"Ollama chat model '{model}' is not available. "
                f"Pull it first with: ollama pull {model}"
            ) from e
        raise
    content = (resp.get("message") or {}).get("content") or ""
    return content.strip()


def chat_completion(
    messages: list[dict[str, str]],
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> str:
    """Call chat API (OpenAI or Ollama from config) and return assistant content."""
    settings = get_settings()
    model = model or (settings.llm_model_large if settings.use_large_llm else settings.llm_model)
    temperature = temperature if temperature is not None else settings.llm_temperature
    top_p = top_p if top_p is not None else settings.llm_top_p
    max_tokens = max_tokens if max_tokens is not None else settings.llm_max_tokens
    try:
        if settings.provider == "ollama":
            return _chat_ollama(messages, model, temperature, top_p, max_tokens)
        return _chat_openai(messages, model, temperature, top_p, max_tokens)
    except Exception as e:
        raise RuntimeError(f"LLM call failed: {e}") from e


def extract_json_from_text(text: str) -> Optional[dict[str, Any]]:
    """Try to parse a JSON object from model output (handles markdown code blocks)."""
    text = text.strip()
    # Try to find ```json ... ``` or ``` ... ```
    for pattern in [r"```(?:json)?\s*([\s\S]*?)```", r"\{[\s\S]*\}"]:
        match = re.search(pattern, text)
        if match:
            raw = match.group(1).strip() if "```" in pattern else match.group(0)
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                continue
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None
