"""Embedding generation via OpenAI or Ollama (local)."""

from typing import List

from rag.config import get_settings


def _get_embeddings_openai(texts: List[str], model: str) -> List[List[float]]:
    from openai import OpenAI
    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key)
    resp = client.embeddings.create(input=texts, model=model)
    return [e.embedding for e in resp.data]


def _get_embeddings_ollama(texts: List[str], model: str) -> List[List[float]]:
    import ollama
    settings = get_settings()
    client = ollama.Client(host=settings.ollama_base_url)
    try:
        out = client.embed(model=model, input=texts)
    except Exception as e:
        err_msg = str(e).lower()
        if "not found" in err_msg or "404" in err_msg or "try pulling" in err_msg:
            raise RuntimeError(
                f"Ollama embedding model '{model}' is not available. "
                f"Pull it first with: ollama pull {model}"
            ) from e
        raise
    embeddings = out.get("embeddings") or []
    if not embeddings:
        return []
    if isinstance(embeddings[0], list):
        return embeddings
    return [embeddings]


def get_embeddings(texts: List[str], model: str | None = None) -> List[List[float]]:
    """Get embeddings for a list of texts (OpenAI or Ollama from config)."""
    settings = get_settings()
    model = model or settings.embedding_model
    if not texts:
        return []
    try:
        if settings.provider == "ollama":
            return _get_embeddings_ollama(texts, model)
        return _get_embeddings_openai(texts, model)
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"Embedding failed: {e}") from e


def embed_single(text: str, model: str | None = None) -> List[float]:
    """Get embedding for a single text."""
    return get_embeddings([text], model=model)[0]
