"""Optional cross-encoder re-ranker (e.g. BGE) for better retrieval precision."""

from typing import List, Optional

from rag.config import get_settings
from rag.models import RetrievedChunk


def _load_reranker_model(model_name: str):
    """Lazy-load CrossEncoder to avoid importing torch/sentence_transformers at startup."""
    from sentence_transformers import CrossEncoder
    return CrossEncoder(model_name)


class Reranker:
    """Re-rank retrieved chunks using a cross-encoder (e.g. BAAI/bge-reranker-base)."""

    def __init__(
        self,
        model_name: Optional[str] = None,
    ):
        settings = get_settings()
        self.model_name = model_name or settings.reranker_model
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = _load_reranker_model(self.model_name)
        return self._model

    def rerank(
        self,
        query: str,
        retrieved: List[RetrievedChunk],
        top_k: Optional[int] = None,
    ) -> List[RetrievedChunk]:
        """Re-rank by cross-encoder score and return top_k chunks."""
        if not retrieved:
            return []
        top_k = top_k or len(retrieved)
        pairs = [(query, r.chunk.to_search_text()) for r in retrieved]
        scores = self.model.predict(pairs)
        # Handle single pair (model may return scalar or 0-d array)
        import numpy as np
        scores = np.atleast_1d(np.asarray(scores))
        scores = scores.tolist()
        indexed = list(zip(range(len(retrieved)), scores, retrieved))
        indexed.sort(key=lambda x: x[1], reverse=True)
        return [
            RetrievedChunk(chunk=r.chunk, score=float(score), source="reranker")
            for _, score, r in indexed[:top_k]
        ]
