"""Hybrid retrieval: merge BM25 and vector search, optional re-ranker."""

from typing import List, Optional

from rag.config import get_settings
from rag.models import RetrievedChunk
from rag.retrieval.store import IndexStore, clause_type_from_metadata, clause_types_from_metadata


class HybridRetrieval:
    """Combine BM25 and vector search, merge and deduplicate; optionally re-rank with BGE."""

    def __init__(
        self,
        store: Optional[IndexStore] = None,
        top_k: Optional[int] = None,
        bm25_weight: Optional[float] = None,
        vector_weight: Optional[float] = None,
        use_reranker: Optional[bool] = None,
    ):
        settings = get_settings()
        self.store = store or IndexStore()
        self.top_k = top_k or settings.top_k
        self.bm25_weight = bm25_weight if bm25_weight is not None else settings.bm25_weight
        self.vector_weight = vector_weight if vector_weight is not None else settings.vector_weight
        self.use_reranker = use_reranker if use_reranker is not None else settings.use_reranker
        self._reranker = None
        self._reranker_unavailable = False

    def _get_reranker(self):
        if self._reranker_unavailable:
            return None
        if self._reranker is not None:
            return self._reranker
        try:
            from rag.retrieval.reranker import Reranker
            self._reranker = Reranker()
            return self._reranker
        except ImportError:
            self._reranker_unavailable = True
            return None

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        metadata_filter: Optional[dict] = None,
        document_filter: Optional[List[str]] = None,
        hard_clause_type: Optional[str] = None,
    ) -> List[RetrievedChunk]:
        """
        Hybrid strategy:
        - If document_filter: HARD filter by document (vector where + BM25 post-filter).
        - If hard_clause_type (e.g. for service_availability): HARD filter by clause_type as well.
        - If metadata_filter has clause_type: SOFT boost only (clause_type may be str or list of types).
        """
        k = top_k or self.top_k
        query = query.strip()
        if not query:
            return []

        chunks = self.store.get_chunks()
        if not chunks:
            return []

        # Build Chroma where: single doc = {"document": "Name"} (not list); else $in
        if document_filter is None or len(document_filter) == 0:
            where_filter = None
        elif len(document_filter) == 1:
            where_filter = {"document": document_filter[0]}
        else:
            where_filter = {"document": {"$in": document_filter}}

        if where_filter and hard_clause_type:
            valid_keys = self.store.get_metadata_keys()
            if "clause_type" in valid_keys:
                where_filter = {"$and": [where_filter, {"clause_type": hard_clause_type}]}
            # else: index has no clause_type (old or missing); omit clause_type from filter so results still return

        allowed_docs: Optional[set[str]] = None
        if document_filter:
            allowed_docs = set(document_filter)

        # BM25: hard filter by document (and by clause_type when hard_clause_type set)
        bm25_hits = self.store.bm25_search(query, top_k=k * 4)
        if allowed_docs is not None:
            bm25_hits = [(i, s) for i, s in bm25_hits if i < len(chunks) and chunks[i].document in allowed_docs]
        if hard_clause_type:
            bm25_hits = [(i, s) for i, s in bm25_hits if i < len(chunks) and clause_type_from_metadata(chunks[i]) == hard_clause_type]
        bm25_scores = {i: s for i, s in bm25_hits if i < len(chunks)}
        bm25_max = max(bm25_scores.values()) if bm25_scores else 1.0
        bm25_norm = {i: s / bm25_max for i, s in bm25_scores.items()}

        # Vector: pipeline calls ONLY store.retrieve(query, where_filter) — never collection.query(where=...)
        vector_hits = self.store.retrieve(query, where_filter=where_filter, k=k * 4)
        vector_sim = {}
        for chunk_id, sim in vector_hits:
            try:
                idx = int(chunk_id.replace("chunk_", ""))
            except ValueError:
                continue
            if idx >= len(chunks):
                continue
            if allowed_docs is not None and chunks[idx].document not in allowed_docs:
                continue
            if hard_clause_type and clause_type_from_metadata(chunks[idx]) != hard_clause_type:
                continue
            vector_sim[idx] = sim

        # Soft merge: boost by clause_type (single type or list of types)
        preferred_type = (metadata_filter or {}).get("clause_type")
        preferred_types: list = [preferred_type] if isinstance(preferred_type, str) else (preferred_type if isinstance(preferred_type, list) else [])
        BOOST = 1.25
        all_indices = set(bm25_norm) | set(vector_sim)
        combined = []
        for idx in all_indices:
            b = bm25_norm.get(idx, 0.0)
            v = vector_sim.get(idx, 0.0)
            score = self.bm25_weight * b + self.vector_weight * v
            if preferred_types and idx < len(chunks):
                tags = clause_types_from_metadata(chunks[idx])
                if any(t in tags for t in preferred_types):
                    score *= BOOST
            combined.append((idx, score))

        combined.sort(key=lambda x: x[1], reverse=True)
        seen = set()
        results: List[RetrievedChunk] = []
        candidate_limit = k * 3 if self.use_reranker else k
        for idx, score in combined[:candidate_limit]:
            if idx in seen:
                continue
            seen.add(idx)
            if idx < len(chunks):
                results.append(
                    RetrievedChunk(chunk=chunks[idx], score=score, source="hybrid")
                )

        if self.use_reranker and results:
            reranker = self._get_reranker()
            if reranker is not None:
                reranker_top_k = get_settings().reranker_top_k or (k * 3)
                candidates = results[: min(len(results), reranker_top_k)]
                results = reranker.rerank(query, candidates, top_k=k)

        return results[:k]
