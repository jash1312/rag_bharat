"""Index store: ChromaDB for vectors, BM25 for keyword search."""

import json
import math
import pickle
from pathlib import Path
from typing import List, Optional

from rag.config import get_settings
from rag.models import ClauseChunk
from rag.retrieval.embeddings import get_embeddings


def clause_type_from_metadata(c: ClauseChunk) -> str:
    """Primary clause_type for Chroma filter. Uses primary_clause_type when set (ingestion), else infers from clause_types/booleans."""
    m = c.metadata
    primary = getattr(m, "primary_clause_type", None)
    if primary:
        return primary
    types = getattr(m, "clause_types", None) or []
    if "service_availability" in types:
        return "uptime"
    if "service_credits" in types:
        return "service_credit"
    if m.governing_law:
        return "governing_law"
    if m.liability_related or "liability" in types or "damage_exclusion" in types:
        return "liability"
    if m.termination_related:
        return "termination"
    if m.confidentiality_related:
        return "confidentiality"
    if m.indemnification_related:
        return "indemnification"
    return "other"


def clause_types_from_metadata(c: ClauseChunk) -> list[str]:
    """All applicable tags for soft-merge boosting. Maps service_availability -> uptime, service_credits -> service_credit for filter compatibility."""
    m = c.metadata
    if getattr(m, "clause_types", None) and len(m.clause_types) > 0:
        tags = list(m.clause_types)
        if "service_availability" in tags and "uptime" not in tags:
            tags.append("uptime")
        if "service_credits" in tags and "service_credit" not in tags:
            tags.append("service_credit")
        return tags
    tags: list[str] = []
    if m.governing_law:
        tags.append("governing_law")
    if m.liability_related:
        tags.append("liability")
    if m.termination_related:
        tags.append("termination")
    if m.confidentiality_related:
        tags.append("confidentiality")
    if m.indemnification_related:
        tags.append("indemnification")
    if not tags:
        tags.append("other")
    return tags


class IndexStore:
    """Persist clause chunks in ChromaDB and BM25 index."""

    def __init__(
        self,
        persist_directory: Optional[Path] = None,
        collection_name: str = "legal_clauses",
    ):
        settings = get_settings()
        self.persist_directory = Path(persist_directory or settings.persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self._chroma = None
        self._bm25_corpus: List[str] = []
        self._bm25_index = None
        self._chunks: List[ClauseChunk] = []
        self._load_or_create()

    def _load_or_create(self) -> None:
        import chromadb
        from chromadb.config import Settings as ChromaSettings
        self._chroma = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self._chroma.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Legal clause chunks"},
        )
        bm25_path = self.persist_directory.parent / "bm25_index"
        bm25_path.mkdir(parents=True, exist_ok=True)
        self._bm25_path = bm25_path
        self._chunks_path = bm25_path / "chunks.json"
        self._load_bm25()

    def _load_bm25(self) -> None:
        if self._chunks_path.exists():
            data = json.loads(self._chunks_path.read_text(encoding="utf-8"))
            self._chunks = [ClauseChunk.model_validate(c) for c in data]
            self._bm25_corpus = [c.to_search_text() for c in self._chunks]
            bm25_pkl = self._bm25_path / "bm25.pkl"
            if bm25_pkl.exists():
                with open(bm25_pkl, "rb") as f:
                    self._bm25_index = pickle.load(f)
            else:
                self._bm25_index = None
        else:
            self._chunks = []
            self._bm25_corpus = []
            self._bm25_index = None

    def index_chunks(self, chunks: List[ClauseChunk]) -> None:
        """Index chunks into ChromaDB and BM25."""
        if not chunks:
            return
        texts = [c.to_search_text() for c in chunks]
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        for i, c in enumerate(chunks):
            c.chunk_id = ids[i]
        embeddings = get_embeddings(texts)
        # clause_type enables metadata_filter in search (e.g. clause_retrieval + topic=termination). Re-index required for existing collections.
        metadatas = []
        for c in chunks:
            meta = {
                "document": c.document,
                "section": c.section,
                "title": c.title,
                "clause_text": c.clause_text[:10000],
                "governing_law": c.metadata.governing_law,
                "liability_related": c.metadata.liability_related,
                "clause_type": clause_type_from_metadata(c),
            }
            if getattr(c.metadata, "document_type", None):
                meta["document_type"] = c.metadata.document_type
            metadatas.append(meta)
        self.collection.upsert(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
        self._chunks = chunks
        self._bm25_corpus = texts
        from rank_bm25 import BM25Okapi
        tokenized = [t.split() for t in texts]
        self._bm25_index = BM25Okapi(tokenized)
        self._bm25_path.mkdir(parents=True, exist_ok=True)
        self._chunks_path.write_text(
            json.dumps([c.model_dump(mode="json") for c in chunks], indent=0),
            encoding="utf-8",
        )
        with open(self._bm25_path / "bm25.pkl", "wb") as f:
            pickle.dump(self._bm25_index, f)

    def get_chunks(self) -> List[ClauseChunk]:
        """Return currently loaded chunks (from last BM25 load/index)."""
        return self._chunks

    def get_all_documents(self) -> List[str]:
        """Return unique document names in the index (for debug: verify SLA / Service Level Agreement is ingested)."""
        if not self._chunks:
            return []
        return sorted(set(c.document for c in self._chunks if c.document))

    def get_metadata_keys(self) -> set:
        """Return metadata keys from one stored document. Used to avoid filtering on clause_type when index was built without it."""
        try:
            res = self.collection.get(include=["metadatas"])
            metas = res.get("metadatas") or []
            if metas and metas[0]:
                return set(metas[0].keys())
        except Exception:
            pass
        return set()

    def similarity_search(self, query: str, k: int = 5, where: Optional[dict] = None) -> List[ClauseChunk]:
        """Run vector similarity search; uses retrieve(query, where_filter=where) only."""
        hits = self.retrieve(query, where_filter=where, k=k)
        out: List[ClauseChunk] = []
        for chunk_id, _ in hits:
            try:
                idx = int(chunk_id.replace("chunk_", ""))
            except ValueError:
                continue
            if 0 <= idx < len(self._chunks):
                out.append(self._chunks[idx])
        return out

    def bm25_search(self, query: str, top_k: int = 10) -> List[tuple[int, float]]:
        """Return list of (chunk_index, score) from BM25."""
        if not self._bm25_index:
            return []
        tokenized_query = query.split()
        scores = self._bm25_index.get_scores(tokenized_query)
        indexed = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(i, float(scores[i])) for i in indexed if scores[i] > 0]

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Cosine similarity in [0, 1] for non-negative embeddings; 0 if either norm is 0."""
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        if na <= 0 or nb <= 0:
            return 0.0
        sim = dot / (na * nb)
        return max(0.0, min(1.0, sim))

    def restricted_vector_search(
        self,
        query: str,
        where_filter: dict,
        top_k: int = 10,
    ) -> List[tuple[str, float]]:
        """Filtered path: embed query once, use Chroma query(where=...) — no document re-embedding."""
        from rag.retrieval.embeddings import embed_single

        query_emb = embed_single(query)
        try:
            res = self.collection.query(
                query_embeddings=[query_emb],
                where=where_filter,
                n_results=top_k,
                include=["metadatas"],
            )
        except Exception:
            return []

        if not res or not res.get("ids") or not res["ids"][0]:
            return []

        ids = res["ids"][0]
        dists = res["distances"][0] if res.get("distances") else [0.0] * len(ids)
        # Same 0–1 scale as unfiltered: L2 distance → cosine-like similarity
        return [
            (cid, max(0.0, min(1.0, 1.0 - (float(d) ** 2) / 2.0)) if d is not None else 1.0)
            for cid, d in zip(ids, dists)
        ]

    def retrieve(
        self,
        query: str,
        where_filter: Optional[dict] = None,
        k: int = 5,
    ) -> List[tuple[str, float]]:
        """
        Single entry point. Returns (chunk_id, similarity) in 0–1 range.
        Filtered: one query embed + collection.query(where=..., n_results=k). Unfiltered: same, no where.
        """
        if where_filter:
            return self.restricted_vector_search(query, where_filter, top_k=k)
        from rag.retrieval.embeddings import embed_single
        query_emb = embed_single(query)
        hits = self._vector_search_unfiltered(query_emb, top_k=k)
        # Standardize to 0–1 similarity everywhere: Chroma L2 distance → cosine-like (1 - d²/2 for normalized)
        return [
            (cid, max(0.0, min(1.0, 1.0 - (float(dist) ** 2) / 2.0)) if dist is not None else 1.0)
            for cid, dist in hits
        ]

    def _vector_search_unfiltered(
        self,
        query_embedding: List[float],
        top_k: int = 10,
    ) -> List[tuple[str, float]]:
        """Unfiltered path only. Uses collection.query with NO where — never use for document scoping."""
        res = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["metadatas"],
        )
        if not res or not res["ids"] or not res["ids"][0]:
            return []
        ids = res["ids"][0]
        dists = res["distances"][0] if res.get("distances") else [0.0] * len(ids)
        return list(zip(ids, [float(d) for d in dists]))
