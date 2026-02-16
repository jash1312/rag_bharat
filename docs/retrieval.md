# Retrieval

## Overview

Retrieval is **hybrid**: BM25 (keyword) + vector similarity (semantic), then merge, deduplicate, and return top-k chunks.

## Components

- **Embeddings** (`rag/retrieval/embeddings.py`) – OpenAI embedding API
- **Index store** (`rag/retrieval/store.py`) – ChromaDB (vectors) + persisted BM25 index
- **Hybrid search** (`rag/retrieval/hybrid_search.py`) – combine BM25 and vector scores with configurable weights

## Flow

1. **Query** – user or intent-refined query string
2. **BM25** – tokenize query, score against corpus, return top candidates
3. **Vector** – embed query, query ChromaDB, get top by distance
4. **Merge** – normalize scores, compute `bm25_weight * bm25_norm + vector_weight * vector_sim`
5. **Deduplicate** – by chunk index, keep highest combined score
6. **Top-K** – return up to `top_k` `RetrievedChunk` items

## Config

- `top_k` – number of chunks returned (default: 5)
- `bm25_weight`, `vector_weight` – blend (default 0.5 / 0.5)

## Usage

**Programmatic:**

```python
from rag.retrieval.store import IndexStore
from rag.retrieval.hybrid_search import HybridRetrieval

store = IndexStore(persist_directory=Path("data/chroma"))
retrieval = HybridRetrieval(store=store, top_k=5)
results = retrieval.search("What is the notice period for termination?")
```

**Notebook:** `notebooks/02_retrieval.ipynb` – set query and top_k in config, then call `retrieval.search()`.

## Re-ranking (BGE)

When `USE_RERANKER=true` (default), a cross-encoder re-ranker (**BAAI/bge-reranker-base**) runs after hybrid search: more candidates are retrieved (e.g. `top_k * 3`), then the re-ranker scores each (query, chunk) pair and returns the top-k. This improves precision significantly. Disable with `USE_RERANKER=false` or in code by passing `use_reranker=False` to `HybridRetrieval`.
