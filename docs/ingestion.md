# Ingestion

## Overview

The ingestion pipeline reads legal documents from `input_data/`, parses them into clause-level chunks, and indexes them for retrieval (ChromaDB + BM25).

## Steps

1. **Document parsing** (`rag/ingestion/parser.py`)
   - Extract section headings and clause numbers
   - Normalize formatting
   - Infer metadata (governing law, liability, termination, confidentiality, indemnification) from keywords

2. **Chunking** (`rag/ingestion/chunker.py`)
   - Keep clauses as primary units
   - If a clause exceeds `max_chunk_tokens`, split with overlap
   - Preserve document, section, title, and metadata on each chunk

3. **Indexing** (`rag/retrieval/store.py`)
   - Embed chunk text via OpenAI
   - Upsert into ChromaDB (vector store)
   - Build BM25 index over the same corpus and persist to `data/bm25_index/`

## Chunk schema

Each chunk follows the structure in `rag/models.py`:

- `document` – e.g. "Nda Acme Vendor"
- `section` – section or clause number
- `title` – section/clause title
- `clause_text` – full text of the clause
- `metadata` – boolean flags (governing_law, liability_related, etc.)

## Usage

**CLI:**

```bash
python ingest.py
```

**Programmatic:**

```python
from rag.ingestion.pipeline import run_ingestion
from pathlib import Path

chunks = run_ingestion(
    input_dir=Path("input_data"),
    persist_directory=Path("data/chroma"),
    index_to_store=True,
)
```

**Notebook:** Use `notebooks/01_ingestion.ipynb` with config (paths, `index_to_store`) and call `run_ingestion()`.

## Config

- `input_data_dir` – source directory
- `persist_directory` – ChromaDB path
- `max_chunk_tokens`, `overlap_tokens` – chunking behavior
