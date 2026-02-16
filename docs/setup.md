# Setup Guide

## Prerequisites

- Python 3.10+
- **Local (Ollama):** [Ollama](https://ollama.com) installed and running (no API key)
- **Cloud (OpenAI):** OpenAI API key

## Virtual environment and dependencies

```bash
# From project root
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Provider: local (Ollama) vs cloud (OpenAI)

The app supports two backends. **Default is Ollama** (fully local).

### Local with Ollama (no API key)

1. Install [Ollama](https://ollama.com) and start the server.
2. Pull models:
   - **Embeddings:** `nomic-embed-text` — good semantic performance, lightweight, works well for legal text.
   - **LLM (development):** `llama3:8b`. For stronger quality when GPU is available: `llama3:70b`.

   ```bash
   ollama pull nomic-embed-text
   ollama pull llama3:8b
   # optional, when GPU available:
   ollama pull llama3:70b
   ```

3. In `.env` (optional; these are the defaults):

   ```env
   PROVIDER=ollama
   EMBEDDING_MODEL=nomic-embed-text
   LLM_MODEL=llama3:8b
   LLM_MODEL_LARGE=llama3:70b
   USE_LARGE_LLM=false
   LLM_TEMPERATURE=0.1
   LLM_TOP_P=0.9
   LLM_MAX_TOKENS=2048
   USE_RERANKER=true
   RERANKER_MODEL=BAAI/bge-reranker-base
   OLLAMA_BASE_URL=http://localhost:11434
   ```

Set `USE_LARGE_LLM=true` when using a GPU and `llama3:70b`.

### Cloud with OpenAI

Set provider and API key in `.env`:

```env
PROVIDER=openai
OPENAI_API_KEY=sk-...
EMBEDDING_MODEL=text-embedding-3-small
LLM_MODEL=gpt-4o-mini
```

## Environment variables

Optional overrides (see `rag/config.py`):

- `PROVIDER` – `ollama` (default) or `openai`
- `EMBEDDING_MODEL` – `nomic-embed-text` (Ollama, recommended for legal)
- `LLM_MODEL` – development chat model (default: `llama3:8b`)
- `LLM_MODEL_LARGE` – larger model when GPU available (default: `llama3:70b`)
- `USE_LARGE_LLM` – set `true` to use `LLM_MODEL_LARGE`
- `LLM_TEMPERATURE` – `0.1` for deterministic outputs
- `LLM_TOP_P` – `0.9` for nucleus sampling
- `LLM_MAX_TOKENS` – max tokens per response (default: `2048`)
- `USE_RERANKER` – use BGE re-ranker after hybrid search (default: `true`)
- `RERANKER_MODEL` – `BAAI/bge-reranker-base` (improves precision)
- `INPUT_DATA_DIR`, `PERSIST_DIRECTORY`, `OLLAMA_BASE_URL`, `TOP_K`

## Data

Place contract documents as plain text (`.txt`) in `input_data/`. Supported examples:

- NDA
- Vendor Services Agreement
- Service Level Agreement
- Data Processing Agreement

## Run ingestion then CLI

```bash
python ingest.py
python main.py
```

## Notebooks

From project root, start Jupyter and open notebooks under `notebooks/`:

```bash
jupyter notebook notebooks/
```

Or use the kernel tied to `venv` so that `import rag` resolves (project root is the current working directory when opening `notebooks/`; notebooks add `../` to `sys.path`).
