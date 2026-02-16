"""Configuration for the RAG system."""

from pathlib import Path
from typing import Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment and .env."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Provider: "openai" (cloud) or "ollama" (local, no API key needed)
    provider: Literal["openai", "ollama"] = Field(
        default="ollama",
        description="LLM/embedding provider: openai or ollama (local)",
    )

    # Paths
    input_data_dir: Path = Field(default=Path("input_data"), description="Directory containing contract documents")
    persist_directory: Path = Field(default=Path("data/chroma"), description="ChromaDB persist path")
    bm25_index_path: Optional[Path] = Field(default=Path("data/bm25_index"), description="BM25 index directory")

    # Embedding: nomic-embed-text — good semantic performance, lightweight, works well for legal text
    embedding_model: str = Field(
        default="nomic-embed-text",
        description="Embedding model (Ollama/OpenAI); nomic-embed-text recommended for legal",
    )
    embedding_dim: int = Field(default=768, description="Embedding dimension (768 for nomic-embed-text)")

    # LLM: llama3:8b for development; llama3:70b when GPU available
    llm_model: str = Field(
        default="llama3:8b",
        description="Chat model for development (e.g. llama3:8b)",
    )
    llm_model_large: str = Field(
        default="llama3:70b",
        description="Chat model when GPU available (e.g. llama3:70b)",
    )
    use_large_llm: bool = Field(
        default=False,
        description="If True, use llm_model_large (set when GPU available)",
    )
    llm_temperature: float = Field(default=0.1, description="Temperature for deterministic outputs")
    llm_top_p: float = Field(default=0.9, description="Top-p (nucleus) sampling for LLM")
    llm_max_tokens: int = Field(default=2048, description="Max tokens per LLM response")

    # Ollama (only used when provider=ollama)
    ollama_base_url: str = Field(default="http://localhost:11434", description="Ollama server URL")

    # Retrieval
    top_k: int = Field(default=5, description="Number of chunks to retrieve")
    bm25_weight: float = Field(default=0.5, description="Weight for BM25 in hybrid search (0-1)")
    vector_weight: float = Field(default=0.5, description="Weight for vector search in hybrid (0-1)")

    # Re-ranker (optional; disabled by default)
    use_reranker: bool = Field(default=False, description="Use cross-encoder re-ranker after hybrid search")
    reranker_model: str = Field(
        default="BAAI/bge-reranker-base",
        description="Re-ranker model (e.g. BAAI/bge-reranker-base)",
    )
    reranker_top_k: Optional[int] = Field(
        default=None,
        description="Max candidates to pass to re-ranker (default: top_k * 3); final count is top_k",
    )

    # Chunking
    max_chunk_tokens: int = Field(default=512, description="Max tokens per chunk (approximate)")
    overlap_tokens: int = Field(default=50, description="Overlap between chunks")

    # OpenAI (only used when provider=openai)
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key (or set OPENAI_API_KEY)")


def get_settings() -> Settings:
    """Return application settings."""
    return Settings()
