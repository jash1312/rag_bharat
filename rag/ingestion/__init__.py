"""Ingestion pipeline: parsing and chunking of legal documents."""

from rag.ingestion.parser import DocumentParser
from rag.ingestion.chunker import ClauseChunker
from rag.ingestion.pipeline import run_ingestion

__all__ = ["DocumentParser", "ClauseChunker", "run_ingestion"]
