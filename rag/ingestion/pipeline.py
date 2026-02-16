"""Ingestion pipeline: read input_data, parse, chunk, and optionally index."""

from pathlib import Path
from typing import Callable, Optional

from tqdm import tqdm

from rag.config import get_settings
from rag.ingestion.parser import DocumentParser, load_document
from rag.ingestion.chunker import ClauseChunker
from rag.models import ClauseChunk


def collect_chunks_from_directory(
    input_dir: Path,
    document_id_from_path: Optional[Callable[[Path], str]] = None,
    file_glob: str = "*.txt",
) -> list[ClauseChunk]:
    """
    Scan input_dir for files matching file_glob, parse each into clause chunks,
    and return a flat list of ClauseChunk.
    """
    settings = get_settings()
    input_path = Path(input_dir) if isinstance(input_dir, str) else input_dir
    if not input_path.is_absolute():
        input_path = Path.cwd() / input_path
    files = sorted(input_path.glob(file_glob))
    chunker = ClauseChunker(
        max_chunk_tokens=settings.max_chunk_tokens,
        overlap_tokens=settings.overlap_tokens,
    )
    all_chunks: list[ClauseChunk] = []
    for path in tqdm(files, desc="Parsing documents"):
        doc_id = (document_id_from_path or _default_doc_id)(path)
        raw = load_document(path, doc_id)
        parser = DocumentParser(doc_id)
        for clause in parser.parse_to_chunks(raw):
            for sub in chunker.chunk_clause(clause):
                all_chunks.append(sub)
    return all_chunks


def _default_doc_id(path: Path) -> str:
    """Default: use stem, normalized (e.g. nda_acme_vendor -> NDA Acme Vendor)."""
    stem = path.stem
    return stem.replace("_", " ").title()


def run_ingestion(
    input_dir: Optional[Path] = None,
    persist_directory: Optional[Path] = None,
    index_to_store: bool = True,
) -> list[ClauseChunk]:
    """
    Run full ingestion: collect chunks from input_dir, then optionally
    index them into the vector store and BM25. Returns list of chunks.
    """
    settings = get_settings()
    input_path = input_dir or settings.input_data_dir
    if not input_path.is_absolute():
        input_path = Path.cwd() / input_path
    chunks = collect_chunks_from_directory(input_path)
    if index_to_store and chunks:
        from rag.retrieval.store import IndexStore
        store = IndexStore(persist_directory=persist_directory or settings.persist_directory)
        store.index_chunks(chunks)
    return chunks
