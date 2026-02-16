"""Ingestion entrypoint: load contracts from input_data, parse, chunk, and index."""

import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rag.config import get_settings
from rag.ingestion.pipeline import run_ingestion


def main() -> None:
    settings = get_settings()
    input_dir = Path(settings.input_data_dir)
    if not input_dir.is_absolute():
        input_dir = Path.cwd() / input_dir
    if not input_dir.exists():
        print(f"Input directory not found: {input_dir}")
        sys.exit(1)
    chunks = run_ingestion(input_dir=input_dir, index_to_store=True)
    print(f"Ingestion complete. Indexed {len(chunks)} clause chunks.")


if __name__ == "__main__":
    main()
