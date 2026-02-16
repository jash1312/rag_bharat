"""CLI entrypoint: interactive console for legal contract Q&A."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rag.config import get_settings
from rag.orchestration import Orchestrator
from rag.retrieval.store import IndexStore


def main() -> None:
    settings = get_settings()
    store = IndexStore(persist_directory=settings.persist_directory)
    if not store.get_chunks():
        print("No indexed data found. Run: python ingest.py")
        sys.exit(1)
    orchestrator = Orchestrator(index_store=store)
    print("Legal Contract Q&A. Type 'quit' or 'exit' to stop.\n")
    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q or q.lower() in ("quit", "exit"):
            break
        resp = orchestrator.run(q)
        print("\nAnswer:", resp.direct_answer)
        if resp.cited_clauses:
            print("Cited:", resp.cited_clauses)
        if resp.risk_flags:
            print("Risks:", resp.risk_flags)
        if resp.follow_up_suggestions:
            print("Follow-up:", resp.follow_up_suggestions)
        print()
    print("Goodbye.")


if __name__ == "__main__":
    main()
