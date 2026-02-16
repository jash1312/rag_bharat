"""Load prompt templates and evaluation queries from the prompts directory."""

import json
from pathlib import Path

_PROMPTS_DIR = Path(__file__).resolve().parent


def load_prompt(name: str) -> str:
    """Load a prompt by name (e.g. 'intent_system') from rag/prompts/<name>.txt."""
    path = _PROMPTS_DIR / f"{name}.txt"
    return path.read_text(encoding="utf-8").strip()


def load_evaluation_queries() -> list[str]:
    """Load the list of evaluation query strings from evaluation_queries.json."""
    path = _PROMPTS_DIR / "evaluation_queries.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(data)
