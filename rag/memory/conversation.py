"""Conversation memory: dialogue history, document focus, previous risks."""

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Turn:
    """Single turn: user query and system response summary."""

    query: str
    response_summary: str = ""
    referenced_documents: list[str] = field(default_factory=list)
    risks_mentioned: list[str] = field(default_factory=list)


class ConversationMemory:
    """Short-term session memory: turns plus current semantic topic and document for multi-turn rewriting."""

    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        self._turns: list[Turn] = []
        self._current_topic: str | None = None
        self._current_document: str | None = None
        self._last_intent: str | None = None

    def add(
        self,
        query: str,
        response_summary: str = "",
        documents: list[str] | None = None,
        risks: list[str] | None = None,
        topic: str | None = None,
        document: str | None = None,
        last_intent: str | None = None,
    ) -> None:
        """Append a turn and update current_topic / current_document / last_intent for follow-up rewriting."""
        self._turns.append(
            Turn(
                query=query,
                response_summary=response_summary[:500],
                referenced_documents=documents or [],
                risks_mentioned=risks or [],
            )
        )
        if len(self._turns) > self.max_turns:
            self._turns = self._turns[-self.max_turns :]
        if topic is not None:
            self._current_topic = topic
        if document is not None:
            self._current_document = document
        if last_intent is not None:
            self._last_intent = last_intent

    def get_current_topic(self) -> str | None:
        return self._current_topic

    def get_current_document(self) -> str | None:
        return self._current_document

    def get_last_intent(self) -> str | None:
        return self._last_intent

    def get_context_for_intent(self, last_n: int = 3) -> str:
        """Return recent context string for intent/refinement."""
        if not self._turns:
            return ""
        recent = self._turns[-last_n:]
        parts = []
        for t in recent:
            parts.append(f"Q: {t.query}\nA: {t.response_summary[:200]}...")
        return "\n".join(parts)

    def get_referenced_documents(self) -> list[str]:
        """Union of documents referenced in recent turns."""
        out: set[str] = set()
        for t in self._turns:
            out.update(t.referenced_documents)
        return list(out)

    def get_last_turn(self) -> Optional["Turn"]:
        """Return the most recent turn (query + response summary) for contextual rewriting."""
        if not self._turns:
            return None
        return self._turns[-1]

    def clear(self) -> None:
        """Reset conversation and topic state."""
        self._turns.clear()
        self._current_topic = None
        self._current_document = None
        self._last_intent = None
