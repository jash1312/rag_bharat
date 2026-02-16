"""Chunking: split long clauses if needed, enforce max token approx."""

import re
from typing import Iterator

from rag.models import ClauseChunk, ChunkMetadata


def _approx_tokens(text: str) -> int:
    """Approximate token count (4 chars per token)."""
    return max(1, len(text) // 4)


class ClauseChunker:
    """Optionally split long clauses into smaller chunks with overlap."""

    def __init__(self, max_chunk_tokens: int = 512, overlap_tokens: int = 50):
        self.max_chunk_tokens = max_chunk_tokens
        self.overlap_tokens = overlap_tokens

    def chunk_clause(self, clause: ClauseChunk) -> Iterator[ClauseChunk]:
        """
        Yield one or more ClauseChunk(s). If clause text is short enough, yield as-is.
        Otherwise split by paragraphs/sentences with overlap.
        """
        text = clause.clause_text
        if _approx_tokens(text) <= self.max_chunk_tokens:
            yield clause
            return

        # Split by double newline first, then by sentence
        parts = re.split(r"\n\n+", text)
        current = []
        current_tokens = 0
        sub_idx = 0

        for part in parts:
            part_tokens = _approx_tokens(part)
            if current_tokens + part_tokens > self.max_chunk_tokens and current:
                sub_idx += 1
                yield ClauseChunk(
                    document=clause.document,
                    section=clause.section,
                    title=clause.title + (f" (part {sub_idx})" if sub_idx > 1 else ""),
                    clause_text="\n\n".join(current),
                    metadata=clause.metadata,
                )
                # Overlap: keep last N tokens
                overlap_text = _take_last_tokens("\n\n".join(current), self.overlap_tokens)
                current = [overlap_text] if overlap_text else []
                current_tokens = _approx_tokens(overlap_text)

            current.append(part)
            current_tokens += part_tokens

        if current:
            sub_idx += 1
            yield ClauseChunk(
                document=clause.document,
                section=clause.section,
                title=clause.title + (f" (part {sub_idx})" if sub_idx > 1 else ""),
                clause_text="\n\n".join(current),
                metadata=clause.metadata,
            )


def _take_last_tokens(text: str, n_tokens: int) -> str:
    """Keep approximately the last n_tokens of text."""
    approx_chars = n_tokens * 4
    if len(text) <= approx_chars:
        return text
    return text[-approx_chars:].lstrip()
