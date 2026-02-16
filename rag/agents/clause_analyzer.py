"""Clause analyzer: strict extraction only. One LLM call. Temperature 0. No summarization."""

import re

from rag.models import ExtractedClauses, RetrievedChunk
from rag.agents.llm import chat_completion, extract_json_from_text
from rag.prompts import load_prompt


def extract_damage_exclusion(text: str) -> str | None:
    """Rule-based extraction of damage exclusion clauses. Strong patterns to avoid random lines (fixes RK2)."""
    if not text or not text.strip():
        return None
    patterns = [
        r"shall not be liable for (?:any )?(?:indirect|incidental|consequential|special|punitive).*?damages",
        r"excluding (?:indirect|incidental|consequential|special|punitive).*?damages",
        r"in no event shall .*? be liable for .*? damages",
        r"not liable for (?:any )?(?:indirect|incidental|consequential|special|punitive) damages",
        r"exclusion of (?:indirect|incidental|consequential|special) damages",
        r"sole and exclusive remedy",
        r"sole remedy.*?for.*?failure",
    ]
    for p in patterns:
        match = re.search(p, text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(0).strip()
    return None


def _str_or_none(v) -> str | None:
    if v is None:
        return None
    if isinstance(v, str) and v.strip():
        return v.strip()
    return None


class ClauseAnalyzerAgent:
    """Extract verbatim clauses only. Returns ExtractedClauses (notice_period, liability_cap, survival_clause, governing_law, uptime_commitment)."""

    def analyze(self, retrieved: list[RetrievedChunk]) -> ExtractedClauses:
        """Extract strict JSON from retrieved clause text. No summarization."""
        if not retrieved:
            return ExtractedClauses()
        context = "\n\n---\n\n".join(
            f"[CHUNK_{i}] [{r.chunk.document} - {r.chunk.title}]\n{r.chunk.clause_text}"
            for i, r in enumerate(retrieved[:5])
        )
        content = chat_completion(
            [
                {"role": "system", "content": load_prompt("clause_extractor_system")},
                {"role": "user", "content": f"Extract from these clauses:\n\n{context}"},
            ],
            max_tokens=512,
            temperature=0.0,
        )
        data = extract_json_from_text(content)
        damage_exclusion = _str_or_none(data.get("damage_exclusion")) if data else None
        if not damage_exclusion:
            damage_exclusion = extract_damage_exclusion(context)
        if damage_exclusion and not damage_exclusion.strip():
            damage_exclusion = None
        if data:
            return ExtractedClauses(
                notice_period=_str_or_none(data.get("notice_period")),
                liability_cap=_str_or_none(data.get("liability_cap")),
                survival_clause=_str_or_none(data.get("survival_clause")),
                governing_law=_str_or_none(data.get("governing_law")),
                uptime_commitment=_str_or_none(data.get("uptime_commitment")),
                damage_exclusion=damage_exclusion,
                notice_period_source=_str_or_none(data.get("notice_period_source")),
                liability_cap_source=_str_or_none(data.get("liability_cap_source")),
                survival_clause_source=_str_or_none(data.get("survival_clause_source")),
                governing_law_source=_str_or_none(data.get("governing_law_source")),
                uptime_commitment_source=_str_or_none(data.get("uptime_commitment_source")),
                damage_exclusion_source=_str_or_none(data.get("damage_exclusion_source")),
            )
        return ExtractedClauses()

    def extract_governing_law_per_document(self, retrieved: list[RetrievedChunk]) -> dict[str, str]:
        """
        For cross_document intent: extract governing_law from each document independently.
        Returns dict document_name -> governing_law clause text (or "Not found" if absent).
        """
        from collections import defaultdict
        by_doc: dict[str, list[RetrievedChunk]] = defaultdict(list)
        for r in retrieved:
            by_doc[r.chunk.document].append(r)
        result: dict[str, str] = {}
        for doc_name, chunks in by_doc.items():
            context = "\n\n---\n\n".join(
                f"[{r.chunk.document} - {r.chunk.title}]\n{r.chunk.clause_text}" for r in chunks[:2]
            )
            content = chat_completion(
                [
                    {"role": "system", "content": load_prompt("clause_extractor_system")},
                    {"role": "user", "content": f"Extract only the governing_law key (search for 'governed by the laws of') from these clauses. Return JSON with key governing_law.\n\n{context}"},
                ],
                max_tokens=256,
                temperature=0.0,
            )
            data = extract_json_from_text(content)
            gl = _str_or_none(data.get("governing_law")) if data else None
            result[doc_name] = gl if gl else "Not found"
        return result

    def extract_all_per_document(self, retrieved: list[RetrievedChunk]) -> dict[str, dict[str, str | None]]:
        """
        For cross_document: extract governing_law, liability_cap, damage_exclusion, etc. per document.
        Returns liability_summary = { doc_name: { "governing_law": ..., "liability_cap": ..., "damage_exclusion": ..., ... } }.
        """
        from collections import defaultdict
        by_doc: dict[str, list[RetrievedChunk]] = defaultdict(list)
        for r in retrieved:
            by_doc[r.chunk.document].append(r)
        result: dict[str, dict[str, str | None]] = {}
        for doc_name, chunks in by_doc.items():
            context = "\n\n---\n\n".join(
                f"[{r.chunk.document} - {r.chunk.title}]\n{r.chunk.clause_text}" for r in chunks[:3]
            )
            content = chat_completion(
                [
                    {"role": "system", "content": load_prompt("clause_extractor_system")},
                    {"role": "user", "content": f"Extract from these clauses. Return JSON with keys: governing_law, liability_cap, damage_exclusion, notice_period, survival_clause, uptime_commitment.\n\n{context}"},
                ],
                max_tokens=512,
                temperature=0.0,
            )
            data = extract_json_from_text(content)
            if data:
                result[doc_name] = {
                    "governing_law": _str_or_none(data.get("governing_law")),
                    "liability_cap": _str_or_none(data.get("liability_cap")),
                    "damage_exclusion": _str_or_none(data.get("damage_exclusion")),
                    "notice_period": _str_or_none(data.get("notice_period")),
                    "survival_clause": _str_or_none(data.get("survival_clause")),
                    "uptime_commitment": _str_or_none(data.get("uptime_commitment")),
                }
            else:
                result[doc_name] = {
                    "governing_law": None,
                    "liability_cap": None,
                    "damage_exclusion": None,
                    "notice_period": None,
                    "survival_clause": None,
                    "uptime_commitment": None,
                }
        return result
