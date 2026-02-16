"""Pydantic models for chunks, queries, and agent outputs."""

from typing import Any, Optional

from pydantic import BaseModel, Field


class ChunkMetadata(BaseModel):
    """Metadata attached to a clause chunk. Multiple tags per clause via clause_types."""

    governing_law: bool = False
    liability_related: bool = False
    termination_related: bool = False
    confidentiality_related: bool = False
    indemnification_related: bool = False
    clause_types: list[str] = Field(default_factory=list, description="All applicable tags, e.g. ['termination', 'confidentiality']")
    keywords: list[str] = Field(default_factory=list, description="e.g. ['uptime', 'availability'] for SLA Service Availability")
    document_type: Optional[str] = Field(default=None, description="e.g. SLA, NDA, DPA, VSA for hybrid filtering")
    primary_clause_type: Optional[str] = Field(default=None, description="Single clause_type for Chroma filter: uptime, liability, governing_law, etc.")
    custom: dict[str, Any] = Field(default_factory=dict)


class ClauseChunk(BaseModel):
    """A single clause-level chunk for indexing and retrieval."""

    document: str = Field(..., description="Document identifier (e.g. NDA, SLA)")
    section: str = Field(..., description="Section number or id")
    title: str = Field(..., description="Section/clause title")
    clause_text: str = Field(..., description="Full clause text")
    metadata: ChunkMetadata = Field(default_factory=ChunkMetadata)
    chunk_id: Optional[str] = Field(default=None, description="Unique id after indexing")

    def to_search_text(self) -> str:
        """Text used for BM25 and display."""
        return f"{self.document} | {self.title}\n{self.clause_text}"


class IntentResult(BaseModel):
    """Output of intent classification."""

    intent_type: str = Field(..., description="e.g. clause_retrieval, liability_assessment, governing_law")
    document_focus: list[str] = Field(default_factory=list, description="Relevant document names if inferred")
    is_legal_advice: bool = Field(default=False, description="If true, should refuse")
    requires_clarification: bool = Field(default=False, description="Query ambiguous")
    refined_query: Optional[str] = Field(default=None)
    topic: Optional[str] = Field(
        default=None,
        description="Semantic topic for clause_retrieval: termination, liability, governing_law, confidentiality, service_level, indemnification, general",
    )


class ClauseAnalysis(BaseModel):
    """Structured output from clause analyzer agent (legacy list-based)."""

    obligations: list[str] = Field(default_factory=list)
    exceptions: list[str] = Field(default_factory=list)
    caps: list[str] = Field(default_factory=list)
    survival_clauses: list[str] = Field(default_factory=list)
    remedies: list[str] = Field(default_factory=list)
    raw_summary: Optional[str] = None


class ExtractedClauses(BaseModel):
    """Strict extraction: verbatim sentences only. No summarization. Null if not present."""

    notice_period: Optional[str] = None
    liability_cap: Optional[str] = None
    survival_clause: Optional[str] = None
    governing_law: Optional[str] = None
    uptime_commitment: Optional[str] = None
    damage_exclusion: Optional[str] = None
    notice_period_source: Optional[str] = Field(default=None, description="CHUNK_X id from extractor")
    liability_cap_source: Optional[str] = None
    survival_clause_source: Optional[str] = None
    governing_law_source: Optional[str] = None
    uptime_commitment_source: Optional[str] = None
    damage_exclusion_source: Optional[str] = None
    governing_law_by_document: Optional[dict[str, str]] = Field(
        default=None,
        description="When intent is cross_document: governing law clause text per document name.",
    )
    extracted_by_document: Optional[dict[str, dict[str, str | None]]] = Field(
        default=None,
        description="When intent is cross_document: liability_summary etc. per document for comparison.",
    )


class RiskAssessment(BaseModel):
    """Output from risk assessment (rule engine or agent)."""

    risk_level: str = Field(..., description="LOW, MEDIUM, HIGH")
    reason: str = Field(default="")
    financial_exposure: Optional[str] = None
    risk_patterns: list[str] = Field(default_factory=list)
    # Deterministic: when potential_unlimited_liability, use this wording (no binary yes/no)
    prescribed_liability_wording: Optional[str] = Field(
        default=None,
        description="Legally nuanced wording when no cap is found (e.g. 'The agreement does not specify a liability cap. This may result in uncapped liability exposure.')",
    )


class RetrievedChunk(BaseModel):
    """Chunk with retrieval score for pipeline use."""

    chunk: ClauseChunk
    score: float
    source: str = Field(default="hybrid", description="bm25, vector, or hybrid")


class SynthesizedResponse(BaseModel):
    """Final response from response composer."""

    direct_answer: str
    cited_clauses: list[dict[str, str]] = Field(default_factory=list)
    risk_flags: list[dict[str, Any]] = Field(default_factory=list)
    follow_up_suggestions: list[str] = Field(default_factory=list)
    insufficient_info: bool = Field(default=False)
