"""100% rule-based risk engine. No LLM. Deterministic from extracted clauses and retrieved chunks."""

from rag.models import ExtractedClauses, RetrievedChunk, RiskAssessment


UNLIMITED_LIABILITY_WORDING = (
    "No explicit limitation of liability is specified."
)


def _has_liability_content(retrieved: list[RetrievedChunk]) -> bool:
    if not retrieved:
        return False
    for r in retrieved:
        if r.chunk.metadata.liability_related:
            return True
        t = (r.chunk.clause_text or "").lower()
        if "liability" in t or "cap" in t or "damages" in t or "limit" in t:
            return True
    return False


def _jurisdictions_in_text(text: str) -> set[str]:
    """Extract jurisdiction names from text (e.g. governing law clause)."""
    t = (text or "").lower()
    out = set()
    if "california" in t or "california law" in t:
        out.add("California")
    if "england" in t or "wales" in t or "english law" in t:
        out.add("England and Wales")
    if "gdpr" in t or "eu " in t or "european" in t:
        out.add("EU/GDPR")
    return out


def _multiple_governing_laws(retrieved: list[RetrievedChunk]) -> bool:
    if not retrieved:
        return False
    jurisdictions = set()
    for r in retrieved:
        jurisdictions |= _jurisdictions_in_text(r.chunk.clause_text or "")
    return len(jurisdictions) > 1


def _unique_governing_laws_from_extracted(extracted: ExtractedClauses) -> set[str]:
    """Unique jurisdictions from extracted governing_law and governing_law_by_document."""
    out = set()
    if extracted.governing_law:
        out |= _jurisdictions_in_text(extracted.governing_law)
    by_doc = getattr(extracted, "governing_law_by_document", None) or {}
    for text in by_doc.values():
        if text and text != "Not found":
            out |= _jurisdictions_in_text(text)
    return out


def _sole_remedy_detected(retrieved: list[RetrievedChunk]) -> bool:
    if not retrieved:
        return False
    for r in retrieved:
        t = (r.chunk.clause_text or "").lower()
        if "sole remedy" in t or "exclusive remedy" in t or "sole and exclusive remedy" in t:
            return True
    return False


def evaluate_risk(
    extracted: ExtractedClauses,
    retrieved: list[RetrievedChunk],
    intent_type: str | None = None,
) -> RiskAssessment:
    """
    100% rule-based. No LLM. risk_level and risk_patterns are computed here (call evaluate_risk(...); no .detect).
    - POTENTIAL_UNCAPPED_LIABILITY only when intent in (liability, cross_document); not for termination.
    - JURISDICTION_FRAGMENTATION when multiple governing laws.
    - SOLE_REMEDY_LIMITATION when sole/exclusive remedy in retrieved text and intent is liability-related.
    """
    intent = (intent_type or "").lower()
    risk_patterns: list[str] = []
    risk_level = "LOW"
    reason = ""
    prescribed_liability_wording: str | None = None

    # Liability: only when intent is liability or cross_document. Do NOT run for termination, confidentiality, governing_law.
    # Cross-document: check all documents in extracted_by_document before flagging uncapped liability.
    liability_intents = ("liability", "cross_document")
    if intent in liability_intents:
        liability_clause_present = _has_liability_content(retrieved)
        no_cap_in_extracted = extracted.liability_cap is None or not (extracted.liability_cap or "").strip()
        if intent == "cross_document":
            by_doc = getattr(extracted, "extracted_by_document", None) or {}
            if by_doc:
                any_cap = any((d.get("liability_cap") or "").strip() for d in by_doc.values())
                no_cap_in_extracted = no_cap_in_extracted and not any_cap
        if no_cap_in_extracted and liability_clause_present:
            risk_patterns.append("POTENTIAL_UNCAPPED_LIABILITY")
            prescribed_liability_wording = UNLIMITED_LIABILITY_WORDING
            reason = "No liability cap specified in the retrieved clauses."
            risk_level = "HIGH"

    # Governing law / cross_document: when multiple distinct governing laws detected
    unique_from_extracted = _unique_governing_laws_from_extracted(extracted)
    multiple_from_retrieved = _multiple_governing_laws(retrieved)
    if intent in ("governing_law", "cross_document") and (len(unique_from_extracted) > 1 or multiple_from_retrieved):
        risk_patterns.append("JURISDICTION_FRAGMENTATION")
        if not reason:
            reason = "Multiple governing laws or jurisdictions mentioned."
        risk_level = "MEDIUM" if risk_level == "LOW" else risk_level

    # Sole remedy: only when intent is liability-related
    if intent in liability_intents and _sole_remedy_detected(retrieved):
        risk_patterns.append("SOLE_REMEDY_LIMITATION")
        if risk_level == "LOW":
            risk_level = "MEDIUM"
        if not reason:
            reason = "Sole or exclusive remedy clause detected."

    if not reason:
        reason = "No high-risk patterns identified." if retrieved else "Insufficient information to assess."

    return RiskAssessment(
        risk_level=risk_level,
        reason=reason,
        risk_patterns=risk_patterns,
        prescribed_liability_wording=prescribed_liability_wording,
    )
