"""Intent classification: rule-based only. No LLM. Latency <1ms."""

from rag.models import IntentResult

# Map query keywords to actual document names in the index (no abbreviations; must match Chroma metadata)
DOCUMENT_KEYWORDS = {
    "nda": "Nda Acme Vendor",
    "non-disclosure": "Nda Acme Vendor",
    "confidential": "Nda Acme Vendor",
    "sla": "Service Level Agreement",
    "uptime": "Service Level Agreement",
    "service level": "Service Level Agreement",
    "service credit": "Service Level Agreement",
    "dpa": "Data Processing Agreement",
    "data processing": "Data Processing Agreement",
    "vendor": "Vendor Services Agreement",
    "agreement": "Vendor Services Agreement",
}
# Fallback document lists use canonical names only (never NDA/SLA/DPA)
FALLBACK_NDA_SLA = ["Nda Acme Vendor", "Service Level Agreement"]
FALLBACK_NDA_SLA_DPA = ["Nda Acme Vendor", "Service Level Agreement", "Data Processing Agreement"]
FALLBACK_NDA = ["Nda Acme Vendor"]
FALLBACK_SLA = ["Service Level Agreement"]
FALLBACK_DPA = ["Data Processing Agreement"]


def _document_focus_from_query(q: str) -> list[str]:
    """Override document_focus from DOCUMENT_KEYWORDS so uptime/SLA -> Service Level Agreement, etc."""
    focus: list[str] = []
    for keyword, doc_name in DOCUMENT_KEYWORDS.items():
        if keyword in q and doc_name not in focus:
            focus.append(doc_name)
    return focus


def classify_intent(query: str, _conversation_context: str = "") -> IntentResult:
    """
    Rule-based intent classification. Returns IntentResult with intent_type, topic, document_focus, is_legal_advice, requires_clarification.
    No LLM. Latency <1ms.
    """
    q = (query or "").strip().lower()
    if not q:
        return IntentResult(intent_type="general", refined_query=query)

    # Legal advice: refuse
    if "sign" in q or "strategy" in q or "should i " in q or "recommend" in q or "what should" in q:
        return IntentResult(
            intent_type="legal_advice",
            is_legal_advice=True,
            refined_query=query,
        )

    # document_focus from keyword mapping (uptime/sla -> Service Level Agreement; nda -> Nda Acme Vendor)
    document_focus = _document_focus_from_query(q)

    # Cross-document: compare governing law across all docs; no document filter
    if any(x in q for x in ("compare governing", "across document", "each agreement", "each document", "governing law across", "conflicting governing", "jurisdiction in each", "which jurisdiction")):
        return IntentResult(
            intent_type="cross_document",
            document_focus=[],  # no filter; retrieve from all
            topic="governing_law",
            refined_query=query,
        )
    if "governing law" in q or "governing" in q and "law" in q or "jurisdiction" in q:
        return IntentResult(
            intent_type="governing_law",
            document_focus=document_focus or FALLBACK_NDA_SLA_DPA,
            topic="governing_law",
            refined_query=query,
        )
    if "liability" in q or "cap" in q and ("liability" in q or "damages" in q):
        return IntentResult(
            intent_type="liability",
            document_focus=document_focus or FALLBACK_NDA_SLA,
            topic="liability",
            refined_query=query,
        )
    if "notice period" in q or "terminate" in q or "termination" in q:
        return IntentResult(
            intent_type="termination",
            document_focus=document_focus or FALLBACK_NDA_SLA,
            topic="termination",
            refined_query=query,
        )
    if "uptime" in q or "sla" in q or "service level" in q or "availability" in q:
        return IntentResult(
            intent_type="service_availability",
            document_focus=document_focus or FALLBACK_SLA,
            topic="service_level",
            refined_query=query,
        )
    if "breach" in q or "notification" in q and "data" in q:
        return IntentResult(
            intent_type="data_breach",
            document_focus=document_focus or FALLBACK_DPA,
            topic="general",
            refined_query=query,
        )
    if "survive" in q or "survival" in q or "confidential" in q or "how long" in q:
        return IntentResult(
            intent_type="clause_retrieval",
            document_focus=document_focus or FALLBACK_NDA,
            topic="confidentiality",
            refined_query=query,
        )

    return IntentResult(
        intent_type="general",
        document_focus=document_focus,
        topic="general",
        refined_query=query,
    )


class IntentAgent:
    """Rule-based intent classifier. No LLM."""

    def classify(self, query: str, conversation_context: str = "") -> IntentResult:
        """Return IntentResult from rule-based classification."""
        return classify_intent(query, conversation_context)
