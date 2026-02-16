"""Query rewriter: rule-based only. No LLM. Rewrite from last_topic / last_intent / last_document."""

from rag.memory.conversation import Turn

FOLLOWUP_HOW_LONG = ("for how long?", "how long?", "and how long?", "how long", "for how long")


def _rewrite_from_state(
    query: str,
    current_topic: str | None,
    current_document: str | None,
    last_intent: str | None,
) -> str | None:
    """
    Deterministic rewrite from conversation state. No LLM.
    If user says "For how long?" and last_topic == confidentiality_survival -> full standalone query.
    """
    if not query:
        return None
    q = query.strip().lower()
    doc = (current_document or "the agreement").strip()
    if q in FOLLOWUP_HOW_LONG or (len(q) < 25 and "how long" in q):
        # Confidentiality first: do not switch to notice period / VSA when user was on NDA confidentiality
        if current_topic in ("confidentiality_survival", "confidentiality"):
            return f"How long do confidentiality obligations survive termination under the {doc}?"
        if current_topic in ("termination", "notice_period"):
            return f"How long is the notice period for terminating the {doc}?"
        if current_topic == "liability":
            return f"What is the liability cap in the {doc}?"
        if last_intent == "termination":
            return f"How long is the notice period for terminating the {doc}?"
        if last_intent in ("clause_retrieval", "general") and current_topic == "confidentiality":
            return f"How long do confidentiality obligations survive termination under the {doc}?"
    if q in ("and the cap?", "what about the cap?", "the cap?") and (current_topic == "liability" or last_intent == "liability"):
        return f"What is the liability cap in the {doc}?"
    return None


class QueryRewriter:
    """Rule-based rewriter only. No LLM."""

    def rewrite(
        self,
        query: str,
        conversation_context: str = "",
        last_turn: Turn | None = None,
        current_topic: str | None = None,
        current_document: str | None = None,
        last_intent: str | None = None,
    ) -> str:
        """Return standalone query from state only. If no rewrite rule matches, return query unchanged."""
        query = query.strip()
        if not query:
            return query
        deterministic = _rewrite_from_state(query, current_topic, current_document, last_intent)
        if deterministic:
            return deterministic
        return query
