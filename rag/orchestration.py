"""Orchestration: rule-based intent → rule-based rewrite → metadata filter → retrieval → 1 LLM (extraction) → rule-based risk → rule-based compose → validation."""

import time
from typing import Any

from rag.memory.conversation import ConversationMemory
from rag.agents.intent import IntentAgent
from rag.agents.query_rewriter import QueryRewriter
from rag.retrieval.hybrid_search import HybridRetrieval
from rag.retrieval.store import IndexStore
from rag.agents.clause_analyzer import ClauseAnalyzerAgent
from rag.agents.risk_rule_engine import evaluate_risk
from rag.agents.response_composer import ResponseComposerAgent
from rag.config import get_settings
from rag.models import IntentResult, RetrievedChunk, SynthesizedResponse, ExtractedClauses
from rag.prompts import load_prompt


# Abbreviations -> canonical names matching index metadata (do not pass NDA/SLA to Chroma)
DOC_CANONICAL = {
    "NDA": "Nda Acme Vendor",
    "SLA": "Service Level Agreement",
    "DPA": "Data Processing Agreement",
    "VSA": "Vendor Services Agreement",
}


def normalize_document_focus(document_focus: list[str]) -> list[str]:
    """Map NDA/SLA/DPA/VSA to canonical names so filter matches index metadata."""
    if not document_focus:
        return []
    return [DOC_CANONICAL.get((d or "").strip(), d) for d in document_focus]


def resolve_document_focus(document_focus: list[str], all_document_names: list[str]) -> list[str]:
    """
    Resolve intent document_focus (e.g. ['Nda Acme Vendor']) to actual document names in the index.
    Call normalize_document_focus first if intent may return NDA/SLA/DPA/VSA.
    """
    if not document_focus or not all_document_names:
        return []
    allowed = set()
    for doc_name in all_document_names:
        doc_l = (doc_name or "").lower()
        for focus in document_focus:
            f = (focus or "").lower().strip()
            if not f:
                continue
            if f in doc_l or doc_l.startswith(f) or f in doc_l.split()[0]:
                allowed.add(doc_name)
                break
    return list(allowed)


def build_filter(intent_type: str, document_focus: list[str]) -> dict | None:
    """
    Build metadata filter for retrieval. Only clause_type (soft merge in retrieval).
    Document filter is applied separately as document_filter (hard) in search().
    """
    intent = (intent_type or "").lower()
    filters: dict = {}
    if intent == "termination":
        filters["clause_type"] = "termination"
    elif intent == "liability":
        # Soft-boost both liability and damage_exclusion (e.g. SLA damages exclusion)
        filters["clause_type"] = ["liability", "damage_exclusion"]
    elif intent == "governing_law":
        filters["clause_type"] = "governing_law"
    elif intent == "cross_document":
        # Retrieve liability, indemnification, damage_exclusion, governing_law; then aggregate
        filters["clause_type"] = ["liability", "indemnification", "damage_exclusion", "governing_law"]
    elif intent == "service_availability":
        filters["clause_type"] = "uptime"
    elif intent == "data_breach":
        filters["clause_type"] = "general"
    elif intent == "clause_retrieval":
        pass
    elif intent == "general":
        return None
    else:
        return None
    if not filters:
        return None
    return filters


LEGAL_ADVICE_REFUSAL = (
    "I can only summarize and cite what is in your contracts. "
    "I cannot provide legal advice or recommend whether you should sign or how to negotiate. "
    "Please consult a qualified lawyer for advice."
)

CLARIFICATION_MSG = "Your question is a bit broad. Could you specify which document (e.g. NDA, SLA, DPA) or which topic (e.g. liability, termination) you mean?"

# Topic names for conversation state and metadata filter
VALID_TOPICS = ("termination", "liability", "governing_law", "confidentiality", "indemnification", "general", "service_level")


def _infer_topic_from_query(query: str) -> str | None:
    """Infer semantic topic from query text when intent has no topic."""
    q = (query or "").lower()
    if "terminat" in q or "notice period" in q:
        return "termination"
    if "liability" in q or "cap" in q or "damages" in q:
        return "liability"
    if "governing" in q or "jurisdiction" in q or "law" in q:
        return "governing_law"
    if "confidential" in q or "survive" in q or "survival" in q:
        return "confidentiality"
    if "indemnif" in q:
        return "indemnification"
    if "uptime" in q or "sla" in q or "service level" in q:
        return "service_level"
    return "general"


def _latency_ms(t0: float, t1: float) -> float:
    return round((t1 - t0) * 1000, 2)


class Orchestrator:
    """Pipeline: intent → query rewrite (follow-up → standalone) → retrieval → clause analysis → risk rules → response."""

    def __init__(
        self,
        index_store: IndexStore | None = None,
        top_k: int | None = None,
        use_reranker: bool | None = None,
    ):
        settings = get_settings()
        self.store = index_store or IndexStore()
        self.retrieval = HybridRetrieval(
            store=self.store,
            top_k=top_k or settings.top_k,
            use_reranker=use_reranker if use_reranker is not None else settings.use_reranker,
        )
        self.intent_agent = IntentAgent()
        self.query_rewriter = QueryRewriter()
        self.clause_analyzer = ClauseAnalyzerAgent()
        self.composer = ResponseComposerAgent()
        self.memory = ConversationMemory()
        self.last_latency_ms: dict[str, float] = {}

    def run(self, query: str) -> SynthesizedResponse:
        """Execute full pipeline: intent → rewrite → retrieve → clause analysis → risk rules → compose → validate."""
        query = query.strip()
        latency: dict[str, float] = {"intent_ms": 0, "rewrite_ms": 0, "retrieval_ms": 0, "clause_analysis_ms": 0, "risk_ms": 0, "compose_ms": 0, "total_ms": 0}
        t_start = time.perf_counter()
        if not query:
            self.last_latency_ms = {"total_ms": 0, **latency}
            return SynthesizedResponse(direct_answer="Please ask a question about your contracts.")

        context = self.memory.get_context_for_intent()
        t0 = time.perf_counter()
        intent = self.intent_agent.classify(query, context)
        latency["intent_ms"] = _latency_ms(t0, time.perf_counter())

        if intent.is_legal_advice:
            latency["total_ms"] = _latency_ms(t_start, time.perf_counter())
            self.last_latency_ms = latency
            return SynthesizedResponse(direct_answer=LEGAL_ADVICE_REFUSAL, insufficient_info=True)

        if intent.requires_clarification:
            latency["total_ms"] = _latency_ms(t_start, time.perf_counter())
            self.last_latency_ms = latency
            return SynthesizedResponse(direct_answer=CLARIFICATION_MSG)

        last_turn = self.memory.get_last_turn()
        t0 = time.perf_counter()
        standalone_query = self.query_rewriter.rewrite(
            intent.refined_query or query,
            context,
            last_turn=last_turn,
            current_topic=self.memory.get_current_topic(),
            current_document=self.memory.get_current_document(),
            last_intent=self.memory.get_last_intent(),
        )
        latency["rewrite_ms"] = _latency_ms(t0, time.perf_counter())
        search_query = standalone_query.strip() or query
        metadata_filter = build_filter(intent.intent_type, intent.document_focus or [])
        if metadata_filter is None and intent.intent_type == "clause_retrieval" and getattr(intent, "topic", None) and intent.topic in ("termination", "liability", "governing_law", "confidentiality", "indemnification", "general"):
            metadata_filter = {"clause_type": intent.topic}
        all_chunks = self.store.get_chunks()
        all_doc_names = list({c.document for c in all_chunks} if all_chunks else [])
        raw_focus = intent.document_focus or []
        normalized_focus = normalize_document_focus(raw_focus) if raw_focus else []
        resolved_focus = resolve_document_focus(normalized_focus, all_doc_names) if normalized_focus else []
        document_filter = resolved_focus if resolved_focus else None
        hard_clause_type: str | None = None
        if intent.intent_type == "cross_document":
            document_filter = None
        elif intent.intent_type == "service_availability":
            doc_focus = normalized_focus or ["Service Level Agreement"]
            document_filter = resolve_document_focus(doc_focus, all_doc_names) or [
                d for d in all_doc_names if ("service level" in (d or "").lower() or ("vendor" in (d or "").lower() and "agreement" in (d or "").lower()))
            ]
            if document_filter:
                hard_clause_type = "uptime"
        elif intent.intent_type in ("termination", "liability", "confidentiality", "governing_law", "indemnification"):
            hard_clause_type = getattr(intent, "topic", None) or intent.intent_type
            metadata_filter = None  # deterministic: hard filter like SLA
        t0 = time.perf_counter()
        retrieved = self.retrieval.search(search_query, metadata_filter=metadata_filter, document_filter=document_filter, hard_clause_type=hard_clause_type)
        latency["retrieval_ms"] = _latency_ms(t0, time.perf_counter())
        if not retrieved:
            self.memory.add(query, "No relevant clauses found.")
            latency["total_ms"] = _latency_ms(t_start, time.perf_counter())
            self.last_latency_ms = latency
            return SynthesizedResponse(
                direct_answer="No relevant clauses were found. Try rephrasing or specifying the document (e.g. NDA, SLA).",
                insufficient_info=True,
            )

        focused_docs = set(resolved_focus) if resolved_focus else None
        if intent.intent_type == "cross_document":
            clause_input = retrieved
        elif focused_docs:
            clause_input = [r for r in retrieved if r.chunk.document in focused_docs]
        else:
            clause_input = retrieved
        t0 = time.perf_counter()
        if intent.intent_type == "cross_document":
            extracted_by_doc = self.clause_analyzer.extract_all_per_document(retrieved)
            governing_law_by_doc = {doc: (d.get("governing_law") or "Not found") for doc, d in extracted_by_doc.items()}
            extracted = ExtractedClauses(
                governing_law=None,
                governing_law_by_document=governing_law_by_doc,
                extracted_by_document=extracted_by_doc,
            )
        else:
            extracted = self.clause_analyzer.analyze(clause_input)
        source_map = {f"CHUNK_{i}": r for i, r in enumerate(clause_input)}
        latency["clause_analysis_ms"] = _latency_ms(t0, time.perf_counter())
        t0 = time.perf_counter()
        risk = evaluate_risk(extracted, retrieved, intent_type=intent.intent_type)
        latency["risk_ms"] = _latency_ms(t0, time.perf_counter())
        t0 = time.perf_counter()
        response = self.composer.compose(query, intent, retrieved, extracted, risk, source_map=source_map)
        latency["compose_ms"] = _latency_ms(t0, time.perf_counter())

        doc = retrieved[0].chunk.document if retrieved else None
        docs = list({r.chunk.document for r in retrieved})
        risks_str = [risk.reason] if risk.risk_level != "LOW" else []
        topic = getattr(intent, "topic", None) or _infer_topic_from_query(search_query)
        topic_effective = (
            "confidentiality_survival"
            if topic == "confidentiality" and ("survive" in search_query.lower() or "survival" in search_query.lower())
            else ("notice_period" if topic == "termination" and ("notice" in search_query.lower() or "period" in search_query.lower()) else topic)
        )
        self.memory.add(query, response.direct_answer, documents=docs, risks=risks_str, topic=topic_effective, document=doc, last_intent=intent.intent_type)
        latency["total_ms"] = _latency_ms(t_start, time.perf_counter())
        self.last_latency_ms = latency
        return response

    def run_with_retrieval(self, query: str) -> tuple[SynthesizedResponse, list[RetrievedChunk]]:
        """Run full pipeline and return (response, retrieved_chunks). Latency in self.last_latency_ms."""
        query = query.strip()
        latency: dict[str, float] = {"intent_ms": 0, "rewrite_ms": 0, "retrieval_ms": 0, "clause_analysis_ms": 0, "risk_ms": 0, "compose_ms": 0, "total_ms": 0}
        t_start = time.perf_counter()
        if not query:
            self.last_latency_ms = {"total_ms": 0, **latency}
            return (SynthesizedResponse(direct_answer="Please ask a question about your contracts."), [])
        context = self.memory.get_context_for_intent()
        t0 = time.perf_counter()
        intent = self.intent_agent.classify(query, context)
        latency["intent_ms"] = _latency_ms(t0, time.perf_counter())
        if intent.is_legal_advice:
            latency["total_ms"] = _latency_ms(t_start, time.perf_counter())
            self.last_latency_ms = latency
            return (SynthesizedResponse(direct_answer=LEGAL_ADVICE_REFUSAL, insufficient_info=True), [])
        if intent.requires_clarification:
            latency["total_ms"] = _latency_ms(t_start, time.perf_counter())
            self.last_latency_ms = latency
            return (SynthesizedResponse(direct_answer=CLARIFICATION_MSG), [])
        last_turn = self.memory.get_last_turn()
        t0 = time.perf_counter()
        standalone_query = self.query_rewriter.rewrite(
            intent.refined_query or query,
            context,
            last_turn=last_turn,
            current_topic=self.memory.get_current_topic(),
            current_document=self.memory.get_current_document(),
            last_intent=self.memory.get_last_intent(),
        )
        latency["rewrite_ms"] = _latency_ms(t0, time.perf_counter())
        search_query = standalone_query.strip() or query
        metadata_filter = build_filter(intent.intent_type, intent.document_focus or [])
        if metadata_filter is None and intent.intent_type == "clause_retrieval" and getattr(intent, "topic", None) and intent.topic in ("termination", "liability", "governing_law", "confidentiality", "indemnification", "general"):
            metadata_filter = {"clause_type": intent.topic}
        all_chunks = self.store.get_chunks()
        all_doc_names = list({c.document for c in all_chunks} if all_chunks else [])
        raw_focus = intent.document_focus or []
        normalized_focus = normalize_document_focus(raw_focus) if raw_focus else []
        resolved_focus = resolve_document_focus(normalized_focus, all_doc_names) if normalized_focus else []
        document_filter = resolved_focus if resolved_focus else None
        hard_clause_type = None
        if intent.intent_type == "cross_document":
            document_filter = None
        elif intent.intent_type == "service_availability":
            doc_focus = normalized_focus or ["Service Level Agreement"]
            document_filter = resolve_document_focus(doc_focus, all_doc_names) or [
                d for d in all_doc_names if ("service level" in (d or "").lower() or ("vendor" in (d or "").lower() and "agreement" in (d or "").lower()))
            ]
            if document_filter:
                hard_clause_type = "uptime"
        elif intent.intent_type in ("termination", "liability", "confidentiality", "governing_law", "indemnification"):
            hard_clause_type = getattr(intent, "topic", None) or intent.intent_type
            metadata_filter = None  # deterministic: hard filter like SLA
        t0 = time.perf_counter()
        retrieved = self.retrieval.search(search_query, metadata_filter=metadata_filter, document_filter=document_filter, hard_clause_type=hard_clause_type)
        latency["retrieval_ms"] = _latency_ms(t0, time.perf_counter())
        if not retrieved:
            self.memory.add(query, "No relevant clauses found.")
            latency["total_ms"] = _latency_ms(t_start, time.perf_counter())
            self.last_latency_ms = latency
            return (
                SynthesizedResponse(
                    direct_answer="No relevant clauses were found. Try rephrasing or specifying the document (e.g. NDA, SLA).",
                    insufficient_info=True,
                ),
                [],
            )
        focused_docs = set(resolved_focus) if resolved_focus else None
        if intent.intent_type == "cross_document":
            clause_input = retrieved
        elif focused_docs:
            clause_input = [r for r in retrieved if r.chunk.document in focused_docs]
        else:
            clause_input = retrieved
        t0 = time.perf_counter()
        if intent.intent_type == "cross_document":
            extracted_by_doc = self.clause_analyzer.extract_all_per_document(retrieved)
            governing_law_by_doc = {doc: (d.get("governing_law") or "Not found") for doc, d in extracted_by_doc.items()}
            extracted = ExtractedClauses(
                governing_law=None,
                governing_law_by_document=governing_law_by_doc,
                extracted_by_document=extracted_by_doc,
            )
        else:
            extracted = self.clause_analyzer.analyze(clause_input)
        source_map = {f"CHUNK_{i}": r for i, r in enumerate(clause_input)}
        latency["clause_analysis_ms"] = _latency_ms(t0, time.perf_counter())
        t0 = time.perf_counter()
        risk = evaluate_risk(extracted, retrieved, intent_type=intent.intent_type)
        latency["risk_ms"] = _latency_ms(t0, time.perf_counter())
        t0 = time.perf_counter()
        response = self.composer.compose(query, intent, retrieved, extracted, risk, source_map=source_map)
        latency["compose_ms"] = _latency_ms(t0, time.perf_counter())
        doc = retrieved[0].chunk.document if retrieved else None
        docs = list({r.chunk.document for r in retrieved})
        risks_str = [risk.reason] if risk.risk_level != "LOW" else []
        topic = getattr(intent, "topic", None) or _infer_topic_from_query(search_query)
        topic_effective = (
            "confidentiality_survival"
            if topic == "confidentiality" and ("survive" in search_query.lower() or "survival" in search_query.lower())
            else ("notice_period" if topic == "termination" and ("notice" in search_query.lower() or "period" in search_query.lower()) else topic)
        )
        self.memory.add(query, response.direct_answer, documents=docs, risks=risks_str, topic=topic_effective, document=doc, last_intent=intent.intent_type)
        latency["total_ms"] = _latency_ms(t_start, time.perf_counter())
        self.last_latency_ms = latency
        return (response, retrieved)

    def run_verbose(
        self,
        query: str,
        include_prompt_content: bool = False,
    ) -> dict[str, Any]:
        """
        Run full pipeline and return every intermediate state for debugging and evaluation.
        Keys: intent, rewritten_query, retrieved_summary, clause_analysis, risk, response,
        prompt_names, prompts (if include_prompt_content=True).
        """
        query = query.strip()
        out: dict[str, Any] = {
            "intent": None,
            "rewritten_query": None,
            "retrieved_summary": [],
            "clause_analysis": None,
            "risk": None,
            "response": None,
            "latency_ms": {},
            "prompt_names": ["clause_extractor_system"],
        }
        if include_prompt_content:
            out["prompts"] = {n: load_prompt(n) for n in out["prompt_names"]}
        t_start = time.perf_counter()
        lat: dict[str, float] = {"intent_ms": 0, "rewrite_ms": 0, "retrieval_ms": 0, "clause_analysis_ms": 0, "risk_ms": 0, "compose_ms": 0, "total_ms": 0}

        if not query:
            out["response"] = SynthesizedResponse(direct_answer="Please ask a question about your contracts.")
            out["latency_ms"] = {**lat, "total_ms": 0}
            return out

        t0 = time.perf_counter()
        context = self.memory.get_context_for_intent()
        intent = self.intent_agent.classify(query, context)
        lat["intent_ms"] = _latency_ms(t0, time.perf_counter())
        out["intent"] = intent

        if intent.is_legal_advice:
            out["response"] = SynthesizedResponse(direct_answer=LEGAL_ADVICE_REFUSAL, insufficient_info=True)
            lat["total_ms"] = _latency_ms(t_start, time.perf_counter())
            out["latency_ms"] = lat
            return out
        if intent.requires_clarification:
            out["response"] = SynthesizedResponse(direct_answer=CLARIFICATION_MSG)
            lat["total_ms"] = _latency_ms(t_start, time.perf_counter())
            out["latency_ms"] = lat
            return out

        t0 = time.perf_counter()
        last_turn = self.memory.get_last_turn()
        standalone_query = self.query_rewriter.rewrite(
            intent.refined_query or query,
            context,
            last_turn=last_turn,
            current_topic=self.memory.get_current_topic(),
            current_document=self.memory.get_current_document(),
            last_intent=self.memory.get_last_intent(),
        )
        lat["rewrite_ms"] = _latency_ms(t0, time.perf_counter())
        out["rewritten_query"] = standalone_query.strip() or query
        search_query = out["rewritten_query"]
        metadata_filter = build_filter(intent.intent_type, intent.document_focus or [])
        if metadata_filter is None and intent.intent_type == "clause_retrieval" and getattr(intent, "topic", None) and intent.topic in ("termination", "liability", "governing_law", "confidentiality", "indemnification", "general"):
            metadata_filter = {"clause_type": intent.topic}
        all_chunks = self.store.get_chunks()
        all_doc_names = list({c.document for c in all_chunks} if all_chunks else [])
        raw_focus = intent.document_focus or []
        normalized_focus = normalize_document_focus(raw_focus) if raw_focus else []
        resolved_focus = resolve_document_focus(normalized_focus, all_doc_names) if normalized_focus else []
        document_filter = resolved_focus if resolved_focus else None
        hard_clause_type = None
        if intent.intent_type == "cross_document":
            document_filter = None
        elif intent.intent_type == "service_availability":
            doc_focus = normalized_focus or ["Service Level Agreement"]
            document_filter = resolve_document_focus(doc_focus, all_doc_names) or [
                d for d in all_doc_names if ("service level" in (d or "").lower() or ("vendor" in (d or "").lower() and "agreement" in (d or "").lower()))
            ]
            if document_filter:
                hard_clause_type = "uptime"
        elif intent.intent_type in ("termination", "liability", "confidentiality", "governing_law", "indemnification"):
            hard_clause_type = getattr(intent, "topic", None) or intent.intent_type
            metadata_filter = None  # deterministic: hard filter like SLA
        t0 = time.perf_counter()
        retrieved = self.retrieval.search(search_query, metadata_filter=metadata_filter, document_filter=document_filter, hard_clause_type=hard_clause_type)
        lat["retrieval_ms"] = _latency_ms(t0, time.perf_counter())

        if not retrieved:
            self.memory.add(query, "No relevant clauses found.")
            out["response"] = SynthesizedResponse(
                direct_answer="No relevant clauses were found. Try rephrasing or specifying the document (e.g. NDA, SLA).",
                insufficient_info=True,
            )
            lat["total_ms"] = _latency_ms(t_start, time.perf_counter())
            out["latency_ms"] = lat
            return out

        out["retrieved_summary"] = [
            {
                "document": r.chunk.document,
                "section": r.chunk.section,
                "title": r.chunk.title,
                "snippet": (r.chunk.clause_text or "")[:200] + ("..." if len(r.chunk.clause_text or "") > 200 else ""),
                "score": r.score,
            }
            for r in retrieved
        ]

        focused_docs = set(resolved_focus) if resolved_focus else None
        if intent.intent_type == "cross_document":
            clause_input = retrieved
        elif focused_docs:
            clause_input = [r for r in retrieved if r.chunk.document in focused_docs]
        else:
            clause_input = retrieved
        t0 = time.perf_counter()
        if intent.intent_type == "cross_document":
            extracted_by_doc = self.clause_analyzer.extract_all_per_document(retrieved)
            governing_law_by_doc = {doc: (d.get("governing_law") or "Not found") for doc, d in extracted_by_doc.items()}
            extracted = ExtractedClauses(
                governing_law=None,
                governing_law_by_document=governing_law_by_doc,
                extracted_by_document=extracted_by_doc,
            )
        else:
            extracted = self.clause_analyzer.analyze(clause_input)
        lat["clause_analysis_ms"] = _latency_ms(t0, time.perf_counter())
        out["clause_analysis"] = extracted
        t0 = time.perf_counter()
        risk = evaluate_risk(extracted, retrieved, intent_type=intent.intent_type)
        lat["risk_ms"] = _latency_ms(t0, time.perf_counter())
        out["risk"] = risk
        t0 = time.perf_counter()
        response = self.composer.compose(query, intent, retrieved, extracted, risk)
        lat["compose_ms"] = _latency_ms(t0, time.perf_counter())
        out["response"] = response

        doc = retrieved[0].chunk.document if retrieved else None
        docs = list({r.chunk.document for r in retrieved})
        risks_str = [risk.reason] if risk.risk_level != "LOW" else []
        topic = getattr(intent, "topic", None) or _infer_topic_from_query(search_query)
        topic_effective = (
            "confidentiality_survival"
            if topic == "confidentiality" and ("survive" in search_query.lower() or "survival" in search_query.lower())
            else ("notice_period" if topic == "termination" and ("notice" in search_query.lower() or "period" in search_query.lower()) else topic)
        )
        self.memory.add(query, response.direct_answer, documents=docs, risks=risks_str, topic=topic_effective, document=doc, last_intent=intent.intent_type)
        lat["total_ms"] = _latency_ms(t_start, time.perf_counter())
        out["latency_ms"] = lat
        self.last_latency_ms = lat
        return out
