"""Evaluation helpers: retrieval (recall, precision, MRR, avg_rank), faithfulness, citation accuracy, hallucination, risk, safety."""

import re
from typing import Callable

from rag.models import RetrievedChunk, SynthesizedResponse

__all__ = [
    "make_retrieval_fn_for_eval",
    "run_retrieval_eval",
    "has_citation",
    "citation_correct",
    "refusal_for_legal_advice",
    "answer_matches_gold",
    "risk_matches",
    "detect_hallucination",
    "multi_turn_memory_ok",
    "cross_document_mentions",
    "assert_sla_uptime_retrieval",
]


def _normalize_section(s: str) -> str:
    return (s or "").strip().lower()


def _cosine_sim(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na * nb == 0:
        return 0.0
    return dot / (na * nb)


def _doc_section_match(expected_doc: str, expected_sec: str, actual_doc: str, actual_sec: str) -> bool:
    """Loose match: expected doc substring of actual or vice versa; section/title overlap."""
    ed, ad = (expected_doc or "").lower(), (actual_doc or "").lower()
    if ed not in ad and ad not in ed:
        return False
    es, as_ = _normalize_section(expected_sec), _normalize_section(actual_sec or "")
    if not es:
        return True
    return es in as_ or as_ in es


def retrieval_precision_at_k(retrieved: list[RetrievedChunk], relevant_doc_sections: set[tuple[str, str]]) -> float:
    """Precision@k: fraction of retrieved chunks that are relevant (document, section)."""
    if not retrieved:
        return 0.0
    hits = sum(
        1 for r in retrieved
        if any(_doc_section_match(rd, rs, r.chunk.document, r.chunk.section or r.chunk.title) for rd, rs in relevant_doc_sections)
    )
    return hits / len(retrieved)


def retrieval_recall_at_k(
    retrieved: list[RetrievedChunk],
    relevant_doc_sections: set[tuple[str, str]],
) -> float:
    """Recall@k: fraction of relevant (document, section) that appear in retrieved."""
    if not relevant_doc_sections:
        return 1.0
    hits = sum(
        1 for (rd, rs) in relevant_doc_sections
        if any(_doc_section_match(rd, rs, r.chunk.document, r.chunk.section or r.chunk.title) for r in retrieved)
    )
    return hits / len(relevant_doc_sections)


def retrieval_mrr_at_k(retrieved: list[RetrievedChunk], relevant_doc_sections: set[tuple[str, str]]) -> float:
    """Reciprocal rank of first relevant chunk (1/rank)."""
    for i, r in enumerate(retrieved):
        if any(_doc_section_match(rd, rs, r.chunk.document, r.chunk.section or r.chunk.title) for rd, rs in relevant_doc_sections):
            return 1.0 / (i + 1)
    return 0.0


def retrieval_avg_rank(retrieved: list[RetrievedChunk], relevant_doc_sections: set[tuple[str, str]], k: int = 10) -> float:
    """Rank of first relevant chunk (1-based); k+1 if none."""
    for i, r in enumerate(retrieved):
        if any(_doc_section_match(rd, rs, r.chunk.document, r.chunk.section or r.chunk.title) for rd, rs in relevant_doc_sections):
            return float(i + 1)
    return float(k + 1)


def make_retrieval_fn_for_eval(orchestrator) -> Callable[[str], list[RetrievedChunk]]:
    """
    Return a retrieval function that uses intent -> document_filter so the evaluation
    path calls retrieve(query, where_filter) and never bypasses with raw query.
    Use this instead of lambda q: retrieval.search(q, top_k=5) in RUN_EVALUATION.
    """
    from rag.orchestration import normalize_document_focus, resolve_document_focus
    store = orchestrator.store
    retrieval = orchestrator.retrieval
    all_doc_names = store.get_all_documents()

    def retrieval_fn(query: str) -> list[RetrievedChunk]:
        intent = orchestrator.intent_agent.classify(query)

        raw_focus = intent.document_focus or []
        normalized_focus = normalize_document_focus(raw_focus) if raw_focus else []
        document_filter = resolve_document_focus(normalized_focus, all_doc_names) if normalized_focus else None

        hard_clause_type = intent.topic if intent.topic in (
            "termination",
            "liability",
            "confidentiality",
            "governing_law",
            "indemnification",
            "uptime",
        ) else None

        return retrieval.search(
            query,
            document_filter=document_filter,
            hard_clause_type=hard_clause_type,
            top_k=5,
        )

    return retrieval_fn


def run_retrieval_eval(
    retrieval_fn: Callable[[str], list[RetrievedChunk]],
    test_queries: list[dict],
) -> dict:
    """
    test_queries: list of {"query": str, "relevant": [{"document": str, "section": str}]}
    For cross-document queries, include all expected (doc, section) in relevant.
    Returns: recall_at_k, precision_at_k, mrr, avg_rank, n_queries.
    """
    precisions, recalls, mrrs, ranks = [], [], [], []
    k = 10
    for item in test_queries:
        relevant = set((r["document"], r.get("section") or r.get("title") or "") for r in item.get("relevant", []))
        retrieved = retrieval_fn(item["query"])
        precisions.append(retrieval_precision_at_k(retrieved, relevant))
        recalls.append(retrieval_recall_at_k(retrieved, relevant))
        mrrs.append(retrieval_mrr_at_k(retrieved, relevant))
        ranks.append(retrieval_avg_rank(retrieved, relevant, k=k))
    n = len(test_queries)
    rec = sum(recalls) / n if n else 0.0
    prec = sum(precisions) / n if n else 0.0
    return {
        "recall_at_k": rec,
        "precision_at_k": prec,
        "mean_recall_at_k": rec,
        "mean_precision_at_k": prec,
        "mrr": sum(mrrs) / n if n else 0.0,
        "avg_rank": sum(ranks) / n if n else 0.0,
        "n_queries": n,
    }


def has_citation(response: SynthesizedResponse) -> bool:
    """True if response includes at least one cited clause or answer text contains inline (Source: ...)."""
    if response.cited_clauses:
        return True
    direct = response.direct_answer or ""
    return "(Source:" in direct


def citation_correct(
    response: SynthesizedResponse,
    expected_document: str,
    expected_section: str,
) -> bool:
    """True if at least one citation matches expected document and section (legal discipline)."""
    expected_doc_n = (expected_document or "").lower()
    expected_sec_n = _normalize_section(expected_section)
    for c in response.cited_clauses or []:
        doc = (c.get("document") or "").lower()
        sec = _normalize_section(c.get("section") or c.get("title") or "")
        if expected_doc_n in doc or doc in expected_doc_n:
            if not expected_sec_n or expected_sec_n in sec or sec in expected_sec_n:
                return True
    return False


def answer_matches_gold(
    system_answer: str,
    gold_answer: str,
    embedding_fn: Callable[[str], list[float]] | None = None,
    semantic_threshold: float = 0.6,
) -> dict:
    """
    Faithfulness: exact match, key phrase overlap, numeric consistency, optional semantic similarity.
    Returns dict: exact_match, key_phrase_ok, numeric_ok, semantic_score, overall_ok.
    """
    sys = (system_answer or "").strip()
    gold = (gold_answer or "").strip()
    out = {"exact_match": False, "key_phrase_ok": False, "numeric_ok": False, "semantic_score": 0.0, "overall_ok": False}
    if not sys:
        return out
    out["exact_match"] = sys.lower() == gold.lower() or gold in sys or sys in gold
    gold_words = set(w for w in re.findall(r"\w+", gold.lower()) if len(w) > 2)
    sys_lower = sys.lower()
    overlap = sum(1 for w in gold_words if w in sys_lower) / len(gold_words) if gold_words else 1.0
    out["key_phrase_ok"] = overlap >= 0.5
    gold_nums = re.findall(r"\d+\.?\d*%?|\d+\s*%\s*|\d+\s*days?|\d+\s*years?|\d+\s*hours?", gold)
    sys_nums = re.findall(r"\d+\.?\d*%?|\d+\s*%\s*|\d+\s*days?|\d+\s*years?|\d+\s*hours?", sys_lower)
    out["numeric_ok"] = (not gold_nums) or all(any(n in " ".join(sys_nums) or n in sys_lower for n in gold_nums))
    if gold:
        emb_fn = embedding_fn
        if emb_fn is None:
            try:
                from rag.retrieval.embeddings import embed_single
                emb_fn = embed_single
            except Exception:
                pass
        if emb_fn:
            try:
                e_g = emb_fn(gold)
                e_s = emb_fn(sys)
                out["semantic_score"] = _cosine_sim(e_g, e_s)
            except Exception:
                pass
    out["overall_ok"] = out["exact_match"] or (out["key_phrase_ok"] and out["numeric_ok"]) or (out["semantic_score"] >= semantic_threshold)
    return out


def detect_hallucination(answer: str, retrieved_text: str) -> tuple[bool, list[str]]:
    """
    Heuristic: answer mentions penalty/fine/damages/compensation/numbers not in retrieved text → suspicious.
    Returns (is_hallucination, list of suspicious phrases or numbers).
    """
    if not answer or not retrieved_text:
        return False, []
    suspicious = []
    answer_lower = answer.lower()
    retrieved_lower = retrieved_text.lower()
    danger_phrases = ["penalty", "fine", "damages", "compensation", "indemnity", "liquidated"]
    for p in danger_phrases:
        if p in answer_lower and p not in retrieved_lower:
            suspicious.append(p)
    nums_in_answer = set(re.findall(r"\d+\.?\d*%?|\d+\s*(?:days?|years?|hours?)", answer_lower))
    nums_in_retrieved = set(re.findall(r"\d+\.?\d*%?|\d+\s*(?:days?|years?|hours?)", retrieved_lower))
    for n in nums_in_answer:
        if n not in nums_in_retrieved and not any(n in r for r in nums_in_retrieved):
            suspicious.append(n)
    return len(suspicious) > 0, suspicious


def risk_matches(
    system_risk_flags: list[str] | list[dict],
    expected_risks: list[str],
    penalize_false_positive: bool = True,
) -> float:
    """Score 1 if expected ⊆ system; penalize extra (false positives)."""
    exp = {r.strip().upper().replace(" ", "_") for r in (expected_risks or [])}
    sys = set()
    for r in system_risk_flags or []:
        if isinstance(r, dict):
            msg = (r.get("message") or r.get("level") or "").strip().upper().replace(" ", "_")
            if msg:
                sys.add(msg)
        else:
            sys.add((r or "").strip().upper().replace(" ", "_"))
    if not exp:
        return 1.0 if not sys else (max(0.0, 1.0 - 0.5 * len(sys)) if penalize_false_positive else 1.0)
    if exp <= sys:
        extra = sys - exp
        return 1.0 if not extra else max(0.0, 1.0 - 0.5 * len(extra)) if penalize_false_positive else 1.0
    return 0.0


def marked_insufficient_info(response: SynthesizedResponse) -> bool:
    """True if model correctly declined to answer for lack of context."""
    return response.insufficient_info


def refusal_for_legal_advice(response: SynthesizedResponse) -> bool:
    """Check if response is a safe refusal (no legal advice given)."""
    text = (response.direct_answer or "").lower()
    refusal_phrases = ("cannot provide legal advice", "consult a lawyer", "qualified lawyer", "do not provide")
    return any(p in text for p in refusal_phrases)


def multi_turn_memory_ok(
    first_query: str,
    first_response: SynthesizedResponse,
    second_query: str,
    second_response: SynthesizedResponse,
    required_phrases: list[str],
    required_doc_in_citations: str | None = None,
) -> dict:
    """
    After a follow-up (e.g. "For how long?"), check second response contains required phrases
    and optionally still cites expected document. Returns dict with phrase_ok, citation_ok, overall_ok.
    """
    text = (second_response.direct_answer or "").lower()
    phrase_ok = all(p.lower() in text for p in required_phrases)
    citation_ok = True
    if required_doc_in_citations:
        doc_lower = required_doc_in_citations.lower()
        cited = second_response.cited_clauses or []
        citation_ok = any((c.get("document") or "").lower() in doc_lower or doc_lower in (c.get("document") or "").lower() for c in cited)
    return {"phrase_ok": phrase_ok, "citation_ok": citation_ok, "overall_ok": phrase_ok and citation_ok}


def cross_document_mentions(response: SynthesizedResponse, required_mentions: list[str]) -> dict:
    """Check that answer or citations mention all required items (e.g. jurisdictions)."""
    text = (response.direct_answer or "").lower()
    cited_docs = " ".join((c.get("document") or "").lower() for c in (response.cited_clauses or []))
    combined = text + " " + cited_docs
    found = [m for m in required_mentions if m.lower() in combined]
    missing = [m for m in required_mentions if m.lower() not in combined]
    return {"found": found, "missing": missing, "all_ok": len(missing) == 0}


def assert_sla_uptime_retrieval(retrieval, store=None, document_filter=None) -> bool:
    """
    Test that retrieval returns the SLA Service Availability clause for uptime query.
    Uses document_filter so retrieve(query, where_filter) is called — do not call retrieval.search(q, top_k=5) only.
    If store is provided, uses store.retrieve with where_filter={"document": "Service Level Agreement"}.
    Otherwise uses retrieval.search with document_filter=["Service Level Agreement"].
    """
    where_filter = {"document": "Service Level Agreement"} if (document_filter is None) else ({"document": document_filter[0]} if document_filter else None)
    if store and where_filter:
        hits = store.retrieve("99.5% monthly uptime", where_filter=where_filter, k=5)
        chunks = store.get_chunks()
        results = []
        for cid, _ in hits:
            try:
                idx = int(cid.replace("chunk_", ""))
            except ValueError:
                continue
            if 0 <= idx < len(chunks):
                results.append(RetrievedChunk(chunk=chunks[idx], score=1.0, source="eval"))
    else:
        results = retrieval.search("99.5% monthly uptime", document_filter=document_filter or ["Service Level Agreement"], top_k=5)
    for r in results:
        doc = (r.chunk.document or "").lower()
        title = (r.chunk.title or "").lower()
        text = (r.chunk.clause_text or "").lower()
        if "service level" in doc or "sla" in doc:
            if "service availability" in title or "uptime" in text or "99.5" in text:
                return True
    raise AssertionError(
        "retrieval.search('99.5% monthly uptime') did not return the SLA Service Availability clause. "
        "Re-run ingestion so the chunk has document='Service Level Agreement', section='Service Availability', "
        "clause_type=['service_availability'], keywords=['uptime', 'availability', 'monthly uptime']."
    )
