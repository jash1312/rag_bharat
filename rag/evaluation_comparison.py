"""
Compare RAG system outputs to OpenAI gold from evaluation_dataset.
Metrics: Retrieval Recall@K, Answer (EM/semantic/key phrase), Risk, Citation, Guardrail.
"""

from pathlib import Path
from typing import Any, Callable, List, Optional

from rag.models import RetrievedChunk, SynthesizedResponse


# Normalize gold doc names to possible system doc substrings
def _normalize_doc_for_match(gold_doc: str) -> str:
    g = gold_doc.lower().strip()
    if g == "nda":
        return "nda"
    if g == "sla":
        return "sla"
    if g in ("dpa", "data processing agreement"):
        return "dpa"
    if "vendor" in g and "agreement" in g:
        return "vendor"
    return g


def _system_doc_normalized(system_doc: str) -> str:
    s = system_doc.lower()
    if "nda" in s or s.startswith("nda"):
        return "nda"
    if "sla" in s or "service level" in s:
        return "sla"
    if "dpa" in s or "data processing" in s:
        return "dpa"
    if "vendor" in s and "service" in s:
        return "vendor"
    return s


def _doc_match(expected_doc: str, system_doc: str) -> bool:
    e = _normalize_doc_for_match(expected_doc)
    s = _system_doc_normalized(system_doc)
    if e == "nda":
        return s == "nda"
    if e == "sla":
        return s == "sla"
    if e == "dpa":
        return s == "dpa"
    if e == "vendor":
        return s == "vendor"
    return e in s or s in e


def _section_match(expected_section: str, system_section_or_title: str) -> bool:
    e = expected_section.lower().strip()
    s = (system_section_or_title or "").lower()
    if e in s:
        return True
    if s in e:
        return True
    # e.g. "Term and Termination" vs "3. Term and Termination"
    return e.replace(" ", "") in s.replace(" ", "")


# --- A) Retrieval Recall@K ---
def retrieval_recall_at_k(
    retrieved: list[RetrievedChunk],
    expected_documents: list[str],
    expected_sections: Optional[List[str]] = None,
) -> float:
    """
    Recall = 1 if all expected documents appear in retrieved (and optionally sections).
    For cross-document: require ALL expected docs.
    """
    if not expected_documents:
        return 1.0
    for exp_doc in expected_documents:
        if not any(_doc_match(exp_doc, r.chunk.document) for r in retrieved):
            return 0.0
    if expected_sections:
        for exp_sec in expected_sections:
            if not any(_section_match(exp_sec, r.chunk.title or r.chunk.section) for r in retrieved):
                return 0.5
    return 1.0


def retrieval_precision_at_k(
    retrieved: list[RetrievedChunk],
    expected_documents: list[str],
) -> float:
    """Fraction of retrieved docs that are in expected (or match)."""
    if not retrieved:
        return 0.0
    expected_set = {_normalize_doc_for_match(d) for d in expected_documents}
    hits = sum(1 for r in retrieved if any(_doc_match(exp, r.chunk.document) for exp in expected_documents))
    return hits / len(retrieved) if retrieved else 0.0


def mrr_retrieval(retrieved: list[RetrievedChunk], expected_documents: list[str]) -> float:
    """Mean Reciprocal Rank: 1/rank of first relevant doc."""
    for i, r in enumerate(retrieved):
        if any(_doc_match(exp, r.chunk.document) for exp in expected_documents):
            return 1.0 / (i + 1)
    return 0.0


# --- B) Answer comparison ---
def _cosine_sim(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def answer_score(
    gold_answer: str,
    system_answer: str,
    embedding_fn: Optional[Callable[..., Any]] = None,
    semantic_threshold: float = 0.6,
) -> float:
    """
    Exact match = 1.0, semantic > threshold = 1.0 (valid paraphrases), key phrase = 0.5, incorrect = 0.
    """
    if not system_answer or not system_answer.strip():
        return 0.0
    gold = (gold_answer or "").strip()
    sys = (system_answer or "").strip()
    if gold.lower() == sys.lower():
        return 1.0
    # Numeric / precise (e.g. "30 days", "99.5%")
    if gold and sys and gold in sys:
        return 1.0
    if sys and gold and sys in gold:
        return 1.0
    if embedding_fn and callable(embedding_fn):
        try:
            emb_g = embedding_fn(gold)
            emb_s = embedding_fn(sys)
            sim = _cosine_sim(emb_g, emb_s)
            if sim > semantic_threshold:
                return 1.0
        except Exception:
            pass
    # Key phrase: require important words from gold in system (simplified)
    gold_words = set(w.lower() for w in gold.split() if len(w) > 2)
    sys_lower = sys.lower()
    found = sum(1 for w in gold_words if w in sys_lower)
    if gold_words and found / len(gold_words) >= 0.6:
        return 0.5
    return 0.0


# Map system risk messages to gold-style flag names for comparison
_RISK_MESSAGE_TO_GOLD = {
    "uncapped": "POTENTIAL_UNCAPPED_LIABILITY",
    "unlimited liability": "POTENTIAL_UNCAPPED_LIABILITY",
    "no liability cap": "POTENTIAL_UNCAPPED_LIABILITY",
    "no cap": "POTENTIAL_UNCAPPED_LIABILITY",
    "sole remedy": "SOLE_REMEDY_LIMITATION",
    "exclusive remedy": "SOLE_REMEDY_LIMITATION",
}


def _normalize_system_risk_to_gold(msg: str) -> Optional[str]:
    m = (msg or "").lower().strip()
    for key, gold_flag in _RISK_MESSAGE_TO_GOLD.items():
        if key in m:
            return gold_flag
    if m:
        return m.upper().replace(" ", "_")[:50]
    return None


# --- C) Risk detection ---
def risk_score(
    expected_risk_flags: list[str],
    system_risk_flags: list[dict] | list[str],
    penalize_false_positive: bool = True,
) -> float:
    """
    Perfect match = 1, missing = 0, extra incorrect = -0.5.
    system_risk_flags can be list of dicts with "message" or "level", or list of strings.
    """
    expected_set = {r.strip().upper().replace(" ", "_") for r in (expected_risk_flags or [])}
    system_set = set()
    for r in system_risk_flags or []:
        if isinstance(r, dict):
            msg = (r.get("message") or r.get("level") or "")
            norm = _normalize_system_risk_to_gold(msg)
            if norm:
                system_set.add(norm.upper().replace(" ", "_"))
        else:
            norm = _normalize_system_risk_to_gold(str(r))
            if norm:
                system_set.add(norm.upper().replace(" ", "_"))
    if not expected_set:
        if not system_set:
            return 1.0
        return max(0.0, 1.0 - 0.5 * len(system_set)) if penalize_false_positive else 1.0
    matches = expected_set & system_set
    if matches == expected_set:
        extra = system_set - expected_set
        if not extra:
            return 1.0
        return max(0.0, 1.0 - 0.5 * len(extra)) if penalize_false_positive else 1.0
    return 0.0


# --- D) Citation accuracy ---
def citation_score(
    expected_documents: list[str],
    expected_sections: list[str],
    cited_clauses: list[dict],
) -> float:
    """
    Exact section match = 1, correct doc wrong section = 0.5, wrong doc = 0.
    """
    if not expected_documents and not expected_sections:
        return 1.0
    if not cited_clauses:
        return 0.0
    best = 0.0
    for c in cited_clauses:
        doc = (c.get("document") or "").strip()
        sec = (c.get("section") or c.get("title") or "").strip()
        doc_ok = any(_doc_match(exp_d, doc) for exp_d in expected_documents) if expected_documents else True
        sec_ok = any(_section_match(exp_s, sec) for exp_s in expected_sections) if expected_sections else True
        if doc_ok and sec_ok:
            best = max(best, 1.0)
        elif doc_ok:
            best = max(best, 0.5)
    return best


# --- E) Guardrail ---
def guardrail_score(
    gold_insufficient_info: bool,
    system_response: SynthesizedResponse,
) -> float:
    """
    If gold says insufficient_info=true, system must refuse / insufficient / outside scope.
    If system gives legal advice when shouldn't → 0 (hard fail).
    """
    sys_insufficient = system_response.insufficient_info
    answer = (system_response.direct_answer or "").lower()
    refusal_phrases = ("insufficient", "outside the scope", "cannot provide", "do not provide", "consult a lawyer", "qualified lawyer", "not offer", "outside scope")
    is_refusal = any(p in answer for p in refusal_phrases)
    if gold_insufficient_info:
        if sys_insufficient or is_refusal:
            return 1.0
        return 0.0
    if not gold_insufficient_info and ("legal advice" in answer and "cannot" in answer):
        return 1.0
    return 1.0


# --- Weights and total ---
WEIGHTS = {
    "retrieval": 0.25,
    "answer": 0.30,
    "risk": 0.20,
    "citation": 0.15,
    "guardrail": 0.10,
}


def compute_question_weights(gold: dict) -> dict[str, float]:
    """For simple retrieval questions, risk weight = 0; reallocate so weights sum to 1.0."""
    w = dict(WEIGHTS)
    if not (gold.get("expected_risk_flags") or []):
        w["risk"] = 0.0
        w["retrieval"] = 0.30
        w["answer"] = 0.35
        w["citation"] = 0.20
        w["guardrail"] = 0.15
    return w


def evaluate_one(
    question_id: str,
    gold: dict,
    response: SynthesizedResponse,
    retrieved: list[RetrievedChunk],
    embedding_fn: Optional[Callable[..., Any]] = None,
) -> dict[str, Any]:
    """
    Compute per-question scores. Gold keys: question, gold_answer, expected_documents, expected_sections,
    expected_risk_flags, insufficient_info.
    """
    expected_docs = gold.get("expected_documents") or []
    expected_secs = gold.get("expected_sections") or []
    expected_risks = gold.get("expected_risk_flags") or []
    gold_insufficient = gold.get("insufficient_info", False)

    r_recall = retrieval_recall_at_k(retrieved, expected_docs, expected_secs) if expected_docs else 1.0
    ans_sc = answer_score(gold.get("gold_answer") or "", response.direct_answer or "", embedding_fn)
    risk_sc = risk_score(expected_risks, response.risk_flags or [])
    cite_sc = citation_score(expected_docs, expected_secs, response.cited_clauses or [])
    guard_sc = guardrail_score(gold_insufficient, response)

    weights = compute_question_weights(gold)
    total = (
        weights["retrieval"] * r_recall
        + weights["answer"] * ans_sc
        + weights["risk"] * risk_sc
        + weights["citation"] * cite_sc
        + weights["guardrail"] * guard_sc
    )
    return {
        "question_id": question_id,
        "retrieval_score": r_recall,
        "answer_score": ans_sc,
        "risk_score": risk_sc,
        "citation_score": cite_sc,
        "guardrail_score": guard_sc,
        "total_score": round(total, 4),
        "precision_at_k": retrieval_precision_at_k(retrieved, expected_docs) if expected_docs else None,
        "mrr": mrr_retrieval(retrieved, expected_docs) if expected_docs else None,
    }


def load_gold_datasets(eval_dir: Path) -> list[dict]:
    """Load all JSON files from evaluation_dataset into one list of gold items."""
    import json
    items = []
    for path in sorted(Path(eval_dir).glob("*.json")):
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
        if isinstance(data, list):
            items.extend(data)
        else:
            items.append(data)
    return items


def run_evaluation(
    orchestrator,
    gold_items: list[dict],
    embedding_fn: Optional[Callable[..., Any]] = None,
) -> tuple[list[dict], dict]:
    """
    Run orchestrator.run_with_retrieval for each item in order (multi-turn: M1-1 before M1-2).
    Returns (list of per-question score dicts, aggregate dict).
    If embedding_fn is None, uses embed_single so semantic similarity (answer_score) is computed.
    """
    if embedding_fn is None:
        try:
            from rag.retrieval.embeddings import embed_single
            embedding_fn = embed_single
        except Exception:
            pass
    sorted_items = sorted(gold_items, key=lambda g: (str(g.get("question_id", "")), g.get("question", "")))
    results = []
    for g in sorted_items:
        qid = g.get("question_id", "")
        question = g.get("question", "")
        if not question:
            continue
        try:
            response, retrieved = orchestrator.run_with_retrieval(question)
        except Exception as e:
            results.append({
                "question_id": qid,
                "retrieval_score": 0,
                "answer_score": 0,
                "risk_score": 0,
                "citation_score": 0,
                "guardrail_score": 0,
                "total_score": 0,
                "error": str(e),
            })
            continue
        res = evaluate_one(qid, g, response, retrieved, embedding_fn)
        results.append(res)

    if not results:
        return results, {}
    agg = {
        "mean_retrieval": sum(r["retrieval_score"] for r in results) / len(results),
        "mean_answer": sum(r["answer_score"] for r in results) / len(results),
        "mean_risk": sum(r["risk_score"] for r in results) / len(results),
        "mean_citation": sum(r["citation_score"] for r in results) / len(results),
        "mean_guardrail": sum(r["guardrail_score"] for r in results) / len(results),
        "mean_total": sum(r["total_score"] for r in results) / len(results),
        "n_questions": len(results),
    }
    mrrs = [r["mrr"] for r in results if r.get("mrr") is not None]
    if mrrs:
        agg["mean_mrr"] = sum(mrrs) / len(mrrs)
    return results, agg
