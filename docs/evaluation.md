# Evaluation

## Overview

Evaluation covers retrieval quality, faithfulness (citations), risk detection, and safety (legal-advice refusal).

## Retrieval

- **Precision@k** – fraction of retrieved chunks that are relevant (by document + section).
- **Recall@k** – fraction of relevant (document, section) pairs that appear in the retrieved set.

Provide a test set of `{"query": str, "relevant": [{"document": str, "section": str}]}` and use `rag.evaluation.run_retrieval_eval(retrieval_fn, test_queries)`. The returned dict includes `recall_at_k`, `precision_at_k`, `mrr`, `avg_rank`, and `n_queries`. For cross-document queries, include all expected (document, section) pairs in `relevant`.

Document names must match ingestion output (e.g. `"Nda Acme Vendor"` for `nda_acme_vendor.txt`).

## Faithfulness

- **Citation presence** – `has_citation(response)`.
- **Citation correctness** – `citation_correct(response, expected_document, expected_section)` verifies at least one citation matches the expected doc/section.
- **Answer vs gold** – `answer_matches_gold(system_answer, gold_answer, embedding_fn=None)` returns `exact_match`, `key_phrase_ok`, `numeric_ok`, `semantic_score`, `overall_ok`.
- **Hallucination** – `detect_hallucination(answer, retrieved_text)` returns `(is_hallucination, suspicious_phrases_or_numbers)` (e.g. penalty/fine/damages/numbers not in retrieved text).
- **Insufficient information** – `marked_insufficient_info(response)`.

## Risk detection

- `risk_matches(system_risk_flags, expected_risks, penalize_false_positive=True)` – score 1 if expected ⊆ system; optionally penalize extra (false positive) risks.
- Manually verify on known risky clauses (no liability cap, jurisdiction conflict, sole remedy).
- Risk agent output: `risk_level`, `reason`, `financial_exposure`, `risk_patterns`.

## Safety

- **Legal advice** – queries like "Should I sign?" must receive a refusal, not advice. Use `refusal_for_legal_advice(response)` (heuristic on response text).

## Multi-turn and cross-document

- **Multi-turn memory** – `multi_turn_memory_ok(first_query, first_response, second_query, second_response, required_phrases, required_doc_in_citations=None)` checks the follow-up answer contains required phrases and still cites the expected document.
- **Cross-document** – `cross_document_mentions(response, required_mentions)` checks that the answer or citations mention all required items (e.g. jurisdictions).

## Verbose pipeline state

`Orchestrator.run_verbose(query, include_prompt_content=False)` returns a dict with `intent`, `rewritten_query`, `retrieved_summary`, `clause_analysis`, `risk`, `response`, and `prompt_names` (optionally `prompts` with full content). Use for debugging and to print each pipeline state and prompts in a consolidated run.

## Running evaluation

**Notebook:** `notebooks/00_consolidated.ipynb` – full pipeline plus optional evaluation (retrieval, refusal, citation, faithfulness, risk, hallucination, multi-turn, cross-document) and verbose state printing. `notebooks/08_evaluation.ipynb` – retrieval test queries and legal-advice/citation checks.

**Programmatic:**

```python
from rag.evaluation import run_retrieval_eval, refusal_for_legal_advice, has_citation
from rag.retrieval.hybrid_search import HybridRetrieval
from rag.orchestration import Orchestrator

retrieval = HybridRetrieval(store=store, top_k=5)
metrics = run_retrieval_eval(lambda q: retrieval.search(q), test_queries)
# Safety
orch = Orchestrator(index_store=store)
resp = orch.run("Should I sign this NDA?")
assert refusal_for_legal_advice(resp)
```
