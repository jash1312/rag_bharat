# Evaluating Against a Premium OpenAI Model

Use the same questions and a single prompt so you can compare the RAG system to a latest OpenAI model (e.g. GPT-4o) on the same evaluation format.

## Files

- **`rag/prompts/openai_evaluation_prompt.txt`** – Single prompt that defines the output format and lists all questions. Paste your contract text first, then this prompt.
- **`rag/prompts/evaluation_queries.json`** – Same 34 questions as a JSON list (for running the RAG pipeline on the same set).

## Output Format (for evaluation)

Each answer must be a JSON object with:

- **direct_answer** (string)
- **cited_clauses** (array of `{document, section, title, excerpt}`)
- **risk_flags** (array of `{level, message}`)
- **follow_up_suggestions** (array of strings)
- **insufficient_info** (boolean)

The prompt asks the model to return a **single JSON array** with one object per question, in order. That array can be parsed and compared to `SynthesizedResponse`-style outputs (e.g. for citation, insufficient_info, refusal on legal-advice questions).

## How to run

1. Load your contract documents (e.g. NDA, SLA, DPA, Vendor Agreement) into the chat as context (e.g. system message or first user message).
2. Send the contents of `openai_evaluation_prompt.txt` as the user message (or append it after the contracts).
3. Copy the model’s JSON array response and parse it (e.g. `json.loads(...)`).
4. Run your evaluation logic (citation presence, insufficient_info, legal-advice refusal) on each object, and optionally compare to RAG outputs for the same questions via `load_evaluation_queries()`.

## Loading queries in code

```python
from rag.prompts import load_evaluation_queries, load_prompt

queries = load_evaluation_queries()  # list of 34 question strings
openai_prompt = load_prompt("openai_evaluation_prompt")  # full prompt with instructions + questions
```

Use `queries` to run the RAG orchestrator on each question and collect `SynthesizedResponse` objects, then compare to the parsed OpenAI array.
