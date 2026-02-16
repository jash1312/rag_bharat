Multi-Agent RAG System for Legal Contract Analysis
1. Overview

This project implements a multi-agent Retrieval-Augmented Generation (RAG) system designed to analyze and reason over legal contracts.

The system enables users to:

Ask natural language questions about contracts

Receive grounded, well-cited answers

Identify potential legal risks

Perform cross-document reasoning

Engage in multi-turn interactive dialogue via console

The application is intentionally architected using separated reasoning agents to ensure:

Controlled inference

Reduced hallucination

Legal clause-level grounding

Structured risk assessment

2. Problem Statement

Legal agreements such as:

NDAs

Vendor Services Agreements

Service Level Agreements

Data Processing Agreements

contain complex clauses related to:

Termination

Confidentiality

Liability caps

Indemnification

Governing law

Data breach obligations

Manual review is time-consuming and error-prone.

This system provides an intelligent legal analysis assistant built on a multi-agent RAG architecture to:

Retrieve relevant clauses

Analyze obligations

Detect liability exposure

Surface risk indicators

Maintain contextual memory across conversation turns

3. System Capabilities
✔ Clause-Level Question Answering

Example:

“What is the notice period for terminating the NDA?”

✔ Cross-Document Reasoning

Example:

“Are there conflicting governing laws across agreements?”

✔ Risk Detection

Example:

“Is there any unlimited liability?”

✔ Hypothetical Scenario Analysis

Example:

“What happens if Vendor delays breach notification beyond 72 hours?”

✔ Multi-Turn Context Handling

Example:
User:

“Does confidentiality survive termination?”

Follow-up:

“For how long?”

The system remembers the NDA context.

4. Architecture Overview

The system is built using a multi-agent modular architecture.

User (CLI)
   ↓
Conversation Agent
   ↓
Intent & Query Router
   ↓
Retrieval Agent (Hybrid Search)
   ↓
Clause Analyzer Agent
   ↓
Risk Assessment Agent
   ↓
Response Synthesizer Agent

Each agent has clearly defined responsibilities to prevent over-agentification.

5. RAG Design Decisions
Document Chunking Strategy

Clause-aware hierarchical chunking

Section-based segmentation

Metadata tagging (document, clause number, topic)

Embedding Model

OpenAI text-embedding-3-large OR

Local embedding model via Ollama

Retrieval Strategy

Hybrid search:

BM25 (keyword precision)

Vector similarity (semantic recall)

Metadata filtering

LLM Model

Deterministic configuration

Low temperature (0.1)

Structured JSON extraction prompts

Hallucination control instructions

6. Evaluation Strategy

The system is evaluated on:

Retrieval recall (Top-K clause hit rate)

Faithfulness (no hallucination)

Risk detection accuracy

Cross-document reasoning correctness

Refusal behavior for legal advice queries

7. Setup Instructions

Create and activate a virtual environment, then install dependencies:

```bash
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Local (default):** Use [Ollama](https://ollama.com) — no API key. Set `PROVIDER=ollama`, pull `nomic-embed-text` and a chat model (e.g. `llama3.2`).  
**Cloud:** Set `PROVIDER=openai` and `OPENAI_API_KEY` in `.env`. See `docs/setup.md`.

Run ingestion pipeline, then start the console application:

```bash
python ingest.py
python main.py
```

- **Notebooks** – Under `notebooks/`: component notebooks (ingestion, retrieval, intent, clause analyzer, risk, response synthesis), full flow, and evaluation. All logic lives in the `rag` package; notebooks use config and call the modules.
- **Docs** – In `docs/`: `setup.md`, `ingestion.md`, `retrieval.md`, `evaluation.md`.
- **Data** – Contract `.txt` files go in `input_data/`.
8. Known Limitations

Small evaluation corpus

No adversarial document noise

Limited real-world drafting ambiguity

No clause-level legal ontology enrichment

9. Production Improvements

If deployed at enterprise scale:

Clause graph knowledge base

Legal taxonomy tagging

Version diffing

Audit logging

Risk scoring dashboard

Automated redlining