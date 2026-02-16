"""Risk assessment agent: evaluate liability, jurisdiction, sole remedy, etc."""

from rag.models import RetrievedChunk, RiskAssessment
from rag.agents.llm import chat_completion, extract_json_from_text
from rag.prompts import load_prompt


class RiskAssessmentAgent:
    """Evaluate risk from retrieved clauses."""

    def assess(self, retrieved: list[RetrievedChunk]) -> RiskAssessment:
        """Produce RiskAssessment from retrieved chunks."""
        if not retrieved:
            return RiskAssessment(risk_level="LOW", reason="No clauses retrieved.")
        context = "\n\n---\n\n".join(
            f"[{r.chunk.document} - {r.chunk.title}]\n{r.chunk.clause_text}" for r in retrieved[:5]
        )
        content = chat_completion(
            [
                {"role": "system", "content": load_prompt("risk_system")},
                {"role": "user", "content": f"Assess risk for these clauses:\n\n{context}"},
            ],
            max_tokens=512,
        )
        data = extract_json_from_text(content)
        if data:
            return RiskAssessment(
                risk_level=data.get("risk_level", "LOW"),
                reason=data.get("reason", ""),
                financial_exposure=data.get("financial_exposure"),
                risk_patterns=data.get("risk_patterns") or [],
            )
        return RiskAssessment(risk_level="LOW", reason="Could not parse risk assessment.")
