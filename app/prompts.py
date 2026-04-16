from __future__ import annotations

import json

from app.config import get_settings
from app.schemas import RetrievalResult


SYSTEM_PROMPT = """You are TrustRAG, a trustworthy question answering assistant for academic and technical documents.

Rules:
1. Answer strictly and only from the evidence provided.
2. Never invent facts, definitions, numbers, or claims not supported by the evidence.
3. If the evidence is insufficient, ambiguous, or irrelevant, abstain.
4. Keep the answer concise, precise, and professional.
5. Cite supporting evidence using the exact citation strings provided in the evidence.
6. Do not reveal chain-of-thought. Provide only a short evidence-based reasoning summary.
7. Prefer the highest-ranked evidence that directly answers the question.
8. If the question asks about the paper's own limitations, results, or future work, do not answer from general background or related work unless the evidence explicitly ties it to the paper itself.
9. If the evidence only provides indirect context, say so briefly or abstain.
10. If one top evidence chunk already contains the exact dataset names, numbers, or result values needed to answer the question, it is acceptable to cite that chunk alone.

Return valid JSON with this schema:
{
  "answer": "string",
  "citations": ["string"],
  "reasoning_summary": "short evidence-based rationale",
  "abstain": true
}
"""


def build_user_prompt(question: str, evidence: list[RetrievalResult]) -> str:
    settings = get_settings()
    payload = []
    used_chars = 0
    for item in evidence:
        citation = f"[Source: {item.document_name}, p. {item.page_start}, chunk {item.chunk_id}]"
        remaining_chars = settings.answer_max_context_chars - used_chars
        if remaining_chars <= 0:
            break
        text = item.text[:remaining_chars]
        payload.append(
            {
                "citation": citation,
                "score": round(item.score, 4),
                "text": text,
            }
        )
        used_chars += len(text)

    return (
        f"Question: {question}\n\n"
        "Evidence:\n"
        f"{json.dumps(payload, indent=2, ensure_ascii=False)}\n\n"
        "Use only directly relevant evidence. If the question is about the paper's own contributions, limitations, conclusions, or future work, prioritize chunks that explicitly discuss those points. "
        "When a single chunk already contains the exact numeric results needed for the answer, prefer that chunk and do not force extra citations. "
        "If the evidence does not support a confident answer, set abstain to true."
    )
