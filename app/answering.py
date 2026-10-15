"""Question answering orchestration: evidence retrieval, abstention, response assembly.

Single entry point ``answer_question(db, question)`` used by the FastAPI
``/ask`` handler and the offline eval harness alike, so measured behavior
is the shipped behavior.
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from .langchain_pipeline import retrieve_evidence, synthesize_with_langchain
from .schemas import AskResponse, Citation
from .services import (
    InsufficientEvidence,
    QueryService,
    classify,
    evaluate,
    grounded_answer,
    lexical_overlap,
)

ABSTAIN_MESSAGE = (
    "I don't have evidence to answer that. I can answer questions about "
    "Q2 vs Q3 revenue by segment, the revenue metric definition, and the "
    "ingested business reports — all with cited sources."
)

# Question terms proving the governed structured data covers the question,
# even when no document chunk shares vocabulary with it (e.g. "How did SMB
# perform?" — the segment revenue rows answer it, the report text does not).
# A bare period ("Q3") is NOT enough: "Why did customers churn in Q3?" is
# about churn, not revenue, and must abstain.
TOPIC_ANCHORS = frozenset({
    "revenue", "segment", "enterprise", "smb", "mid-market", "midmarket",
    "metric", "decline", "drop",
})


def answer_question(db: Session, question: str) -> AskResponse:
    """Answer with cited evidence, or abstain when evidence is insufficient."""
    intent = classify(question)
    try:
        evidence = retrieve_evidence(db, question)
        data, docs, definition = evidence["data"], evidence["docs"], evidence["definition"]
        if definition is None:
            raise InsufficientEvidence("revenue metric definition missing")
        if intent.name == "knowledge_lookup" and docs:
            lowered = question.lower()
            best_overlap = max(lexical_overlap(question, d["content"]) for d in docs)
            has_anchor = any(anchor in lowered for anchor in TOPIC_ANCHORS)
            if best_overlap == 0 and not has_anchor:
                raise InsufficientEvidence("no retrieved chunk shares terms with the question")
        citations = [
            Citation(kind="sql", title="Governed revenue comparison",
                     locator="revenue_facts: Q2 vs Q3 by segment",
                     excerpt="Reviewed template; no generated SQL executed."),
            Citation(kind="metric", title=definition.name,
                     locator=f"{definition.source_table} | {definition.grain}",
                     excerpt=definition.definition),
        ]
        for doc in docs:
            citations.append(Citation(kind="document", title="Q3 Commercial Performance Review",
                                      locator=f"document_chunks:{doc['id']}", excerpt=doc["content"]))
        fallback = grounded_answer(data, docs)
        if intent.name == "knowledge_lookup" and definition is not None:
            # Metric questions are otherwise answered with numbers only;
            # verbalize the cited definition so the answer stands alone.
            fallback += (f" Metric used: {definition.name} — {definition.definition}"
                         f" (grain: {definition.grain}).")
        answer = synthesize_with_langchain(evidence, fallback)
        return AskResponse(answer=answer, intent=intent.name,
                           citations=citations, sql=QueryService.QOQ_REVENUE_SQL,
                           data=data, evaluation=evaluate(answer, [c.model_dump() for c in citations]))
    except InsufficientEvidence as exc:
        return AskResponse(answer=f"{ABSTAIN_MESSAGE} (Reason: {exc}.)",
                           intent=intent.name, citations=[], sql="",
                           data={}, evaluation={"grounded": False, "citation_count": 0,
                                                "checks": ["insufficient evidence — abstained"]})
