"""Offline eval harness for the BI RAG agent (deterministic, no LLM, no network).

Builds a fresh in-memory SQLite DB, seeds the demo knowledge, runs every
case in ``qa_test_set.json`` through the same ``answer_question()`` service
the ``/ask`` endpoint uses, and writes ``eval/report.json``.

    python eval/run_eval.py

Metrics per case: intent match, retrieval recall@2, answer-phrase coverage,
citation-kind coverage, abstention correctness, latency. Summary aggregates
plus the known-gap list (cases the system is EXPECTED to fail today).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

for var in ("OPENAI_API_KEY", "SEMANTIC_RETRIEVAL_ENABLED"):
    os.environ.pop(var, None)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

from app.answering import ABSTAIN_MESSAGE, answer_question  # noqa: E402
from app.main import seed  # noqa: E402
from app.models import Base, DocumentChunk, MetricDefinition, RevenueFact  # noqa: E402
from app.services import retrieve  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def fresh_db(seeded: bool = True):
    # StaticPool: one shared :memory: connection, so the LCEL worker threads
    # used by retrieve_evidence see the same tables as the seeding thread.
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False},
                           poolclass=StaticPool)
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine, autoflush=False)()
    if seeded:
        seed(session)
    return session


def run_case(db, case: dict) -> dict:
    if case.get("setup") == "empty_db":
        db.query(RevenueFact).delete()
        db.query(DocumentChunk).delete()
        db.query(MetricDefinition).delete()
        db.commit()
    question = case["question"]
    docs = retrieve(db, question)
    t0 = time.perf_counter()
    response = answer_question(db, question)
    latency_ms = round((time.perf_counter() - t0) * 1000, 1)
    answer = response.answer
    abstained = answer.startswith(ABSTAIN_MESSAGE[:20])

    expected_docs = case.get("expected_doc_contains", [])
    recall = (
        sum(any(s.lower() in d["content"].lower() for d in docs) for s in expected_docs) / len(expected_docs)
        if expected_docs
        else None
    )
    expected_phrases = case.get("expected_phrases", [])
    coverage = (
        sum(p.lower() in answer.lower() for p in expected_phrases) / len(expected_phrases)
        if expected_phrases
        else None
    )
    expected_kinds = set(case.get("expected_citation_kinds", []))
    actual_kinds = {c.kind for c in response.citations}
    return {
        "id": case["id"],
        "intent": response.intent,
        "intent_ok": response.intent == case["intent"],
        "abstained": abstained,
        "abstention_ok": abstained == case["expect_abstain"],
        "recall_at_2": round(recall, 3) if recall is not None else None,
        "coverage": round(coverage, 3) if coverage is not None else None,
        "citations_ok": expected_kinds <= actual_kinds,
        "citation_kinds": sorted(actual_kinds),
        "latency_ms": latency_ms,
        "answer_preview": answer[:220],
    }


def main() -> None:
    suite = json.loads((ROOT / "eval" / "qa_test_set.json").read_text(encoding="utf-8"))
    db = fresh_db()
    results = [run_case(db, case) for case in suite["cases"]]

    def avg(key: str, filt=None):
        vals = [r[key] for r in results if r[key] is not None and (filt is None or filt(r))]
        return round(sum(vals) / len(vals), 3) if vals else None

    answerable = [r for r in results if r["id"] in
                  {c["id"] for c in suite["cases"] if c["answerable"]}]
    report = {
        "n_cases": len(results),
        "intent_accuracy": round(sum(r["intent_ok"] for r in results) / len(results), 3),
        "abstention_accuracy": sum(r["abstention_ok"] for r in results) / len(results),
        "mean_recall_at_2": avg("recall_at_2"),
        "mean_coverage_answerable": (
            sum(r["coverage"] for r in answerable if r["coverage"] is not None)
            / max(1, sum(1 for r in answerable if r["coverage"] is not None))
        ),
        "mean_latency_ms": avg("latency_ms"),
        "llm": "none (deterministic fallback; OPENAI_API_KEY unset)",
        "cost_per_query_usd": 0.0,
        "cases": results,
    }
    out = ROOT / "eval" / "report.json"
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")

    print(f"{'case':24s} {'intent':5s} {'abstain':7s} {'recall':6s} {'cover':5s} {'cite':4s} {'ms':>6s}")
    for r in results:
        print(f"{r['id']:24s} {str(r['intent_ok']):5s} {str(r['abstention_ok']):7s} "
              f"{str(r['recall_at_2']):6s} {str(r['coverage']):5s} {str(r['citations_ok']):4s} {r['latency_ms']:>6.1f}")
    print(f"\nintent_acc={report['intent_accuracy']:.3f} abstention_acc={report['abstention_accuracy']:.3f} "
          f"recall@2={report['mean_recall_at_2']} coverage(answerable)={report['mean_coverage_answerable']:.3f} "
          f"p50_latency~{report['mean_latency_ms']}ms llm=none cost=$0")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
