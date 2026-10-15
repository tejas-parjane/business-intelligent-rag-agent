"""Tests for the answer service and eval behavior (SQLite in-memory, no LLM)."""

from __future__ import annotations

import os
import sys
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
from app.services import QueryService, lexical_overlap  # noqa: E402


def make_db(seeded: bool = True):
    # StaticPool shares one :memory: connection across the LCEL threads.
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False},
                           poolclass=StaticPool)
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine, autoflush=False)()
    if seeded:
        seed(db)
    return db


def test_demo_question_answered_with_all_citation_kinds():
    resp = answer_question(make_db(), "Why did revenue decline in Q3?")
    assert resp.intent == "metric_driver_analysis"
    assert "21.7" in resp.answer and "Enterprise" in resp.answer
    assert {c.kind for c in resp.citations} == {"sql", "metric", "document"}
    assert resp.evaluation["grounded"] is True


def test_off_topic_questions_abstain():
    for q in ("What is our current headcount?", "Why did customers churn in Q3?",
              "Did we change pricing in Q3?", "asdf qwerty zxcv blorp"):
        resp = answer_question(make_db(), q)
        assert resp.answer.startswith(ABSTAIN_MESSAGE[:20]), q
        assert resp.citations == []
        assert resp.evaluation["grounded"] is False


def test_empty_database_abstains_without_crash():
    db = make_db()
    db.query(RevenueFact).delete()
    db.query(DocumentChunk).delete()
    db.query(MetricDefinition).delete()
    db.commit()
    resp = answer_question(db, "Why did revenue decline in Q3?")
    assert resp.answer.startswith(ABSTAIN_MESSAGE[:20])
    assert resp.citations == []


def test_segment_question_answered_from_structured_data():
    resp = answer_question(make_db(), "How did SMB perform in Q3?")
    assert not resp.answer.startswith(ABSTAIN_MESSAGE[:20])
    assert "SMB" in resp.answer


def test_metric_definition_is_verbalized():
    resp = answer_question(make_db(), "How is revenue defined?")
    assert "excluding tax" in resp.answer


def test_governed_sql_never_contains_question_text():
    resp = answer_question(make_db(), "Why did revenue decline in Q3?'; DROP TABLE revenue_facts; --")
    assert "DROP TABLE" not in resp.sql
    assert resp.sql == QueryService.QOQ_REVENUE_SQL


def test_lexical_overlap_counts_terms():
    assert lexical_overlap("Why did revenue decline?", "Enterprise revenue fell") >= 1
    assert lexical_overlap("headcount benefits", "Enterprise revenue fell") == 0
