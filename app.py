"""Hugging Face Spaces entrypoint (Streamlit SDK).

Runs the RAG pipeline in-process — no separate FastAPI server — so the public
demo is a single container that shows real LangChain retrieval, governed SQL,
citations, and evaluation deterministically, with no LLM key and zero API cost.

The FastAPI service (``app/main.py``) and its ``ui.py`` client remain the
two-service Docker demo; this file reuses the same ``answer_question`` service
so the hosted behavior matches the tested behavior.
"""

from __future__ import annotations

import os

os.environ.setdefault("DATABASE_URL", "sqlite:////tmp/bi_rag.db")
os.environ.setdefault("SEMANTIC_RETRIEVAL_ENABLED", "false")

import streamlit as st
from sqlalchemy import select

from app.answering import answer_question
from app.main import SessionLocal, engine
from app.models import Base, MetricDefinition

EXAMPLES = [
    "Why did revenue decline in Q3?",
    "What is the revenue metric definition?",
    "How did SMB perform?",
    "Why did customers churn in Q4?",
]


@st.cache_resource(show_spinner="Preparing the demo knowledge base...")
def _bootstrap() -> bool:
    Base.metadata.create_all(engine)
    with SessionLocal() as db:
        if db.scalar(select(MetricDefinition.id).limit(1)) is None:
            from app.main import seed

            seed(db)
    return True


_bootstrap()

st.set_page_config(page_title="BI RAG Agent", page_icon="📊", layout="wide")

st.title("Business Intelligence RAG Agent")
st.caption(
    "Evidence-first answers over business documents, KPI definitions, and governed "
    "SQL data — every answer carries its sources. Deterministic demo (no LLM key): "
    "LangChain retrieval, reviewed SQL templates, and citations."
)

if "asked" not in st.session_state:
    st.session_state.asked = None
    st.session_state.result = None

with st.sidebar:
    st.header("Try a question")
    preset = st.radio("Examples", EXAMPLES, index=0, label_visibility="collapsed")
    custom = st.text_input("Or type your own", "")
    question = custom.strip() or preset
    clicked = st.button("Analyze", type="primary", use_container_width=True)
    st.caption(
        "Out-of-scope questions abstain on purpose — that is the safety "
        "behavior, not a failure."
    )

if clicked or st.session_state.asked is None:
    with st.spinner("Retrieving governed evidence..."):
        with SessionLocal() as db:
            st.session_state.result = answer_question(db, question)
        st.session_state.asked = question

result = st.session_state.result

if result.data:
    st.subheader("Grounded answer")
    st.write(result.answer)

    c1, c2, c3 = st.columns(3)
    c1.metric("Q2 revenue", f"${result.data['q2_revenue']:,.0f}")
    c2.metric("Q3 revenue", f"${result.data['q3_revenue']:,.0f}", f"{result.data['change_pct']}%")
    c3.metric("Evidence checks", result.evaluation.get("citation_count", 0))

    st.subheader("Segment contribution")
    st.bar_chart({row["segment"]: row["change"] for row in result.data["drivers"]})
else:
    st.subheader("Grounded answer")
    st.info(result.answer)

if result.citations:
    with st.expander("Evidence & citations", expanded=True):
        for citation in result.citations:
            st.markdown(f"**{citation.kind.upper()} · {citation.title}**")
            st.caption(f"{citation.locator} — {citation.excerpt or ''}")

if result.sql:
    with st.expander("Reviewed SQL template (no generated SQL is executed)"):
        st.code(result.sql, language="sql")

with st.expander("Evaluation trace"):
    st.json(result.evaluation)
