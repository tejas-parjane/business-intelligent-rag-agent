import os
from datetime import date
from fastapi import Depends, FastAPI
from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import Session, sessionmaker
from .models import Base, Document, DocumentChunk, MetricDefinition, RevenueFact
from .schemas import AskRequest, AskResponse, IngestRequest, IngestResponse
from .ingestion import ingest_text_document
from .answering import answer_question

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./bi_rag.db")
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False)
app = FastAPI(title="Business Intelligence RAG Agent", version="0.1.0")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def seed(db: Session):
    if db.scalar(select(MetricDefinition.id).limit(1)):
        return
    db.add(MetricDefinition(name="Revenue", definition="Recognized subscription revenue, excluding tax and credits.", grain="Quarter x customer segment", source_table="revenue_facts"))
    db.add_all([
        RevenueFact(period="Q2", segment="Enterprise", revenue=700000), RevenueFact(period="Q3", segment="Enterprise", revenue=500000),
        RevenueFact(period="Q2", segment="Mid-market", revenue=300000), RevenueFact(period="Q3", segment="Mid-market", revenue=280000),
        RevenueFact(period="Q2", segment="SMB", revenue=200000), RevenueFact(period="Q3", segment="SMB", revenue=160000),
    ])
    report = Document(title="Q3 Commercial Performance Review", source_uri="s3://bi-rag-demo/reports/q3-commercial-review.pdf", published_on=date(2026, 10, 5))
    db.add(report); db.flush()
    db.add(DocumentChunk(document_id=report.id, chunk_index=0, content="Enterprise revenue fell as several large renewals moved into Q4 after procurement delays. SMB new-logo conversion softened after paid-search efficiency declined. Mid-market remained broadly stable."))
    db.commit()


@app.on_event("startup")
def startup():
    if engine.dialect.name == "postgresql":
        with engine.begin() as connection:
            connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(engine)
    with SessionLocal() as db:
        seed(db)


@app.get("/health")
def health():
    return {"status": "ok", "llm_configured": bool(os.getenv("OPENAI_API_KEY")), "semantic_retrieval_enabled": os.getenv("SEMANTIC_RETRIEVAL_ENABLED", "false")}


@app.post("/ingest/text", response_model=IngestResponse, status_code=201)
def ingest_text(request: IngestRequest, db: Session = Depends(get_db)):
    document, chunks_created, semantic = ingest_text_document(db, request.title, request.source_uri, request.content)
    return IngestResponse(document_id=document.id, chunks_created=chunks_created, semantic_embeddings_created=semantic)


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest, db: Session = Depends(get_db)):
    return answer_question(db, request.question)


@app.post("/evaluate")
def evaluation(request: AskRequest, db: Session = Depends(get_db)):
    return ask(request, db).evaluation
