# Business Intelligence RAG Agent — Applied AI (complements core ML work)

> **Independent Applied-AI project** (self-directed demo). My core Data Science work is predictive modeling / forecasting in `attention-os`, `forecasting-engine`, and `growth-decision-engine`; this repo demonstrates applied LLM engineering alongside it.

> A production-shaped, evidence-first GenAI application for answering business questions from documents, KPI definitions, and structured revenue data.

> 🚀 **Live demo (no signup, no API key):** https://business-intelligent-rag-agent-anwnwmvr3efyz2y9getagj.streamlit.app

This is a focused portfolio project demonstrating how to build a safe RAG workflow beyond a generic “chat with PDFs” demo. It combines **LangChain orchestration**, governed SQL retrieval, a vector-ready knowledge layer, citations, and evaluation behind a typed FastAPI service.

## Portfolio highlights

- **Applied RAG engineering:** LangChain LCEL orchestration retrieves business-report chunks, structured metric data, and KPI definitions in parallel.
- **Grounded answers:** every response includes its approved SQL template, underlying data, metric-definition context, and document-level citations.
- **Safe structured-data access:** no model-generated SQL is executed. The application selects reviewed query templates rather than allowing arbitrary database access.
- **Production-minded design:** Pydantic contracts, service-layer separation, Docker deployment, health checks, deterministic fallbacks, and evaluation signals.
- **Semantic retrieval:** local SentenceTransformers/PyTorch embeddings and pgvector cosine similarity are enabled with one environment flag; lexical retrieval remains the no-cost fallback.
- **Applied ML evidence:** a small PyTorch contrastive metric-learning experiment demonstrates retriever-training mechanics separately from the production embedding model.

## The business problem

Business teams often need to search reports, metric definitions, and a warehouse separately before they can answer a simple question such as:

> **Why did revenue decline in Q3?**

This agent resolves the relevant metric, compares Q2 and Q3 by segment using a governed data query, retrieves supporting business-report evidence, and returns a concise explanation with traceable sources.

1. **Business documents** – chunked reports with source citations.
2. **SQL data** – governed revenue facts, queried through approved templates.
3. **KPI definitions** – a semantic layer that explains the metric and grain.

The demo answers: **“Why did revenue decline in Q3?”** by comparing Q2 and Q3, identifying contributing segments, retrieving an analyst report, and returning evidence with SQL and document citations.

## Architecture

```text
Question -> intent classification -> metric resolution
         -> LangChain LCEL evidence graph
         -> governed SQL + document retriever + KPI definition
         -> evidence-only LLM synthesis / deterministic fallback
         -> answer + citations + evaluation trace
```

```text
Business documents / SQL data / KPI definitions
                    ↓
          ingestion + chunk metadata
                    ↓
      embeddings + pgvector (production extension)
                    ↓
        LangChain custom retriever + SQL evidence
                    ↓
        LLM reasoning constrained to retrieved evidence
                    ↓
          grounded answer, citations, evaluation
```

The database has `documents`, `document_chunks`, `metric_definitions`, and `revenue_facts` tables. PostgreSQL is bundled with pgvector, and `DocumentChunk.embedding` maps to a native `vector(384)` column when PostgreSQL is active. Set `SEMANTIC_RETRIEVAL_ENABLED=true` to embed newly ingested documents with the local PyTorch-backed `all-MiniLM-L6-v2` model and retrieve them with pgvector cosine distance. The default remains deterministic lexical retrieval so the public demo runs without model downloads or a paid API account.

## Technical implementation

- **LangChain**: LCEL orchestration, custom retriever, structured evidence context, optional `ChatOpenAI` synthesis.
- **FastAPI + Pydantic**: typed API contract and service boundary.
- **SQLAlchemy + PostgreSQL/pgvector**: governed SQL and a vector-ready document schema.
- **Streamlit + Docker**: usable UI and reproducible local deployment.
- **Evaluation**: response-level checks for evidence coverage, citation count, and grounded-answer constraints.
- **PyTorch + SentenceTransformers**: local embedding inference; a compact contrastive retriever-projection training experiment in [`scripts/train_retriever.py`](scripts/train_retriever.py).

LlamaIndex is intentionally not included: it overlaps with LangChain for this scope. A focused LangChain implementation is easier to explain in an interview than using both frameworks without a clear responsibility split.

## What the live demo proves

| Capability | Evidence in this repository |
| --- | --- |
| Python application design | FastAPI service, SQLAlchemy models, typed Pydantic request/response schemas |
| RAG orchestration | [`app/langchain_pipeline.py`](app/langchain_pipeline.py): LCEL `RunnableParallel`, custom `BaseRetriever`, optional `ChatOpenAI` chain |
| Data engineering | document chunk model, metadata, KPI semantic layer, revenue fact model, seeded ingestion path |
| Embeddings and vector search | `/ingest/text` chunks documents and creates local embeddings; PostgreSQL executes pgvector cosine-distance retrieval when enabled |
| Neural-network fundamentals | `scripts/train_retriever.py` trains a PyTorch projection with contrastive retrieval loss |
| GenAI safety | evidence-only system prompt, governed SQL templates, no arbitrary generated SQL execution |
| Evaluation / observability | `/evaluate` endpoint and response-level grounding/citation checks |
| Deployment practice | Dockerfile, Docker Compose, Render Blueprint, service health endpoint |

## Run it

1. Copy `.env.example` to `.env` and set your existing `OPENAI_API_KEY` locally if you want LLM synthesis. The service remains useful without it.
2. Start the stack:

   ```bash
   docker compose up --build
   ```

3. Open Streamlit at http://localhost:8501, or FastAPI docs at http://localhost:8000/docs.

For a local API-only demo, run `pip install -r requirements.txt` then `uvicorn app.main:app --reload`. It uses SQLite automatically when `DATABASE_URL` is absent.

`requirements.txt` is the light runtime set (API, Streamlit, evaluation). The optional PyTorch/SentenceTransformers semantic-retrieval stack lives in `requirements-semantic.txt` (`pip install -r requirements-semantic.txt`) and is only needed with `SEMANTIC_RETRIEVAL_ENABLED=true`.

## Deploy to Streamlit Community Cloud (live demo)

[`app.py`](app.py) is a self-contained Streamlit entrypoint that runs the RAG service **in-process** — no separate API server — so the hosted demo is a single free app. It answers deterministically without an `OPENAI_API_KEY`, so there is nothing to leak and no per-query cost.

1. Sign in at https://share.streamlit.io with your GitHub account.
2. Click **Create app** → **Yup, I have an app**.
3. Repository `tejas-parjane/business-intelligent-rag-agent`, branch `main`, main file path `app.py`.
4. (Optional) In **Advanced settings**, set the app URL subdomain to `business-intelligent-rag-agent`.
5. Click **Deploy**.

The app is **public** by default (deployed from a public repo). If you see component boxes show a `Failed to fetch dynamically imported module` error on a cold start, that is a known Streamlit Cloud chunk-loading race — a refresh clears it.

Streamlit Community Cloud installs `requirements.txt`, which is intentionally light (no PyTorch). Free apps sleep after 12 hours of inactivity and wake on the next visit.

To wire the LLM synthesis path, add `OPENAI_API_KEY` under **App settings → Secrets** — the public demo intentionally runs without it.

## Deploy to Hugging Face Spaces (optional, paid)

Hugging Face deprecated the built-in Streamlit SDK (2025-04-30) and, as of 2026, gated Docker/CPU Spaces behind a paid plan, so a Streamlit Space now requires **PRO** (or a paid org). If you have PRO, the root [`Dockerfile`](Dockerfile) already installs the light `requirements.txt` and launches Streamlit on port 8501 — create a Docker Space, add the front matter `sdk: docker` / `app_port: 8501`, and `git push`.

## Deploy to Render

This repository includes [`render.yaml`](render.yaml), which creates two services: the FastAPI evidence API and the Streamlit UI. In Render, select **New +** → **Blueprint**, connect this repository, and approve the detected `render.yaml`. The hosted version works without `OPENAI_API_KEY`; leave that variable blank for a no-cost, deterministic portfolio demo. The UI will be available at the `bi-rag-agent-ui` service URL after deployment.

## Safety and governance

- The API never executes model-generated SQL. `QueryService` selects reviewed query templates from structured intent.
- Every response carries query, KPI, and document evidence.
- The LLM prompt is evidence-only and explicitly forbids invented drivers.
- `/evaluate` checks citation coverage and whether claims remain grounded in supplied evidence.

## Enable semantic search locally

Start PostgreSQL + pgvector, set `SEMANTIC_RETRIEVAL_ENABLED=true`, then ingest a report through the API:

```powershell
curl -X POST http://localhost:8000/ingest/text -H "Content-Type: application/json" -d '{"title":"Q3 review","source_uri":"s3://demo/q3-review.md","content":"Your business report text, at least fifty characters long..."}'
```

The first embedding request downloads the local model. Run the PyTorch training demonstration separately with:

```powershell
python scripts/train_retriever.py
```

Run the automated retrieval/evaluation checks with `pytest`.

## Evaluation (measured, deterministic — no LLM needed)

`eval/qa_test_set.json` holds 12 labeled cases (answerable + unanswerable +
a wiped-database robustness case). `eval/run_eval.py` runs each through the
same `answer_question()` service as `/ask` and writes `eval/report.json`:

```bash
python eval/run_eval.py
```

| Measure | Result (seeded demo KB) |
|---------|------------------------:|
| Intent accuracy | 12/12 |
| Abstention accuracy | 12/12 (answers when evidence exists, abstains otherwise) |
| Retrieval recall@2 | 1.00 |
| Answer-phrase coverage (answerable) | 1.00 |
| Mean latency | ~9ms (deterministic fallback; LLM off, cost $0/query) |

The harness caught two real issues that are now fixed and regression-tested:
an `IndexError` on empty segment data (now abstains) and metric definitions
cited but never verbalized (now stated in the answer).

### Failure matrix — how the system behaves when things go wrong

| Situation | Behavior (tested) |
|-----------|-------------------|
| Question outside the KB (headcount, churn, pricing, gibberish) | Abstains with an insufficient-evidence message, zero citations |
| All facts/documents deleted | Abstains, no 500 (`test_empty_database_abstains_without_crash`) |
| Future question ("revenue in Q4?") | **Known gap:** answered with past Q2/Q3 data; no temporal guard yet |
| LLM provider down or unconfigured | Deterministic evidence-backed answer; never blocks on the LLM |
| Semantic retrieval unavailable | Silent fallback to lexical retrieval |

## Limitations

- Single-metric, single-template scope: only the Revenue Q2-vs-Q3 comparison is governed; the KPI layer holds one definition.
- Rule-based intent classification (two branches) — brittle outside revenue phrasing; adversarial or paraphrased intents are untested.
- Lexical retrieval by default; semantic search needs PostgreSQL + a local embedding download, and the two paths can diverge.
- Groundedness check is citation-presence + a fixed caveat phrase, not NLI claim verification; citation *accuracy* (does the cited chunk support the claim?) is measured only at the coarse phrase-coverage level.
- No read-only DB role, audit log, or per-query cost accounting yet (see roadmap).

## Production extension roadmap

- Add S3-triggered document ingestion, chunking, and metadata enrichment.
- Connect a governed warehouse read replica with role-based policies, query cost limits, and query audit persistence.
- Persist evaluation traces and add a curated benchmark set for retrieval relevance, citation faithfulness, and answer quality.
