# Architecture

## System Context

```mermaid
flowchart LR
    User[Browser] --> Frontend[Static frontend<br/>HTML / CSS / JavaScript]
    Frontend -->|REST API| API[FastAPI + Uvicorn]
    API --> Graph[LangGraph agent workflow]
    Graph --> Knowledge[Knowledge search tool]
    Graph --> Study[Study plan tool]
    Graph --> Match[Resume and JD match tool]
    Knowledge --> Chroma[(Chroma vectors)]
    Knowledge --> BM25[(BM25 index)]
    API --> Sessions[(SQLite sessions)]
    Knowledge --> Model[Qwen OpenAI-compatible API]
    Study --> Model
    Match --> Model
```

## Document Ingestion

```mermaid
flowchart LR
    Upload[PDF upload] --> Parse[Parse with PyPDFLoader]
    Parse --> Split[Split into 500-character chunks<br/>50-character overlap]
    Split --> Staging[Build staging Chroma collection]
    Split --> Tokens[Build staged Chinese bigram BM25 index]
    Staging --> Verify{Verify chunk count}
Verify -->|valid| Switch[Persist active collection<br/>and swap in-memory state]
Verify -->|invalid| Cleanup[Delete staging collection]
Switch --> Chroma[(Active Chroma)]
Switch --> BM25[(Active BM25)]
Delete[Delete PDF request] --> Backup[Move PDF to hidden backup]
Backup --> Parse
```

The upload endpoint writes the validated PDF to a temporary path, rebuilds both
indexes in a staging collection, checks the stored chunk count, and only then
updates the active collection pointer. A failed rebuild removes the staging
collection and restores the previous PDF instead of damaging the active index.
The delete endpoint follows the same build-and-switch path after moving the PDF
to a hidden backup. A failed rebuild restores the PDF and keeps the old indexes.
The health endpoint verifies that the Chroma collection and BM25 document
counts agree.

## Hybrid Retrieval

```mermaid
flowchart LR
    Question --> Vector[Vector search top 12]
    Question --> Lexical[BM25 search top 12]
    Vector --> RRF[Weighted RRF fusion]
    Lexical --> RRF
    RRF --> Threshold[Relevance threshold]
    Threshold --> Rerank[Optional Cross-Encoder rerank]
    Rerank --> Context[Top 4 context chunks]
    Context --> Answer[Grounded answer and citations]
```

If no chunk passes the relevance threshold, the knowledge tool returns the
configured refusal response without calling the generation model.

## Agent Workflow

```mermaid
flowchart LR
    Message[User message] --> Route[Model-backed tool selection]
    Route -->|knowledge_search| Knowledge[Knowledge search]
    Route -->|create_study_plan| Study[Study plan]
    Route -->|resume_match_guidance| Match[Resume match guidance]
    Route -->|invalid or unavailable| Fallback[Keyword fallback]
    Knowledge --> End[Answer, sources, routing metadata]
    Study --> End
    Match --> End
    Fallback --> End
```

The current graph is intentionally a bounded single-tool workflow. A future
iteration can add query rewriting, evidence verification, and bounded retry
nodes while keeping the same tool interface.

## Request Observability

```mermaid
flowchart LR
    Request[HTTP request] --> RequestID[X-Request-ID middleware]
    RequestID --> Route[FastAPI route]
    Route --> Model[Central model call wrapper]
    Model --> Timeout[Timeout and retry policy]
    Model --> Logs[(JSON request and model logs)]
    RequestID --> Logs
```

Every request receives a validated or generated request ID. JSON logs capture the
method, path, status code, request duration, model operation, model duration, and
error type. A model timeout returns HTTP 504 with the same request ID so operators
can locate the corresponding log entries without exposing model credentials.

## Failure Boundaries

- Model calls are isolated inside the generation and agent-routing helpers.
- Reranker initialization and inference failures fall back to RRF ordering.
- Index rebuilds use staged collections and switch only after count validation.
- Concurrent retrieval waits for the index swap to complete.
- Unsupported, empty, oversized, or unreadable uploads return explicit errors.
- Session history is stored in SQLite with a maximum retained history length.
- `/health` reports collection count, document count, and reranker status.
