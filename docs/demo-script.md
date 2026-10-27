# Demo Script

Target length: 2-3 minutes.

## 1. Introduction (15 seconds)

Show the project title and state the goal:

> This is a user-scoped RAG and LangGraph project for enterprise knowledge
> search and job preparation. It supports bearer-token authentication,
> per-user document and session isolation, grounded Q&A, source citations,
> resume matching, and study planning.

## 2. Login and Workspace Isolation (15 seconds)

Show the login page, enter the demo workspace, and point out that each account
has a separate PDF directory, Chroma collection, BM25 index, and conversation
history.

## 3. Knowledge Base Q&A (30 seconds)

Ask:

> 这个项目采用了哪些检索和重排技术？

Show:

- The answer
- File names and page numbers in the sources
- Retrieval method and hybrid score if available in the source details

## 4. Reliable Refusal (20 seconds)

Ask a question that is not covered by the knowledge base:

> 公司今年的营业收入是多少？

Show that the system refuses to answer and returns no unrelated sources.

## 5. Resume and JD Matching (30 seconds)

Upload a resume PDF and a job description PDF, then run the match.

Show:

- Match score
- Matched skills
- Skill gaps
- Resume suggestions
- Predicted interview questions

## 6. Study Plan (15 seconds)

Generate a study plan from the identified skill gaps.

Show the four-week structure and one concrete deliverable per week.

## 7. Document Lifecycle (15 seconds)

Open the document management page and show:

- Per-file chunk counts, size, and modification time
- The delete confirmation flow
- That successful deletion rebuilds both indexes atomically

Do not delete the only remaining document: the system intentionally rejects an
empty index and restores the deleted file.

## 8. Engineering Evidence (20 seconds)

Show:

- `/workspace/health` returning `status: ok`
- Both Docker containers reporting `healthy`
- A failed fake-PDF upload and an HTTP 429 rate-limit response
- The GitHub Actions workflow
- A generated Markdown report in `eval/reports/` showing Hit@4, refusal accuracy,
  P95 latency, and failure details

## 9. Architecture (15 seconds)

Show `docs/architecture.md` and briefly explain:

> Authentication resolves a Bearer Token to one user. Uploads, vector and BM25
> indexes, and sessions are scoped to that user. Retrieval fuses Chroma and
> BM25 with weighted RRF, optionally reranks the candidates, and only calls the
> generation model when the evidence passes the relevance threshold.

## Recording Checklist

- Use a clean browser window at 1280x720 or larger.
- Do not expose API keys, personal documents, or terminal environment values.
- Use a dedicated demo account and show that another account cannot see its documents.
- Upload only synthetic documents.
- Keep the video under three minutes.
- Add the final video URL to the README.
