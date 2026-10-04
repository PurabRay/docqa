# DocQA: GitHub Reference Repositories (MongoDB stack)

> Converted from the original Word document. The .docx is the authoring source; this file is the copy agents read.

Oct 4, 2026 · @ray

These 20 repos cover every box in the revised, MongoDB-based DocQA architecture, in pipeline order. Six are new for MongoDB: three small open-source RAG apps already built on Atlas Vector Search, MongoDB's official cookbook, the LangChain integration, and the PyMongo driver. Qdrant, R2R (Postgres-based), rerankers, llm-guard, SlowAPI and DeepEval made way for them. Use means install it as a dependency; Reference means read its code and copy the pattern.

All three Atlas RAG apps fuse results on the application side rather than with $rankFusion; one says $rankFusion is unavailable on free clusters, while MongoDB's own docs say it runs on 8.0+, which free clusters use. Smoke-test $rankFusion on your free cluster in M1, and keep their app-side RRF as the fallback (retrieval.fusion: app).

## The 20 repos

| # | Repo | Verdict | DocQA part it solves | Where it differs from DocQA | Modification for DocQA |
|---|---|---|---|---|---|
| 1 | adrianofratelli-glitch/atlas-rag-multitenant | Reference | Closest whole-system match: FastAPI + SSE, $vectorSearch + Atlas Search with a mandatory tenant filter on both paths, RRF, re-ranking, citations, a golden-set eval with Ragas and refusal accuracy, and a test that injects higher-scoring chunks from another tenant to prove none leak | Voyage embeddings and reranker, Claude as LLM, React UI; RRF done in the app; a new repo (MIT, 0 stars, 3 commits) | Copy tests/test_tenant_isolation.py almost verbatim into your contract suite; swap Voyage for FastEmbed and Claude for your router |
| 2 | sethigoldy/rag-document-assistant | Reference | FastAPI on atlas-local, bge-small on CPU, hybrid $vectorSearch + $search with app-side RRF, citation validation that withholds answers with no valid citation, score-threshold refusal before calling the LLM, per-response tokens/cost/latency, 60 unit + 6 Mongo integration tests | Cites documents, not pages; fixed 0.87 vector-score threshold; demo data; new repo (MIT, 0 stars) | Borrow its integration-test setup on atlas-local and its refusal/false-refusal metrics; calibrate θ on the re-ranker score instead of a fixed vector score |
| 3 | coleam00/MongoDB-RAG-Agent | Reference | PyMongo async, Docling HybridChunker, separate documents / chunks collections, vector + Atlas Search indexes, manual RRF 1/(60+rank) running on the free M0 tier at ~350–600 ms per query | Pydantic AI agent, CLI, no page citations or eval gate; its "no $rankFusion on free tier" claim predates MongoDB's GA notice | Use its collection layout and concurrent two-query RRF as the fusion: app fallback; its latency figure is a useful M0 baseline (115 stars) |
| 4 | mongodb-developer/GenAI-Showcase | Reference | MongoDB's official cookbook: RAG, hybrid search, vector index creation, quantization notebooks | Notebooks, not a service | Lift index-definition and $rankFusion snippets; check them against the free-cluster limits |
| 5 | langchain-ai/langchain-mongodb | Reference | MongoDBAtlasHybridSearchRetriever (vector + full-text fused by RRF with per-branch penalties and a pre_filter), index-creation helpers | LangChain abstractions would leak into your domain layer | Read its pipeline-building code next to your pure hybrid_pipeline(); don't take the dependency |
| 6 | mongodb/mongo-python-driver (PyMongo) | Use | AsyncMongoClient (Motor's replacement), bulk_write with ReplaceOne(upsert=True), create_search_index, Binary.from_vector(..., FLOAT32) | Motor reached end of life in May 2026 | Use only the async API; one client in the FastAPI lifespan; pack every vector as float32 binary |
| 7 | Cinnamon/kotaemon | Reference | Document QA UI, hybrid retriever + re-ranker, citations with relevance score highlighted in a PDF viewer | Gradio; citations displayed, not verified | Borrow the citation-highlight UX for FR-9; put DocQA's validator in front of it |
| 8 | zylon-ai/private-gpt | Reference | Dependency injection; swappable LLM / embedding / vector-store components per settings profile | v1.0 (2026) grew into a broad backend | Mirror its profiles for free.yaml / local.yaml and its DI for bootstrap.py |
| 9 | pymupdf/pymupdf4llm | Use | FR-1 to FR-3: per-page Markdown with tables and page numbers (page_chunks=True), reading order, TOC headings | Newer versions auto-OCR scanned pages, which v1 rejects | Run the text-density check (< 50 chars/page) first, or disable OCR |
| 10 | jsvine/pdfplumber | Use | Table extraction fallback for the 6 table questions | Slower, weaker on multi-column text | Call only on pages with a detected table; one Markdown chunk per table |
| 11 | chonkie-inc/chonkie | Use | Recursive chunker with tiktoken counts and overlap; semantic chunker for the ablation | Works on plain text, so page boundaries are lost | Chunk page by page (or carry offsets) so every chunk keeps page_start / page_end |
| 12 | qdrant/fastembed | Use | ONNX on CPU: nomic-embed-text-v1.5 embeddings and the TextCrossEncoder MiniLM-L-6 re-ranker; works with any database | Its BM25 sparse model is no longer needed; MiniLM truncates each pair at 512 tokens | Use only dense + re-ranker; prefix search_document: / search_query: for nomic; calibrate θ on logits |
| 13 | 567-labs/instructor | Use or reference | Structured JSON over OpenAI-compatible clients; exact-citations example drops quotes not found in the context | Validates against one context string and re-asks | Check each quote against its own chunk_id's text; drop invalid citations instead of re-asking |
| 14 | BerriAI/litellm | Reference | Router with fallbacks, retries and cooldown across Gemini, Groq, Mistral, Ollama | Open issues on streaming fallback after 429 and rate-limit vs quota 429 | Keep your own router; fall back only before the first token; open the breaker until reset on a daily-quota 429 |
| 15 | ollama/ollama | Use | Local SLM path via an OpenAI-compatible /v1 endpoint; with atlas-local it gives a fully offline "local mode" | Thinking mode on by default for some models | think: false, temperature: 0 |
| 16 | microsoft/presidio | Use | PII scrub of questions, logs, traces and the turns collection | spaCy model download | en_core_web_sm + regex recognisers; run before every turns insert |
| 17 | vibrantlabsai/ragas | Use | Faithfulness, context precision / recall on the golden set | Async metrics API; judge JSON can truncate into NaN | Point llm_factory at Mistral via AsyncOpenAI; count NaNs as gate failures |
| 18 | langfuse/langfuse | Use | Traces, spans, tokens, cost, feedback and judge scores, prompt versions | Self-host is six services; SDK now v4 | Cloud Hobby for the demo, NoopTracer in CI; add a db_ms attribute on retrieval spans |
| 19 | lfoppiano/streamlit-pdf-viewer | Use | FR-9: PDF in Streamlit with clickable highlight rectangles | Takes coordinates, not text | Turn each verified quote into boxes with PyMuPDF page.search_for(quote) at click time |
| 20 | locustio/locust | Use | Load test: 8 users, p50 / p99, throughput, error rate | SSE streams need custom timing | Time first token and done separately; read MongoDB opcounters during the run to report peak ops/s |

## Build order by milestone

- M1: private-gpt (layout, settings profiles); PyMongo async + GenAI-Showcase for init_db.py and index definitions; smoke-test $rankFusion on the free cluster.
- M2: pymupdf4llm, pdfplumber, Chonkie, FastEmbed; MongoDB-RAG-Agent's collection layout and ingestion flow.
- M3: langchain-mongodb's hybrid retriever as a reference for hybrid_pipeline(); atlas-rag-multitenant's tenant-isolation test; FastEmbed re-ranker; instructor citations; kotaemon's citation view.
- M4: Langfuse, RAGAS, Presidio; rag-document-assistant's atlas-local integration tests and refusal metrics.
- M5: LiteLLM ideas in the router, Ollama local mode, Locust with MongoDB opcounters.
- M6: streamlit-pdf-viewer polish; the "differs" column above is your trade-off slide material.

## Sources

Each repo name in the table links to its GitHub page, and the three Atlas RAG apps were opened for this review. Also checked: MongoDB hybrid search docs, hybrid search GA announcement, Atlas free cluster limits, LangChain hybrid search with MongoDB, Motor deprecation notice, Qdrant: when a reranker is worth it.
