# DocQA audit (prompt 10, part C)

Date: 2026-10-04. Branch `feat/retrieval-generation`. Read-only: nothing here was fixed
as part of the audit.

**How it was verified.** Locally (Windows, no Docker): `ruff`, `mypy` and the unit
suite. The integration, contract, e2e and load suites ran in GitHub Actions on
`mongodb/mongodb-atlas-local`. Numbers quoted from CI come from the run annotations
(`.github/workflows/ci.yml` publishes them). Nothing ran against the Atlas free
cluster or a hosted LLM: this machine has no Atlas URI and no API keys.

Status: **met** = implemented and verified by a test or run cited here (last CI run: 37194219037, all suites green except the re-rank latency test). **partly** =
implemented but not fully verified, or verified with a miss. **missing** = not built.

## Functional requirements

| ID | Status | Evidence | Gap |
|---|---|---|---|
| FR-1 upload limits | partly | `ingestion/validator.py`; `test_validator.py`; 409/413/415/422 in `test_documents_api.py`, `test_ui_client_and_highlights.py` | the 10-per-session limit is enforced in the UI only (your decision); the API accepts more |
| FR-2 scanned PDFs | met | `NoTextLayerError`; `test_rejects_bad_pdfs[scanned.pdf]`; e2e `test_scanned_pdf_is_rejected` | — |
| FR-3 page text, headings, tables | met | `ingestion/extractor.py`; `test_extractor_sanitizer.py` | pdfplumber fallback triggers only on ragged tables |
| FR-4 async ingest + statuses, ready after sync | met | `services/ingestion_service.py`, `mongo_store.wait_until_searchable`; `test_ingestion_flow.py`; `test_documents_api.py` | — |
| FR-5 hybrid search + re-rank | met | `adapters/mongo/pipelines.py`, `mongo_search.py`; contract suite in both fusion modes; `test_retrieval.py` | — |
| FR-6 structured, verified citations | met | `generation/citation_validator.py`; `test_generation.py`; e2e citations on the right page | — |
| FR-7 abstention | partly | `retrieval/abstention.py`; `test_query_service.py`; e2e abstain tests | θ not calibrated (needs the golden set); `abstain_threshold` is still 0.0 |
| FR-8 streaming | met | `api/sse.py`, `routers/query.py`; e2e `test_query_streams_meta_tokens_citations_done` | — |
| FR-9 click citation → page + highlight | partly | `/documents/{id}/highlights` and `/file`; `ui/streamlit_app.py`; e2e `test_full_ui_flow` | not checked by hand in a browser (no Docker here) |
| FR-10 follow-up rewriting | met | `retrieval/query_rewriter.py`; e2e `test_follow_up_*` | a follow-up can be answered from the cache before rewriting (open item) |
| FR-11 feedback | partly | `POST /feedback` → `feedback` collection + Langfuse score; e2e `test_feedback_is_stored` | DESIGN's 404 for an unknown trace is not implemented (needs a repository method) |
| FR-12 delete within 1 min | met | `document_service.delete` purges chunks and cache; e2e `test_delete_removes_chunks_cache_and_access_within_60s` | re-uploading the same file after delete returns 409 (open item) |
| FR-13 REST endpoints | met | `api/routers/*`; e2e suites | — |

## Non-functional requirements

| NFR | Status | Evidence | Gap |
|---|---|---|---|
| Latency p50/p95/p99 | missing | `eval/report.py` computes them | needs a real-LLM eval run |
| Time to first token | partly | Locust reports it; stub runs p50 3.8 / 2.1 s, p95 6.0 / 3.5 s at 8 users | stub numbers on shared CI CPU, not the real provider |
| Ingestion ≤ 60 s for 100 pages | met (barely) | `test_100_page_pdf_is_ready_within_target`: 38–57 s across CI runs | synthetic PDF, not a real report |
| Throughput ≥ 2 q/s at 8 users | **partly (miss)** | CI stub load, two runs: **1.19 and 1.78 req/s**, 0 % errors, done p50 5.8 / 4.2 s | CPU-bound embedding, re-ranking and Presidio on a 4-vCPU runner |
| DB ≤ 60 ops/s at peak | **partly (miss)** | CI stub load: **peak 64 and 61 ops/s** (cap 100) | per query: doc-ready lookups, two turn inserts, history read |
| Storage ≤ 400 MB, 500 docs | met (guard) | `guardrails/storage_guard.py`; e2e `test_..._storage_cap_returns_507`; ~3.4–4.1 KB per chunk | current Atlas size not measured |
| Cost ≤ $0.002 / query | missing | equivalent cost computed per query in eval reports | needs an eval run |
| Context cap 6,000 tokens | met | `PromptBuilder`; `test_builder_drops_lowest_ranked_chunks_to_fit_the_token_cap` | — |
| Faithfulness / Recall@5 / citation accuracy | missing | metrics in `eval/metrics/`; unit tests | needs the golden set |
| Privacy: owner pre-filter in both branches | met | pipeline builders raise without an `AccessFilter`; contract `test_another_owners_chunks_never_return` (B's chunks more similar, k = 1/5/20) | — |
| Privacy: PII scrubbed from logs/traces/turns | met | `guardrails/pii.py`, `observability/logging.py`; `test_a_logged_email_never_reaches_the_log_output`; `test_turns_are_stored_scrubbed` | a live Langfuse trace not inspected (no keys) |
| Security: chunks wrapped, injection flagged | met | `prompt_builder.document_block` (untrusted flag); `sanitizer.py`; tests | — |
| Least-privilege Atlas user, IP list | missing | documented in `docs/runbooks/deploy.md` | must be set in Atlas by you |
| Reproducibility: config, prompts, indexes versioned | met | `config/`, `prompts/`, `config/indexes/`; `config_hash()` on every trace and report | — |
| One image; CI tests + eval gate | partly | `Dockerfile`, `ci.yml`, `eval.yml`, `release.yml` | image never built here; eval gate needs golden set + baseline |

## Guardrails, monitoring, risks

| Item | Status | Evidence | Gap |
|---|---|---|---|
| Question length, rate limit 10/min → 429 | met | `input_guard.py`, `rate_limiter.py`; unit + e2e `test_bad_requests_*` | — |
| Injection heuristics on questions (flag) | met | `InputGuard.check` → `injection_flag` on the trace | no alerting on spikes |
| Re-ranker below θ → abstain, no LLM call | met | `test_low_relevance_abstains_without_calling_the_llm` | θ uncalibrated |
| All LLMs down → passages | met | e2e `test_all_llms_down_degrades_*` | — |
| MongoDB down → /health 503, fail fast | met | e2e `test_mongo_down_health_503_and_query_fails_fast` (passes locally too) | — |
| Search index missing → 503 | met | e2e `test_missing_search_index_makes_health_503` | — |
| Alert checks (latency, errors, abstention, storage, ops/s, indexes) | partly | `scripts/check_alerts.py` | never run against a live system; drift, cost and judge alerts not implemented |
| Online judge 20 % | partly | `scripts/online_judge.py` | never run (needs Langfuse + Mistral keys); not scheduled |
| Backups + keep-alive | partly | `.github/workflows/backup.yml`, `docs/runbooks/restore.md` | needs the `MONGODB_URI_ATLAS` secret; no manual run yet |
| `$rankFusion` on the free cluster | partly | accepted on atlas-local (server-mode tests pass); `scripts/smoke_rankfusion.py` | not run on the Atlas free cluster |

## Deliverables

| Deliverable | Status | Gap |
|---|---|---|
| Live deployment | missing | `docs/runbooks/deploy.md` + in-process UI mode ready; needs your Streamlit Cloud account and secrets |
| README with numbers | partly | renderer works; every number currently reads "not measured yet" |
| Hand-written eval set | missing | tooling done; `eval/golden.jsonl` is yours to write (rule 11) |
| Observability | partly | tracer, logs, alerts built; no live trace inspected |
| Ablation table | missing | 10 profiles + `eval/run_ablations.py`; needs the golden set |

## Code rules

| Rule | Status | Evidence |
|---|---|---|
| No vendor imports in domain/ports/services | met | `test_import_boundaries.py` (passes) |
| Every chunks aggregation via the pipeline builders | met | the only `.aggregate(` calls are in `mongo_search.py`, `mongo_store.py` (count pipelines) and `smoke_rankfusion.py`, all built by `pipelines.py` from an `AccessFilter` |
| Vectors stored as binary float32 | met | integration `test_vectors_are_binary_float32_of_the_configured_size` |
| Files under ~200 lines | partly | over: `bootstrap.py` 300, `config_schema.py` 280, `domain/models.py` 265, `query_service.py` 260, `settings.py` 223, `ui/streamlit_app.py` 213 |
| Re-rank 20 pairs < 400 ms | **miss** | CI: 741–969 ms; the strict test fails CI by design until you choose a remedy |

## Prioritised list

**Fix before presenting**
1. Write `eval/golden.jsonl` (40 records) and the corpus; run `make eval`, calibrate θ, commit `baseline.json`. Every quality number depends on this.
2. Decide the re-rank remedy (20 → 10 candidates per the PRD, the quantised model, or measure on the demo machine), then rerun the latency test.
3. Run the Atlas checks with your URI: `make initdb`, `smoke_rankfusion.py`, and one manual `backup.yml` run.
4. Deploy with the runbook; run `scripts/warmup.py --url ...` after a cold start.
5. Do the prompt-8 browser walkthrough: upload, ready, ask, click citation, thumbs-down.

**Name as a limitation**
- Throughput 1.19–1.78 req/s and peak 61–64 ops/s on CI's 4-vCPU runner (stub LLM). The CPU-bound models dominate; a bigger instance or fewer candidates would help.
- 10-PDF limit enforced only in the UI. Re-upload after delete returns 409. Follow-ups can hit the cache. No 404 for an unknown feedback trace.
- Six files over the ~200-line guideline.
