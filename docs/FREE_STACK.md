# Free stack for DocQA v1 (MongoDB revision)

> Converted from the original Word document. The .docx is the authoring source; this file is the copy agents read.

Oct 4, 2026 · @ray

DocQA v1 can be built, tested and evaluated for $0. Use Gemini Flash-class models on the Google AI Studio free tier as the main generator, Groq's free open-weight models as the fallback, Mistral's free Experiment plan as the LLM judge, and Qwen3.5 or Gemma 4 small models through Ollama as the local SLM path. Embeddings and re-ranking run on your own CPU with open models; MongoDB Atlas's free cluster holds every document, chunk and vector, with the atlas-local Docker image for development and CI. Free-tier limits were checked in late September and early October 2026; they change often, so re-check before the final README.

## Recommended zero-cost stack

| Layer | Free choice (default) | Free alternative | Main limit to plan around |
|---|---|---|---|
| Generator (hosted LLM) | Gemini Flash / Flash-Lite via Google AI Studio API | Groq: openai/gpt-oss-20b or gpt-oss-120b | Per-minute and per-day request caps; free-tier prompts may be used by Google to improve products |
| Fallback LLM | Groq gpt-oss-20b | OpenRouter :free models, GitHub Models | Groq: 8K tokens/min, 200K tokens/day |
| Local SLM | qwen3.5:4b via Ollama | gemma4:e2b, phi4-mini, qwen3.5:2b | Speed on CPU; needs ~8 GB RAM |
| LLM-as-judge | Mistral Small/Medium on the free Experiment plan (different family from the generator) | Groq gpt-oss-120b | ~1 request/s; phone verification at sign-up |
| Embeddings | nomic-embed-text-v1.5 (768-d) via FastEmbed, ONNX on CPU | bge-small-en-v1.5 (faster), Qwen3-Embedding-0.6B (better) | CPU time at ingestion; vector size on the 512 MB free cluster |
| Keyword search | Atlas Search full-text index (Lucene BM25) inside MongoDB | rank_bm25 in memory (unit tests only) | Counts toward the free cluster's 3 search indexes |
| Re-ranker | ms-marco-MiniLM-L-6-v2 cross-encoder (ONNX, CPU) | bge-reranker-v2-m3 (higher quality, much slower on CPU) | ~5 ms per pair for MiniLM |
| Database (vectors + metadata) | MongoDB Atlas free cluster for the demo | mongodb/mongodb-atlas-local Docker image | 512 MB storage, 100 ops/s, 3 search indexes, no backups, pauses after 30 days without connections |
| Local database (dev, CI) | mongodb/mongodb-atlas-local container | Atlas CLI local deployment | Single node, for development and testing only |
| PDF parsing | PyMuPDF + pdfplumber | Docling (better layout, heavier) | PyMuPDF is AGPL; fine for a public repo |
| Tracing | Langfuse self-hosted (Docker) | Langfuse Cloud Hobby: 50K units/month, 30-day retention, 2 users | Cloud retention is 30 days |
| Evaluation | RAGAS + pytest, judge via Mistral | DeepEval | Judge rate limits |
| CI | GitHub Actions (free for public repos), atlas-local as a service container | None needed | Store API keys and MONGODB_URI as repo secrets |
| Hosting | Local docker compose for all tests | Streamlit Community Cloud or Render free tier for a public demo | Hugging Face no longer hosts new Docker/Streamlit Spaces on free CPU |
| Load testing | Locust against a stub LLM, plus a low-rate run against the real provider | k6 | Provider rate caps, and the free cluster's 100 ops/s, cap real throughput |

## Hosted LLMs on free tiers

Gemini is the best primary because its free tier allows far more tokens per minute than Groq, and a RAG prompt is ~3,500 tokens. Groq is the fastest fallback but its 8K tokens/min allows only about 2 RAG queries a minute. All five providers below expose an OpenAI-compatible endpoint, so one client class covers them all (see the Design doc).

| Provider | Free models worth using | Free limits | Catches | Role in DocQA |
|---|---|---|---|---|
| Google AI Studio | Gemini 3.x Flash, Gemini 3.5 / 3.1 Flash-Lite | Set per project and shown in AI Studio; third-party trackers report Flash-class models around 10–30 requests/min and up to ~1,500/day with ~1M tokens/min (approximate, tracker) | Free-tier content is used to improve Google products: no confidential PDFs | Primary generator, query rewriter |
| Groq | openai/gpt-oss-20b, openai/gpt-oss-120b, qwen/qwen3.8-27b; Llama Prompt Guard 2 (22M / 86M) | Chat models: 30 req/min, 1,000 req/day, 8K tokens/min, 200K tokens/day | Llama 3.x models left the free tier in Aug 2026 (report) | Fallback generator; Prompt Guard as an optional injection classifier |
| Mistral (Experiment plan) | Mistral Small, Medium, Large, Ministral | ~1 request/s, ~500K tokens/min, ~1B tokens/month | Phone verification; opt out of training in privacy settings | LLM-as-judge (different model family from Gemini) |
| OpenRouter | Rotating :free models | 20 req/min; 50 req/day (1,000/day after a one-time $10 top-up) | Free model list changes; providers may log prompts | Emergency fallback only |
| GitHub Models | OpenAI mini models and other low-tier models | Low tier: 15 req/min, 150 req/day, 8K tokens in per request; high tier: 10 req/min, 50/day | 8K input cap per request; meant for prototyping | Quick experiments, second judge for cross-checks |
| Cerebras | gpt-oss-120b, qwen-3.8-27b | $5 trial credit, expires after 30 days; 5 req/min | Not permanently free | Skip |

Rule for free tiers: never put a single provider on the critical path. The generator sits behind a router that tries Gemini, then Groq, then the local SLM, and every trace records which one answered.

## Small language models (local, via Ollama)

Run qwen3.5:4b as the default local model, and compare it with the hosted LLM as one of your ablations. "SLM vs LLM on the same retrieval" is a strong trade-off slide. Small models are fine at grounded answering when retrieval is good, but they follow citation formats less reliably and are slow on CPU, so they are the offline and fallback path, not the default.

| Model (Ollama tag) | Download | Context | Strength for DocQA | Use it as |
|---|---|---|---|---|
| qwen3.5:4b | 3.4 GB | 256K | Good instruction and JSON following; tool calling; multilingual | Default local generator |
| qwen3.5:2b | 2.7 GB | 256K | Faster on weak laptops | Query rewriter; low-RAM machines |
| qwen3.5:0.8b | 1.0 GB | 256K | Very fast | Query classifier or rewriter only, not answers |
| gemma4:e2b | 4.6–7.5 GB | 128K | Apache 2.0 licence (Gemma 4); strong for its size | Second local generator for the ablation |
| phi4-mini (3.8B) | 2.5 GB | 128K | Reasoning-dense, smallest download | Comparison point |
| qwen3.5:9b | 6.6 GB | 256K | Closest to hosted quality | Only on a machine with a GPU or 16 GB+ RAM |

What to expect:

- RAM: plan for ~8 GB free RAM for 2–4B models at 4-bit quantisation.
- Speed: a 2–4B model on a laptop CPU generates roughly 5–20 tokens/s (approximate; measure it on your machines). A 300-token answer then takes 15–60 s, so SLM latency gets its own target (see the last section). An 8 GB GPU or Apple Silicon is several times faster.
- Thinking mode: Qwen3.5 can emit reasoning tokens. Turn thinking off for answering (it adds latency and breaks JSON parsing) and set temperature: 0.
- Prompt size: keep the system prompt short for SLMs and pass at most 4–5 chunks; small models lose information from long prompts faster.

## Embeddings and re-rankers

Use nomic-embed-text-v1.5 for embeddings and ms-marco-MiniLM-L-6-v2 for re-ranking, both through FastEmbed (ONNX, no PyTorch, CPU-friendly). Nomic beats bge-small here because its 8,192-token window fits our 512-token chunks without truncation; bge-small caps input at 512 BERT tokens, which a 512-tiktoken chunk can exceed. FastEmbed's BM25 sparse model is no longer needed, because Atlas Search scores keywords inside MongoDB.

Vector size now matters for storage: a 768-d float32 vector is 3 KB as BSON binary but ~8.4 KB as an array of doubles, so always pack vectors with Binary.from_vector(..., BinaryVectorDtype.FLOAT32). Nomic is Matryoshka-trained, so truncating to 512 or 256 dims is the fallback if the free cluster fills.

| Embedding model | Params | Dims | Max input | Quality (MTEB, as reported) | Fit |
|---|---|---|---|---|---|
| nomic-embed-text-v1.5 | 137M | 768 (Matryoshka down to 256) | 8,192 | 62.3 English | Default |
| BAAI/bge-small-en-v1.5 | 33M | 384 | 512 | Lower, fastest | Fast option if chunks stay ≤ 400 tokens; halves vector storage |
| Qwen3-Embedding-0.6B | 0.6B | 1,024 | 32K | 64.3 multilingual | Quality ablation; ~3–4× slower ingestion; 4 KB vectors |
| embeddinggemma | 300M | 768 | 2K | 61.2 multilingual | Alternative |
| bge-m3 | 567M | 1,024 + sparse | 8,192 | Dense + sparse in one model | Too heavy for a CPU-only demo |

Pin the model name and version in config; changing it means re-indexing everything.

| Re-ranker | Params | Quality (BEIR nDCG@10, as reported) | CPU cost | Fit |
|---|---|---|---|---|
| ms-marco-MiniLM-L-6-v2 | 22M | ~60% (L-12 variant) | ~5 ms/pair: 20 pairs ≈ 0.1–0.2 s | Default |
| bge-reranker-v2-m3 | ~0.6B | ~71.5% | Seconds for 20 pairs on CPU | Quality ablation; GPU or offline eval only |
| Qwen3-Reranker-0.6B | 0.6B | Strong family (4B / 8B variants lead the board) | Similar to bge-m3 | Optional ablation |

Atlas's native $rerank stage is not an option on the free cluster: it needs MongoDB 8.3+, and free clusters run 8.0.

## Infrastructure on free tiers

Run everything locally with docker compose (API, UI, atlas-local, optionally Langfuse and Ollama). That costs nothing, has no limits except the LLM's, and is how all evaluation numbers and ablations should be produced. The Atlas free cluster is for the public demo only; the hosting rows below give the honest options.

### MongoDB Atlas free cluster limits that shape DocQA

| Limit | Value | What DocQA does about it |
|---|---|---|
| Storage (documents + indexes) | 512 MB | Packed float32 vectors; 500-document cap; alert at 400 MB |
| Throughput | 100 read/write operations per second, then throttled | Bulk writes of 64; one aggregate per query; ingest one document at a time |
| Search indexes | 3 on the free cluster (vector and full-text count together) | Uses 2; an embedding-model change is a planned re-index, not blue-green |
| MongoDB version | 8.0, upgraded by Atlas, not configurable | $rankFusion works (8.0+); $scoreFusion and $rerank (8.3+) do not |
| Connections | 500 | Pool size 20 per API process |
| Collections | 100 databases, 500 collections | One docqa database, ~6 collections |
| Data transfer | 10 GB in and 10 GB out per rolling 7 days | ~50 KB per query; project vectors out of results |
| Backups | None | Nightly mongodump in GitHub Actions |
| Idle pause | After 30 days with zero connections | Weekly scheduled ping |
| Monitoring | Connections, logical size, network, opcounters only | Admin page reads dbStats and $listSearchIndexes |
| Region | A subset of AWS, Google Cloud and Azure regions | Pick the one nearest the app host |

### Other free infrastructure

| Need | Free option | Limits | Notes |
|---|---|---|---|
| Database (local) | mongodb/mongodb-atlas-local Docker image | Your machine; single node | Bundles mongod + mongot, so $vectorSearch, $search and $rankFusion behave as on Atlas; for development and testing only |
| Database (hosted) | MongoDB Atlas free cluster | See the table above | Same driver and index definitions as local, so code does not change |
| Tracing | Langfuse self-hosted via Docker | Your machine (needs ClickHouse, Redis, MinIO, Postgres) | Full features, no retention limit, but heavy on a laptop |
| Tracing (hosted) | Langfuse Cloud Hobby | 50K units/month, 30-day data access, 2 users | ~6 observations per query → roughly 8K traced queries/month |
| PDF parsing | PyMuPDF (AGPL-3.0), pdfplumber (MIT) | None | Docling (MIT) handles complex layouts and tables better but pulls in heavy models |
| CI | GitHub Actions | Free on public repos | atlas-local runs as a service container; run the golden-set eval only when prompts, config or retrieval code change, to save free LLM quota |
| Public demo hosting | Render free web service | 512 MB RAM, 0.1 CPU, sleeps after 15 min idle, 5 GB bandwidth/month | Too small to run local embedding and re-rank models; only works if those also run elsewhere |
| Public demo hosting | Streamlit Community Cloud | Hibernates after long idle; limited RAM | Easiest public link for a Streamlit app; run the pipeline in-process with FastEmbed models + hosted LLM + Atlas free cluster |
| Public demo hosting | Hugging Face Spaces | New Docker, Gradio and Streamlit Spaces now need a paid plan; free CPU Spaces created before June 2026 still run | No longer a free option for us |
| Hosting with a card on file | Oracle Cloud Always Free, Google Cloud Run | Generous always-free compute | Only if a team member is willing to add a card; set billing alerts |

## What the free stack changes in the PRD

Four PRD targets need restating: cost becomes $0 with a quota budget instead of a dollar budget, real throughput is capped by provider rate limits and the free cluster's 100 ops/s rather than our code, storage becomes a tracked budget, and the local SLM path gets its own latency target. Everything else in the PRD stands.

| PRD item | Paid-API version | Free-stack version |
|---|---|---|
| Generator | gpt-4o-mini | Gemini Flash-class (free tier) → Groq gpt-oss-20b → Ollama qwen3.5:4b |
| Judge | Gemini Flash | Mistral Small/Medium (Experiment plan); still a different family from the generator |
| Embeddings | text-embedding-3-small, 512-d | nomic-embed-text-v1.5, 768-d, local CPU; no per-query embedding cost or network call |
| Re-ranker | MiniLM-L-6 on CPU | Unchanged |
| Database | MongoDB Atlas free cluster | Unchanged; atlas-local for dev, CI and ablations |
| Storage budget | ~0.5 MB per document at 512-d | ~0.6 MB per document at 768-d; 500 documents ≈ 300 MB, under the 400 MB alert. Truncate nomic to 512-d if it grows |
| Cost target | ≤ $0.002 per query; ≤ $20 total | $0. Track tokens per query anyway and report the equivalent paid cost, so the cost slide still has numbers |
| Quota budget | Not needed | One golden-set eval run ≈ 40 generator calls + ~160 judge calls; fits a single day's free quota. Cache generator outputs by (question, prompt version, model) so re-running metrics costs nothing |
| Latency, hosted path | p95 ≤ 4 s | Unchanged; measured on Gemini. Add a 429 retry budget of at most 1 retry |
| Latency, SLM path | Not in scope | Report separately: time to first token and total time on the team's actual hardware; target p95 ≤ 30 s on CPU, ≤ 8 s with a GPU |
| Throughput | ≥ 2 queries/s on one instance | Two numbers: (1) pipeline throughput with a stub LLM that sleeps for the measured Gemini latency (shows our system's capacity), (2) real end-to-end throughput at the provider's cap. Report peak database ops/s against the 100 ops/s cap. State the caps as constraints, not failures |
| Privacy | Warn users not to upload confidential files | Stronger warning: Gemini free-tier prompts may be used to improve Google's products. Offer "local mode" (Ollama + atlas-local) for private documents, so nothing leaves the machine |

New trade-offs worth presenting: "We chose a hosted free LLM over a local SLM as default because p95 latency is ~5–10× lower on CPU hardware; the cost is that document text leaves the machine, so local mode exists for private files." "We chose to put three providers behind a router over a single provider because free tiers fail with HTTP 429 under load." And: "We chose MongoDB for vectors and metadata over a dedicated vector DB plus SQLite because one store halves the moving parts; the cost is the free cluster's 512 MB and 100 ops/s, which we size for and monitor."

## Sources

- MongoDB: Atlas free cluster limits, Atlas pricing, hybrid search with $rankFusion, hybrid search GA announcement, search index limits on M0, local development with atlas-local, PyMongo Async migration
- Carried over from the original free-stack review: Groq rate limits and free tier changes (Sep 2026); Gemini API pricing and rate limits; Mistral free Experiment plan; OpenRouter limits; GitHub Models free limits; Cerebras rate limits; Ollama model pages for Qwen3.5, Gemma 4, Phi-4-mini; embedding model comparison (Jun 2026); open-weight rerankers (May 2026); FastEmbed; Langfuse pricing; Hugging Face Spaces free CPU change; Render free tier.