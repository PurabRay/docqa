# Deploy and roll back

## Streamlit Community Cloud (public demo)

The UI runs the whole pipeline in-process (FastEmbed models + hosted LLM router + the
Atlas free cluster): set `DOCQA_API_URL=inprocess` and the UI builds the API in the same
process instead of calling it over HTTP.

1. New app → this repo → main file `ui/streamlit_app.py`.
2. Secrets (platform store, never in the repo): `MONGODB_URI` (Atlas, least-privilege
   readWrite on `docqa`), `GEMINI_API_KEY`, `GROQ_API_KEY`, `LANGFUSE_*`, `DOCQA_API_URL=inprocess`.
3. Atlas → Network Access: allow the platform's egress (or 0.0.0.0/0 for the demo only).
4. Before the demo: `uv run python -m scripts.warmup --url <app url>`.

## Render (alternative)

Two web services from the same image (`ghcr.io/<owner>/docqa:<sha>`): the API with the
default command, the UI with `streamlit run ui/streamlit_app.py --server.port=$PORT`
and `DOCQA_API_URL` pointing at the API. Render's free 512 MB instance is too small for
the local models, so this needs a paid instance.

## Release and rollback

`.github/workflows/release.yml` pushes `ghcr.io/<owner>/docqa:<commit sha>` on every merge
to main. Rollback in one line: redeploy the previous tag, e.g.

```bash
docker run -p 8000:8000 --env-file .env ghcr.io/<owner>/docqa:<previous sha>
```

A prompt or model rollback needs no rebuild: flip `prompts.answer` or `llm.router` back in config.
