# Changing the embedding model

Chunks store `embed_model` and live in `chunks_<model>`; the API refuses a collection
whose model differs from its config (EmbeddingVersionMismatchError).

## Locally (atlas-local): blue-green

1. Set the new `embedding.dense_model` / `embedding.dim` in a profile.
2. `make initdb` with that profile: a new `chunks_<new model>` collection and its two
   search indexes are created next to the old ones.
3. Re-ingest the corpus with the new profile; run `make eval` and compare with the old report.
4. Switch the default profile; drop the old collection when satisfied.

## On the Atlas free cluster: a planned window

The free cluster allows 3 search indexes and DocQA uses 2, so both versions cannot be
indexed at once.

1. Announce a window; take a fresh backup (`backup.yml` → run workflow).
2. Drop the old chunks collection's search indexes.
3. Deploy the new config, run `make initdb`, then re-ingest every document.
4. Verify on the golden dev split; if it is worse, restore the backup and redeploy the
   previous image tag (see deploy.md).
