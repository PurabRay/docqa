"""Collection names in the docqa database (docs/DESIGN.md, "Collections")."""

DOCUMENTS = "documents"
SESSIONS = "sessions"
TURNS = "turns"
FEEDBACK = "feedback"
ANSWER_CACHE = "answer_cache"  # only with cache.backend: mongo

# In config/indexes/btree.yaml, "chunks" stands for the active chunks_<model> collection.
CHUNKS_ALIAS = "chunks"
