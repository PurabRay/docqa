"""Collection names in the docqa database (docs/DESIGN.md, "Collections")."""

DOCUMENTS = "documents"
SESSIONS = "sessions"
TURNS = "turns"
FEEDBACK = "feedback"

# In config/indexes/btree.yaml, "chunks" stands for the active chunks_<model> collection.
CHUNKS_ALIAS = "chunks"
