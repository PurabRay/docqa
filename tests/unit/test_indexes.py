import pytest

from docqa.adapters.mongo.indexes import definition_matches, documents_validator
from docqa.domain.errors import ConfigurationError
from docqa.settings import load_settings, render_template


def vector_field(settings):
    fields = settings.search_indexes["chunks_vector"].definition["fields"]
    return next(f for f in fields if f["type"] == "vector")


def test_vector_index_takes_dimensions_from_embedding_config():
    assert vector_field(load_settings(env_file=None))["numDimensions"] == 768
    small = load_settings(overrides={"embedding": {"dim": 512}}, env_file=None)
    assert vector_field(small)["numDimensions"] == 512


def test_both_indexes_filter_on_owner_and_document():
    settings = load_settings(env_file=None)
    vector_filters = {
        f["path"]
        for f in settings.search_indexes["chunks_vector"].definition["fields"]
        if f["type"] == "filter"
    }
    text_fields = settings.search_indexes["chunks_text"].definition["mappings"]["fields"]
    assert vector_filters == {"owner_id", "doc_id"}
    assert text_fields["owner_id"] == text_fields["doc_id"] == {"type": "token"}


def test_unknown_placeholder_is_rejected():
    with pytest.raises(ConfigurationError, match="nope"):
        render_template({"a": "{nope}"}, {})


def test_definition_matches_ignores_server_defaults():
    desired = {"mappings": {"dynamic": False, "fields": {"text": {"type": "string"}}}}
    actual = {"mappings": {"dynamic": False, "fields": {"text": {"type": "string", "store": True}}}}
    assert definition_matches(desired, actual)


def test_definition_matches_detects_changes():
    assert not definition_matches(
        {"fields": [{"numDimensions": 768}]}, {"fields": [{"numDimensions": 512}]}
    )
    assert not definition_matches({"fields": [{"a": 1}]}, {"fields": [{"a": 1}, {"b": 2}]})
    assert not definition_matches({"a": 1}, None)


def test_documents_validator_lists_every_status():
    enum = documents_validator()["$jsonSchema"]["properties"]["status"]["enum"]
    assert "syncing" in enum and "deleted" in enum and len(enum) == 7
