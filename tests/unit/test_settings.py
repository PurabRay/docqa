import shutil
from pathlib import Path

import pytest

from docqa.domain.errors import ConfigurationError
from docqa.settings import DEFAULT_CONFIG_DIR, deep_merge, load_settings


@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    """A private copy of config/ that a test may edit."""
    return Path(shutil.copytree(DEFAULT_CONFIG_DIR, tmp_path / "config"))


def test_free_profile_routes_through_three_providers():
    settings = load_settings("free", env_file=None)
    assert settings.llm.router == ["gemini_flash", "groq_oss_20b", "ollama_qwen"]
    assert settings.tracing.backend == "langfuse"


def test_local_profile_overlays_base():
    settings = load_settings("local", env_file=None)
    assert settings.llm.router == ["ollama_qwen"]
    assert settings.tracing.backend == "noop"
    assert settings.retrieval.top_k == 5  # untouched keys come from base.yaml


def test_profile_comes_from_env(monkeypatch):
    monkeypatch.setenv("DOCQA_PROFILE", "local")
    assert load_settings(env_file=None).profile == "local"


def test_ablation_file_is_a_profile(config_dir):
    (config_dir / "ablations").mkdir(exist_ok=True)
    (config_dir / "ablations" / "top3.yaml").write_text("retrieval: {top_k: 3}\n")
    settings = load_settings("top3", config_dir=config_dir, env_file=None)
    assert settings.retrieval.top_k == 3


def test_unknown_profile_is_a_clear_error():
    with pytest.raises(ConfigurationError, match="Profile 'nope' not found"):
        load_settings("nope", env_file=None)


def test_typo_in_config_is_rejected():
    with pytest.raises(ConfigurationError, match="top_kk"):
        load_settings(overrides={"retrieval": {"top_kk": 3}}, env_file=None)


def test_secret_comes_from_env(monkeypatch):
    monkeypatch.setenv("MONGODB_URI", "mongodb://example:27017")
    assert load_settings(env_file=None).mongodb_uri() == "mongodb://example:27017"


def test_missing_secret_is_a_clear_error(monkeypatch):
    monkeypatch.delenv("MONGODB_URI", raising=False)
    settings = load_settings(env_file=None)
    with pytest.raises(ConfigurationError, match=r"MONGODB_URI is not set.*\.env\.example"):
        settings.mongodb_uri()


def test_secrets_are_masked_in_repr(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "super-secret")
    assert "super-secret" not in repr(load_settings(env_file=None))


def test_config_hash_is_stable(config_dir):
    first = load_settings(config_dir=config_dir, env_file=None).config_hash()
    assert first == load_settings(config_dir=config_dir, env_file=None).config_hash()


def test_config_hash_changes_with_config(config_dir):
    before = load_settings(config_dir=config_dir, env_file=None).config_hash()
    after = load_settings(
        config_dir=config_dir, overrides={"retrieval": {"top_k": 8}}, env_file=None
    ).config_hash()
    assert before != after


def test_config_hash_changes_with_index_files(config_dir):
    before = load_settings(config_dir=config_dir, env_file=None).config_hash()
    path = config_dir / "indexes" / "chunks_text.json"
    path.write_text(path.read_text().replace("lucene.english", "lucene.standard"))
    assert load_settings(config_dir=config_dir, env_file=None).config_hash() != before


def test_config_hash_ignores_secrets(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "a")
    first = load_settings(env_file=None).config_hash()
    monkeypatch.setenv("GEMINI_API_KEY", "b")
    assert load_settings(env_file=None).config_hash() == first


def test_chunks_collection_is_named_after_the_model():
    assert load_settings(env_file=None).chunks_collection == "chunks_nomic_ai_nomic_embed_text_v1_5"


def test_deep_merge_merges_nested_and_replaces_leaves():
    merged = deep_merge({"a": {"x": 1, "y": 2}, "b": [1]}, {"a": {"y": 3}, "b": [2]})
    assert merged == {"a": {"x": 1, "y": 3}, "b": [2]}
