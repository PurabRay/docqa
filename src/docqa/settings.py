"""Settings = config/base.yaml + a profile + secrets from env vars.

The only module that reads environment variables.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import SecretStr, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

from docqa.config_schema import AppConfig, BTreeIndex, RetrievalCfg, SearchIndex
from docqa.domain.errors import ConfigurationError

DEFAULT_PROFILE = "free"
DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"
PROFILE_ENV = "DOCQA_PROFILE"
CONFIG_DIR_ENV = "DOCQA_CONFIG_DIR"


class Secrets(BaseSettings):
    """API keys and connection strings, read from env vars (or a local .env file)."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", frozen=True)

    mongodb_uri: SecretStr | None = None
    gemini_api_key: SecretStr | None = None
    groq_api_key: SecretStr | None = None
    mistral_api_key: SecretStr | None = None
    langfuse_public_key: SecretStr | None = None
    langfuse_secret_key: SecretStr | None = None
    langfuse_host: SecretStr | None = None
    docqa_api_url: str | None = None  # not a secret: deployment URL of the API for the UI

    def require(self, env_name: str) -> str:
        """Return the secret ``env_name`` (e.g. "MONGODB_URI") or raise ConfigurationError."""
        field = env_name.lower()
        if field not in type(self).model_fields:
            raise ConfigurationError(f"{env_name} is not a known secret.")
        value: SecretStr | None = getattr(self, field)
        if value is None or not value.get_secret_value():
            raise ConfigurationError(
                f"{env_name} is not set. Export it or add it to .env (see .env.example)."
            )
        return value.get_secret_value()


class Settings(AppConfig):
    """The merged configuration plus index definitions and secrets."""

    profile: str
    config_dir: Path
    btree_indexes: dict[str, list[BTreeIndex]]
    search_indexes: dict[str, SearchIndex]
    secrets: Secrets

    @property
    def chunks_collection(self) -> str:
        """Name of the chunks collection for the active embedding model."""
        slug = re.sub(r"[^a-z0-9]+", "_", self.embedding.dense_model.lower()).strip("_")
        return self.mongodb.chunks_collection.format(embed_model_slug=slug)

    @property
    def prompts_dir(self) -> Path:
        """Folder with prompts/<name>/<version>.yaml, next to config/."""
        return self.config_dir.parent / "prompts"

    def optional_secret(self, env_name: str) -> str | None:
        """The secret's value, or None if it is not set."""
        value: SecretStr | None = getattr(self.secrets, env_name.lower(), None)
        return value.get_secret_value() if value is not None and value.get_secret_value() else None

    def ui_api_url(self) -> str:
        """The API base URL for the UI: $DOCQA_API_URL, else ui.api_url."""
        return self.secrets.docqa_api_url or self.ui.api_url

    def retrieval_cfg(self, **overrides: object) -> RetrievalCfg:
        """The vector store's settings; ``overrides`` replace fields (e.g. fusion="app")."""
        r, m = self.retrieval, self.mongodb
        values: dict[str, object] = {
            "fusion": r.fusion,
            "vector_index": m.vector_index,
            "text_index": m.text_index,
            "num_candidates": r.num_candidates,
            "vector_k": r.vector_k,
            "text_k": r.text_k,
            "fused_k": r.fused_k,
            "w_vector": r.weights.vector,
            "w_text": r.weights.text,
            "bulk_batch": m.bulk_batch,
            "sync_poll_interval_s": m.sync_poll_interval_s,
            "index_ready_timeout_s": m.index_ready_timeout_s,
            "index_poll_interval_s": m.index_poll_interval_s,
            "search_indexes": list(self.search_indexes.values()),
        }
        return RetrievalCfg.model_validate(values | overrides)

    def mongodb_uri(self) -> str:
        """Return the MongoDB connection string. Never log it: it carries credentials."""
        return self.secrets.require(self.mongodb.uri_env)

    def config_hash(self) -> str:
        """SHA-256 of the merged settings plus config/indexes/*. Excludes secrets."""
        digest = hashlib.sha256()
        config = self.model_dump(mode="json", include=set(AppConfig.model_fields))
        digest.update(json.dumps(config, sort_keys=True).encode())
        for path in sorted((self.config_dir / "indexes").iterdir()):
            digest.update(path.name.encode())
            # Normalise line endings so Windows and Linux checkouts hash the same.
            digest.update(path.read_text(encoding="utf-8").replace("\r\n", "\n").encode())
        return digest.hexdigest()


def load_settings(
    profile: str | None = None,
    *,
    config_dir: Path | None = None,
    overrides: dict[str, Any] | None = None,
    env_file: str | None = ".env",
) -> Settings:
    """Build Settings. Raises ConfigurationError on a missing file or invalid value.

    Args:
        profile: "free", "local", an ablation name or a YAML path ($DOCQA_PROFILE, else "free").
        config_dir: Folder with base.yaml ($DOCQA_CONFIG_DIR, else the repo's config/).
        overrides: Values merged last; used by tests.
        env_file: .env file for secrets; None reads real env vars only.
    """
    profile = profile or os.environ.get(PROFILE_ENV) or DEFAULT_PROFILE
    config_dir = config_dir or Path(os.environ.get(CONFIG_DIR_ENV) or DEFAULT_CONFIG_DIR)
    merged = deep_merge(
        _read_yaml(config_dir / "base.yaml"), _read_yaml(_profile_path(profile, config_dir))
    )
    merged = deep_merge(merged, overrides or {})
    try:
        app = AppConfig.model_validate(merged)
        return Settings(
            **app.model_dump(),
            profile=profile,
            config_dir=config_dir,
            btree_indexes=_load_btree(config_dir / "indexes" / "btree.yaml"),
            search_indexes=_load_search_indexes(config_dir / "indexes", app),
            secrets=Secrets(_env_file=env_file),
        )
    except ValidationError as err:
        raise ConfigurationError(f"Invalid configuration (profile {profile!r}):\n{err}") from err


def deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Return ``base`` with ``overlay`` merged in; nested dicts merge, other values replace."""
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def render_template(value: Any, variables: dict[str, Any]) -> Any:
    """Replace every string that is exactly "{name}" with ``variables[name]``, recursively."""
    if isinstance(value, dict):
        return {key: render_template(item, variables) for key, item in value.items()}
    if isinstance(value, list):
        return [render_template(item, variables) for item in value]
    if isinstance(value, str) and value.startswith("{") and value.endswith("}"):
        name = value[1:-1]
        if name not in variables:
            raise ConfigurationError(f"Unknown placeholder {value!r} in an index definition.")
        return variables[name]
    return value


def _profile_path(profile: str, config_dir: Path) -> Path:
    candidates = [
        Path(profile),
        config_dir / f"{profile}.yaml",
        config_dir / "ablations" / f"{profile}.yaml",
    ]
    for path in candidates:
        if path.suffix in {".yaml", ".yml"} and path.is_file():
            return path
    raise ConfigurationError(f"Profile {profile!r} not found; looked in {config_dir}.")


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigurationError(f"Config file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ConfigurationError(f"{path} must contain a mapping at the top level.")
    return data


def _load_btree(path: Path) -> dict[str, list[BTreeIndex]]:
    raw = _read_yaml(path)
    return {
        collection: [BTreeIndex(**index) for index in specs] for collection, specs in raw.items()
    }


def _load_search_indexes(folder: Path, app: AppConfig) -> dict[str, SearchIndex]:
    """Read config/indexes/*.json, filling "{embedding.dim}" from the embedding config."""
    variables = {"embedding.dim": app.embedding.dim}
    indexes = {}
    for path in sorted(folder.glob("*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        index = SearchIndex(**render_template(raw, variables))
        indexes[index.name] = index
    for required in (app.mongodb.vector_index, app.mongodb.text_index):
        if required not in indexes:
            raise ConfigurationError(f"No definition for search index {required!r} in {folder}.")
    return indexes
