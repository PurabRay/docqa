"""Load versioned prompts from prompts/<name>/<version>.yaml.

A prompt change is a new file (v2.yaml), never an edit, so traces and eval
reports that name a version stay comparable.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ValidationError

from docqa.domain.errors import ConfigurationError

# Placeholders each prompt's user template must contain.
REQUIRED_PLACEHOLDERS = {
    "answer": ["{documents}", "{question}"],
    "rewrite": ["{history}", "{question}"],
    "judge": ["{question}", "{answer}"],
}


class PromptTemplate(BaseModel):
    """One prompt file."""

    name: str
    version: str
    changelog: str
    system: str
    user: str


class PromptRegistry:
    """Reads and validates prompt files on demand."""

    def __init__(self, prompts_dir: Path) -> None:
        self._dir = prompts_dir

    def get(self, name: str, version: str) -> PromptTemplate:
        """Load prompts/<name>/<version>.yaml.

        Raises:
            ConfigurationError: Missing file, bad YAML, or a name/version/placeholder mismatch.
        """
        path = self._dir / name / f"{version}.yaml"
        if not path.is_file():
            raise ConfigurationError(f"Prompt not found: {path}")
        try:
            template = PromptTemplate.model_validate(
                yaml.safe_load(path.read_text(encoding="utf-8"))
            )
        except (ValidationError, yaml.YAMLError) as err:
            raise ConfigurationError(f"Invalid prompt file {path}: {err}") from err
        if (template.name, template.version) != (name, version):
            raise ConfigurationError(f"{path} declares {template.name}/{template.version}.")
        missing = [p for p in REQUIRED_PLACEHOLDERS.get(name, []) if p not in template.user]
        if missing:
            raise ConfigurationError(f"{path} user template is missing {missing}.")
        return template
