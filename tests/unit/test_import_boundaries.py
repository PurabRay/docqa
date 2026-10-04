"""domain/, ports/ and services/ may import only the stdlib, pydantic and each other."""

import ast
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "docqa"
ALLOWED_THIRD_PARTY = {"pydantic"}
ALLOWED_INTERNAL = {"docqa.domain", "docqa.ports"}


def imported_modules(path: Path) -> list[str]:
    modules = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            modules += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            modules.append(node.module)
    return modules


def is_allowed(module: str) -> bool:
    top = module.split(".")[0]
    if top == "docqa":
        return any(module == p or module.startswith(p + ".") for p in ALLOWED_INTERNAL)
    return top in sys.stdlib_module_names or top in ALLOWED_THIRD_PARTY or top == "__future__"


INNER_FILES = [p for layer in ("domain", "ports", "services") for p in (SRC / layer).rglob("*.py")]


@pytest.mark.parametrize("path", INNER_FILES, ids=lambda p: str(p.relative_to(SRC)))
def test_inner_layers_import_no_vendor_code(path):
    bad = [m for m in imported_modules(path) if not is_allowed(m)]
    assert not bad, f"{path.name} imports {bad}"


def test_checker_catches_a_vendor_import():
    assert not is_allowed("pymongo")
    assert not is_allowed("docqa.adapters.mongo.client")
    assert is_allowed("pydantic") and is_allowed("typing") and is_allowed("docqa.domain.models")
