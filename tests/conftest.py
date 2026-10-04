"""Mark each test by its folder, so `pytest -m unit` runs tests/unit and so on."""

from pathlib import Path

import pytest

FOLDER_MARKERS = {"unit", "integration", "e2e"}


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        folder = Path(item.fspath).parent.name
        if folder in FOLDER_MARKERS:
            item.add_marker(folder)
