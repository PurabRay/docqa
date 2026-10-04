import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_agents_md_matches_claude_md():
    assert (ROOT / "AGENTS.md").read_text(encoding="utf-8") == (ROOT / "CLAUDE.md").read_text(
        encoding="utf-8"
    ), "Edit CLAUDE.md, then copy it to AGENTS.md"


def test_claude_md_lists_twelve_rules():
    text = (ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    numbers = [int(n) for n in re.findall(r"^(\d+)\. ", text, flags=re.MULTILINE)]
    assert numbers == list(range(1, 13))
