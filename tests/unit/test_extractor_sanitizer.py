from pathlib import Path

import pytest

from docqa.domain.models import Page
from docqa.ingestion.extractor import (
    extract_pages,
    is_table_garbled,
    markdown_headings,
    replace_tables,
    to_markdown_table,
)
from docqa.ingestion.sanitizer import compile_patterns, sanitize
from docqa.settings import load_settings
from tests.fixtures.make_fixtures import INJECTION, KNOWN_SENTENCES, TABLES

FIXTURES = Path(__file__).parents[1] / "fixtures"
PATTERNS = compile_patterns(load_settings(env_file=None).ingestion.injection_patterns)


@pytest.fixture(scope="module")
def text_pages() -> list[Page]:
    return extract_pages(FIXTURES / "text.pdf", doc_id="d1")


def test_known_sentences_are_on_their_pages(text_pages):
    assert [page.number for page in text_pages] == list(range(1, 13))
    for number, sentence in KNOWN_SENTENCES.items():
        assert sentence in text_pages[number - 1].text
        others = [p.number for p in text_pages if sentence in p.text]
        assert others == [number]


def test_headings_come_from_the_toc(text_pages):
    assert text_pages[2].headings == ["Section 3: Policy area 3"]
    assert text_pages[5].headings == []  # continuation page


def test_two_column_page_keeps_both_columns(text_pages):
    text = text_pages[8].text
    assert text.index("Left column.") < text.index("Right column.")


def test_tables_are_extracted_as_markdown():
    text = extract_pages(FIXTURES / "tables.pdf", doc_id="d2")[0].text
    for _title, header, rows in TABLES:
        assert "|" + "|".join(header) + "|" in text
        for row in rows:
            assert "|" + "|".join(row) + "|" in text


def test_garbled_table_detection():
    good = "|a|b|\n|---|---|\n|1|2|"
    assert not is_table_garbled(good)
    assert is_table_garbled("|a|b|\n|---|---|\n|1|2|3|4|")


def test_replace_tables_keeps_prose():
    page = "Intro text\n\n|a|b|\n|1|2|3|\n\nOutro"
    fixed = replace_tables(page, [to_markdown_table([["a", "b"], ["1", None]])])
    assert "Intro text" in fixed and "Outro" in fixed
    assert "|a|b|\n|---|---|\n|1||" in fixed and "|1|2|3|" not in fixed


def test_markdown_headings():
    assert markdown_headings("# **Title**\ntext\n## Sub") == ["Title", "Sub"]


def test_sanitizer_flags_the_injection_page_and_keeps_its_text():
    pages = [sanitize(p, PATTERNS) for p in extract_pages(FIXTURES / "injection.pdf", "d3")]
    assert [p.flagged_injection for p in pages] == [False, True]
    assert INJECTION in pages[1].text


def test_sanitizer_strips_control_characters_only():
    page = Page(doc_id="d", number=1, text="a\x00b\x07c\n\td\x1fe")
    assert sanitize(page, PATTERNS).text == "abc\n\tde"
