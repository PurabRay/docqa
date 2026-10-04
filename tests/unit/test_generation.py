import json

import pytest

from docqa.adapters.llm.stub import StubLLM
from docqa.domain.errors import AllProvidersUnavailableError, ConfigurationError
from docqa.domain.models import Chunk, RetrievedChunk, Turn
from docqa.generation.answer_generator import NO_CITATIONS_WARNING, AnswerGenerator, Final
from docqa.generation.citation_validator import validate_citations
from docqa.generation.json_stream import FieldStreamer
from docqa.generation.prompt_builder import PromptBuilder
from docqa.generation.prompt_registry import PromptRegistry
from docqa.generation.schemas import CitedClaim, LLMAnswer
from docqa.ingestion.text_split import token_counter
from docqa.retrieval.query_rewriter import QueryRewriter
from docqa.settings import load_settings

SETTINGS = load_settings(env_file=None)
PROMPTS = PromptRegistry(SETTINGS.prompts_dir)
COUNT = token_counter("cl100k_base")


def retrieved(cid, text, start=1, end=1, offsets=(), flagged=False, filename="report.pdf"):
    chunk = Chunk(
        id=cid,
        doc_id="d",
        owner_id="u",
        filename=filename,
        text=text,
        page_start=start,
        page_end=end,
        content_hash="h",
        flagged_injection=flagged,
        page_offsets=list(offsets),
    )
    return RetrievedChunk(chunk=chunk)


# ---------------------------------------------------------------- registry and builder


def test_registry_loads_both_v1_prompts():
    answer = PROMPTS.get("answer", "v1")
    assert answer.version == "v1" and "[BEGIN DOCUMENT]" in answer.system
    assert "{history}" in PROMPTS.get("rewrite", "v1").user


def test_registry_rejects_a_file_that_names_another_version(tmp_path):
    (tmp_path / "answer").mkdir()
    (tmp_path / "answer" / "v2.yaml").write_text(
        "name: answer\nversion: v1\nchangelog: x\nsystem: s\nuser: '{documents} {question}'\n"
    )
    with pytest.raises(ConfigurationError, match="declares"):
        PromptRegistry(tmp_path).get("answer", "v2")


def test_builder_order_and_markers():
    chunks = [
        retrieved("c1", "First {braces} text.", 4, 5),
        retrieved("c2", "Risky.", flagged=True),
    ]
    system, user = PromptBuilder(6000, COUNT).build(PROMPTS.get("answer", "v1"), chunks, "What?")
    assert system.role == "system" and system.content == PROMPTS.get("answer", "v1").system
    assert (
        "[BEGIN DOCUMENT id=c1 doc=report.pdf page=4-5]\nFirst {braces} text.\n[END DOCUMENT]"
        in user.content
    )
    assert "[BEGIN DOCUMENT id=c2 doc=report.pdf page=1 untrusted=true]" in user.content
    assert (
        user.content.index("id=c1")
        < user.content.index("id=c2")
        < user.content.index("Question: What?")
    )
    assert user.content.rstrip().endswith("Question: What?")


def test_builder_drops_lowest_ranked_chunks_to_fit_the_token_cap():
    chunks = [retrieved(f"c{i}", "word " * 200) for i in range(5)]
    _, user = PromptBuilder(700, COUNT).build(PROMPTS.get("answer", "v1"), chunks, "Q?")
    assert "id=c0" in user.content and "id=c1" in user.content and "id=c4" not in user.content
    assert COUNT(user.content) <= 700


# ---------------------------------------------------------------- streaming and parsing


def feed_all(pieces):
    streamer = FieldStreamer("answer")
    return "".join(streamer.feed(p) for p in pieces)


def test_streamer_emits_only_the_answer_field_even_when_escapes_are_split():
    raw = json.dumps(
        {"answer": 'He said "30 days"\nthen é', "citations": [], "sufficient_context": True}
    )
    pieces = [raw[i : i + 3] for i in range(0, len(raw), 3)]
    assert feed_all(pieces) == 'He said "30 days"\nthen é'


def test_streamer_passes_plain_text_through():
    assert feed_all(["  The refund ", "window is 30 days."]) == "The refund window is 30 days."


async def generate(reply):
    events = [e async for e in AnswerGenerator(StubLLM(reply, piece_size=5), 600).generate([])]
    tokens = "".join(e.text for e in events[:-1])
    final = events[-1]
    assert isinstance(final, Final) and all(not isinstance(e, Final) for e in events[:-1])
    return tokens, final


ANSWER = {
    "answer": "30 days.",
    "citations": [{"chunk_id": "c1", "quote": "30 days"}],
    "sufficient_context": True,
}


async def test_structured_json_streams_only_answer_text():
    tokens, final = await generate(json.dumps(ANSWER))
    assert (
        tokens == "30 days."
        and final.style == "structured"
        and final.parsed.citations[0].chunk_id == "c1"
    )
    assert final.provider == "stub"


async def test_fenced_json_is_parsed():
    tokens, final = await generate(f"```json\n{json.dumps(ANSWER)}\n```")
    assert tokens == "30 days." and final.style == "fenced" and final.warning is None


async def test_plain_text_gets_no_citations_and_a_warning():
    tokens, final = await generate("The window is 30 days [c1].")
    assert tokens == "The window is 30 days [c1]."
    assert (
        final.style == "plain"
        and final.parsed.citations == []
        and final.warning == NO_CITATIONS_WARNING
    )


# ---------------------------------------------------------------- citation validation


SPANNING = retrieved(
    "c5",
    "This paragraph starts on page five and keeps going until it finally ends on page six.",
    start=5,
    end=6,
    offsets=[56],
)


def test_validator_drops_a_fabricated_chunk_and_a_fake_quote():
    chunks = [retrieved("c1", "The refund window is 30 days from delivery.", 3, 3)]
    answer = LLMAnswer(
        answer="...",
        sufficient_context=True,
        citations=[
            CitedClaim(chunk_id="made-up", quote="30 days"),
            CitedClaim(chunk_id="c1", quote="the refund window is 90 days"),
            CitedClaim(
                chunk_id="c1", quote="refund   window is\n30 days"
            ),  # whitespace differs: kept
        ],
    )
    result = validate_citations(answer, chunks)
    assert result.dropped == 2 and result.verified
    assert [(c.chunk_id, c.page, c.doc_name) for c in result.citations] == [("c1", 3, "report.pdf")]


def test_curly_quotes_match_straight_quotes():
    chunks = [retrieved("c1", 'The policy says "final sale" items can\'t be returned.')]
    answer = LLMAnswer(
        answer="...",
        sufficient_context=True,
        citations=[
            CitedClaim(
                chunk_id="c1",
                quote="\N{LEFT DOUBLE QUOTATION MARK}final sale\N{RIGHT DOUBLE QUOTATION MARK} "
                "items can\N{RIGHT SINGLE QUOTATION MARK}t",
            )
        ],
    )
    assert validate_citations(answer, chunks).verified


@pytest.mark.parametrize(
    "quote, page", [("starts on page five", 5), ("finally ends on page six", 6)]
)
def test_validator_picks_the_page_that_holds_the_quote(quote, page):
    answer = LLMAnswer(
        answer="...", sufficient_context=True, citations=[CitedClaim(chunk_id="c5", quote=quote)]
    )
    assert validate_citations(answer, [SPANNING]).citations[0].page == page


def test_no_surviving_citation_means_unverified():
    answer = LLMAnswer(
        answer="...", sufficient_context=True, citations=[CitedClaim(chunk_id="x", quote="y")]
    )
    result = validate_citations(answer, [SPANNING])
    assert (result.verified, result.dropped, result.citations) == (False, 1, [])


# ---------------------------------------------------------------- rewriter


def rewriter(llm):
    return QueryRewriter(llm, PROMPTS.get("rewrite", "v1"), max_tokens=100)


async def test_rewriter_is_a_no_op_on_the_first_turn():
    llm = StubLLM("should not be called")
    assert (
        await rewriter(llm).rewrite("What is the refund window?", [])
        == "What is the refund window?"
    )
    assert llm.calls == []


async def test_rewriter_uses_history_and_keeps_the_first_line():
    llm = StubLLM('"What is the refund window for international orders?"\nextra chatter')
    history = [
        Turn(session_id="s", role="user", content_scrubbed="What is the refund window?"),
        Turn(session_id="s", role="assistant", content_scrubbed="30 days."),
    ]
    result = await rewriter(llm).rewrite("And for international orders?", history)
    assert result == "What is the refund window for international orders?"
    assert "user: What is the refund window?" in llm.calls[0][1].content


async def test_rewriter_falls_back_to_the_raw_question_without_an_llm():
    class Down:
        name = "down"

        async def complete(self, *a, **k):
            raise AllProvidersUnavailableError()

    history = [Turn(session_id="s", role="user", content_scrubbed="hi")]
    assert await rewriter(Down()).rewrite("And then?", history) == "And then?"
