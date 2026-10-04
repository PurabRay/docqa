"""Build the answer prompt: static system prompt, document blocks, question last.

The system prompt comes first and never changes, so providers can cache it.
Each chunk is wrapped in markers so the model treats it as data. If the prompt is
over max_context_tokens, the lowest-ranked chunks are dropped first.
"""

from __future__ import annotations

from docqa.domain.models import RetrievedChunk
from docqa.generation.prompt_registry import PromptTemplate
from docqa.ingestion.text_split import TokenCounter
from docqa.ports.llm import Message


def document_block(item: RetrievedChunk) -> str:
    """One chunk inside [BEGIN DOCUMENT ...] ... [END DOCUMENT] markers."""
    chunk = item.chunk
    pages = (
        str(chunk.page_start)
        if chunk.page_start == chunk.page_end
        else f"{chunk.page_start}-{chunk.page_end}"
    )
    untrusted = " untrusted=true" if chunk.flagged_injection else ""
    header = f"[BEGIN DOCUMENT id={chunk.id} doc={chunk.filename} page={pages}{untrusted}]"
    return f"{header}\n{chunk.text}\n[END DOCUMENT]"


class PromptBuilder:
    """Builds answer prompts within a token budget (limits.max_context_tokens)."""

    def __init__(self, max_context_tokens: int, count: TokenCounter) -> None:
        self._max_tokens = max_context_tokens
        self._count = count

    def build(
        self, template: PromptTemplate, chunks: list[RetrievedChunk], question: str
    ) -> list[Message]:
        """System + user messages, keeping as many top-ranked chunks as fit the budget."""
        kept = list(chunks)
        while True:
            documents = "\n\n".join(document_block(c) for c in kept)
            # replace(), not format(): document text may contain braces.
            user = template.user.replace("{documents}", documents).replace("{question}", question)
            if not kept or self._count(template.system) + self._count(user) <= self._max_tokens:
                return [
                    Message(role="system", content=template.system),
                    Message(role="user", content=user),
                ]
            kept.pop()  # drop the lowest-ranked chunk
