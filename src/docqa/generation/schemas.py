"""The JSON the LLM must return (docs/DESIGN.md, "LLM output schema")."""

from __future__ import annotations

from pydantic import BaseModel, Field


class CitedClaim(BaseModel):
    """One citation: which chunk, and the exact words the claim rests on."""

    chunk_id: str
    quote: str = Field(max_length=300)


class LLMAnswer(BaseModel):
    """The model's whole reply."""

    answer: str = Field(max_length=1200)
    citations: list[CitedClaim]
    sufficient_context: bool
