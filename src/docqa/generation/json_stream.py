"""Pull the text of one string field ("answer") out of JSON that arrives in pieces.

The model streams something like {"answer": "The refund window is...", "citations": [...]}.
The user should see the answer text as it is written, not the JSON around it.
If the reply turns out not to be JSON at all, the text is passed through unchanged.
"""

from __future__ import annotations

import json
import re

ESCAPE_LENGTH = {"u": 6}  # \uXXXX is 6 characters; every other escape is 2


class FieldStreamer:
    """Feed raw deltas in; get back the newly decoded characters of the field."""

    def __init__(self, field: str = "answer") -> None:
        self._start = re.compile(r'"' + re.escape(field) + r'"\s*:\s*"')
        self.raw = ""
        self._pos: int | None = None  # where undecoded field text starts in ``raw``
        self._mode = "detect"  # detect | json | plain | done

    def feed(self, delta: str) -> str:
        """Add a delta; return the field text it completed (may be empty)."""
        self.raw += delta
        if self._mode == "detect":
            self._detect()
            if self._mode == "plain":
                return self.raw.lstrip()  # first visible text, minus leading whitespace
        if self._mode == "plain":
            return delta
        if self._mode == "json" and self._pos is None:
            match = self._start.search(self.raw)
            self._pos = match.end() if match else None
        return self._decode() if self._mode == "json" and self._pos is not None else ""

    def _detect(self) -> None:
        stripped = self.raw.lstrip()
        if stripped:
            self._mode = "json" if stripped[0] in "{`" else "plain"

    def _decode(self) -> str:
        """Decode characters up to the closing quote or an incomplete escape."""
        assert self._pos is not None
        out, pos = [], self._pos
        while pos < len(self.raw):
            char = self.raw[pos]
            if char == '"':
                self._mode = "done"
                break
            if char == "\\":
                length = ESCAPE_LENGTH.get(self.raw[pos + 1 : pos + 2], 2)
                if pos + length > len(self.raw):
                    break  # wait for the rest of the escape sequence
                out.append(json.loads(f'"{self.raw[pos : pos + length]}"'))
                pos += length
                continue
            out.append(char)
            pos += 1
        self._pos = pos
        return "".join(out)
