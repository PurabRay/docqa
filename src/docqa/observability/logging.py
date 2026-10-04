"""structlog JSON logs on stdout, with PII and secrets kept out (docs/DESIGN.md, Logging).

Every record (structlog or stdlib logging) passes ``redact``: fields named question,
text, content or uri are replaced, and emails/phones/ID numbers in any other string are
scrubbed. trace_id, request_id and doc_id come from contextvars bound per request.
PyMongo command monitoring stays off: no listeners are registered anywhere.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import MutableMapping
from typing import Any

import structlog

from docqa.guardrails.pii import regex_scrub

SENSITIVE_FIELDS = {"question", "text", "content", "uri", "mongodb_uri", "api_key"}
REDACTED = "[redacted]"


def redact(
    _logger: Any, _method: str, event_dict: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    """Drop sensitive fields and scrub PII from every other string value."""
    for key, value in list(event_dict.items()):
        if key in SENSITIVE_FIELDS:
            event_dict[key] = REDACTED
        elif isinstance(value, str):
            event_dict[key] = regex_scrub(value)
    return event_dict


SHARED_PROCESSORS: list[Any] = [
    structlog.contextvars.merge_contextvars,
    structlog.stdlib.add_log_level,
    structlog.processors.TimeStamper(fmt="iso", utc=True),
    structlog.stdlib.ExtraAdder(),
    redact,
]


def configure_logging(level: str = "INFO") -> None:
    """Send structlog and stdlib logging through the same JSON renderer."""
    structlog.configure(
        processors=[*SHARED_PROCESSORS, structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=SHARED_PROCESSORS,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            redact,  # again, after %-style messages are formatted into "event"
            structlog.processors.JSONRenderer(),
        ],
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)
