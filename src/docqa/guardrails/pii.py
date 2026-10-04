"""PII scrubbing for logs, turns and traces: emails, phone numbers, ID-like numbers.

Two layers: fast regexes (always, also used by the log processor), then Presidio with
the small spaCy model for anything the regexes miss. Matches become <TYPE> tags.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

REGEXES = [
    ("EMAIL", re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")),
    ("PHONE", re.compile(r"(?<!\w)\+?\d[\d ()./-]{7,}\d(?!\w)")),
    ("ID_NUMBER", re.compile(r"\b[A-Z]{0,3}\d{6,}\b")),
]
PRESIDIO_ENTITIES = [
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "CREDIT_CARD",
    "IBAN_CODE",
    "US_SSN",
    "IP_ADDRESS",
]
SPACY_MODEL = "en_core_web_sm"


def regex_scrub(text: str) -> str:
    """Replace emails, phone numbers and long digit runs with <TYPE> tags."""
    for tag, pattern in REGEXES:
        text = pattern.sub(f"<{tag}>", text)
    return text


def scrub(text: str) -> str:
    """Regex scrub, then Presidio. Use before anything is logged, stored or traced."""
    text = regex_scrub(text)
    analyzer, anonymizer = _presidio()
    results = analyzer.analyze(text=text, entities=PRESIDIO_ENTITIES, language="en")
    return str(anonymizer.anonymize(text=text, analyzer_results=results).text) if results else text


@lru_cache(maxsize=1)
def _presidio() -> tuple[Any, Any]:
    """Build Presidio once (loading spaCy takes about a second)."""
    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer.nlp_engine import NlpEngineProvider
    from presidio_anonymizer import AnonymizerEngine

    nlp = NlpEngineProvider(
        nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": SPACY_MODEL}],
        }
    ).create_engine()
    return AnalyzerEngine(nlp_engine=nlp, supported_languages=["en"]), AnonymizerEngine()
