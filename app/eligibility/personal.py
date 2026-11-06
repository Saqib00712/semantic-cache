"""Detect requests that likely contain or depend on one person's data. Such requests are never looked up and never stored:
a cached answer about "my order 12345" must not be served to someone else. The detector is deliberately conservative (it prefers
to skip caching); the client can also declare `personalized: true`. A generic "my" is NOT a signal ("Can I get my money back?" is a
policy question); specific personal nouns, identifiers, emails and phone numbers are."""

import re

_PATTERNS = {
    "email": re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"),
    "phone": re.compile(r"(?<!\d)(?:\+?\d[\s\-().]?){9,}\d(?!\d)"),
    "personal_noun": re.compile(
        r"\bmy\s+(?:order|orders|account|invoice|invoices|subscription|password|card|ticket|booking|reservation|balance|address|email|"
        r"phone|number|id|profile|history|payment|package|delivery|shipment|appointment|prescription|record|records)\b",
        re.I,
    ),
    "self_statement": re.compile(r"\b(?:i\s+(?:ordered|bought|paid|purchased|booked|signed\s+up)|my\s+name\s+is|i\s+am\s+called)\b", re.I),
    "identifier": re.compile(r"\b(?=[A-Za-z0-9_\-]*\d)(?=[A-Za-z0-9_\-]*[A-Za-z])[A-Za-z0-9_\-]{5,}\b"),
    "long_number": re.compile(r"\d{5,}"),
}


def personal_data_reasons(text: str) -> list[str]:
    """Names of the signals found (never the matched text)."""
    return [name for name, pattern in _PATTERNS.items() if pattern.search(text)]
