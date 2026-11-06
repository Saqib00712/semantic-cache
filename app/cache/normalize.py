"""Request normalization: make trivially different spellings of the SAME request identical, and nothing more.

Done: Unicode NFC (the same text written with composed or decomposed accents), line endings to \\n, trailing whitespace per line
removed, runs of spaces/tabs collapsed, leading/trailing blank space removed.
NOT done, on purpose: lower-casing ("US" vs "us", "Apple" vs "apple" can change meaning), stripping punctuation ("Is it safe?" vs
"It is safe."), removing numbers or stop words, stemming, spelling correction. Wording differences are the semantic layer's job, and
that layer is guarded by the eligibility engine. Over-normalizing here would silently merge different requests."""

import hashlib
import re
import unicodedata

_SPACES = re.compile(r"[ \t ]+")
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
_QUOTED = re.compile(r"\"([^\"]{1,200})\"|“([^”]{1,200})”")
_IDENT = re.compile(r"\b(?=[A-Za-z0-9_\-]*\d)(?=[A-Za-z0-9_\-]*[A-Za-z])[A-Za-z0-9_\-]{3,}\b")
_WORD = re.compile(r"[A-Za-z']+")
NEGATIONS = frozenset({"not", "no", "never", "without", "cannot", "none", "nothing", "nobody", "neither", "nor", "cant", "dont", "wont"})
_CONTROL = set(range(0, 9)) | {11, 12} | set(range(14, 32)) | {127}


def has_control_chars(s: str) -> bool:
    return any(ord(c) in _CONTROL for c in s)


def normalize_query(text: str) -> str:
    text = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    lines = [_SPACES.sub(" ", line).strip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def critical_signature(normalized: str) -> dict:
    """The parts of a request where a small wording change flips the meaning: numbers, negation, quoted strings, identifiers.
    Two requests may share a cached answer only if their signatures match (see the eligibility engine). This is a heuristic guard
    against the classic failure ("within 30 days" vs "within 60 days", "can" vs "cannot"); it is NOT a proof of equivalence."""
    numbers = sorted({m.replace(",", "") for m in _NUMBER.findall(normalized)})
    words = [w.lower().replace("'", "") for w in _WORD.findall(normalized)]
    negated = any(
        w in NEGATIONS
        or w.endswith("nt")
        and w[:-2] in {"do", "does", "did", "is", "are", "was", "were", "ca", "wo", "has", "have", "had", "should", "would", "could"}
        for w in words
    )  # noqa: E501
    quoted = sorted({sha256_hex(a or b)[:16] for a, b in _QUOTED.findall(normalized)})
    idents = sorted({m.lower() for m in _IDENT.findall(normalized)})
    return {"numbers": numbers, "negated": negated, "quoted": quoted, "idents": idents}
