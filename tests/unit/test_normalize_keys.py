import pytest

from app.cache.keys import exact_key, namespace, params_hash
from app.cache.models import QueryRequest
from app.cache.normalize import critical_signature, normalize_query
from app.eligibility.personal import personal_data_reasons


def test_normalize_is_safe_and_minimal():
    assert normalize_query("  Hello   world \r\n  next\t line  ") == "Hello world\nnext line"
    assert normalize_query("é") == normalize_query("é")  # NFC
    assert normalize_query("US") != normalize_query("us")  # no lower-casing
    assert normalize_query("Is it safe?") != normalize_query("Is it safe")  # punctuation kept
    assert normalize_query("30 days") != normalize_query("60 days")


def _k(q="hello", **kw):
    req = QueryRequest(query=q, **{k: v for k, v in kw.items() if k in QueryRequest.model_fields})
    return exact_key(
        kw.get("tenant", "t1"),
        req,
        model=kw.get("model", "m"),
        system_prompt_version=kw.get("sp", "sp-1"),
        max_output_tokens=kw.get("mt", 512),
    )


def test_exact_key_changes_with_every_answer_defining_input():
    base = _k()
    assert _k() == base
    assert _k("hello  ") == base  # whitespace-only difference
    for changed in (_k("Hello"), _k(tenant="t2"), _k(model="m2"), _k(sp="sp-2"), _k(mt=256), _k(context_version="c2"),
                    _k(prompt_template_version="v2"), _k(tool_version="tv1"), _k(temperature=0.2)):  # fmt: skip
        assert changed != base


def test_namespace_is_stable_and_hides_tenant_id():
    assert namespace("t_abc") == namespace("t_abc") != namespace("t_abd")
    assert "t_abc" not in namespace("t_abc") and len(namespace("x")) == 24


def test_params_hash():
    assert params_hash(0.0, 100) != params_hash(0.0, 101) != params_hash(0.1, 101)


@pytest.mark.parametrize("a,b,field", [
    ("refund within 30 days", "refund within 60 days", "numbers"),
    ("can I cancel", "I cannot cancel", "negated"),
    ('what is "alpha"', 'what is "beta"', "quoted"),
    ("status of order AB123", "status of order AB124", "idents"),
])  # fmt: skip
def test_signature_differs_when_meaning_flips(a, b, field):
    assert critical_signature(a)[field] != critical_signature(b)[field]


def test_signature_same_for_paraphrase():
    assert critical_signature("how to get a refund in 30 days") == critical_signature("refund me within 30 days please")


@pytest.mark.parametrize("text,expected", [
    ("Can I get my money back?", []),
    ("where is my order 12345", ["personal_noun", "long_number"]),
    ("write to bob@example.com", ["email"]),
    ("what is the capital of France", []),
])  # fmt: skip
def test_personal_data(text, expected):
    got = personal_data_reasons(text)
    assert all(e in got for e in expected) and (bool(got) == bool(expected))
