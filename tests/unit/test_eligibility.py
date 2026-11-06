import time
import uuid

import pytest

from app.cache.keys import params_hash
from app.cache.models import QueryRequest
from app.cache.normalize import critical_signature, sha256_hex
from app.config import Settings
from app.eligibility import engine

S = Settings(_env_file=None, backend="memory", embedding_dim=256, similarity_threshold=0.9, log_level="ERROR")
NOW = 1_000_000.0


def meta(**over):
    sig = critical_signature("refund in 30 days")
    m = dict(schema_version=1, entry_id=uuid.uuid4().hex, tenant_id="t1", exact_key="a" * 64, query_hash=sha256_hex("x"), signature=sig,
             model=S.generation_model, system_prompt_version=S.system_prompt_version, prompt_template_version="v1", context_version="none",
             tool_version=None, params_hash=params_hash(0.0, 512), embedding_model=S.embedding_model, embedding_dim=256, personalized=False,
             scope="tenant", status="active", created_at=NOW - 10, expires_at=NOW + 1000, ttl_seconds=1010, prompt_tokens=1,
             completion_tokens=1, finish_reason="stop", gen_cost_pico=100)  # fmt: skip
    m.update(over)
    return m


def sem(raw, score=0.95, req=None, text="refund in 30 days"):
    return engine.check_semantic(raw, score, req=req or QueryRequest(query=text), tenant_id="t1", signature=critical_signature(text),
                                 model=S.generation_model, max_tokens=512, settings=S, now=NOW)  # fmt: skip


def test_hit_when_everything_matches():
    v = sem(meta())
    assert v.decision == "semantic_hit" and v.meta is not None


@pytest.mark.parametrize("over,reason", [
    ({"tenant_id": "t2"}, "tenant_mismatch"),
    ({"status": "invalidated"}, "invalidated"),
    ({"personalized": True}, "entry_personalized"),
    ({"embedding_model": "other"}, "embedding_model_mismatch"),
    ({"embedding_dim": 8}, "embedding_model_mismatch"),
    ({"model": "gpt-x"}, "model_mismatch"),
    ({"system_prompt_version": "sp-9"}, "system_prompt_version_mismatch"),
    ({"prompt_template_version": "v9"}, "template_version_mismatch"),
    ({"context_version": "c9"}, "context_version_mismatch"),
    ({"tool_version": "tool1"}, "tool_version_mismatch"),
    ({"params_hash": "0" * 32}, "params_mismatch"),
])  # fmt: skip
def test_each_incompatibility_is_rejected_with_its_reason(over, reason):
    v = sem(meta(**over))
    assert (v.decision, v.reason) == ("rejected_incompatible", reason)


def test_expired_below_threshold_and_malformed():
    assert sem(meta(expires_at=NOW - 1)).decision == "expired"
    assert sem(meta(), score=0.5).reason == "below_threshold"
    assert sem({"junk": 1}).reason == "malformed_metadata"
    bad = meta()
    del bad["exact_key"]
    assert sem(bad).reason == "malformed_metadata"
    assert sem(meta(entry_id="nothex")).reason == "malformed_metadata"


@pytest.mark.parametrize("text,reason", [
    ("refund in 60 days", "number_mismatch"),
    ("no refund in 30 days", "negation_mismatch"),
    ('refund in 30 days "gold"', "quoted_text_mismatch"),
    ("refund in 30 days AB-30", "identifier_mismatch"),
])  # fmt: skip
def test_critical_signature_guard(text, reason):
    assert sem(meta(), text=text).reason == reason


def test_request_bypass_reasons():
    assert engine.request_bypass_reason(QueryRequest(query="hi", personalized=True), S) == "caller_marked_personalized"
    assert engine.request_bypass_reason(QueryRequest(query="hi", temperature=0.9), S) == "temperature_above_cacheable_limit"
    assert engine.request_bypass_reason(QueryRequest(query="my email is a@b.co"), S).startswith("personal_data")
    assert engine.request_bypass_reason(QueryRequest(query="x" * 5000), S) == "query_too_long"
    assert engine.request_bypass_reason(QueryRequest(query="what is DNS"), S) is None


def test_exact_check_rejects_cross_tenant_and_expired():
    assert engine.check_exact(meta(), tenant_id="t1", now=NOW).meta is not None
    assert engine.check_exact(meta(), tenant_id="t2", now=NOW).reason == "tenant_mismatch"
    assert engine.check_exact(meta(expires_at=time.time() - 5), tenant_id="t1", now=time.time()).decision == "expired"
