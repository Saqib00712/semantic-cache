"""Generation through the official OpenAI SDK. With SEMCACHE_GENERATION_BASE_URL set, the same call goes to an OpenAI-compatible
LLM gateway (see app/integrations/gateway.py) and the gateway's policy applies. Errors are classified; only transient ones retry."""

import time
from dataclasses import dataclass
from decimal import Decimal

from app.config import Settings
from app.integrations.gateway import request_headers


class GenerationError(Exception):
    def __init__(self, code: str, status: int, message: str):
        super().__init__(code)
        self.code, self.status, self.message = code, status, message


@dataclass
class Generated:
    answer: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str


def make_generation_client(settings: Settings):
    import openai

    key = settings.generation_api_key or settings.openai_api_key
    return openai.OpenAI(
        api_key=key.get_secret_value() if key else None,
        base_url=settings.generation_base_url or settings.openai_base_url,
        timeout=settings.openai_timeout_seconds,
        max_retries=0,
    )


def cost_of(settings: Settings, prompt_tokens: int, completion_tokens: int) -> Decimal:
    return (Decimal(prompt_tokens) * settings.price_input_per_mtok + Decimal(completion_tokens) * settings.price_output_per_mtok) / Decimal(
        1_000_000
    )


def embedding_cost(settings: Settings, tokens: int) -> Decimal:
    return Decimal(tokens) * settings.price_embedding_per_mtok / Decimal(1_000_000)


def classify(e: Exception) -> tuple[GenerationError, bool]:
    """(error to surface, is_transient)."""
    import openai

    if isinstance(e, openai.RateLimitError):
        return GenerationError("upstream_rate_limited", 503, "the language model provider is rate limiting requests"), True
    if isinstance(e, openai.APITimeoutError):
        return GenerationError("upstream_timeout", 504, "the language model provider timed out"), True
    if isinstance(e, openai.APIConnectionError):
        return GenerationError("upstream_unavailable", 503, "the language model provider is unreachable"), True
    if isinstance(e, openai.PermissionDeniedError):
        # Through a gateway, 403 is the gateway's POLICY decision. It is surfaced as-is and never retried or routed around.
        return GenerationError("policy_denied", 403, "the request was denied by the language model provider or gateway policy"), False
    if isinstance(e, openai.AuthenticationError):
        return GenerationError("upstream_auth_failed", 502, "the language model provider rejected our credentials"), False
    if isinstance(e, openai.BadRequestError | openai.NotFoundError | openai.UnprocessableEntityError):
        return GenerationError("upstream_rejected_request", 502, "the language model provider rejected the request"), False
    if isinstance(e, openai.APIStatusError):
        return GenerationError("upstream_error", 502, "the language model provider returned an error"), e.status_code >= 500
    return GenerationError("upstream_error", 502, "the language model provider failed"), False


def generate(
    settings: Settings, client, *, model: str, query: str, max_tokens: int, temperature: float, request_id: str, sleep=time.sleep
) -> Generated:
    import openai

    if client is None:
        raise GenerationError("generation_not_configured", 503, "generation is not configured")
    last: GenerationError | None = None
    for attempt in range(1, settings.max_attempts + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": settings.system_prompt}, {"role": "user", "content": query}],
                max_tokens=max_tokens,
                temperature=temperature,
                extra_headers=request_headers(request_id),  # preserved end to end through a gateway
            )
            choice = resp.choices[0]
            text = choice.message.content
            if not isinstance(text, str) or not text.strip():
                raise GenerationError("empty_generation", 502, "the language model returned an empty answer")
            u = resp.usage
            return Generated(
                text,
                int(getattr(u, "prompt_tokens", 0) or 0),
                int(getattr(u, "completion_tokens", 0) or 0),
                str(choice.finish_reason or "unknown"),
            )
        except GenerationError:
            raise
        except (IndexError, AttributeError, TypeError):
            raise GenerationError("malformed_generation", 502, "the language model returned a malformed response") from None
        except openai.OpenAIError as e:
            last, transient = classify(e)
            if not transient or attempt == settings.max_attempts:
                break
            sleep(settings.retry_base_seconds * (2 ** (attempt - 1)))
    assert last is not None
    raise last
