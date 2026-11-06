"""OpenTelemetry tracing. Spans use the OTel API (a no-op unless an SDK is configured). Attributes are low-cardinality and contain
no prompts, answers, keys or tenant ids."""

from contextlib import contextmanager

from opentelemetry import trace

_tracer = trace.get_tracer("semcache")


@contextmanager
def span(name: str, **attrs):
    with _tracer.start_as_current_span(name) as s:
        for k, v in attrs.items():
            s.set_attribute(k, v)
        yield s


def configure_tracing(exporter: str) -> None:
    """exporter: none | console | otlp. The SDK is only imported when an exporter is requested (it is a dev/optional dependency)."""
    if exporter == "none":
        return
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

    provider = TracerProvider(resource=Resource.create({"service.name": "semantic-cache"}))
    if exporter == "console":
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    else:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter  # needs opentelemetry-exporter-otlp-proto-http

        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)
