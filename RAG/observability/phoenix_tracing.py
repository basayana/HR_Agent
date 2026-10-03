"""Configure OpenTelemetry tracing for LlamaIndex and Arize Phoenix."""

import os
import threading

_CONFIGURED = False
_CONFIG_LOCK = threading.Lock()
_INSTRUMENTOR = None


def configure_phoenix_tracing() -> bool:
    """Enable the LlamaIndex Phoenix instrumentor once per process.

    Set ``PHOENIX_TRACING_ENABLED=false`` to disable tracing. Phoenix itself
    runs separately; see ``RAG/observability/README.md`` for local setup and cloud
    configuration.
    """
    global _CONFIGURED, _INSTRUMENTOR

    enabled = os.getenv("PHOENIX_TRACING_ENABLED", "true").strip().lower()
    if enabled in {"0", "false", "no", "off"}:
        return False

    with _CONFIG_LOCK:
        if _CONFIGURED:
            return True

        from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
        from phoenix.otel import register

        tracer_provider = register()
        instrumentor = LlamaIndexInstrumentor()
        instrumentor.instrument(tracer_provider=tracer_provider)
        _INSTRUMENTOR = instrumentor
        _CONFIGURED = True

    return True
