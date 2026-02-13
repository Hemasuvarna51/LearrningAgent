# src/tracing.py
from __future__ import annotations

from contextlib import contextmanager
from typing import Any

@contextmanager
def trace_run(name: str, metadata: dict[str, Any] | None = None):
    """
    Real LangSmith span context.
    Works if LANGCHAIN_TRACING_V2 + LANGCHAIN_API_KEY are set.
    Safe no-op locally if langsmith isn't installed.
    """
    try:
        from langsmith import Client
    except Exception:
        Client = None

    if Client is None:
        yield
        return

    client = Client()

    # Create a run/span
    run = client.create_run(
        name=name,
        run_type="tool",   # good enough for custom steps
        inputs=metadata or {},
    )

    try:
        yield
        # mark success
        client.update_run(
            run_id=run.id,
            outputs={"ok": True},
            error=None,
        )
    except Exception as e:
        # mark failure
        client.update_run(
            run_id=run.id,
            error=str(e),
        )
        raise
