from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any

@contextmanager
def trace_run(name: str, metadata: dict[str, Any] | None = None):
    # Only enable tracing if explicitly configured
    if os.getenv("LANGCHAIN_TRACING_V2") != "true":
        yield
        return

    try:
        from langsmith import Client
    except Exception:
        yield
        return

    try:
        client = Client()
        run = client.create_run(
            name=name,
            run_type="tool",
            inputs=metadata or {},
        )
    except Exception:
        yield
        return

    try:
        yield
        client.update_run(
            run_id=run.id,
            outputs={"ok": True},
            error=None,
        )
    except Exception as e:
        client.update_run(
            run_id=run.id,
            error=str(e),
        )
        raise
