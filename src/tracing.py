# src/tracing.py
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any

@contextmanager
def trace_run(name: str, metadata: dict[str, Any] | None = None):
    """
    Safe LangSmith tracing.
    - Never crashes your app if tracing is misconfigured.
    - Only enables tracing when LANGCHAIN_TRACING_V2=true.
    """
    if os.getenv("LANGCHAIN_TRACING_V2", "").lower() != "true":
        yield
        return

    try:
        from langsmith import Client
    except Exception:
        yield
        return

    client = None
    run_id = None

    # Try to create a run
    try:
        client = Client()
        run = client.create_run(
            name=name,
            run_type="tool",
            inputs=metadata or {},
        )
        # Some environments can return None; guard it
        if run is not None and getattr(run, "id", None) is not None:
            run_id = run.id
    except Exception:
        # If anything fails, tracing becomes no-op
        run_id = None

    try:
        yield
        # update only if we truly have a run_id
        if client is not None and run_id is not None:
            client.update_run(run_id=run_id, outputs={"ok": True}, error=None)
    except Exception as e:
        if client is not None and run_id is not None:
            client.update_run(run_id=run_id, error=str(e))
        raise
