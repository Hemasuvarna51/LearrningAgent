# src/tracing.py
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any

@contextmanager
def trace_run(name: str, metadata: dict[str, Any] | None = None):
    """
    Bulletproof LangSmith tracing.
    Will NEVER crash your app.
    """

    # Only trace if explicitly enabled
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

    try:
        client = Client()
        run = client.create_run(
            name=name,
            run_type="tool",
            inputs=metadata or {},
        )

        if run and hasattr(run, "id"):
            run_id = run.id

    except Exception:
        # Any tracing failure becomes silent
        run_id = None

    try:
        yield

        if client and run_id:
            client.update_run(
                run_id=run_id,
                outputs={"ok": True},
                error=None,
            )

    except Exception as e:
        if client and run_id:
            client.update_run(
                run_id=run_id,
                error=str(e),
            )
        raise
