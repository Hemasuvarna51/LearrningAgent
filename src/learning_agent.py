# src/learning_agent.py
from __future__ import annotations

from learning_graph import build_learning_graph

_graph = None

def build_agent():
    global _graph
    if _graph is None:
        _graph = build_learning_graph()

    def agent(state: dict):
        out = _graph.invoke(state)
        return {
            "checkpoint": out.get("checkpoint", state.get("checkpoint")),
            "context": out.get("context", ""),
            "relevance_score": int(out.get("relevance_score", 0)),
            "feedback": out.get("feedback", []),
            "is_valid": bool(out.get("is_valid", False)),
        }

    return agent
