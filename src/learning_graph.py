# src/learning_graph.py
from __future__ import annotations

from typing import Any, Dict, TypedDict

from langgraph.graph import StateGraph, END

from context_gathering import gather_context
from scoring_engine import calculate_relevance_score
from formatter import format_as_gpt_style
from tracing import trace_run

 # reuse your formatter


class AgentState(TypedDict, total=False):
    checkpoint: Any
    attempt: int

    raw_context: str
    relevance_score: int
    feedback: list
    is_valid: bool
    context: str


def node_gather_context(state: AgentState) -> Dict[str, Any]:
    checkpoint = state["checkpoint"]

    with trace_run("gather_context", {"query_topic": checkpoint.topic}):
        raw_context, meta = gather_context(checkpoint)

    with trace_run("retrieved_document", meta):
        pass

    return {"raw_context": raw_context}

def node_score_relevance(state: AgentState) -> Dict[str, Any]:
    checkpoint = state["checkpoint"]
    attempt = state.get("attempt", 1)
    raw_context = state.get("raw_context", "")

    with trace_run(
        "score_relevance",
        {"topic": checkpoint.topic, "attempt": attempt, "context_len": len(raw_context or "")},
    ):
        result = calculate_relevance_score(raw_context, checkpoint.topic, attempt=attempt)

    score = int(result.get("score", 0))
    return {
        "relevance_score": score,
        "feedback": result.get("feedback", []),
        "is_valid": score >= 60,
    }


def node_format_output(state: AgentState) -> Dict[str, Any]:
    checkpoint = state["checkpoint"]
    attempt = state.get("attempt", 1)
    raw_context = state.get("raw_context", "")

    with trace_run(
        "format_output",
        {"topic": checkpoint.topic, "attempt": attempt, "raw_context_len": len(raw_context or "")},
    ):
        gpt_explanation = format_as_gpt_style(
            topic=checkpoint.topic,
            raw_text=raw_context,
            attempt=attempt,
        )

    # Log output size (useful in manual review)
    with trace_run(
        "output_stats",
        {
            "topic": checkpoint.topic,
            "attempt": attempt,
            "explanation_chars": len(gpt_explanation or ""),
            "explanation_lines": len((gpt_explanation or "").splitlines()),
        },
    ):
        pass

    return {"context": gpt_explanation}


def build_learning_graph():
    g = StateGraph(AgentState)

    g.add_node("gather_context", node_gather_context)
    g.add_node("score_relevance", node_score_relevance)
    g.add_node("format_output", node_format_output)

    g.set_entry_point("gather_context")
    g.add_edge("gather_context", "score_relevance")
    g.add_edge("score_relevance", "format_output")
    g.add_edge("format_output", END)

    return g.compile()
