# src/learning_graph.py
from __future__ import annotations

from typing import Any, Dict, TypedDict

from langgraph.graph import StateGraph, END

from context_gathering import gather_context
from scoring_engine import calculate_relevance_score
from formatter import format_as_gpt_style
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
    raw_context = gather_context(checkpoint)
    return {"raw_context": raw_context}


def node_score_relevance(state: AgentState) -> Dict[str, Any]:
    checkpoint = state["checkpoint"]
    attempt = state.get("attempt", 1)
    raw_context = state.get("raw_context", "")

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

    gpt_explanation = format_as_gpt_style(
        topic=checkpoint.topic,
        raw_text=raw_context,
        attempt=attempt,
    )
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
