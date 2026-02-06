from scoring_engine import calculate_relevance_score
from context_gathering import gather_context


def build_agent():
    def agent(state: dict):
        checkpoint = state["checkpoint"]
        attempt = state.get("attempt", 1)

        raw_context = gather_context(checkpoint)

        # Score relevance using raw context
        result = calculate_relevance_score(raw_context, checkpoint.topic, attempt=attempt)
        score = int(result.get("score", 0))

        # Format for display (varies by attempt)
        gpt_explanation = format_as_gpt_style(
            topic=checkpoint.topic,
            raw_text=raw_context,
            attempt=attempt
        )

        return {
            "checkpoint": checkpoint,
            "context": gpt_explanation,
            "relevance_score": score,
            "feedback": result.get("feedback", []),
            "is_valid": score >= 60
        }

    return agent


# ------------------ Helpers ------------------

def _clean_lines(raw_text: str) -> list[str]:
    """Remove underline headings, empty lines, and [Image ...] placeholders."""
    lines = []
    for line in raw_text.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("[Image"):
            continue
        if set(s) <= {"-", "="}:  # underline-only lines like ---- or ====
            continue
        lines.append(s)
    return lines


def _topic_analogy(topic: str) -> str:
    t = topic.lower()
    if "supervised" in t:
        return "Think of it like a teacher giving you questions **with the answer key** — you learn by comparing your answer to the correct one."
    if "unsupervised" in t:
        return "Think of it like sorting a pile of items **without labels** — you group similar things and discover patterns."
    if "reinforcement" in t:
        return "Think of it like training a pet — actions get **rewards/penalties**, and you learn what maximizes reward over time."
    if "neural" in t:
        return "Think of it like layered decision-making — each layer learns a better representation before passing it forward."
    if "deep learning" in t:
        return "Think of it like stacking many feature-finders — early layers learn simple patterns, later layers learn complex ones."
    if "overfitting" in t:
        return "Think of it like memorizing one question paper — looks great in practice, fails on a new paper."
    if "underfitting" in t:
        return "Think of it like using a straight line for a curve — the model is too simple to capture real patterns."
    if "bias vs variance" in t:
        return "Bias = too simple (miss patterns). Variance = too sensitive (memorize noise). You want the balance."
    if "model evaluation" in t:
        return "Think of it like testing students on **new questions**, not the same homework."
    return "Think of it like learning patterns from examples — the goal is to do well on new, unseen data."


def _attempt_style(attempt: int) -> int:
    """Rotate 4 different explanation styles for attempts 2+."""
    return (attempt - 2) % 4


def _extract_key_concepts(lines: list[str], limit: int = 7) -> str:
    """
    Pull likely "key concept" lines (lines with ':' or arrows or strong keywords).
    Falls back to first N if nothing matches.
    """
    picks = []
    keywords = ("definition", "core", "goal", "features", "labels", "train", "test", "metric", "overfit", "underfit", "generalization")

    for s in lines:
        low = s.lower()
        if ":" in s or "→" in s or any(k in low for k in keywords):
            picks.append(s)
        if len(picks) >= limit:
            break

    if not picks:
        picks = lines[:limit]

    return "\n".join([f"- {x}" for x in picks])


def _mini_example(topic: str) -> str:
    t = topic.lower()
    if "machine learning" in t:
        return (
            "- **Spam filter:** emails (X) → spam / not spam (Y)\n"
            "- Train on labeled emails, then test on new emails to check generalization."
        )
    if "supervised" in t:
        return (
            "- **House price prediction:** features (X) → price (Y)\n"
            "- You have labels (actual prices), so the model learns from examples."
        )
    if "unsupervised" in t:
        return (
            "- **Customer segmentation:** user behavior (X) → clusters (no Y)\n"
            "- The model groups similar customers without labels."
        )
    if "overfitting" in t:
        return (
            "- **Symptom:** 98% train accuracy, 55% test accuracy\n"
            "- **Fix:** regularization, more data, early stopping, simpler model."
        )
    if "deep learning" in t:
        return (
            "- **Image classification:** pixels (X) → class label (Y)\n"
            "- CNN learns features automatically (edges → shapes → objects)."
        )
    return "- Pick a real dataset, define X and Y, and verify performance on unseen test data."


def _simple_explanation_variant(topic: str, raw_text: str, attempt: int) -> str:
    lines = _clean_lines(raw_text)
    if not lines:
        return (
            f"No notes found for '{topic}'.\n"
            f"Add a '{topic}' section to data/notes/machine_learning.txt."
        )

    style = _attempt_style(attempt)
    analogy = _topic_analogy(topic)

    # ✅ MORE CONTENT than before
    picked = lines[:20]

    if style == 0:
        bullets = "\n".join([f"- {x}" for x in picked[:8]])
        return f"""{analogy}

Detailed summary:
{bullets}
""".strip()

    if style == 1:
        flow = picked[:8]
        steps = []
        for idx, s in enumerate(flow, start=1):
            steps.append(f"{idx}) {s}")
        steps_txt = "\n".join(steps)

        return f"""How it works (step-by-step):
{steps_txt}

{analogy}
""".strip()

    if style == 2:
        return f"""Common mistakes:
- Confusing labeled vs unlabeled data
- Testing only on training data
- Ignoring overfitting/underfitting
- Picking the wrong metric

Fixes that actually work:
- Keep a validation/test set
- Track the right metric (accuracy/F1/MAE etc.)
- Start simple, then increase complexity
- Regularize if the model memorizes

{analogy}
""".strip()

    return f"""Mini example + intuition:
{_mini_example(topic)}

Why this matters:
- Models should learn **patterns**, not memorize examples
- The real test is performance on **new, unseen data**

{analogy}
""".strip()


# ------------------ Formatter ------------------

def format_as_gpt_style(topic: str, raw_text: str, attempt: int) -> str:
    lines = _clean_lines(raw_text)

    if not lines:
        return f"## 🧠 Deep Dive: {topic}\n\nNo notes found for '{topic}'. Add it to data/notes/machine_learning.txt."

    # ✅ Attempt 1: MUCH MORE explanation
    if attempt == 1:
        explanation = "\n".join(lines[:30]).strip()
        key_concepts = _extract_key_concepts(lines, limit=7)
        example = _mini_example(topic)

        return f"""
## 🧠 Deep Dive: {topic}

### Concept Explanation
{explanation}

### Key Concepts
{key_concepts}

### Quick Example
{example}

---
💡 **Pro Tip:** Read the explanation once, then try to explain it in your own words in 2–3 lines.
""".strip()

    # Attempt 2+: deeper explanation with rotating styles
    simple = _simple_explanation_variant(topic, raw_text, attempt)

    return f"""
## 🧠 Deep Dive: {topic}

### Deeper Explanation (Attempt {attempt})
{simple}

---
📌 **Exam Tip:** Questions often test *generalization, evaluation, and edge cases*, not just definitions.
""".strip()
