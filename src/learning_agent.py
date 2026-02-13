# src/learning_agent.py
from __future__ import annotations

import re
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


# ============================================================
# Helpers (clean + dedupe + reduce noisy headings)
# ============================================================

def _clean_lines(raw_text: str) -> list[str]:
    """Remove underline headings, empty lines, and [Image ...] placeholders."""
    lines: list[str] = []
    for line in (raw_text or "").splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("[Image"):
            continue
        if set(s) <= {"-", "="}:  # underline-only lines like ---- or ====
            continue
        lines.append(s)
    return lines


def _looks_like_section_label(s: str) -> bool:
    """
    Filters out noisy headings/labels that pollute explanations + MCQ extraction.
    """
    low = (s or "").strip().lower()
    if not low:
        return True

    # short "Label:" lines like "Core Concepts:" / "Learning Pipeline:"
    if low.endswith(":") and len(low.split()) <= 5:
        return True

    # common headings found in notes
    if low in {
        "machine learning context & core concepts",
        "machine learning overview",
        "core concepts",
        "core concepts:",
        "learning pipeline",
        "learning pipeline:",
        "limitations",
        "limitations:",
        "training",
        "evaluation",
        "deployment",
        "monitoring and retraining",
        "quick example",
        "quick example:",
        "concept explanation",
        "concept explanation:",
        "key concepts",
        "key concepts:",
        "detailed summary",
        "detailed summary:",
    }:
        return True

    # very short ALL CAPS headings
    if len(low.split()) <= 3 and s.isupper():
        return True

    return False


def _dedupe_keep_order(lines: list[str]) -> list[str]:
    seen = set()
    out = []
    for x in lines:
        k = (x or "").strip().lower()
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(x.strip())
    return out


def _drop_near_duplicates(lines: list[str]) -> list[str]:
    """
    Cheap near-duplicate remover:
    if a line shares 90%+ words with a recent line, drop it.
    Helps remove repeated ML definition lines in notes.
    """
    out: list[str] = []
    recent: list[str] = []

    for s in lines:
        words = set(re.findall(r"[a-zA-Z]{3,}", s.lower()))
        if not words:
            continue

        dup = False
        for prev in recent[-6:]:
            pwords = set(re.findall(r"[a-zA-Z]{3,}", prev.lower()))
            if not pwords:
                continue
            overlap = len(words & pwords) / max(1, len(words))
            if overlap >= 0.90:
                dup = True
                break

        if not dup:
            out.append(s)
            recent.append(s)

    return out


def _clean_for_display(raw_text: str) -> list[str]:
    """Stronger cleanup for display: remove labels + dedupe + near-dedupe."""
    lines = _clean_lines(raw_text)
    lines = [x for x in lines if not _looks_like_section_label(x)]
    lines = _dedupe_keep_order(lines)
    lines = _drop_near_duplicates(lines)
    return lines

def _chunk(lines: list[str], n: int) -> list[str]:
    return [x.strip() for x in lines[:n] if x.strip()]

def _bullets(lines: list[str], n: int = 6) -> str:
    picked = _chunk(lines, n)
    return "\n".join([f"- {x}" for x in picked]) if picked else "- (No points found.)"

def _find_first(lines: list[str], keywords: tuple[str, ...], fallback: str = "") -> str:
    for s in lines:
        low = s.lower()
        if any(k in low for k in keywords):
            return s.strip()
    return fallback

def _pipeline_from_notes(lines: list[str]) -> list[str]:
    kws = (
        "data collection", "data preprocessing", "preprocessing", "cleaning", "normalization",
        "feature engineering", "features", "model selection", "training", "validation",
        "evaluation", "deployment", "monitoring", "retraining", "concept drift"
    )
    hits = []
    for s in lines:
        low = s.lower()
        if any(k in low for k in kws):
            hits.append(s)
    hits = _dedupe_keep_order(hits)
    return hits[:7]

def _format_pipeline(lines: list[str]) -> str:
    flow = _pipeline_from_notes(lines)
    if not flow:
        # generic but technical
        flow = [
            "Collect dataset D",
            "Preprocess (clean/normalize/encode)",
            "Define features X and target Y (if supervised)",
            "Train model fθ by minimizing loss L(θ)",
            "Validate / tune hyperparameters",
            "Test on unseen data",
            "Deploy + monitor drift + retrain",
        ]
    return "\n".join([f"{i+1}) {x}" for i, x in enumerate(flow, start=1)])

def _tech_notation_block(topic: str) -> str:
    t = (topic or "").lower()

    # Default ML notation (works for most topics)
    base = (
        "- Dataset: **D = {(xᵢ, yᵢ)}** (supervised)\n"
        "- Features: **X**, Labels/Targets: **Y**\n"
        "- Model: **ŷ = f(x; θ)**\n"
        "- Training objective: **θ\* = argmin₍θ₎  Σ L(f(xᵢ; θ), yᵢ)**\n"
        "- Generalization gap: **E_test − E_train** (large ⇒ overfitting risk)"
    )

    if "unsupervised" in t:
        return (
            "- Dataset: **D = {xᵢ}** (no labels)\n"
            "- Goal: learn structure **g(x)** (clusters / embeddings)\n"
            "- Clustering: minimize within-cluster distance (e.g., K-means)\n"
            "- Dimensionality reduction: **z = h(x)** (e.g., PCA)\n"
            "- Evaluation: silhouette / inertia / domain validation"
        )

    if "reinforcement" in t:
        return (
            "- Agent interacts with environment **E**\n"
            "- State **s**, action **a**, reward **r**\n"
            "- Policy: **π(a|s)**\n"
            "- Objective: maximize expected return **E[Σ γᵗ rₜ]**\n"
            "- Exploration vs exploitation trade-off"
        )

    if "overfitting" in t:
        return (
            "- Symptom: **Acc_train ↑**, **Acc_test ↓**\n"
            "- Generalization gap: **E_test − E_train** large\n"
            "- Fix: regularization (L2), early stopping, more data, simpler model\n"
            "- Goal: minimize test error, not train error"
        )

    if "bias vs variance" in t:
        return (
            "- **Bias**: error from overly-simple assumptions (underfit)\n"
            "- **Variance**: sensitivity to noise (overfit)\n"
            "- Tradeoff: increase model capacity ↓bias but ↑variance\n"
            "- Control: regularization, more data, ensembling"
        )

    return base

def _keyword_pack(topic: str) -> str:
    t = (topic or "").lower()
    packs = {
        "machine learning": ["X/Y", "train-test split", "loss", "generalization", "overfitting", "concept drift", "monitoring"],
        "supervised learning": ["labels", "classification", "regression", "loss function", "ground truth", "metrics"],
        "unsupervised learning": ["clustering", "PCA", "embeddings", "similarity", "no labels", "structure discovery"],
        "deep learning": ["layers", "backpropagation", "parameters", "GPU", "CNN/RNN", "representation learning"],
        "neural networks": ["weights", "bias", "activation", "gradient", "backprop", "hidden layers"],
        "model evaluation": ["accuracy", "precision/recall", "F1", "ROC-AUC", "MAE/MSE", "cross-validation"],
        "overfitting": ["generalization gap", "regularization", "early stopping", "data leakage"],
        "underfitting": ["high bias", "low capacity", "feature insufficiency"],
        "reinforcement learning": ["policy", "reward", "episode", "Q-learning", "exploration"],
        "bias vs variance": ["bias", "variance", "tradeoff", "regularization", "ensembling"],
    }

    words = packs.get(t, ["data", "model", "loss", "train", "test", "metrics", "generalization"])
    return ", ".join([f"**{w}**" for w in words])



# ============================================================
# Explanation style helpers
# ============================================================

def _topic_analogy(topic: str) -> str:
    t = (topic or "").lower()
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


def _extract_key_concepts(lines: list[str], limit: int = 10) -> str:
    """
    Pull high-signal concept lines.
    Skips section labels so Key Concepts aren't polluted.
    """
    picks: list[str] = []
    keywords = (
        "definition", "core", "goal", "features", "labels",
        "train", "test", "metric", "overfit", "underfit",
        "generalization", "concept drift", "monitor", "bias", "variance",
        "validation", "deployment", "feedback"
    )

    for s in lines:
        if _looks_like_section_label(s):
            continue
        low = s.lower()
        if ":" in s or "→" in s or any(k in low for k in keywords):
            picks.append(s)
        if len(picks) >= limit:
            break

    if not picks:
        picks = lines[:limit]

    return "\n".join([f"- {x}" for x in picks])


def _mini_example(topic: str) -> str:
    t = (topic or "").lower()
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
    """
    Attempt 2+ rotating variants.
    IMPORTANT: bullets are generated explicitly to help MCQ extraction.
    """
    lines = _clean_for_display(raw_text)

    if not lines:
        return (
            f"No notes found for '{topic}'.\n"
            f"Add a '{topic}' section to data/notes/machine_learning.txt."
        )

    style = _attempt_style(attempt)
    analogy = _topic_analogy(topic)
    picked = lines[:24]

    if style == 0:
        bullets = "\n".join([f"- {x}" for x in picked[:10]])
        return f"""{analogy}

Detailed summary:
{bullets}
""".strip()

    if style == 1:
        flow = picked[:10]
        steps_txt = "\n".join([f"{idx}) {s}" for idx, s in enumerate(flow, start=1)])
        return f"""How it works (step-by-step):
{steps_txt}

{analogy}
""".strip()

    if style == 2:
        mistakes = [
            "Confusing labeled vs unlabeled data",
            "Testing only on training data",
            "Ignoring overfitting/underfitting",
            "Picking the wrong metric",
            "Deploying without monitoring",
        ]
        fixes = [
            "Keep a validation/test set",
            "Track the right metric (accuracy/F1/MAE etc.)",
            "Start simple, then increase complexity",
            "Regularize if the model memorizes",
            "Monitor for drift and retrain",
        ]
        mistakes_txt = "\n".join([f"- {m}" for m in mistakes])
        fixes_txt = "\n".join([f"- {f}" for f in fixes])

        return f"""Common mistakes:
{mistakes_txt}

Fixes that actually work:
{fixes_txt}

{analogy}
""".strip()

    # style == 3
    return f"""Mini example + intuition:
{_mini_example(topic)}

Why this matters:
- Models should learn **patterns**, not memorize examples
- The real test is performance on **new, unseen data**

{analogy}
""".strip()


# ============================================================
# Formatter
# ============================================================

def format_as_gpt_style(topic: str, raw_text: str, attempt: int) -> str:
    lines = _clean_for_display(raw_text)

    if not lines:
        return f"## 🧠 Deep Dive: {topic}\n\nNo notes found for '{topic}'. Add it to `data/notes/machine_learning.txt`."

    # Technical definition: try to pull a "is/means/refers" sentence, else first line
    definition = _find_first(
        lines,
        keywords=(" is ", " means ", " refers to ", " can be defined", " enables "),
        fallback=lines[0]
    )

    notation = _tech_notation_block(topic)
    pipeline = _format_pipeline(lines)
    key_concepts = _extract_key_concepts(lines, limit=10)
    example = _mini_example(topic)
    keywords = _keyword_pack(topic)

    # Attempt-1: compact technical summary
    if attempt == 1:
        return f"""
## 🧠 Deep Dive: {topic}

### Definition
- {definition}

### Notation / Core Formulation
{notation}

### Learning Pipeline (high level)
{pipeline}

### Key Terms
{keywords}

### Key Concepts
{key_concepts}

### Quick Example
{example}
""".strip()

    # Attempt 2+: more technical + pitfalls + exam emphasis
    pitfalls = _simple_explanation_variant(topic, raw_text, attempt)

    return f"""
## 🧠 Deep Dive: {topic}

### Re-Explanation (Attempt {attempt}) — Technical View
- {definition}

### Notation / Core Formulation
{notation}

### Learning Pipeline (high level)
{pipeline}

### Key Terms
{keywords}

### Key Concepts (High Signal)
{key_concepts}

### Quick Example
{example}

### Common Pitfalls / Fixes
{pitfalls}

---
📌 **Exam Lens:** Define **X/Y**, write the **objective (loss)**, explain **generalization**, and mention **evaluation + drift monitoring**.
""".strip()

