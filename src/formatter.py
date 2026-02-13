# src/formatter.py
from __future__ import annotations
import re

# -----------------------
# Helpers (clean + dedupe)
# -----------------------

def _clean_lines(raw_text: str) -> list[str]:
    lines: list[str] = []
    for line in (raw_text or "").splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("[Image"):
            continue
        if set(s) <= {"-", "="}:
            continue
        lines.append(s)
    return lines


def _looks_like_section_label(s: str) -> bool:
    low = (s or "").strip().lower()
    if not low:
        return True

    if low.endswith(":") and len(low.split()) <= 5:
        return True

    if low in {
        "machine learning context & core concepts",
        "machine learning overview",
        "core concepts",
        "learning pipeline",
        "limitations",
        "training",
        "evaluation",
        "deployment",
        "monitoring and retraining",
        "quick example",
        "concept explanation",
        "key concepts",
        "detailed summary",
        "tutor tips",
    }:
        return True

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
    lines = _clean_lines(raw_text)
    lines = [x for x in lines if not _looks_like_section_label(x)]
    lines = _dedupe_keep_order(lines)
    lines = _drop_near_duplicates(lines)
    return lines


# -----------------------
# Core formatting blocks
# -----------------------

def _find_first(lines: list[str], keywords: tuple[str, ...], fallback: str = "") -> str:
    for s in lines:
        low = s.lower()
        if any(k in low for k in keywords):
            return s.strip()
    return fallback


def _tech_notation_block(topic: str) -> str:
    t = (topic or "").lower()
    if "unsupervised" in t:
        return (
            "- Dataset: **D = {xᵢ}** (no labels)\n"
            "- Goal: learn structure **g(x)** (clusters / embeddings)\n"
            "- Example: K-means minimizes within-cluster distance\n"
            "- Dimensionality reduction: **z = h(x)** (e.g., PCA)"
        )
    if "reinforcement" in t:
        return (
            "- State **s**, action **a**, reward **r**\n"
            "- Policy: **π(a|s)**\n"
            "- Objective: maximize expected return **E[Σ γᵗ rₜ]**"
        )
    if "overfitting" in t:
        return (
            "- Symptom: **Acc_train ↑**, **Acc_test ↓**\n"
            "- Generalization gap: **E_test − E_train** large\n"
            "- Fix: regularization / early stopping / more data"
        )

    return (
        "- Dataset: **D = {(xᵢ, yᵢ)}** (supervised)\n"
        "- Model: **ŷ = f(x; θ)**\n"
        "- Objective: **θ\* = argmin₍θ₎ Σ L(f(xᵢ; θ), yᵢ)**"
    )


def _keyword_pack(topic: str) -> str:
    t = (topic or "").lower()
    packs = {
        "machine learning": ["X/Y", "loss", "generalization", "overfitting", "train-test split"],
        "supervised learning": ["labels", "classification", "regression", "loss", "metrics"],
        "unsupervised learning": ["clustering", "PCA", "embeddings", "similarity", "no labels"],
        "deep learning": ["layers", "backprop", "parameters", "CNN", "representation learning"],
        "neural networks": ["weights", "bias", "activation", "gradient", "backprop"],
        "model evaluation": ["accuracy", "precision/recall", "F1", "ROC-AUC", "cross-validation"],
        "overfitting": ["generalization gap", "regularization", "early stopping", "data leakage"],
        "underfitting": ["high bias", "low capacity", "feature insufficiency"],
        "reinforcement learning": ["policy", "reward", "episode", "Q-learning", "exploration"],
        "bias vs variance": ["bias", "variance", "tradeoff", "regularization", "ensembling"],
    }
    words = packs.get(t, ["data", "model", "loss", "train", "test", "metrics"])
    return ", ".join([f"**{w}**" for w in words])


def _extract_key_concepts(lines: list[str], limit: int = 10) -> str:
    picks: list[str] = []
    keywords = ("definition", "goal", "features", "labels", "train", "test", "metric", "overfit", "underfit", "bias", "variance")
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
    t = (topic or "").lower()
    if "unsupervised" in t:
        return "- Customer segmentation: behavior vectors → clusters (no labels)"
    if "supervised" in t:
        return "- Spam filter: email text → spam / not spam (labeled)"
    if "overfitting" in t:
        return "- 98% train acc, 55% test acc → regularize / early stop / more data"
    return "- Pick dataset → define X/Y → train → evaluate on unseen test set"


# -----------------------
# Public API
# -----------------------

def format_as_gpt_style(topic: str, raw_text: str, attempt: int) -> str:
    lines = _clean_for_display(raw_text)

    if not lines:
        return f"## 🧠 Deep Dive: {topic}\n\nNo notes found for '{topic}'. Add it to `data/notes/machine_learning.txt`."

    definition = _find_first(
        lines,
        keywords=(" is ", " means ", " refers to ", " can be defined", " enables "),
        fallback=lines[0],
    )

    notation = _tech_notation_block(topic)
    keywords = _keyword_pack(topic)
    key_concepts = _extract_key_concepts(lines, limit=10)
    example = _mini_example(topic)

    return f"""
## 🧠 Deep Dive: {topic}

### Definition
- {definition}

### Notation / Core Formulation
{notation}

### Key Terms
{keywords}

### Key Concepts
{key_concepts}

### Quick Example
{example}
""".strip()
