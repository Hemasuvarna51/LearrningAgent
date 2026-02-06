# src/question_generator.py
# ✅ Hybrid MCQs (EXPLANATION + BANK)
# ✅ Returns exactly n (supports 5–10+)
# ✅ More explanation-based questions when n is bigger (ex: 10 => less repetitive)
# ✅ Stable shuffle (seeded RNG) — no global random.shuffle
# ✅ Avoids paragraph dumps / list blocks / pro tips / headings
# ✅ No duplicate question text inside one quiz

from __future__ import annotations

import hashlib
import random
import re
from typing import Callable


# ============================================================
# Stable RNG
# ============================================================

def _stable_seed(*parts: str) -> int:
    msg = "|".join([(p or "").strip().lower() for p in parts])
    digest = hashlib.sha256(msg.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def _rng(topic: str, attempt: int, explanation: str, salt: str = "checkpoint-agent-v6") -> random.Random:
    seed = _stable_seed(topic, str(attempt), (explanation or "")[:2000], salt)
    return random.Random(seed)


# ============================================================
# MCQ Builder (stable option shuffle)
# ============================================================

def _mcq(r: random.Random, question: str, correct: str, wrongs: list[str]) -> dict:
    wrongs = (wrongs or [])[:3]
    while len(wrongs) < 3:
        wrongs.append("This option is not supported by the explanation.")

    options = wrongs + [correct]
    r.shuffle(options)

    opt_map = {"A": options[0], "B": options[1], "C": options[2], "D": options[3]}
    ans = next(k for k, v in opt_map.items() if v == correct)
    return {"question": question, "options": opt_map, "answer": ans}


# ============================================================
# Explanation Parsing (safe candidates only)
# ============================================================

_STOP = {
    "the","a","an","and","or","to","of","in","for","on","with","is","are","that",
    "this","it","as","by","from","be","will","not","only","too"
}

def _strip_markdown(s: str) -> str:
    s = re.sub(r"(?m)^\s*#{1,6}\s*", "", s)
    s = re.sub(r"`+", "", s)
    s = re.sub(r"\*\*|\*", "", s)
    return s.strip()

def _is_bullet(line: str) -> bool:
    t = (line or "").lstrip()
    return t.startswith(("-", "•", "*"))

def _is_heading_like(line: str) -> bool:
    t = (line or "").strip()
    if not t:
        return True
    low = t.lower()

    # never turn these into options
    if "pro tip" in low or "tutor tip" in low or "exam tip" in low:
        return True
    if t.startswith(("💡", "📌", "🏁", "—")):
        return True

    # short section labels
    if len(t) <= 35 and t.endswith(":"):
        return True
    if len(t.split()) <= 3 and t.isupper():
        return True
    if low in {
        "core concepts", "core concepts:",
        "key concepts", "key concepts:",
        "learning pipeline", "learning pipeline:",
        "quick example", "quick example:",
        "concept explanation", "concept explanation:",
    }:
        return True

    return False

def _sanitize_option_text(s: str, max_len: int = 140) -> str:
    s = re.sub(r"\s+", " ", (s or "")).strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - 1].rstrip() + "…"


def _sentence_candidates(explanation: str) -> list[str]:
    """
    Extract clean candidate sentences without merging bullets into mega paragraphs.
    """
    raw = _strip_markdown(explanation or "")
    raw = raw.replace("\r\n", "\n")
    raw = re.sub(r"\n{3,}", "\n\n", raw).strip()

    lines = [ln.rstrip() for ln in raw.split("\n")]

    cleaned: list[str] = []
    for ln in lines:
        s = ln.strip()
        if not s:
            cleaned.append("")
            continue
        if s.startswith("[Image"):
            continue
        if _is_heading_like(s):
            continue
        if set(s) <= {"-", "="}:
            continue

        if _is_bullet(s):
            s = re.sub(r"^[-•*]\s*", "", s).strip()
            if s:
                cleaned.append(s)
            continue

        cleaned.append(s)

    text = "\n".join(cleaned)
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]

    candidates: list[str] = []
    for p in paras:
        # skip mega list blocks
        if p.count("\n") >= 4:
            continue

        p = p.replace("\n", " ")
        sents = re.split(r"(?<=[.!?])\s+", p)

        for s in sents:
            s = s.strip()
            if not s:
                continue

            # avoid paragraph dumps
            if len(s.split()) > 26:
                continue
            if s.count(":") >= 2:
                continue

            # must be meaningful
            if len(s) >= 45:
                candidates.append(s)

    return candidates


def _keywords_for_topic(topic: str) -> list[str]:
    return {
        "machine learning": ["data", "model", "training", "testing", "generalization", "feedback", "probabilistic", "concept drift", "evaluation"],
        "deep learning": ["neural", "layers", "features", "gpu", "cnn", "rnn", "parameters"],
        "neural networks": ["weights", "bias", "activation", "backprop", "gradient", "layers"],
        "supervised learning": ["labeled", "classification", "regression", "loss", "ground truth", "train"],
        "unsupervised learning": ["unlabeled", "clustering", "pca", "dimensionality", "structure"],
        "reinforcement learning": ["reward", "agent", "environment", "policy", "exploration", "episode"],
        "overfitting": ["training", "test", "generalization", "memorization", "regularization"],
        "underfitting": ["bias", "simple", "capacity", "features"],
        "model evaluation": ["metrics", "accuracy", "precision", "recall", "f1", "validation", "test"],
        "bias vs variance": ["bias", "variance", "overfitting", "underfitting", "tradeoff"],
    }.get(topic, [])


def _pick_sentences(explanation: str, topic: str, r: random.Random, k: int) -> list[str]:
    sents = _sentence_candidates(explanation)
    if not sents:
        return []

    kws = _keywords_for_topic(topic)
    scored = []
    for s in sents:
        low = s.lower()
        score = sum(1 for kw in kws if kw in low)
        if any(x in low for x in [" is ", " means ", " refers to ", " occurs when "]):
            score += 2
        scored.append((score, s))

    scored.sort(key=lambda x: x[0], reverse=True)

    # take a larger pool so we can sample variety
    pool = [s for _, s in scored[: max(k * 6, 14)]]
    r.shuffle(pool)
    return pool[:k]


# ============================================================
# Explanation-based MCQs
# ============================================================

def _explain_mcq(r: random.Random, sentence: str) -> dict:
    q = "According to the explanation, which statement is correct?"
    correct = _sanitize_option_text(sentence, 140)
    wrongs = [
        "It suggests evaluating only on training data is sufficient.",
        "It implies learning happens without data or feedback.",
        "It claims models always produce perfect predictions.",
    ]
    return _mcq(r, q, correct, wrongs)

def _cloze_mcq(r: random.Random, sentence: str) -> dict:
    words = re.findall(r"[a-zA-Z]{5,}", sentence)
    target = None
    for w in words:
        if w.lower() not in _STOP:
            target = w
            break

    if not target:
        return _explain_mcq(r, sentence)

    blanked = re.sub(rf"\b{re.escape(target)}\b", "_____", sentence, count=1)
    blanked = _sanitize_option_text(blanked, 160)

    return _mcq(
        r,
        f"Fill in the blank based on the explanation:\n\n{blanked}",
        target,
        ["accuracy", "database", "internet"]
    )

def _explanation_mcqs(explanation: str, topic: str, attempt: int, count: int) -> list[dict]:
    r = _rng(topic, attempt, explanation, salt="exp-mcqs-v1")
    chosen = _pick_sentences(explanation, topic, r, count)

    out = []
    for i, s in enumerate(chosen):
        out.append(_explain_mcq(r, s) if i % 2 == 0 else _cloze_mcq(r, s))
    return out


# ============================================================
# Professional Topic Banks (expanded)
# Each template takes (rng) -> mcq
# ============================================================

TemplateFn = Callable[[random.Random], dict]

def _bank_more_ml() -> list[TemplateFn]:
    return [
        lambda r: _mcq(r, "What is the role of features (X) in ML?",
            "They are input variables used to make predictions.",
            ["They are always the labels.", "They replace the model.", "They are only used during testing."]),
        lambda r: _mcq(r, "What is the purpose of labels (Y) in supervised ML?",
            "They represent the correct output the model learns to predict.",
            ["They are the same as features.", "They are created only after deployment.", "They are optional for all ML tasks."]),
        lambda r: _mcq(r, "What is concept drift?",
            "When real-world data patterns change over time, reducing model performance.",
            ["When the model always improves without retraining.", "When training data becomes perfectly clean.", "When the model stops using features."]),
        lambda r: _mcq(r, "Why do ML models need monitoring after deployment?",
            "Because data can shift and performance can degrade over time.",
            ["Because training guarantees lifetime accuracy.", "Because monitoring replaces evaluation.", "Because models cannot run without monitoring."]),
        lambda r: _mcq(r, "Which statement about ML is most accurate?",
            "ML systems are probabilistic and aim to be correct most of the time, not always.",
            ["ML guarantees perfect correctness.", "ML avoids evaluation.", "ML never makes mistakes."]),
        lambda r: _mcq(r, "Why do we split data into train and test sets?",
            "To estimate performance on unseen data and avoid overfitting.",
            ["To increase training accuracy only.", "To avoid collecting data.", "To remove the need for metrics."]),
        lambda r: _mcq(r, "What is overfitting in simple terms?",
            "When a model memorizes training data and fails on new data.",
            ["When a model is too simple.", "When a model has low bias.", "When test accuracy is higher than training."]),
    ]


TEMPLATES: dict[str, list[TemplateFn]] = {
    "machine learning": [
        lambda r: _mcq(r, "What is the primary goal of machine learning?",
            "To learn patterns from data and perform well on unseen inputs.",
            ["To always produce perfect results.", "To avoid using training data.", "To replace evaluation entirely."]),
        lambda r: _mcq(r, "What does generalization mean in machine learning?",
            "The ability to perform well on new, unseen data.",
            ["Perfect training accuracy.", "Running faster on GPUs.", "Memorizing the dataset."]),
        lambda r: _mcq(r, "Which practice best supports reliable evaluation?",
            "Using a train/validation/test split on unseen data.",
            ["Evaluating only on training data.", "Avoiding metrics.", "Increasing complexity blindly."]),
        lambda r: _mcq(r, "Why is a feedback loop important in ML?",
            "It enables iterative improvement by measuring errors and updating the model.",
            ["It guarantees 100% accuracy.", "It removes the need for data.", "It replaces evaluation completely."]),
        lambda r: _mcq(r, "When is ML better than rule-based programming?",
            "When rules are too complex, patterns change, and data is available.",
            ["When rules never change and are easy to write.", "When no data exists at all.", "When outputs must be fixed formulas."]),
        *(_bank_more_ml()),
    ],
}


# ============================================================
# Dedupe (by question text only)
# ============================================================

def _dedupe_by_question(qs: list[dict]) -> list[dict]:
    seen = set()
    out = []
    for q in qs:
        k = (q.get("question") or "").strip().lower()
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(q)
    return out


# ============================================================
# MAIN GENERATOR
# ============================================================

def generate_mcqs_from_explanation(explanation: str, topic: str, attempt: int, n: int = 6) -> list[dict]:
    n = int(n)
    topic_key = (topic or "").strip().lower()
    bank = TEMPLATES.get(topic_key, TEMPLATES.get("machine learning", []))

    r = _rng(topic_key, attempt, explanation or "", salt="main-gen-v1")

    # ✅ More explanation-based share as n grows (reduces repetition)
    # 5 -> 3 exp, 6 -> 3 exp, 7 -> 4 exp, 8 -> 4 exp, 9 -> 5 exp, 10 -> 6 exp
    exp_n = max(3, (n + 1) // 2)
    exp_n = min(exp_n, n)  # safety

    qs: list[dict] = []

    # 1) explanation-driven
    qs.extend(_explanation_mcqs(explanation or "", topic_key, attempt, exp_n))
    qs = _dedupe_by_question(qs)

    # 2) bank-driven (fill)
    if bank:
        start = (max(1, attempt) - 1) % len(bank)
        rotated = bank[start:] + bank[:start]
    else:
        rotated = []

    for fn in rotated:
        if len(qs) >= n:
            break
        qs.append(fn(r))
        qs = _dedupe_by_question(qs)

    # 3) last resort: pull extra explanation sentences (different salt via attempt bump)
    guard = 0
    while len(qs) < n and guard < 40:
        extra = _explanation_mcqs(explanation or "", topic_key, attempt + guard + 1, 2)
        qs.extend(extra)
        qs = _dedupe_by_question(qs)
        guard += 1

    # stable shuffle
    r.shuffle(qs)

    # hard guarantee length
    guard = 0
    while len(qs) < n and guard < 200 and rotated:
        q = rotated[guard % len(rotated)](r)
        if (q.get("question") or "").strip().lower() not in {x["question"].strip().lower() for x in qs}:
            qs.append(q)
        guard += 1

    return qs[:n]
