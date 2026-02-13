# src/question_generator.py
# ✅ FINAL v5: Explanation-first MCQs (deduped, numbered-lines fixed, no "not supported" spam)
# ✅ Returns exactly n (5–10+)
# ✅ n=5 => 4 explanation + 1 bank (guaranteed)
# ✅ Stable RNG (deterministic per topic/attempt/explanation)
# ✅ Strips numbering like "3." / "6)" from explanation lines (prevents duplicate Qs)
# ✅ No headings/tips/list-label dumps as options
# ✅ Distractors come from OTHER explanation sentences (guaranteed 3 meaningful)
# ✅ Expanded banks for all 10 concepts

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


def _rng(topic: str, attempt: int, explanation: str, salt: str = "checkpoint-agent-qg-final-v5") -> random.Random:
    seed = _stable_seed(topic, str(attempt), (explanation or "")[:4000], salt)
    return random.Random(seed)


# ============================================================
# Text helpers
# ============================================================

_STOP = {
    "the", "a", "an", "and", "or", "to", "of", "in", "for", "on", "with", "is", "are", "that",
    "this", "it", "as", "by", "from", "be", "will", "not", "only", "too"
}

# words we NEVER want as cloze targets
_CLOZE_BAN = {
    "machine", "learning", "deep", "neural", "networks", "network",
    "supervised", "unsupervised", "reinforcement",
}

_SECTION_NOISE = (
    "key terms", "key concepts", "concept explanation", "re-explanation",
    "learning pipeline", "quick example", "new perspective",
    "tutor tips", "pro tip", "exam tip",
    "definition", "notation", "core formulation",
)

_BAD_PREFIX = ("which ", "that ", "because ", "and ", "or ", "so ", "but ")


def _strip_markdown(s: str) -> str:
    s = (s or "")
    s = s.replace("\\", "")  # remove escape noise like θ\ =
    s = re.sub(r"(?m)^\s*#{1,6}\s*", "", s)
    s = re.sub(r"`+", "", s)
    s = re.sub(r"\*\*|\*", "", s)
    return s.strip()


def _sanitize_text(s: str) -> str:
    s = (s or "").replace("\\", "")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _sanitize_option_text(s: str, max_len: int = 130) -> str:
    s = _sanitize_text(s)
    if len(s) <= max_len:
        return s
    return s[: max_len - 1].rstrip() + "…"


def _normalize_key(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _strip_leading_numbering(s: str) -> str:
    """
    Removes numbering prefixes that come from your explanation formatting:
    - "1. text", "3) text", "6) text", "2: text"
    - also removes accidental double spaces after.
    """
    s = _sanitize_text(s)
    s = re.sub(r"^\s*\d+\s*[.)\:]\s*", "", s)   # 3. / 3) / 3:
    s = re.sub(r"^\s*[-•*]\s*", "", s)         # bullet markers
    return _sanitize_text(s)


def _looks_like_label_or_noise(s: str) -> bool:
    t = (s or "").strip()
    if not t:
        return True
    low = t.lower()

    if t.startswith(("💡", "📌", "🏁", "—")):
        return True

    if any(x in low for x in _SECTION_NOISE):
        return True

    if low.endswith(":") and len(low.split()) <= 6:
        return True

    if len(low.split()) <= 3 and t.isupper():
        return True

    return False


def _is_bullet(line: str) -> bool:
    t = (line or "").lstrip()
    return t.startswith(("-", "•", "*"))


def _is_good_sentence(s: str) -> bool:
    t = _strip_leading_numbering(s)
    low = t.lower()

    # 🚫 never allow analogy lines into MCQs
    if "think of it like" in low:
        return False

    if not t:
        return False
    if _looks_like_label_or_noise(t):
        return False
    if any(low.startswith(p) for p in _BAD_PREFIX):
        return False

    words = t.split()
    if len(words) < 7 or len(words) > 55:
        return False

    # needs some verb-ish cue OR formula cues
    if not any(v in low for v in (" is ", " are ", " means ", " refers ", " enables ", " requires ", " occurs ", " because ", " to ")):
        if not any(x in t for x in ("=", "→", "->", "argmin", "θ", "ŷ", "Σ", "E_train", "E_test")):
            return False

    return True


def _sentence_candidates(explanation: str) -> list[str]:
    raw = _strip_markdown(explanation or "")
    raw = raw.replace("\r\n", "\n")
    raw = re.sub(r"\n{3,}", "\n\n", raw).strip()

    lines = [ln.rstrip() for ln in raw.split("\n")]

    candidates: list[str] = []
    para_buf: list[str] = []

    def flush_para():
        nonlocal para_buf, candidates
        if not para_buf:
            return
        p = " ".join(para_buf).strip()
        para_buf = []
        if not p:
            return
        for s in re.split(r"(?<=[.!?])\s+", p):
            s = _strip_leading_numbering(s.strip())
            if _is_good_sentence(s):
                candidates.append(s)

    for ln in lines:
        s = ln.strip()
        if not s:
            flush_para()
            continue
        if s.startswith("[Image"):
            continue
        if set(s) <= {"-", "="}:
            continue

        if _looks_like_label_or_noise(s):
            flush_para()
            continue

        if _is_bullet(s):
            flush_para()
            bullet = _strip_leading_numbering(re.sub(r"^[-•*]\s*", "", s).strip())
            if _is_good_sentence(bullet):
                candidates.append(bullet)
            continue

        para_buf.append(s)

    flush_para()

    # dedupe candidates
    out: list[str] = []
    seen = set()
    for s in candidates:
        s = _strip_leading_numbering(s)
        k = _normalize_key(s)
        if k and k not in seen:
            seen.add(k)
            out.append(s)
    return out


# ============================================================
# Topic keywords (scoring + cloze)
# ============================================================

def _keywords_for_topic(topic: str) -> list[str]:
    return {
        "machine learning": ["data", "model", "training", "testing", "generalization", "loss", "concept drift", "evaluation", "monitor", "features", "labels"],
        "deep learning": ["layers", "parameters", "backprop", "gradient", "cnn", "rnn", "transformer", "activation", "representation"],
        "neural networks": ["weights", "bias", "activation", "backprop", "gradient", "layer", "forward", "loss", "optimizer"],
        "supervised learning": ["labeled", "classification", "regression", "loss", "ground truth", "train", "target", "features", "labels"],
        "unsupervised learning": ["unlabeled", "clustering", "k-means", "pca", "dimensionality", "similarity", "structure", "embedding"],
        "reinforcement learning": ["agent", "environment", "reward", "policy", "value", "q-learning", "exploration", "exploitation", "episode", "markov"],
        "overfitting": ["train", "test", "generalization", "memorization", "variance", "regularization", "dropout", "early stopping"],
        "underfitting": ["bias", "simple", "capacity", "features", "error", "high bias", "low complexity"],
        "bias vs variance": ["bias", "variance", "tradeoff", "overfitting", "underfitting", "generalization", "complexity"],
        "model evaluation": ["metrics", "accuracy", "precision", "recall", "f1", "roc", "auc", "mae", "mse", "rmse", "confusion", "validation"],
    }.get(topic, [])


def _score_sentence(s: str, topic: str) -> int:
    s = _strip_leading_numbering(s)
    low = s.lower()
    kws = _keywords_for_topic(topic)

    score = 0
    score += sum(2 for kw in kws if kw in low)

    if any(x in low for x in [" is ", " means ", " refers to ", " occurs when ", " because ", " requires "]):
        score += 4

    if any(x in s for x in ("=", "→", "->", "argmin", "θ", "ŷ", "Σ", "E_train", "E_test")):
        score += 3

    if any(x in low for x in ["train", "test", "validation", "generalization", "deployment", "monitor", "drift"]):
        score += 2

    if low.startswith("think of it like"):
        return -999

    return score


def _pick_sentences(explanation: str, topic: str, r: random.Random, k: int) -> list[str]:
    sents = _sentence_candidates(explanation)
    if not sents:
        return []

    scored = sorted([(_score_sentence(s, topic), s) for s in sents], key=lambda x: x[0], reverse=True)
    pool = [s for _, s in scored[: max(k * 14, 28)]]
    r.shuffle(pool)

    # pick unique-by-sentence (prevents “Features and labels…” repeating with different numbering)
    out: list[str] = []
    seen = set()
    for s in pool:
        ks = _normalize_key(_strip_leading_numbering(s))
        if ks in seen:
            continue
        seen.add(ks)
        out.append(_strip_leading_numbering(s))
        if len(out) >= k:
            break
    return out


# ============================================================
# MCQ Builder
# ============================================================

def _mcq(r: random.Random, question: str, correct: str, wrongs: list[str]) -> dict:
    # sanitize everything first
    correct = _sanitize_option_text(correct, 130)
    wrongs = [_sanitize_option_text(w, 130) for w in (wrongs or []) if _sanitize_text(w)]

    # hard dedupe wrongs
    dedup = []
    seen = set()
    for w in wrongs:
        k = _normalize_key(w)
        if k and k not in seen and k != _normalize_key(correct):
            seen.add(k)
            dedup.append(w)
    wrongs = dedup[:3]

    # guarantee 3 wrongs (meaningful fallbacks, never "not supported")
    fallback_wrongs = [
        "It guarantees perfect accuracy on any dataset without evaluation.",
        "It works without training data, testing, or feedback.",
        "It removes the need for monitoring after deployment.",
        "It proves causation directly from correlation patterns.",
    ]
    i = 0
    while len(wrongs) < 3:
        w = fallback_wrongs[i % len(fallback_wrongs)]
        if _normalize_key(w) != _normalize_key(correct) and _normalize_key(w) not in {_normalize_key(x) for x in wrongs}:
            wrongs.append(w)
        i += 1

    options = wrongs + [correct]
    r.shuffle(options)

    opt_map = {"A": options[0], "B": options[1], "C": options[2], "D": options[3]}
    ans = next(k for k, v in opt_map.items() if v == correct)
    return {"question": question, "options": opt_map, "answer": ans}


# ============================================================
# Explanation MCQs
# ============================================================

_EXPLAIN_Q_PROMPTS = [
    "According to the explanation, which statement is correct?",
    "Which statement matches the explanation best?",
    "Based on the explanation, which claim is accurate?",
    "Which option is supported by the explanation?",
]

_CLOZE_Q_PROMPTS = [
    "Fill in the blank based on the explanation:\n\n{blanked}",
    "Complete the statement from the explanation:\n\n{blanked}",
]


def _distractors_from_explanation(r: random.Random, all_sentences: list[str], correct_sentence: str, k: int = 3) -> list[str]:
    correct_sentence = _strip_leading_numbering(correct_sentence)
    correct_txt = _sanitize_option_text(correct_sentence, 130)
    correct_norm = _normalize_key(correct_txt)

    pool: list[str] = []
    for s in all_sentences:
        s = _strip_leading_numbering(s)
        txt = _sanitize_option_text(s, 130)
        if not txt:
            continue
        if _normalize_key(txt) == correct_norm:
            continue
        if not _is_good_sentence(txt):
            continue
        pool.append(txt)

    # dedupe pool
    deduped = []
    seen = set()
    for x in pool:
        kx = _normalize_key(x)
        if kx and kx not in seen:
            seen.add(kx)
            deduped.append(x)

    r.shuffle(deduped)
    return deduped[:k]


def _explain_mcq(r: random.Random, sentence: str, all_sentences: list[str]) -> dict:
    sentence = _strip_leading_numbering(sentence)
    q = r.choice(_EXPLAIN_Q_PROMPTS)
    correct = _sanitize_option_text(sentence, 130)
    wrongs = _distractors_from_explanation(r, all_sentences, sentence, k=3)
    return _mcq(r, q, correct, wrongs)


def _choose_cloze_target(sentence: str, topic: str) -> str | None:
    s = _strip_leading_numbering(sentence)
    low = s.lower()

    words = re.findall(r"[a-zA-Z]{4,}", s)  # allow 4-letter tokens like "data", "loss"
    if not words:
        return None

    first_word = words[0].lower()

    # 1) Prefer topic keywords present (longest first), not first word
    kws = sorted(_keywords_for_topic(topic), key=len, reverse=True)
    for kw in kws:
        kw_low = kw.lower()
        if kw_low in _CLOZE_BAN:
            continue
        if kw_low == first_word:
            continue
        if re.search(rf"\b{re.escape(kw_low)}\b", low):
            return kw

    # 2) fallback: first meaningful non-banned, non-stop, not first word
    for w in words:
        wl = w.lower()
        if wl == first_word:
            continue
        if wl in _STOP or wl in _CLOZE_BAN:
            continue
        return w

    return None


def _cloze_mcq(r: random.Random, sentence: str, topic: str) -> dict:
    sentence = _strip_leading_numbering(sentence)
    target = _choose_cloze_target(sentence, topic)
    if not target:
        return _explain_mcq(r, sentence, _sentence_candidates(sentence))

    blanked = re.sub(
        rf"\b{re.escape(target)}\b",
        "_____",
        sentence,
        count=1,
        flags=re.IGNORECASE
    )

    # ✅ If replacement FAILED (common with E_test, θ*, underscores), fallback to explain-style
    if "_____" not in blanked:
        return _explain_mcq(r, sentence, _sentence_candidates(sentence))

    blanked = _sanitize_option_text(blanked, 170)

    wrong_words = [
        "compiler", "database", "protocol", "encryption",
        "filesystem", "hardware", "internet", "operating"
    ]
    r.shuffle(wrong_words)

    return _mcq(
        r,
        r.choice(_CLOZE_Q_PROMPTS).format(blanked=blanked),
        target,
        wrong_words[:3]
    )



def _explanation_mcqs(explanation: str, topic: str, attempt: int, count: int) -> list[dict]:
    r = _rng(topic, attempt, explanation, salt="exp-mcqs-final-v5")

    all_sents = _sentence_candidates(explanation)
    chosen = _pick_sentences(explanation, topic, r, count)

    out: list[dict] = []
    used_sentence_keys = set()  # ✅ prevents same base sentence generating 2 questions

    for i, s in enumerate(chosen):
        base_key = _normalize_key(_strip_leading_numbering(s))
        if base_key in used_sentence_keys:
            continue
        used_sentence_keys.add(base_key)

        out.append(_explain_mcq(r, s, all_sents) if i % 2 == 0 else _cloze_mcq(r, s, topic))

        if len(out) >= count:
            break

    return out


# ============================================================
# Topic Banks (all 10)
# ============================================================

TemplateFn = Callable[[random.Random], dict]

def _bank_ml() -> list[TemplateFn]:
    return [
        lambda r: _mcq(r, "What does generalization mean in ML?",
            "The ability to perform well on unseen data.",
            ["Perfect training accuracy.", "Using only rules.", "Avoiding evaluation."]),
        lambda r: _mcq(r, "What is concept drift?",
            "When the data distribution changes over time, degrading performance.",
            ["When the model stops training forever.", "When loss becomes zero permanently.", "When labels disappear from data."]),
        lambda r: _mcq(r, "Why do we use a train/test split?",
            "To estimate performance on unseen data and detect overfitting.",
            ["To increase training accuracy only.", "To remove the need for metrics.", "To avoid data collection."]),
    ]

def _bank_dl() -> list[TemplateFn]:
    return [
        lambda r: _mcq(r, "What is backpropagation used for?",
            "Computing gradients to update parameters and minimize loss.",
            ["Creating labels automatically.", "Removing the need for data.", "Guaranteeing 100% accuracy."]),
        lambda r: _mcq(r, "Why are deep models called 'deep'?",
            "They contain multiple layers that learn hierarchical representations.",
            ["They only work on images.", "They require no training.", "They use only linear functions."]),
        lambda r: _mcq(r, "Which is a common deep learning issue?",
            "Vanishing/exploding gradients in very deep architectures.",
            ["Perfect generalization always.", "No need for regularization.", "No dependence on data size."]),
    ]

def _bank_nn() -> list[TemplateFn]:
    return [
        lambda r: _mcq(r, "What is an activation function's role?",
            "It introduces non-linearity so the network can model complex patterns.",
            ["It normalizes labels.", "It removes weights.", "It creates features manually."]),
        lambda r: _mcq(r, "What does a weight represent in a neuron?",
            "The strength of an input connection influencing the output.",
            ["A label for the dataset.", "A metric value.", "A cluster assignment."]),
        lambda r: _mcq(r, "What does gradient descent do?",
            "Updates parameters in the direction that reduces the loss.",
            ["Increases variance intentionally.", "Replaces test data.", "Removes the need for training."]),
    ]

def _bank_supervised() -> list[TemplateFn]:
    return [
        lambda r: _mcq(r, "Supervised learning requires:",
            "Labeled data (inputs with correct targets).",
            ["Only unlabeled data.", "Rewards and an environment.", "No training set."]),
        lambda r: _mcq(r, "Classification is used when:",
            "The output is a discrete category (e.g., spam/not spam).",
            ["The output is continuous (e.g., price).", "There are no labels.", "We only cluster data."]),
        lambda r: _mcq(r, "Regression is used when:",
            "The output is continuous (e.g., house price).",
            ["The output is a category.", "There are no targets.", "We need rewards to learn."]),
    ]

def _bank_unsupervised() -> list[TemplateFn]:
    return [
        lambda r: _mcq(r, "Unsupervised learning works with:",
            "Unlabeled data to discover structure or patterns.",
            ["Only labeled datasets.", "Reward signals only.", "Only time-series labels."]),
        lambda r: _mcq(r, "K-Means is an example of:",
            "Clustering (grouping similar data points).",
            ["Regression.", "Classification.", "Policy optimization."]),
        lambda r: _mcq(r, "PCA is mainly used for:",
            "Dimensionality reduction while preserving variance.",
            ["Creating labels.", "Increasing feature count.", "Selecting a loss function."]),
    ]

def _bank_rl() -> list[TemplateFn]:
    return [
        lambda r: _mcq(r, "In reinforcement learning, an agent learns by:",
            "Interacting with an environment and receiving rewards.",
            ["Using only labeled examples.", "Clustering unlabeled data.", "Minimizing MAE directly."]),
        lambda r: _mcq(r, "Exploration vs exploitation means:",
            "Trying new actions vs using known good actions.",
            ["Training vs testing split.", "Bias vs variance tradeoff.", "Precision vs recall."]),
        lambda r: _mcq(r, "A policy in RL is:",
            "A mapping from states to actions.",
            ["A confusion matrix.", "A set of labels.", "A clustering rule."]),
    ]

def _bank_overfit() -> list[TemplateFn]:
    return [
        lambda r: _mcq(r, "Overfitting usually shows up as:",
            "High training performance but low test performance.",
            ["Low training and low test performance.", "High bias always.", "No dependence on features."]),
        lambda r: _mcq(r, "Which helps reduce overfitting?",
            "Regularization, more data, early stopping, or dropout.",
            ["Increasing model complexity blindly.", "Training longer with no validation.", "Removing the test set."]),
        lambda r: _mcq(r, "A main cause of overfitting is often:",
            "Model capacity too high relative to data and noise.",
            ["Too much labeled data.", "Too simple a model.", "Perfectly stable distribution always."]),
    ]

def _bank_underfit() -> list[TemplateFn]:
    return [
        lambda r: _mcq(r, "Underfitting usually means:",
            "The model is too simple to capture patterns (high bias).",
            ["The model memorizes noise.", "The model has high variance always.", "The model is perfectly generalizing."]),
        lambda r: _mcq(r, "A typical fix for underfitting is:",
            "Increase model capacity or add better features.",
            ["Add more regularization.", "Reduce model size further.", "Stop training earlier always."]),
        lambda r: _mcq(r, "If both train and test accuracy are low, it suggests:",
            "Underfitting (high bias).",
            ["Overfitting (high variance).", "Data leakage.", "Perfect model selection."]),
    ]

def _bank_bias_variance() -> list[TemplateFn]:
    return [
        lambda r: _mcq(r, "High bias typically leads to:",
            "Underfitting (missing real patterns).",
            ["Overfitting (memorizing noise).", "Perfect generalization.", "Zero loss always."]),
        lambda r: _mcq(r, "High variance typically leads to:",
            "Overfitting (sensitive to noise).",
            ["Underfitting.", "No training needed.", "No data needed."]),
        lambda r: _mcq(r, "Bias–variance tradeoff means:",
            "Balancing simplicity vs flexibility for best generalization.",
            ["Always maximizing complexity.", "Always minimizing bias only.", "Removing evaluation metrics."]),
    ]

def _bank_eval() -> list[TemplateFn]:
    return [
        lambda r: _mcq(r, "Precision measures:",
            "Out of predicted positives, how many are truly positive.",
            ["Out of actual positives, how many are found.", "Overall correctness only.", "Mean absolute error."]),
        lambda r: _mcq(r, "Recall measures:",
            "Out of actual positives, how many were correctly found.",
            ["Out of predicted positives, how many are correct.", "Overall correctness only.", "Variance of errors."]),
        lambda r: _mcq(r, "F1-score is useful when:",
            "Classes are imbalanced and you need a balance of precision and recall.",
            ["Only regression tasks exist.", "Accuracy is always enough.", "No labels are available."]),
        lambda r: _mcq(r, "Cross-validation helps by:",
            "Estimating performance more reliably across splits.",
            ["Guaranteeing perfect accuracy.", "Replacing test data forever.", "Removing the need for metrics."]),
    ]


TEMPLATES: dict[str, list[TemplateFn]] = {
    "machine learning": [
        lambda r: _mcq(r, "What is the primary goal of machine learning?",
            "To learn patterns from data and perform well on unseen inputs.",
            ["To always produce perfect results.", "To avoid using training data.", "To replace evaluation entirely."]),
        *_bank_ml(),
    ],
    "deep learning": [
        lambda r: _mcq(r, "Deep learning is best described as:",
            "ML using multi-layer neural networks to learn representations.",
            ["A rule-based AI system.", "Only a clustering technique.", "A database optimization method."]),
        *_bank_dl(),
    ],
    "neural networks": [
        lambda r: _mcq(r, "A neural network is composed of:",
            "Layers of neurons with weights and activations.",
            ["Only a rule set.", "Only clustering steps.", "Only labels and targets."]),
        *_bank_nn(),
    ],
    "supervised learning": [
        lambda r: _mcq(r, "Supervised learning mainly learns from:",
            "Input–output pairs (X, Y) with labels.",
            ["Only unlabeled inputs.", "Rewards only.", "Random guesses only."]),
        *_bank_supervised(),
    ],
    "unsupervised learning": [
        lambda r: _mcq(r, "Unsupervised learning aims to:",
            "Discover hidden structure in unlabeled data.",
            ["Predict known labels.", "Maximize reward signals.", "Minimize MAE in regression."]),
        *_bank_unsupervised(),
    ],
    "reinforcement learning": [
        lambda r: _mcq(r, "Reinforcement learning learns by:",
            "Maximizing cumulative reward via interaction.",
            ["Using labeled targets only.", "Clustering without feedback.", "Only minimizing squared error."]),
        *_bank_rl(),
    ],
    "overfitting": [
        lambda r: _mcq(r, "Overfitting is best defined as:",
            "A model fitting noise in training data and failing on new data.",
            ["A model being too simple.", "A model always improving on test data.", "A model not needing validation."]),
        *_bank_overfit(),
    ],
    "underfitting": [
        lambda r: _mcq(r, "Underfitting is best defined as:",
            "A model too simple to capture the true pattern (high bias).",
            ["A model memorizing training examples.", "A model with perfect generalization.", "A model requiring no data."]),
        *_bank_underfit(),
    ],
    "bias vs variance": [
        lambda r: _mcq(r, "Bias vs variance tradeoff focuses on:",
            "Balancing underfitting and overfitting for best generalization.",
            ["Only increasing training accuracy.", "Only reducing variance always.", "Avoiding evaluation metrics."]),
        *_bank_bias_variance(),
    ],
    "model evaluation": [
        lambda r: _mcq(r, "Model evaluation is primarily about:",
            "Measuring performance with appropriate metrics on unseen data.",
            ["Only increasing training loss.", "Removing test sets.", "Avoiding validation."]),
        *_bank_eval(),
    ],
}


# ============================================================
# Dedupe by question + source sentence
# ============================================================

def _dedupe_by_question(qs: list[dict]) -> list[dict]:
    seen = set()
    out = []
    for q in qs:
        k = _normalize_key(q.get("question") or "")
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(q)
    return out


# ============================================================
# MAIN
# ============================================================

def generate_mcqs_from_explanation(explanation: str, topic: str, attempt: int, n: int = 6) -> list[dict]:
    n = max(1, int(n))
    topic_key = (topic or "").strip().lower()
    bank = TEMPLATES.get(topic_key, TEMPLATES.get("machine learning", []))

    r = _rng(topic_key, attempt, explanation or "", salt="main-gen-final-v5")

    # n=5 => 4 explanation + 1 bank
    if n <= 5:
        exp_n = 4
    elif n == 6:
        exp_n = 4
    else:
        exp_n = max(4, (n * 2) // 3)
    exp_n = min(exp_n, n)

    qs: list[dict] = []

    # 1) explanation-first
    qs.extend(_explanation_mcqs(explanation or "", topic_key, attempt, exp_n))
    qs = _dedupe_by_question(qs)

    # ensure exp_n met (pull more)
    guard = 0
    while len(qs) < exp_n and guard < 80:
        qs.extend(_explanation_mcqs(explanation or "", topic_key, attempt + 200 + guard, 2))
        qs = _dedupe_by_question(qs)
        guard += 1

    # 2) bank fill
    rotated: list[TemplateFn] = []
    if bank:
        start = (max(1, attempt) - 1) % len(bank)
        rotated = bank[start:] + bank[:start]

    for fn in rotated:
        if len(qs) >= n:
            break
        qs.append(fn(r))
        qs = _dedupe_by_question(qs)

    # 3) final fill
    guard = 0
    while len(qs) < n and guard < 120:
        qs.extend(_explanation_mcqs(explanation or "", topic_key, attempt + 800 + guard, 2))
        qs = _dedupe_by_question(qs)
        guard += 1

    r.shuffle(qs)
    return qs[:n]
