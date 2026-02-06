import re

def _normalize(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"\[image[^\]]*\]", " ", text)
    text = re.sub(r"[-=]{3,}", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def calculate_relevance_score(context: str, concept: str, attempt: int = 1) -> dict:
    c = _normalize(context)
    concept_norm = _normalize(concept)

    if not c or len(c.split()) < 25:
        return {"score": 0, "feedback": ["Context is too shallow for effective learning."]}

    score = 0
    feedback = []

    # 1) Concept Mention
    concept_tokens = concept_norm.split()
    concept_hit = (concept_norm in c) or any(tok in c for tok in concept_tokens)
    if concept_hit:
        score += 20 if attempt >= 2 else 30
    else:
        feedback.append(f"The term '{concept}' is missing from the core text.")

    # 2) Keyword Hits
    core_keywords = {
        "deep learning": ["neural network", "layers", "hidden", "cnn", "rnn", "gpu", "feature learning", "abstraction"],
        "machine learning": ["data", "model", "patterns", "training", "testing", "features", "generalization"],
        "supervised learning": ["labeled", "ground truth", "classification", "regression", "target"],
        "unsupervised learning": ["unlabeled", "clustering", "pca", "dimensionality", "association"],
        "overfitting": ["high variance", "memorization", "training error", "test error", "regularization", "dropout"],
        "underfitting": ["high bias", "too simple", "complexity", "feature engineering"],
    }

    keywords = core_keywords.get(concept_norm, [])
    hits = [k for k in keywords if k in c]

    if attempt == 1:
        score += min(len(hits) * 8, 40)
    else:
        score += min(len(hits) * 12, 40)

    # 3) Misconception Detection
    if "overfitting" in concept_norm and "too simple" in c:
        feedback.append("⚠️ Confusion Alert: You might be describing Underfitting (too simple) instead of Overfitting.")
        score -= 10
    if "supervised" in concept_norm and "unlabeled" in c:
        feedback.append("⚠️ Confusion Alert: Supervised learning requires 'labeled' data, not 'unlabeled'.")
        score -= 10

    # 4) Attempt-Based Depth Assessment
    wc = len(c.split())
    if attempt >= 2 and wc < 60:
        feedback.append("The explanation is getting shorter. Try to include a specific use-case.")

    if wc >= 100:
        score += 20
    elif wc >= 50:
        score += 10

    # ✅ NEW: 5) Quality boosts (makes score move when explanation improves)
    quality = 0

    # examples
    if ("example" in c) or ("for example" in c) or ("e.g." in c):
        quality += 5

    # evaluation / splits
    if ("train/test" in c) or ("test set" in c) or ("validation" in c) or ("holdout" in c):
        quality += 5

    # metrics
    if any(m in c for m in ["accuracy", "precision", "recall", "f1", "mae", "mse", "rmse"]):
        quality += 5

    # structured explanation
    if any(x in c for x in ["step-by-step", "steps", "1)", "2)", "3)"]):
        quality += 5

    score += min(quality, 15)

    # 6) Visual Support Check
    if "[image" in (context or "").lower():
        score += 10

    final_score = min(max(score, 0), 100)

    # Final Feedback Polishing
    if final_score > 90:
        feedback = ["Excellent depth + strong structure/examples."]
    elif final_score > 85:
        feedback = ["Excellent depth and keyword usage."]
    elif not feedback:
        feedback = ["Good start, but could use more technical detail."]

    return {
        "score": final_score,
        "feedback": feedback
    }
