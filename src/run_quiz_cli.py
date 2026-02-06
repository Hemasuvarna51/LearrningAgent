import argparse
import sys
import secrets
import random

from checkpoint import LearningCheckpoint
from context_gathering import gather_context, CONCEPTS
from scoring_engine import calculate_relevance_score
from question_generator import generate_mcqs_from_explanation
from quiz_engine import testQuizFromQuestions


# ------------------ Text cleanup helpers ------------------

def _clean_lines(raw_text: str) -> list[str]:
    """Normalize notes: remove empty lines, [Image...], underline headings."""
    out = []
    for line in (raw_text or "").splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("[Image"):
            continue
        if set(s) <= {"-", "="}:
            continue
        out.append(s)
    return out


def _strip_leading_dashes(s: str) -> str:
    """Turn '- - foo' or '-- foo' into 'foo'."""
    t = s.strip()
    while t.startswith("-"):
        t = t[1:].lstrip()
    return t


def _dedupe_keep_order(items: list[str]) -> list[str]:
    seen = set()
    out = []
    for x in items:
        key = x.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(x)
    return out


def _is_headerish(s: str) -> bool:
    low = s.lower().strip()
    if low in {
        "core concepts:", "algorithms:", "types:", "overview", 
        "technical overview", "machine learning overview", "deep dive",
    }:
        return True
    if low.endswith(":") and len(low.split()) <= 4:
        return True
    return False


def _topic_analogy(topic: str) -> str:
    t = (topic or "").lower()
    if "supervised" in t:
        return "Think of it like a student learning with an answer key. For every practice problem, they can check if they were right immediately, helping them correct their logic for the next one."
    if "unsupervised" in t:
        return "Think of it like sorting a massive pile of mixed LEGOs into boxes without instructions. You look for similar colors or shapes to create order out of chaos."
    if "reinforcement" in t:
        return "Think of it like training a puppy. You don't tell it exactly how to sit, but you give it a treat when it does, and no treat when it doesn't. Eventually, it optimizes its behavior for the treats."
    if "neural" in t:
        return "Think of it like a series of filters in a coffee machine. Each layer strains out certain parts of the data until only the most important 'essence' reaches the final cup."
    if "overfitting" in t:
        return "Think of it like a student who memorized every specific number in their textbook but doesn't understand the underlying math. If the exam changes one number, they fail."
    return "Think of it like learning to recognize a face: you don't memorize every pixel, you learn the patterns of eyes, nose, and mouth."


def _why_it_matters(topic: str) -> str:
    t = (topic or "").lower()
    if "unsupervised" in t:
        return "It is essential for discovery. It finds patterns humans might miss, like identifying 'hidden' customer segments or detecting weird network traffic that might be a hack."
    if "supervised" in t:
        return "It is the backbone of prediction. It allows us to automate decisions like 'Is this credit card transaction fraudulent?' based on millions of past examples."
    return "It allows software to move beyond 'if-then' logic and start handling the messy, unpredictable data of the real world."


def _format_bullets(lines: list[str]) -> str:
    cleaned = []
    for x in lines:
        x = _strip_leading_dashes(x)
        if not x or _is_headerish(x):
            continue
        cleaned.append(x)
    cleaned = _dedupe_keep_order(cleaned)
    if not cleaned:
        return "- (No additional context points available.)"
    return "\n".join([f"- {x}" for x in cleaned])


# ------------------ Explanation builder ------------------

def format_as_gpt_style(topic: str, raw_text: str, attempt: int, session_seed: int) -> str:
    lines = _clean_lines(raw_text)

    if attempt == 1:
        technical = "\n".join([_strip_leading_dashes(x) for x in lines]).strip() if lines else (
            f"No specific technical notes found for '{topic}'."
        )
        return f"""
## 🧠 Deep Dive: {topic}

### 🛠️ Technical Architecture & Overview
{technical}

---
💡 **Pro Tip:** Look for the relationship between the inputs and the final output layer.
""".strip()

    return f"""
## 🧠 Deep Dive: {topic} (Re-learning Attempt {attempt})
{_dynamic_simple_explanation(topic, lines, attempt, session_seed)}

---
🏁 **Mastery Note:** If you're still stuck, focus on the "Common Pitfalls" section above.
""".strip()


def _dynamic_simple_explanation(topic: str, lines: list[str], attempt: int, session_seed: int) -> str:
    if not lines:
        return f"No notes found for '{topic}'."

    r = random.Random((session_seed + attempt * 7919) & 0xFFFFFFFF)
    content = [_strip_leading_dashes(x) for x in lines if not _is_headerish(x)]
    content = _dedupe_keep_order([x for x in content if x])
    r.shuffle(content)

    # 1. TOPIC-SPECIFIC KNOWLEDGE (Prevents "Unsupervised" answers leaking into "Deep Learning")
    topic_data = {
        "Deep Learning": {
            "challenge": "The massive computational power (GPUs) and huge datasets required to prevent overfitting.",
            "improvement": "Through backpropagation, where the model calculates the error at the output and sends it back through the layers to adjust weights.",
            "visual": ""
        },
        "Supervised Learning": {
            "challenge": "The high cost and time required to manually label thousands of data points for the 'ground truth'.",
            "improvement": "By minimizing a loss function (the difference between the predicted label and the actual label).",
            "visual": ""
        },
        "Unsupervised Learning": {
            "challenge": "Evaluating success, because there is no 'correct answer' or ground truth to compare against.",
            "improvement": "By finding mathematical similarities or 'clusters' within the data points.",
            "visual": ""
        }
    }

    # Fetch data or use generic defaults
    data = topic_data.get(topic, {
        "challenge": "Ensuring the model generalizes well to new, unseen data instead of just memorizing the training set.",
        "improvement": "Iterative training on diverse datasets.",
        "visual": ""
    })

    # 2. STRATEGY ROTATION
    styles = ["conceptual_deep_dive", "qa_extended", "operational_flow"]
    style = styles[(attempt - 2) % len(styles)]

    if style == "conceptual_deep_dive":
        return f"""
### 💡 The Big Picture
{_topic_analogy(topic)}

### 🏗️ How it's Structured
{data['visual']}
**Core Mechanism:** {data['improvement']}

### 📝 Key Concepts from your Notes
{_format_bullets(content[:7])}

### ⚖️ Why this is Critical
{_why_it_matters(topic)}
""".strip()

    if style == "qa_extended":
        return f"""
### ❓ Focused Q&A Breakout

**Q: What is the primary bottleneck for {topic}?**
A: {data['challenge']}

**Q: How do we actually measure success here?**
A: {"By checking if the model generalizes to a test set it has never seen before." if "Supervised" in topic or "Deep" in topic else "By using internal metrics like silhouette scores or domain expert review."}

### 🔬 Technical Nuances from Notes
{_format_bullets(content)}

{data['visual']}

### ⚠️ The "Real World" Trade-off
In {topic}, more complexity isn't always better. If you have too many layers or parameters without enough data, you hit **Overfitting**—where the model memorizes the noise instead of the signal.
""".strip()

    return f"""
### 🛤️ The {topic} Workflow
1. **Identify Objective:** What are we trying to predict or cluster?
2. **Resource Check:** {"Do we have GPUs?" if "Deep" in topic else "Do we have enough labeled data?"}
3. **Training:** Run the algorithm: {_format_bullets(content[:3])}
4. **Validation:** Use a separate test set to calculate accuracy and interpretability.

{data['visual']}

### 📋 Full Concept Summary
{_format_bullets(content)}
""".strip()
# ------------------ Agent ------------------

def agent_run(topic_name: str, attempt: int, session_seed: int) -> dict:
    objectives = (
        [f"Understand {topic_name}", "Provide technical summary"]
        if attempt == 1
        else [f"Simplify {topic_name}", "Fix misconceptions"]
    )

    checkpoint = LearningCheckpoint(
        topic=topic_name,
        objectives=objectives,
        success_criteria=["Clarity", "Relevance"]
    )

    raw_context = gather_context(checkpoint)
    score_result = calculate_relevance_score(raw_context, topic_name, attempt=attempt)

    explanation = format_as_gpt_style(topic_name, raw_context, attempt, session_seed)

    questions = generate_mcqs_from_explanation(
        explanation=explanation,
        topic=topic_name,
        attempt=attempt,
        n=6
    )

    return {
        "topic": topic_name,
        "attempt": attempt,
        "raw_context": raw_context,
        "explanation": explanation,
        "relevance_score": int(score_result.get("score", 0)),
        "feedback": score_result.get("feedback", []),
        "generated_questions": questions
    }


# ------------------ CLI ------------------

parser = argparse.ArgumentParser()
parser.add_argument("--shuffle", action="store_true")
parser.add_argument("--time", type=int, default=None)
args = parser.parse_args()


def main():
    print("\n🌟 WELCOME TO THE ML CHECKPOINT AGENT 🌟")
    mastered = {}
    session_seed = secrets.randbits(32)

    while True:
        print("\nAvailable Concepts:")
        for c in CONCEPTS:
            print(f"  • {c}{' ✅' if c in mastered else ''}")

        try:
            raw_topic = input("\nWhat would you like to master? (or type 'exit'): ").strip()

            if raw_topic.lower() in {"exit", "quit"}:
                print("👋 Session closed. Happy learning!")
                break

            if not raw_topic:
                continue

            topic = next((c for c in CONCEPTS if c.lower() == raw_topic.lower()), None)
            if not topic:
                print(f"❌ '{raw_topic}' is not a valid concept.")
                continue

            attempt = 1
            while True:
                print(f"\n📍 [LEARNING: {topic} | ATTEMPT {attempt}]")
                result = agent_run(topic, attempt, session_seed)

                print("\n🚀" + "=" * 60)
                print(result["explanation"])
                print("-" * 60)

                score = result["relevance_score"]
                bar_len = 20
                filled = int(bar_len * score / 100)
                bar = "█" * filled + "░" * (bar_len - filled)
                print(f"📊 Notes Relevance: [{bar}] {score}%")

                if result["feedback"]:
                    print(f"💡 Tutor Tip: {', '.join(result['feedback'])}")

                print("=" * 60)

                ready = input("\nPress Enter to start the quiz (or type 'skip'): ").strip().lower()
                if ready == "skip":
                    break

                print("\n📝 Starting Generated Quiz...")
                score_quiz = testQuizFromQuestions(
                    topic,
                    result["generated_questions"],
                    shuffle=args.shuffle,
                    time_limit=args.time
                )

                if score_quiz >= 70:
                    mastered[topic] = score_quiz
                    print(f"\n🏆 MASTERY ACHIEVED: {score_quiz}%")
                    break
                else:
                    print(f"\n📉 SCORE: {score_quiz}% — Let's try a more detailed perspective.")
                    attempt += 1

            again = input("\nLearn another concept? (y/n): ").strip().lower()
            if again not in {"y", "yes"}:
                print("👋 Great progress today. See you next time!")
                break

        except KeyboardInterrupt:
            print("\n\n👋 Session interrupted. Goodbye.")
            sys.exit(0)


if __name__ == "__main__":
    main()