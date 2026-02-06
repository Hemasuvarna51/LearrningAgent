from pathlib import Path
import json
import random
import time
import copy

# -----------------------------
# Paths
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUESTIONS_FILE = PROJECT_ROOT / "data" / "questions" / "quiz.json"

# -----------------------------
# Load questions ONCE
# -----------------------------
try:
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        QUESTIONS = json.load(f)
except FileNotFoundError:
    print(f"❌ Error: Could not find questions file at {QUESTIONS_FILE}")
    QUESTIONS = {}

CONCEPTS = list(QUESTIONS.keys())

# -----------------------------
# Quiz Engine (PURE FUNCTION)
# -----------------------------
def testQuizInteractive(topic: str, shuffle=False, time_limit=None) -> float:
    # ✅ deep copy to avoid mutation of the global QUESTIONS dict
    mcqs = copy.deepcopy(QUESTIONS.get(topic, []))

    if not mcqs:
        print(f"❌ No MCQs found for topic: {topic}")
        return 0.0

    if shuffle:
        random.shuffle(mcqs)

    print(f"\n🧪 Interactive Quiz: {topic}")
    print("=" * 50)

    score = 0
    start_time = time.time()
    total_questions = len(mcqs)

    for index, mcq in enumerate(mcqs, start=1):

        # Time Limit Check
        if time_limit and (time.time() - start_time) > time_limit:
            print("\n⏰ Time limit reached!")
            break

        print(f"\nQ{index}: {mcq['question']}")
        for key in ("A", "B", "C", "D"):
            # Use .get() to safely handle missing options if data is messy
            print(f"{key}) {mcq['options'].get(key, 'N/A')}")

        # Input Loop
        while True:
            user_answer = input("\nYour answer (A/B/C/D): ").strip().upper()
            if user_answer in ("A", "B", "C", "D"):
                break
            print("❌ Invalid input. Enter A, B, C, or D.")

        # Check Answer
        if user_answer == mcq["answer"]:
            print("✅ Correct!")
            score += 1
        else:
            correct_opt = mcq["answer"]
            correct_text = mcq["options"].get(correct_opt, "")
            print(f"❌ Wrong! Correct answer: {correct_opt}) {correct_text}")

    # Calculate Score
    if total_questions > 0:
        percentage = round((score / total_questions) * 100, 2)
    else:
        percentage = 0.0

    print("\n" + "=" * 50)
    print(f"Score: {score}/{total_questions}")
    print(f"Percentage: {percentage}%")

    return percentage