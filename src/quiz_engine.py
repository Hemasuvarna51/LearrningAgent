import copy
import random
import time


def testQuizFromQuestions(topic: str, mcqs: list[dict], shuffle: bool = False, time_limit: int | None = None) -> float:
    mcqs = copy.deepcopy(mcqs)

    if not mcqs:
        print(f"❌ No generated questions for topic: {topic}")
        return 0.0

    if shuffle:
        random.shuffle(mcqs)

    print(f"\n🧪 Generated Quiz: {topic}")
    print("=" * 50)

    score = 0
    start_time = time.time()
    total_questions = len(mcqs)

    for index, mcq in enumerate(mcqs, start=1):
        if time_limit and (time.time() - start_time) > time_limit:
            print("\n⏰ Time limit reached!")
            break

        print(f"\nQ{index}: {mcq['question']}")
        for key in ("A", "B", "C", "D"):
            print(f"{key}) {mcq['options'].get(key, 'N/A')}")

        while True:
            user_answer = input("\nYour answer (A/B/C/D): ").strip().upper()
            if user_answer in ("A", "B", "C", "D"):
                break
            print("❌ Invalid input. Enter A, B, C, or D.")

        if user_answer == mcq["answer"]:
            print("✅ Correct!")
            score += 1
        else:
            correct_opt = mcq["answer"]
            correct_text = mcq["options"].get(correct_opt, "")
            print(f"❌ Wrong! Correct answer: {correct_opt}) {correct_text}")

    percentage = round((score / total_questions) * 100, 2) if total_questions else 0.0

    print("\n" + "=" * 50)
    print(f"Score: {score}/{total_questions}")
    print(f"Percentage: {percentage}%")
    return percentage
