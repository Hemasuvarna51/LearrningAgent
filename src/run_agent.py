import argparse
import sys
from checkpoint import LearningCheckpoint
from learning_agent import build_agent
from quiz import testQuizInteractive, CONCEPTS

# ------------------ CLI ARGS ------------------
parser = argparse.ArgumentParser()
parser.add_argument("--topic", type=str)
parser.add_argument("--shuffle", action="store_true")
parser.add_argument("--time", type=int)
args = parser.parse_args()

# ------------------ AGENT ------------------
agent = build_agent()

# ------------------ EXPLANATION ------------------
def explain_topic(topic_name: str, current_attempt: int):
    """
    Calls the learning agent to produce a GPT-style explanation.
    Automatically switches tone after failed attempts.
    """
    if current_attempt == 1:
        objectives = [
            f"Understand {topic_name}",
            "Provide high-level technical summary"
        ]
    else:
        objectives = [
            f"Simplify {topic_name} using intuition and analogies",
            "Address common misconceptions"
        ]

    checkpoint = LearningCheckpoint(
        topic=topic_name,
        objectives=objectives,
        success_criteria=["Clarity", "Relevance"]
    )

    result = agent({
        "checkpoint": checkpoint,
        "attempt": current_attempt
    })

    context = result.get("context", "")
    score = result.get("relevance_score", 0)
    feedback = result.get("feedback", [])

    print("\n🚀" + "=" * 60)
    print(context)
    print("-" * 60)

    bar_len = 20
    filled = int(bar_len * score / 100)
    bar = "█" * filled + "░" * (bar_len - filled)

    print(f"📊 Context Relevance: [{bar}] {score}%")
    if feedback:
        print(f"💡 Tutor Tip: {', '.join(feedback)}")
    print("=" * 60 + "\n")


# ------------------ MAIN LOOP ------------------
def main():
    print("\n🌟 WELCOME TO THE ML CHECKPOINT AGENT 🌟")

    mastered = {}  # topic -> best score

    while True:
        print("\nAvailable Concepts:")
        for c in CONCEPTS:
            status = " ✅" if c in mastered else ""
            print(f"  • {c}{status}")

        try:
            raw_topic = input(
                "\nWhat would you like to master? (or type 'exit'): "
            ).strip()

            if raw_topic.lower() in {"exit", "quit"}:
                print("👋 Session closed. Happy learning!")
                break

            topic = next(
                (c for c in CONCEPTS if c.lower() == raw_topic.lower()),
                None
            )

            if not topic:
                print(f"❌ '{raw_topic}' is not a valid concept.")
                continue

            if topic in mastered:
                print(f"\n✅ You already mastered {topic} ({mastered[topic]}%).")
                print("Pick another topic or restart the session to practice again.")
                continue

            attempt = 1

            while True:
                print(f"\n📍 [LEARNING: {topic} | ATTEMPT {attempt}]")
                explain_topic(topic, attempt)

                ready = input(
                    "Press Enter to start the quiz (or type 'skip' to go back): "
                ).strip().lower()

                if ready == "skip":
                    break

                print("\n📝 Starting Quiz...")
                score = testQuizInteractive(
                    topic,
                    shuffle=args.shuffle,
                    time_limit=args.time
                )

                if score >= 70:
                    mastered[topic] = score
                    print(f"\n🏆 FINAL SCORE: {score}% - MASTERY ACHIEVED!")
                    break
                else:
                    print(
                        f"\n📉 SCORE: {score}% - Let's try again with a simpler explanation."
                    )
                    attempt += 1

            again = input(
                "\nMastery finished! Learn another concept? (y/n): "
            ).strip().lower()

            if again not in {"y", "yes"}:
                print("👋 Great progress today. See you next time!")
                break

        except KeyboardInterrupt:
            print("\n\n👋 Session interrupted. Goodbye.")
            sys.exit(0)


if __name__ == "__main__":
    main()
