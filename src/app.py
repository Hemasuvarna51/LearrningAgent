import streamlit as st
from checkpoint import LearningCheckpoint
from learning_agent import build_agent
from question_generator import generate_mcqs_from_explanation
import learning_agent
import os


# ============================================================
# Page Config (MUST be first Streamlit call)
# ============================================================
st.set_page_config(
    page_title="ML Checkpoint Agent",
    page_icon="🧠",
    layout="wide"
)

# ============================================================
# DEBUG TOGGLE (turn off later)
# ============================================================
DEBUG = True

if DEBUG:
    with st.sidebar:
        st.write("AGENT FILE:", os.path.abspath(learning_agent.__file__))
        st.write("CACHE_VERSION:", "FORCE_v999")


# ============================================================
# Session State Defaults
# ============================================================
DEFAULTS = {
    "attempt": 1,
    "current_topic": "Machine Learning",
    "quiz_active": False,
    "quiz_submitted": False,
    "last_score": None,
    "questions": None,
    "quiz_n": 6,            # ✅ 5–10
    "quiz_version": 0,      # ✅ forces radio reset when regenerated
}

for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


def reset_topic_state(topic: str | None = None):
    """Hard reset for topic or when quiz should restart cleanly."""
    if topic is not None:
        st.session_state.current_topic = topic
    st.session_state.attempt = 1
    st.session_state.quiz_active = False
    st.session_state.quiz_submitted = False
    st.session_state.last_score = None
    st.session_state.questions = None
    st.session_state.quiz_version += 1  # ✅ reset radios


def reset_quiz_only():
    """Reset quiz without resetting attempt/topic."""
    st.session_state.quiz_active = False
    st.session_state.quiz_submitted = False
    st.session_state.last_score = None
    st.session_state.questions = None
    st.session_state.quiz_version += 1  # ✅ reset radios


# ============================================================
# Core Logic (NO caching for agent during dev)
# ============================================================
def run_agent(topic: str, attempt: int):
    agent = build_agent()
    checkpoint = LearningCheckpoint(topic=topic, objectives=[], success_criteria=[])
    return agent({"checkpoint": checkpoint, "attempt": attempt})


@st.cache_data(show_spinner=False)
def make_questions(explanation: str, topic: str, attempt: int, n: int):
    # ✅ clamp to 5–10 at app level too (extra safety)
    n = max(5, min(10, int(n)))
    return generate_mcqs_from_explanation(
        explanation=explanation,
        topic=topic,
        attempt=attempt,
        n=n,
    )


# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.title("Settings")

    concepts = [
        "Machine Learning",
        "Deep Learning",
        "Neural Networks",
        "Supervised Learning",
        "Unsupervised Learning",
        "Overfitting",
    ]

    selected_topic = st.selectbox(
        "Choose a Concept",
        concepts,
        index=concepts.index(st.session_state.current_topic),
    )

    # ✅ NEW: question count slider (5–10)
    quiz_n = st.slider(
        "No. of Questions",
        min_value=5,
        max_value=10,
        value=int(st.session_state.quiz_n),
        step=1
    )

    # Apply quiz_n change
    if quiz_n != st.session_state.quiz_n:
        st.session_state.quiz_n = quiz_n
        reset_quiz_only()  # ✅ restart quiz cleanly (prevents mixed/duplicated radios)

    # Apply topic change
    if selected_topic != st.session_state.current_topic:
        reset_topic_state(selected_topic)
        st.rerun()

    st.write(f"**Current Attempt:** {st.session_state.attempt}")

    if st.button("Reset Topic Progress"):
        reset_topic_state(st.session_state.current_topic)
        st.rerun()


# ============================================================
# Agent Execution
# ============================================================
result = run_agent(
    st.session_state.current_topic,
    st.session_state.attempt
)

context_text = result.get("context", "") or ""


# ============================================================
# Main UI
# ============================================================
st.title(f"🚀 Mastering {st.session_state.current_topic}")

col1, col2 = st.columns([3, 2])

with col1:
    if context_text.strip():
        st.markdown(context_text)
    else:
        st.warning("No context was generated for this topic. Check your notes file or context gathering.")

with col2:
    score_val = int(result.get("relevance_score", 0))
    score_val = max(0, min(100, score_val))
    st.metric("Relevance Score", f"{score_val}%")
    st.progress(score_val / 100)

    if result.get("feedback"):
        st.subheader("💡 Tutor Tips")
        for tip in result["feedback"]:
            st.info(tip)

st.divider()


# ============================================================
# Quiz Logic
# ============================================================
if not st.session_state.quiz_active:
    if st.button("📝 Start Quiz"):
        st.session_state.quiz_active = True
        st.session_state.quiz_submitted = False
        st.session_state.last_score = None

        st.session_state.questions = make_questions(
            context_text,
            st.session_state.current_topic,
            st.session_state.attempt,
            st.session_state.quiz_n,
        )
        st.session_state.quiz_version += 1  # ✅ reset radios each time quiz starts
        st.rerun()

else:
    st.subheader("🧪 Knowledge Check")

    questions = st.session_state.questions

    # ✅ Ensure we always have correct number of questions
    if (not questions) or (len(questions) != int(st.session_state.quiz_n)):
        questions = make_questions(
            context_text,
            st.session_state.current_topic,
            st.session_state.attempt,
            st.session_state.quiz_n,
        )
        st.session_state.questions = questions
        st.session_state.quiz_version += 1  # ✅ forces new radio keys

    with st.form("quiz_form"):
        user_answers = {}

        for i, q in enumerate(questions):
            st.write(f"**Q{i+1}: {q['question']}**")

            # ✅ key includes quiz_version so old selections don’t leak
            user_answers[i] = st.radio(
                f"Select option for Q{i+1}",
                list(q["options"].values()),
                key=f"{st.session_state.current_topic}_a{st.session_state.attempt}_v{st.session_state.quiz_version}_q{i}",
            )

        submitted = st.form_submit_button("Submit Quiz")

    if submitted:
        score = 0
        for i, q in enumerate(questions):
            if user_answers[i] == q["options"][q["answer"]]:
                score += 1

        st.session_state.last_score = (score / len(questions)) * 100 if questions else 0
        st.session_state.quiz_submitted = True

    if st.session_state.quiz_submitted:
        final_score = st.session_state.last_score or 0

        if final_score >= 70:
            st.balloons()
            st.success(f"🏆 MASTERY ACHIEVED: {final_score:.0f}%")

            if st.button("Learn Next Concept"):
                reset_topic_state(st.session_state.current_topic)
                st.rerun()
        else:
            st.error(f"📉 SCORE: {final_score:.0f}% — Below threshold.")
            st.info(f"Preparing a deeper explanation for Attempt {st.session_state.attempt + 1}.")

            if st.button("🔄 Generate New Perspective"):
                st.session_state.attempt += 1
                reset_quiz_only()
                st.rerun()
