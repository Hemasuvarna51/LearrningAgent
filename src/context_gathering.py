import re
from pathlib import Path
from checkpoint import LearningCheckpoint

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTES_FILE = PROJECT_ROOT / "data" / "notes" / "machine_learning.txt"

CONCEPTS = [
    "Machine Learning",
    "Deep Learning",
    "Neural Networks",
    "Supervised Learning",
    "Unsupervised Learning",
    "Reinforcement Learning",
    "Overfitting",
    "Underfitting",
    "Bias vs Variance",
    "Model Evaluation",
]

def clean_header(line: str) -> str:
    line = re.sub(r"[^\w\s]", " ", line)
    line = re.sub(r"\s+", " ", line)
    return line.lower().strip()

def load_full_notes() -> str:
    return NOTES_FILE.read_text(encoding="utf-8") if NOTES_FILE.exists() else ""

def _is_underline(line: str) -> bool:
    s = (line or "").strip()
    return bool(s) and set(s) <= {"-", "="}

def _is_heading(lines: list[str], idx: int) -> bool:
    """A heading is a line followed by an underline (--- or ===)."""
    if idx < 0 or idx >= len(lines) - 1:
        return False
    return bool(lines[idx].strip()) and _is_underline(lines[idx + 1])

def gather_context(checkpoint: LearningCheckpoint) -> str:
    topic_norm = clean_header(checkpoint.topic)
    text = load_full_notes()
    if not text:
        return ""

    lines = text.splitlines()

    # 1) Find exact heading match (line must equal topic after cleaning)
    start_idx = None
    for i, line in enumerate(lines):
        if _is_heading(lines, i) and clean_header(line) == topic_norm:
            start_idx = i + 2  # skip heading + underline
            break

    if start_idx is None:
        return ""

    # 2) Next heading that matches any concept
    concept_norms = {clean_header(t) for t in CONCEPTS}
    end_idx = None

    for j in range(start_idx, len(lines)):
        if _is_heading(lines, j):
            if clean_header(lines[j]) in concept_norms and clean_header(lines[j]) != topic_norm:
                end_idx = j
                break

    block = lines[start_idx:end_idx] if end_idx else lines[start_idx:]
    return "\n".join(block).strip()
