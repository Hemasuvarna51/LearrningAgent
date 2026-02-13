# src/context_gathering.py
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

def gather_context(checkpoint: LearningCheckpoint) -> tuple[str, dict]:
    """
    Returns:
      (context_text, meta)
    meta is trace-friendly (small + useful for manual review).
    """
    topic = checkpoint.topic
    topic_norm = clean_header(topic)

    text = load_full_notes()
    if not text:
        return "", {
            "found": False,
            "reason": "notes_file_missing_or_empty",
            "topic": topic,
            "notes_file": str(NOTES_FILE),
        }

    lines = text.splitlines()

    # 1) Find exact heading match (line must equal topic after cleaning)
    start_idx = None
    heading_line = None
    for i, line in enumerate(lines):
        if _is_heading(lines, i) and clean_header(line) == topic_norm:
            start_idx = i + 2  # skip heading + underline
            heading_line = i
            break

    if start_idx is None:
        return "", {
            "found": False,
            "reason": "heading_not_found",
            "topic": topic,
            "notes_file": str(NOTES_FILE),
        }

    # 2) Next heading that matches any concept
    concept_norms = {clean_header(t) for t in CONCEPTS}
    end_idx = None
    next_heading_line = None

    for j in range(start_idx, len(lines)):
        if _is_heading(lines, j):
            norm = clean_header(lines[j])
            if norm in concept_norms and norm != topic_norm:
                end_idx = j
                next_heading_line = j
                break

    block_lines = lines[start_idx:end_idx] if end_idx else lines[start_idx:]
    context = "\n".join(block_lines).strip()

    # small preview only (avoid dumping all notes into tracing)
    preview = "\n".join([ln for ln in block_lines if ln.strip()][:2]).strip()

    meta = {
        "found": True,
        "topic": topic,
        "notes_file": str(NOTES_FILE),
        "heading_line": heading_line,
        "start_idx": start_idx,
        "end_idx": end_idx,
        "next_heading_line": next_heading_line,
        "retrieved_lines": len(block_lines),
        "retrieved_words": len((context or "").split()),
        "retrieved_chars": len(context),
        "preview": preview[:200],
    }

    return context, meta
