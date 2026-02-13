from dataclasses import dataclass, field
from typing import List

@dataclass
class LearningCheckpoint:
    topic: str
    objectives: List[str] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)

    def get_clean_state(self):
        return {
            "topic": self.topic,
            "current_attempt": 1,
            "history": []
        }
