from typing import Optional
from core.nd import Array


class FeedbackState:
    def __init__(self) -> None:
        self.previous_frame: Optional[Array] = None
        self.feedback_intensity: float = 0.7
        self.time_sum: float = 0.0

    def reset(self) -> None:
        self.previous_frame = None
        self.time_sum = 0.0


# Singleton instance (matches previous module-global behavior)
feedback_state = FeedbackState()


