class FeedbackState:
    def __init__(self):
        self.previous_frame = None
        self.feedback_intensity = 0.7
        self.time_sum = 0.0

    def reset(self):
        self.previous_frame = None
        self.time_sum = 0.0


# Singleton instance (matches previous module-global behavior)
feedback_state = FeedbackState()


