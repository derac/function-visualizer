from core.nd import xp as np
from .state import feedback_state


def compute_feedback_values(x, y, time_val, params):
    if feedback_state.previous_frame is None:
        return np.zeros_like(x)

    prev_height, prev_width = feedback_state.previous_frame.shape[:2]

    x_range = x.max() - x.min()
    y_range = y.max() - y.min()

    x_norm = (x - x.min()) / (x_range + 1e-8)
    y_norm = (y - y.min()) / (y_range + 1e-8)

    x_prev = np.clip(x_norm * (prev_width - 1), 0, prev_width - 1)
    y_prev = np.clip(y_norm * (prev_height - 1), 0, prev_height - 1)

    x_int = x_prev.astype(int)
    y_int = y_prev.astype(int)

    feedback_rgb = feedback_state.previous_frame[y_int, x_int]
    feedback_gray = 0.299 * feedback_rgb[..., 0] + 0.587 * feedback_rgb[..., 1] + 0.114 * feedback_rgb[..., 2]

    feedback_strength = params.get('feedback_strength', 0.7)
    feedback_decay = params.get('feedback_decay', 0.95)
    feedback_mod = 1.0 + 0.1 * np.sin(time_val * params.get('feedback_mod_freq', 0.02))

    effective_feedback = (feedback_gray * feedback_strength * feedback_mod *
                          (feedback_decay ** (time_val - feedback_state.time_sum)))
    return effective_feedback


