import numpy as base_np
from core.nd import xp as np
from core.feedback.state import feedback_state


def apply(x, y, time_val, params, context):
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

    current_frame = feedback_state.previous_frame[y_int, x_int]
    A = (0.299 * current_frame[..., 0] + 0.587 * current_frame[..., 1] + 0.114 * current_frame[..., 2])
    A = np.clip(A * 0.85, 0.05, 0.95)

    B = 1.0 - A
    noise_scale = 0.05
    B += np.random.normal(0, noise_scale, B.shape) * 0.01
    B = np.clip(B, 0.05, 0.95)

    diffusion_a = params.get('reaction_diffusion_diffusion_a', 1.0)
    diffusion_b = params.get('reaction_diffusion_diffusion_b', 0.5)
    feed_rate = params.get('reaction_diffusion_feed_rate', 0.055)
    kill_rate = params.get('reaction_diffusion_kill_rate', 0.062)
    dt = params.get('reaction_diffusion_dt', 0.02)

    time_factor = np.sin(time_val * 0.02) * 0.002
    feed_rate = np.clip(feed_rate + time_factor, 0.054, 0.056)
    kill_rate = np.clip(kill_rate + time_factor, 0.061, 0.063)

    padded_a = np.pad(A, 1, mode='edge')
    padded_b = np.pad(B, 1, mode='edge')

    laplacian_a = (
        padded_a[2:, 1:-1] + padded_a[:-2, 1:-1] +
        padded_a[1:-1, 2:] + padded_a[1:-1, :-2] -
        4 * padded_a[1:-1, 1:-1]
    ) * 0.25

    laplacian_b = (
        padded_b[2:, 1:-1] + padded_b[:-2, 1:-1] +
        padded_b[1:-1, 2:] + padded_b[1:-1, :-2] -
        4 * padded_b[1:-1, 1:-1]
    ) * 0.25

    reaction = A * B * B * 0.8
    dA = (diffusion_a * laplacian_a) - reaction + feed_rate * (1 - A)
    dB = (diffusion_b * laplacian_b) + reaction - (kill_rate + feed_rate) * B

    new_A = np.clip(A + dt * dA, 0.1, 0.9)
    new_B = np.clip(B + dt * dB, 0.1, 0.9)

    rd_output = new_A - new_B

    percentiles = base_np.percentile(rd_output, [15, 85])
    rd_clipped = np.clip(rd_output, percentiles[0], percentiles[1])
    rd_normalized = (rd_clipped - percentiles[0]) / (percentiles[1] - percentiles[0] + 1e-8)
    rd_mapped = rd_normalized * 0.8 + 0.1

    pattern_scale = params.get('reaction_diffusion_scale', 0.8) * 0.85
    time_modulation = 1.0 + 0.05 * np.sin(time_val * 0.01)

    return rd_mapped * pattern_scale * time_modulation


