from typing import Dict
from core.nd import xp as np, Array


def apply(x: Array, y: Array, time_val: float, params: Dict, context: Dict) -> Array:
    time_abs = time_val * params['abs_time_speed']
    abs_wave1 = np.abs(np.sin(x * params['abs_freq_x'] * np.sin(time_abs / 10)))
    abs_wave2 = np.abs(np.cos(y * params['abs_freq_y'] * np.cos(time_abs / 10) * 0.7))
    abs_combo = abs_wave1 * abs_wave2 + np.abs(abs_wave1 - abs_wave2)
    cross_term = np.abs(np.sin((x + y) * params['abs_freq_xy'] + time_abs * 1.5))
    return (abs_combo + cross_term) * 120 * params['abs_strength']


