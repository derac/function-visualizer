from typing import Dict
from core.nd import xp as np, Array


def apply_sin(x: Array, y: Array, time_val: float, params: Dict) -> Array:
    wave1_mult = params['wave1_mult']
    time_speed = params['time_speed']
    x_norm = (x - (time_val * params['time_translate_x'] + params['wave1_translate_x'])) / wave1_mult
    return np.abs(np.sin(x_norm * params['wave1_freq'] + time_val * time_speed))


def apply_cos(x: Array, y: Array, time_val: float, params: Dict) -> Array:
    wave1_mult = params['wave1_mult']
    time_speed = params['time_speed']
    y_norm = (y - (time_val * params['time_translate_y'] + params['wave1_translate_y'])) / wave1_mult
    return np.abs(np.cos(y_norm * params['wave2_freq'] + time_val * time_speed))


