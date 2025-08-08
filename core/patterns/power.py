from typing import Dict
from core.nd import xp as np, Array


def apply(x: Array, y: Array, time_val: float, params: Dict) -> Array:
    time_power = time_val * params['power_time_speed']
    base = (np.sin(x * params['power_freq_x'] + time_power) +
            np.cos(y * params['power_freq_y'] + time_power * 0.8) + 2) / 2
    exp_base = params['power_exponent']
    exp_mod = exp_base * (1 + 0.4 * np.sin(time_power * params['power_exp_mod_freq']))
    power_val = np.power(base, exp_mod)
    power_val = np.nan_to_num(power_val, nan=0.0, posinf=1000.0, neginf=0.0)

    harmonic = np.power(
        (np.sin(x * params['power_freq_x'] * 2 + time_power * 1.5) +
         np.cos(y * params['power_freq_y'] * 2 + time_power * 2) + 2) / 2,
        exp_mod * 0.5
    )
    harmonic = np.nan_to_num(harmonic, nan=0.0, posinf=1000.0, neginf=0.0)

    return (power_val + harmonic * 0.3)


