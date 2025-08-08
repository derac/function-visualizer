from typing import Dict
from core.nd import xp as np, Array


def apply(x: Array, y: Array, time_val: float, params: Dict) -> Array:
    center_x = np.sin(time_val * params['polar_orbit_speed_x']) * params['polar_orbit_range']
    center_y = np.cos(time_val * params['polar_orbit_speed_y']) * params['polar_orbit_range']

    x_rel = x - center_x
    y_rel = y - center_y

    r = np.sqrt(x_rel**2 + y_rel**2) + 1e-8
    theta = np.arctan2(y_rel, x_rel) + time_val * params['polar_rotation_speed']

    freq_mod = 1 + 0.3 * np.sin(time_val * 0.15)
    polar_wave = (
        np.sin(r * params['polar_freq_r'] * freq_mod + theta * params['polar_freq_theta'])
        * np.cos(theta * params['polar_theta_harmonics'] + time_val * params['polar_time_factor'])
    )

    spiral_angle = theta + r * params['polar_spiral_factor']
    spiral_wave = np.sin(spiral_angle * params['polar_spiral_freq'] + time_val * params['polar_spiral_speed'])

    return (polar_wave + spiral_wave * 0.5)


