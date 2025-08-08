from typing import Dict
from core.nd import xp as np, Array


def apply(x: Array, y: Array, time_val: float, params: Dict) -> Array:
    animated_strength = params['domain_warp_strength']
    time_phase = time_val * params['domain_warp_time_factor']
    warped_x = (x + animated_strength * np.sin(y * 0.1 + time_phase + 
                                              np.sin(time_phase * 2) * 0.5)) / (3 * np.sin(time_phase) + 4)
    warped_y = (y + animated_strength * np.cos(x * 0.1 + time_phase * 0.7 + 
                                              np.sin(time_phase * 1.5) * 0.3)) / (3 * np.sin(time_phase) + 4)
    return np.sin(warped_x) * np.cos(warped_y)


