from typing import Dict
from core.nd import xp as np, Array


def apply(x: Array, y: Array, time_val: float, params: Dict, context: Dict) -> Array:
    animated_strength = context['domain_warp_strength_mod']
    time_phase = time_val * 0.5
    warped_x = (x + animated_strength * np.sin(y * 0.1 + time_phase + 
                                              np.sin(time_phase * 2) * 0.5)) / (3 * np.sin(time_val) + 4)
    warped_y = (y + animated_strength * np.cos(x * 0.1 + time_phase * 0.7 + 
                                              np.sin(time_phase * 1.5) * 0.3)) / (3 * np.sin(time_val) + 4)
    return np.sin(warped_x) * np.cos(warped_y) * 25


