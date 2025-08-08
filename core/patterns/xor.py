from typing import Dict
from core.nd import xp as np, Array


def apply(x: Array, y: Array, time_val: float, params: Dict, context: Dict) -> Array:
    time_shift_x = time_val * params.get('xor_translate_x', 0.5)
    time_shift_y = time_val * params.get('xor_translate_y', 0.3)
    morph_phase = time_val * params.get('xor_morph_speed', 0.2)

    trans_x = x + np.sin(time_shift_x) * params.get('xor_translate_range', 50)
    trans_y = y + np.cos(time_shift_y) * params.get('xor_translate_range', 50)

    morph_x = trans_x + np.sin(morph_phase + trans_y * 0.01) * (5 + 3 * np.sin(morph_phase * 0.7))
    morph_y = trans_y + np.cos(morph_phase + trans_x * 0.01) * (5 + 3 * np.cos(morph_phase * 0.5))

    rot_angle = np.sin(time_val * 0.1) * 0.1
    rot_x = morph_x * np.cos(rot_angle) - morph_y * np.sin(rot_angle)
    rot_y = morph_x * np.sin(rot_angle) + morph_y * np.cos(rot_angle)

    mask_exponent = int(np.clip((np.sin(time_val * 0.1) + 1.0) * 1.5, 1, 3))
    xor_mask = np.bitwise_xor(rot_x.astype(np.int32), rot_y.astype(np.int32)) & (1 << mask_exponent)

    intensity = 0.8 + 0.2 * np.sin(time_val * 0.05 + (rot_x + rot_y) * 0.001)
    return xor_mask * intensity


