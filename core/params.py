import random
from typing import Dict
from core.nd import xp as np
from core.color.palettes import PALETTES


def randomize_function_params() -> Dict:
    """Generate new random parameters for the mathematical function.
    Preserves existing keys and selection strategy for compatibility.
    """
    all_operations = ['use_sin', 'use_cos', 'use_xor', 'use_reaction_diffusion',
                     'use_cellular', 'use_domain_warp', 'use_polar', 'use_sinusoidal_field',
                     'use_noise', 'use_abs', 'use_power', 'use_feedback', 'use_voronoi']

    operations = {}
    remaining_ops = [op for op in all_operations if op not in operations]
    ops_to_select = random.randint(4, 6)
    additional_ops = random.sample(remaining_ops, k=ops_to_select)
    operations.update({op: True for op in additional_ops})

    for op in all_operations:
        if op not in operations:
            operations[op] = False

    enabled_ops = [op for op in all_operations if operations.get(op, False)]
    random.shuffle(enabled_ops)

    color_schemes = [
        {'red': 1.0, 'green': 1.4, 'blue': 0.8},
        {'red': 0.8, 'green': 1.0, 'blue': 1.6},
        {'red': 1.5, 'green': 0.8, 'blue': 1.0},
        {'red': 1.2, 'green': 1.2, 'blue': 1.2},
        {'red': 0.9, 'green': 1.3, 'blue': 0.7},
        {'red': 1.6, 'green': 0.9, 'blue': 1.3},
        {'red': 1.3, 'green': 1.5, 'blue': 1.1},
        {'red': 1.1, 'green': 0.7, 'blue': 1.5},
    ]

    color_scheme = random.choice(color_schemes)

    params = {
        'wave1_freq': random.choice([0.618, 1.0, 1.618, 2.5, 3.14, 4.2, 5.8]),
        'wave2_freq': random.choice([0.618, 1.0, 1.618, 2.5, 3.14, 4.2, 5.8]),
        'wave1_mult': random.uniform(50, 300),
        'wave1_translate_x': random.uniform(-100, 100),
        'wave1_translate_y': random.uniform(-100, 100),

        'time_speed': random.uniform(0.2, 3.0),
        'time_translate_x': random.uniform(-50, 50),
        'time_translate_y': random.uniform(-50, 50),
        'time_warp_factor': random.uniform(0.5, 2.0),

        'xor_strength': random.uniform(.7, 1.3),
        'xor_translate_x': random.uniform(0.2, 1.5),
        'xor_translate_y': random.uniform(0.1, 1.0),
        'xor_translate_range': random.uniform(20, 100),
        'xor_morph_speed': random.uniform(0.1, 0.5),

        'cellular_scale': random.uniform(1.0, 20.0),
        'cellular_time_translate': random.uniform(-2.0, 2.0),

        'domain_warp_strength': random.uniform(15.0, 60.0),
        'domain_warp_time_factor': random.uniform(0.3, 2.0),

        'color_hue_segments': random.uniform(1, 2),
        'color_red_mult': color_scheme['red'],
        'color_green_mult': color_scheme['green'],
        'color_blue_mult': color_scheme['blue'],
        'color_phase_red': random.uniform(0, 360),
        'color_phase_green': random.uniform(0, 360),
        'color_phase_blue': random.uniform(0, 360),
        'color_saturation': random.uniform(1.0, 2.0),
        'color_power': random.uniform(1.0, 1.5),
        'color_mode': random.choice(['harmonic', 'palette', 'palette']),
        'palette_name': random.choice(list(PALETTES.keys())),
        'palette_reverse': random.choice([False, True]),
        'color_gamma': random.uniform(0.8, 1.4),
        'color_contrast': random.uniform(0.8, 1.4),
        'color_brightness': random.uniform(0.9, 1.2),
        'color_vibrance': random.uniform(0.8, 1.3),
        'color_auto_normalize': True,
        'color_clip_low': 2.0,
        'color_clip_high': 98.0,
        'palette_clip_low': 2.0,
        'palette_clip_high': 98.0,

        'polar_strength': random.uniform(0.7, 1.3),
        'polar_freq_r': random.choice([0.01, 0.02, 0.05, 0.1]),
        'polar_freq_theta': random.choice([2, 3, 5, 8]),
        'polar_rotation_speed': random.uniform(-0.5, 0.5),
        'polar_orbit_speed_x': random.uniform(-0.3, 0.3),
        'polar_orbit_speed_y': random.uniform(-0.3, 0.3),
        'polar_orbit_range': random.uniform(50, 150),
        'polar_time_factor': random.uniform(0.2, 2.0),
        'polar_theta_harmonics': random.uniform(1, 4),
        'polar_spiral_factor': random.uniform(-0.005, 0.005),
        'polar_spiral_freq': random.uniform(2, 8),
        'polar_spiral_speed': random.uniform(0.1, 1.5),

        'noise_strength': random.uniform(0.7, 1.3),
        'noise_scale': random.uniform(0.005, 0.02),
        'noise_time_speed': random.uniform(0.1, 1.0),
        'noise_octaves': random.randint(3, 6),

        'abs_strength': random.uniform(0.5, 1.0),
        'abs_freq_x': random.choice([0.01, 0.02, 0.05, 0.1]),
        'abs_freq_y': random.choice([0.01, 0.02, 0.05, 0.1]),
        'abs_freq_xy': random.choice([0.01, 0.02, 0.05]),
        'abs_time_speed': random.uniform(0.2, 2.0),

        'power_strength': random.uniform(0.5, 1.0),
        'power_exponent': random.uniform(0.5, 3.0),
        'power_freq_x': random.choice([0.01, 0.02, 0.05, 0.1]),
        'power_freq_y': random.choice([0.01, 0.02, 0.05, 0.1]),
        'power_time_speed': random.uniform(0.2, 1.8),
        'power_exp_mod_freq': random.uniform(0.1, 1.0),

        'feedback_strength': random.uniform(0.95, 0.99),
        'feedback_decay': random.uniform(0.5, 1.0),
        'feedback_zoom_speed': random.uniform(2.0, 5.0),
        'feedback_zoom_freq': random.uniform(0.05, 0.3),
        'feedback_zoom_amp': random.uniform(0.02, 0.15),
        'feedback_rotation_speed': random.uniform(-0.1, 0.1),
        'feedback_pan_x_speed': random.uniform(0, 0.5),
        'feedback_pan_y_speed': random.uniform(0, 0.5),
        'feedback_pan_range': random.uniform(10, 80),
        'feedback_mod_freq': random.uniform(0.02, 0.1),
        'feedback_color_shift': random.uniform(-0.1, 0.1),

        'voronoi_points': random.randint(7, 12),
        'voronoi_strength': random.uniform(0.4, 0.6),
        'voronoi_scale': random.uniform(0.01, 0.1),

        'sinusoidal_a_freq': random.choice([1.618, 2.414, 3.236, 4.236, 2.718, 3.141, 1.414]),
        'sinusoidal_b_freq': random.choice([2.618, 3.414, 4.618, 5.236, 4.442, 2.449, 1.732]),
        'sinusoidal_phase': random.uniform(0.0, 4 * np.pi),
        'sinusoidal_strength': random.uniform(0.7, 1.0),
        'sinusoidal_time_speed': random.uniform(0.3, 1.0),
        'sinusoidal_phase_speed_ratio': random.uniform(0.3, 3.0),
        'sinusoidal_scale': random.uniform(25.0, 50.0),
        'sinusoidal_freq_mod_depth': random.uniform(0.1, 0.4),
        'sinusoidal_freq_mod_speed': random.uniform(0.05, 0.3),
        'sinusoidal_amplitude_mod': random.uniform(0.2, 0.8),
        'sinusoidal_harmonics': random.randint(2, 5),
        'sinusoidal_harmonic_decay': random.uniform(0.3, 0.7),
        'sinusoidal_x_scale': random.uniform(0.5, 2.0),
        'sinusoidal_y_scale': random.uniform(0.5, 2.0),
        'sinusoidal_rotation_speed': random.uniform(-0.2, 0.2),

        'reaction_diffusion_diffusion_a': 1.0,
        'reaction_diffusion_diffusion_b': 0.5,
        'reaction_diffusion_feed_rate': 0.055,
        'reaction_diffusion_kill_rate': 0.062,
        'reaction_diffusion_dt': 0.02,
        'reaction_diffusion_scale': 0.7,

        'function_order': enabled_ops,
    }

    return {**operations, **params}


