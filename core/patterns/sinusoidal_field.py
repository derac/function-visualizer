from core.nd import xp as np


def apply(x, y, time_val, params, context):
    time_liss = time_val * params['sinusoidal_time_speed']
    freq_mod_time = time_val * params['sinusoidal_freq_mod_speed']
    a_freq_dynamic = params['sinusoidal_a_freq'] * (1 + params['sinusoidal_freq_mod_depth'] * np.sin(freq_mod_time))
    b_freq_dynamic = params['sinusoidal_b_freq'] * (1 + params['sinusoidal_freq_mod_depth'] * np.cos(freq_mod_time * 0.7))

    phase_x = time_liss + params['sinusoidal_phase']
    phase_y = time_liss * params['sinusoidal_phase_speed_ratio'] + params['sinusoidal_phase']
    phase_rotation = time_liss * params['sinusoidal_rotation_speed']

    x_normalized = (x - x.mean()) / (params['sinusoidal_scale'] * params['sinusoidal_x_scale'])
    y_normalized = (y - y.mean()) / (params['sinusoidal_scale'] * params['sinusoidal_y_scale'])

    cos_rot = np.cos(phase_rotation)
    sin_rot = np.sin(phase_rotation)
    x_rot = x_normalized * cos_rot - y_normalized * sin_rot
    y_rot = x_normalized * sin_rot + y_normalized * cos_rot

    amp_mod = 1 + params['sinusoidal_amplitude_mod'] * np.sin(time_val * 0.15)
    sinusoidal_x = np.sin(x_rot * a_freq_dynamic + phase_x)
    sinusoidal_y = np.sin(y_rot * b_freq_dynamic + phase_y)
    sinusoidal_pattern = sinusoidal_x * sinusoidal_y

    for harmonic in range(2, params['sinusoidal_harmonics'] + 1):
        decay = params['sinusoidal_harmonic_decay'] ** (harmonic - 1)
        harmonic_amp = amp_mod * decay
        harm_x_freq = a_freq_dynamic * harmonic
        harm_y_freq = b_freq_dynamic * harmonic
        harm_phase_x = phase_x * harmonic
        harm_phase_y = phase_y * harmonic
        cross_freq = (a_freq_dynamic + b_freq_dynamic) * 0.5 * harmonic
        cross_phase = (phase_x + phase_y) * 0.5
        sinusoidal_pattern += (
            harmonic_amp * (
                np.sin(x_rot * harm_x_freq + harm_phase_x) *
                np.sin(y_rot * harm_y_freq + harm_phase_y) +
                np.sin((x_rot + y_rot) * cross_freq + cross_phase) * 0.3 +
                np.cos((x_rot - y_rot) * cross_freq * 0.8 + cross_phase) * 0.2
            )
        )

    noise_layer = np.sin(x_rot * 0.01 + time_liss * 0.1) * np.cos(y_rot * 0.01 + time_liss * 0.15) * 0.1
    sinusoidal_combined = (sinusoidal_pattern + noise_layer) * params['sinusoidal_strength'] * amp_mod
    return sinusoidal_combined * 100


