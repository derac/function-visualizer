from typing import Dict
from core.nd import xp as np, Array


def apply(x: Array, y: Array, time_val: float, params: Dict, context: Dict) -> Array:
    time_noise = time_val * params['noise_time_speed']
    noise_scale = params['noise_scale']

    if x.shape != y.shape:
        target_shape = np.broadcast(x, y).shape
        x = np.broadcast_to(x, target_shape)
        y = np.broadcast_to(y, target_shape)

    noise_val = np.zeros_like(x)
    for i in range(params['noise_octaves']):
        octave_freq = noise_scale * (2 ** i)
        octave_amp = 0.5 ** i
        octave_val = (np.sin(x * octave_freq + time_noise * (1 + i * 0.2)) +
                      np.cos(y * octave_freq + time_noise * (1 + i * 0.3)) +
                      np.sin((x + y) * octave_freq * 0.7 + time_noise * (1 + i * 0.1)) +
                      np.cos((x - y) * octave_freq * 0.8 + time_noise * (1 + i * 0.15)))
        noise_val += octave_val * octave_amp

    noise_val = (noise_val + 4) / 8.0
    return noise_val


