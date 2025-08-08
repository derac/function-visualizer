from typing import Dict
from core.nd import xp as np, Array


def _fract(v: Array) -> Array:
    return v - np.floor(v)


def _hash(ix: Array, iy: Array, seed: float) -> Array:
    # Deterministic pseudo-random in [0,1)
    return _fract(np.sin(ix * 127.1 + iy * 311.7 + seed) * 43758.5453)


def apply(x: Array, y: Array, time_val: float, params: Dict) -> Array:
    # Parameters
    strength = float(params.get('gabor_strength', 0.9))
    scale = float(params.get('gabor_scale', 0.01))
    sigma = float(params.get('gabor_sigma', 0.35))  # gaussian width in cell units
    gabor_freq = float(params.get('gabor_frequency', 2.5))  # cycles per cell
    time_speed = float(params.get('gabor_time_speed', 0.3))

    # Ensure shapes are broadcasted
    if x.shape != y.shape:
        target_shape = np.broadcast(x, y).shape
        x = np.broadcast_to(x, target_shape)
        y = np.broadcast_to(y, target_shape)

    # Normalized grid coordinates
    gx = x * scale
    gy = y * scale

    ix = np.floor(gx)
    iy = np.floor(gy)
    fx = gx - ix
    fy = gy - iy

    # Bilinear weights
    wx0 = 1.0 - fx
    wx1 = fx
    wy0 = 1.0 - fy
    wy1 = fy

    two_pi = 2.0 * np.pi
    t_phase = time_val * time_speed * two_pi

    def cell_contrib(offset_x: float, offset_y: float, w: Array) -> Array:
        cx = ix + offset_x
        cy = iy + offset_y
        # Random orientation and phase per cell
        rnd0 = _hash(cx, cy, 12.9898)
        rnd1 = _hash(cx, cy, 78.233)
        theta = rnd0 * two_pi
        phase = rnd1 * two_pi + t_phase

        # Local coordinates to the cell center (use +0.5 to reduce seams)
        dx = gx - (cx + 0.5)
        dy = gy - (cy + 0.5)

        # Gaussian envelope and sinusoidal carrier
        r2 = dx * dx + dy * dy
        envelope = np.exp(-(r2) / (2.0 * (sigma * sigma) + 1e-8))
        proj = dx * np.cos(theta) + dy * np.sin(theta)
        carrier = np.cos(two_pi * gabor_freq * proj + phase)
        return w * envelope * carrier

    # Sum contributions from 4 neighbors with bilinear weights
    c00 = cell_contrib(0.0, 0.0, wx0 * wy0)
    c10 = cell_contrib(1.0, 0.0, wx1 * wy0)
    c01 = cell_contrib(0.0, 1.0, wx0 * wy1)
    c11 = cell_contrib(1.0, 1.0, wx1 * wy1)

    gabor = c00 + c10 + c01 + c11

    # Map from [-1,1] -> [0,1]
    gabor_norm = gabor * 0.5 + 0.5
    return gabor_norm



