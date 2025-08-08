from typing import Dict
from core.nd import xp as np, Array


def _repeat_to_tile(p: Array, tile: float) -> Array:
    # Map world coordinates to repeated tile coordinates centered at 0
    t = p / tile + 0.5
    t = t - np.floor(t)
    return (t - 0.5) * tile


def _sdf_circle(px: Array, py: Array, radius: float) -> Array:
    return np.sqrt(px * px + py * py) - radius


def _sdf_rounded_box(px: Array, py: Array, half_extent: float, round_radius: float) -> Array:
    ax = np.abs(px) - half_extent
    ay = np.abs(py) - half_extent
    ox = np.maximum(ax, 0.0)
    oy = np.maximum(ay, 0.0)
    outside = np.sqrt(ox * ox + oy * oy)
    inside = np.minimum(np.maximum(ax, ay), 0.0)
    return outside + inside - round_radius


def apply(x: Array, y: Array, time_val: float, params: Dict, context: Dict) -> Array:
    # Parameters
    tile_scale = float(max(1e-3, params.get('sdf_tile_scale', 60.0)))
    repeat = max(1, int(params.get('sdf_repeat', 4)))
    tile = tile_scale / float(repeat)

    # Repeat domain into tiles
    px = _repeat_to_tile(x, tile)
    py = _repeat_to_tile(y, tile)

    # Optional slow rotation for dynamics
    theta = time_val * float(params.get('sdf_time_speed', 0.2))
    c, s = np.cos(theta), np.sin(theta)
    rx = px * c - py * s
    ry = px * s + py * c

    # Base size from tile
    base_radius = float(params.get('sdf_radius', 0.35)) * tile
    round_frac = float(params.get('sdf_box_rounded', 0.15))
    d_circle = _sdf_circle(rx, ry, base_radius)
    d_box = _sdf_rounded_box(rx, ry, base_radius, base_radius * round_frac)

    # Avoid calling xp.clip on a Python float (some backends try obj.clip and fail)
    mix_param = float(params.get('sdf_shape_mix', 0.5))
    mix = min(1.0, max(0.0, mix_param))
    d = (1.0 - mix) * d_circle + mix * d_box

    # Soft boundary to produce smooth contribution
    edge = tile * float(params.get('sdf_softness', 1.5)) * 0.1
    edge = max(edge, 1e-6)
    # Inside -> high values, outside -> low values
    val = 1.0 / (1.0 + np.exp(d / edge * 6.0))

    return val


