import numpy as base_np
from typing import Dict, Tuple
from core.nd import xp as np, to_cpu, Array
from core.compute.registry import get_registry
from core.color.palettes import sample_palette, PALETTES
from core.color.space import apply_vibrance, enforce_min_variance
from core.color.tone import auto_contrast, apply_gamma_contrast_brightness
from core.feedback.state import feedback_state


def _normalize_single_function(output: Array) -> Array:
    """Normalize a single function's output to [0,1] range."""
    try:
        output_cpu = to_cpu(output)
        v_min = float(base_np.min(output_cpu))
        v_max = float(base_np.max(output_cpu))
    except Exception:
        v_min = float(np.min(output))
        v_max = float(np.max(output))
    
    if v_max > v_min:
        return (output - v_min) / (v_max - v_min + 1e-8)
    else:
        return np.zeros_like(output)


def _apply_enabled_operations(x: Array, y: Array, time_val: float, params: Dict) -> Array:
    registry = get_registry()
    operations = params.get('function_order', [])
    combined = np.zeros_like(x, dtype=np.float32)
    
    for op in operations:
        if not params.get(op, False):
            continue
        func = registry.get(op)
        if func is None:
            continue
        
        # Get raw function output
        contribution = func(x, y, time_val, params)
        
        # Normalize each function's output to [0,1] range
        contribution_norm = _normalize_single_function(contribution)
        
        # Apply function-specific strength parameter if available
        strength_param = op.replace('use_', '') + '_strength'
        strength = params.get(strength_param, 1.0)
        contribution_weighted = contribution_norm * strength
        
        # Add to combined result
        combined = combined + contribution_weighted
    
    return combined


def _normalize_combined(combined: Array, params: Dict) -> Tuple[Array, Array, Array]:
    clip_low = params.get('color_clip_low', 2.0)
    clip_high = params.get('color_clip_high', 98.0)
    try:
        combined_cpu = to_cpu(combined)
        v_low, v_high = base_np.percentile(base_np.ravel(combined_cpu), [clip_low, clip_high])
    except Exception:
        v_low, v_high = np.percentile(combined, [clip_low, clip_high])
    combined_clipped = np.clip(combined, v_low, v_high)
    combined_norm = (combined_clipped - v_low) / (v_high - v_low + 1e-8)
    combined_smooth = 1.0 / (1.0 + np.exp(-3.0 * (combined_norm - 0.5)))
    adjusted = np.power(combined_smooth, params['color_power'] * 0.5 + 0.5)
    return combined_norm, combined_smooth, adjusted


def _compute_initial_rgb(adjusted: Array, combined_smooth: Array, time_val: float, params: Dict) -> Tuple[Array, Array, Array, float]:
    time_factor = time_val * 0.08
    time_warped = time_val * params.get('time_warp_factor', 1.0)
    color_mode = params.get('color_mode', 'harmonic')
    if color_mode == 'palette':
        gamma = params.get('color_gamma', 1.0)
        contrast = params.get('color_contrast', 1.0)
        brightness = params.get('color_brightness', 1.0)
        vibrance = params.get('color_vibrance', 1.0)
        t = auto_contrast(adjusted, low_percentile=params.get('palette_clip_low', 2.0), high_percentile=params.get('palette_clip_high', 98.0))
        t = apply_gamma_contrast_brightness(t, gamma=gamma, contrast=contrast, brightness=brightness)
        t = (t + (np.sin(time_factor) * 0.05 + time_warped * 0.01)) % 1.0
        palette_name = params.get('palette_name', 'viridis')
        palette_reverse = params.get('palette_reverse', False)
        red, green, blue = sample_palette(t, name=palette_name, reverse=palette_reverse)
        red, green, blue = apply_vibrance(red, green, blue, vibrance=vibrance, saturation=params.get('color_saturation', 1.2) * 0.7)
        red *= params.get('color_red_mult', 1.0)
        green *= params.get('color_green_mult', 1.0)
        blue *= params.get('color_blue_mult', 1.0)
    else:
        primary_hue = (time_factor + combined_smooth * 4.5 + time_warped * 0.25) % 6.0
        secondary_hue = (time_factor * 0.7 + combined_smooth * 2.8 + time_warped * 0.15 + 2.0) % 6.0
        tertiary_hue = (time_factor * 0.4 + combined_smooth * 1.2 + time_warped * 0.35 + 4.0) % 6.0
        base_saturation = adjusted * params['color_saturation'] * 0.6
        variance = np.sin(combined_smooth * 12.5) * 0.1 + 0.1
        c = base_saturation * (0.85 + variance)

        def hue_to_rgb_soft(hue, intensity):
            segment = hue * params['color_hue_segments']
            rgb_phase = segment * 2 * np.pi
            red = intensity * params['color_red_mult'] * (1 + 0.3 * np.cos(rgb_phase)) * (0.8 + 0.2 * np.sin(secondary_hue))
            green = intensity * params['color_green_mult'] * (1 + 0.3 * np.cos(rgb_phase - 2.1)) * (0.8 + 0.2 * np.sin(tertiary_hue))
            blue = intensity * params['color_blue_mult'] * (1 + 0.3 * np.cos(rgb_phase + 2.1)) * (0.8 + 0.2 * np.sin(primary_hue))
            return red, green, blue

        red, green, blue = hue_to_rgb_soft(primary_hue, c)
    return red, green, blue, time_factor


def _modulate_colors(red: Array, green: Array, blue: Array, combined_smooth: Array, time_factor: float, params: Dict) -> Tuple[Array, Array, Array]:
    modulation_factor = 0.15
    phase_red = params.get('color_phase_red', 0) * np.pi / 180
    phase_green = params.get('color_phase_green', 0) * np.pi / 180
    phase_blue = params.get('color_phase_blue', 0) * np.pi / 180
    slow_mod = np.sin(time_factor * 0.8 + combined_smooth * 2 * np.pi) * modulation_factor
    breathing = (np.sin(time_factor * 0.3) * 0.05 + 0.95)
    red_mod = np.sin(time_factor * 0.7 + phase_red) * modulation_factor * 0.6
    green_mod = np.sin(time_factor * 0.9 + phase_green) * modulation_factor * 0.8
    blue_mod = np.sin(time_factor * 1.1 + phase_blue) * modulation_factor * 0.7
    warm_base = 1 + 0.2 * np.sin(time_factor * 0.15)
    cool_base = 1 + 0.2 * np.cos(time_factor * 0.15)
    red = np.clip(red * breathing * (1 + slow_mod * warm_base + red_mod) * params.get('color_red_mult', 1.0) * 0.85, 0, 0.95)
    green = np.clip(green * breathing * (1 + slow_mod * 0.9 + green_mod) * params.get('color_green_mult', 1.0) * 0.9, 0, 0.95)
    blue = np.clip(blue * breathing * (1 + slow_mod * cool_base + blue_mod) * params.get('color_blue_mult', 1.0) * 0.85, 0, 0.95)
    return red, green, blue


def _sanitize_and_luminance_fallback(red: Array, green: Array, blue: Array, adjusted: Array, params: Dict) -> Tuple[Array, Array, Array]:
    try:
        red = np.nan_to_num(red, nan=0.0, posinf=1.0, neginf=0.0)
        green = np.nan_to_num(green, nan=0.0, posinf=1.0, neginf=0.0)
        blue = np.nan_to_num(blue, nan=0.0, posinf=1.0, neginf=0.0)
    except Exception:
        red = np.where(np.isfinite(red), red, 0.0)
        green = np.where(np.isfinite(green), green, 0.0)
        blue = np.where(np.isfinite(blue), blue, 0.0)

    luminance = 0.299 * red + 0.587 * green + 0.114 * blue
    try:
        lum_max = float((luminance.max().get() if hasattr(luminance, 'get') else luminance.max()))
    except Exception:
        lum_max = float(base_np.max(luminance.get() if hasattr(luminance, 'get') else base_np.asarray(luminance)))
    if not base_np.isfinite(lum_max) or lum_max < 0.02:
        t_fb = auto_contrast(adjusted, low_percentile=2.0, high_percentile=98.0)
        t_fb = np.clip(t_fb + 0.1, 0.0, 1.0)
        palette_name = params.get('palette_name', 'viridis')
        palette_reverse = params.get('palette_reverse', False)
        red, green, blue = sample_palette(t_fb, name=palette_name, reverse=palette_reverse)
    return red, green, blue


def _final_colors(red: Array, green: Array, blue: Array, combined_norm: Array, adjusted: Array, params: Dict) -> Array:
    colors = np.stack([red, green, blue], axis=-1) * 255
    try:
        colors_cpu = colors.get() if hasattr(colors, 'get') else base_np.asarray(colors)
        max_val = float(base_np.max(colors_cpu))
    except Exception:
        max_val = 0.0
    if not base_np.isfinite(max_val) or max_val < 1.0:
        try:
            t_fb2 = auto_contrast(combined_norm, low_percentile=5.0, high_percentile=95.0)
        except Exception:
            t_fb2 = adjusted
        t_fb2 = np.clip(t_fb2 * 0.8 + 0.15, 0.0, 1.0)
        palette_name = params.get('palette_name', 'viridis')
        palette_reverse = params.get('palette_reverse', False)
        rr, gg, bb = sample_palette(t_fb2, name=palette_name, reverse=palette_reverse)
        rr, gg, bb = apply_vibrance(rr, gg, bb, vibrance=params.get('color_vibrance', 1.0), saturation=params.get('color_saturation', 1.2) * 0.7)
        colors = np.stack([rr, gg, bb], axis=-1) * 255
    return colors


def _update_feedback_and_return(colors: Array, time_val: float) -> Array:
    final_colors = colors.astype(np.float32) / 255.0
    feedback_state.previous_frame = final_colors
    feedback_state.time_sum = time_val
    return colors.astype(np.float32)


def compute_function(x: Array, y: Array, time_val: float, params: Dict) -> Array:
    if params is None:
        raise ValueError("params must not be None; call randomize_function_params() first")

    combined = _apply_enabled_operations(x, y, time_val, params)
    combined_norm, combined_smooth, adjusted = _normalize_combined(combined, params)
    red, green, blue, time_factor = _compute_initial_rgb(adjusted, combined_smooth, time_val, params)
    red, green, blue = _modulate_colors(red, green, blue, combined_smooth, time_factor, params)
    red, green, blue = _sanitize_and_luminance_fallback(red, green, blue, adjusted, params)
    if params.get('color_auto_normalize', True):
        red, green, blue = enforce_min_variance(red, green, blue, min_lum_range=0.35, min_sat_mean=0.35)
    colors = _final_colors(red, green, blue, combined_norm, adjusted, params)
    return _update_feedback_and_return(colors, time_val)

# Re-export for convenience
__all__ = [
    'compute_function',
    'PALETTES',
]


