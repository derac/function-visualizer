import numpy as base_np
from core.nd import xp as np, to_cpu


def apply_gamma_contrast_brightness(t, gamma=1.0, contrast=1.0, brightness=1.0):
    t = np.clip(t, 0.0, 1.0)
    t = t ** gamma
    t = (t - 0.5) * contrast + 0.5
    t = np.clip(t * brightness, 0.0, 1.0)
    return t


def auto_contrast(t, low_percentile=5.0, high_percentile=95.0):
    """Stretch values to [0,1] using robust percentiles.
    Uses CPU percentiles to avoid NaN propagation issues.
    """
    try:
        t_cpu = to_cpu(t)
        lo, hi = base_np.percentile(base_np.ravel(t_cpu), [low_percentile, high_percentile])
        if not base_np.isfinite(hi - lo) or (hi - lo) < 1e-6:
            return np.clip((t - lo) / (1e-6), 0.0, 1.0)
    except Exception:
        lo = np.percentile(t, low_percentile)
        hi = np.percentile(t, high_percentile)
    return np.clip((t - lo) / (hi - lo + 1e-6), 0.0, 1.0)


