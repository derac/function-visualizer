"""NumPy/CuPy interop helpers and array module selection."""
from typing import Any
from utils.hardware import get_array_module
import numpy as base_np

# Select array module (numpy or cupy)
xp = get_array_module()

# Generic array type that may be a NumPy or CuPy ndarray
Array = Any

def to_cpu(array: Array) -> base_np.ndarray:
    """Return a NumPy array for robust CPU-side ops like percentiles.

    Handles CuPy arrays by calling .get() when available.
    """
    try:
        return array.get() if hasattr(array, 'get') else base_np.asarray(array)
    except Exception:
        # Last resort: try numpy array conversion directly
        return base_np.asarray(array)


