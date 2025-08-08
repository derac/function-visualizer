"""NumPy/CuPy interop helpers and array module selection."""
from utils.hardware import get_array_module
import numpy as base_np

# Select array module (numpy or cupy)
xp = get_array_module()

def to_cpu(array):
    """Return a NumPy array for robust CPU-side ops like percentiles.

    Handles CuPy arrays by calling .get() when available.
    """
    try:
        return array.get() if hasattr(array, 'get') else base_np.asarray(array)
    except Exception:
        # Last resort: try numpy array conversion directly
        return base_np.asarray(array)


