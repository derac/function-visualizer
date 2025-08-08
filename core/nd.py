"""NumPy/CuPy interop helpers and array module selection."""
from typing import TYPE_CHECKING, Union
from utils.hardware import get_array_module
import numpy as base_np

# Select array module (numpy or cupy)
xp = get_array_module()

if TYPE_CHECKING:
    import cupy as cp  # type: ignore
    CuPyNDArray = cp.ndarray  # type: ignore[attr-defined]
else:
    try:
        import cupy as cp  # type: ignore
        CuPyNDArray = cp.ndarray  # type: ignore[attr-defined]
    except Exception:
        class CuPyNDArray:  # type: ignore
            pass

# Union of NumPy and CuPy ndarrays
Array = Union[base_np.ndarray, "CuPyNDArray"]

def to_cpu(array: Array) -> base_np.ndarray:
    """Return a NumPy array for robust CPU-side ops like percentiles.

    Handles CuPy arrays by calling .get() when available.
    """
    try:
        return array.get() if hasattr(array, 'get') else base_np.asarray(array)
    except Exception:
        # Last resort: try numpy array conversion directly
        return base_np.asarray(array)


