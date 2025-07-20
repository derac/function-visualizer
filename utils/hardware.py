"""Hardware detection utilities for GPU/CPU selection."""

try:
    import cupy as np
    CUPY_AVAILABLE = True
except ImportError:
    import numpy as np
    CUPY_AVAILABLE = False

def get_array_module():
    """Get the appropriate array module (cupy if available, numpy otherwise)."""
    return np

def get_hardware_info():
    """Get hardware information string."""
    return "CUDA" if CUPY_AVAILABLE else "CPU"