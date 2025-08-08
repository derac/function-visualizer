from typing import Dict
from core.nd import xp as np, Array


def apply(x: Array, y: Array, time_val: float, params: Dict) -> Array:
    time_cellular = time_val * params['cellular_time_translate']
    grid_x = ((x + time_cellular * 5) / params['cellular_scale']).astype(int)
    grid_y = ((y + time_cellular * 3) / params['cellular_scale']).astype(int)
    cell_val = np.sin(grid_x * 0.1 + time_cellular) * np.cos(grid_y * 0.1 + time_cellular * 1.5)
    return cell_val


