from core.nd import xp as np
from core.compute.compose import compute_function


def generate_image_data(width, height, time_val, params, full_width=None, full_height=None):
    if full_width is None:
        full_width = width
    if full_height is None:
        full_height = height

    baseline_width = 640.0
    baseline_range = 400.0
    units_per_pixel = baseline_range / baseline_width

    x_world_range = units_per_pixel * max(full_width, 1)
    y_world_range = units_per_pixel * max(full_height, 1)

    x = np.linspace(-x_world_range / 2.0, x_world_range / 2.0, width)[:, None]
    y = np.linspace(-y_world_range / 2.0, y_world_range / 2.0, height)[None, :]

    colors = compute_function(x, y, time_val, params)
    return np.transpose(colors, (1, 0, 2))


