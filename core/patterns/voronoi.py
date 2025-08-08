import random
from core.nd import xp as np
from core.feedback.state import feedback_state


def apply(x, y, time_val, params, context):
    num_points = params.get('voronoi_points', 8)
    voronoi_strength = params.get('voronoi_strength', 1.0)
    voronoi_scale = params.get('voronoi_scale', 1.0)

    if 'voronoi_seeds' not in params:
        seeds_x = np.random.uniform(x.min(), x.max(), num_points)
        seeds_y = np.random.uniform(y.min(), y.max(), num_points)
        params['voronoi_seeds'] = [[float(sx), float(sy)] for sx, sy in zip(seeds_x.tolist(), seeds_y.tolist())]
    else:
        params['voronoi_seeds'] = [[
            float(seeds_x) + 10.0 * (random.randint(0, 4) - 2) * (time_val - feedback_state.time_sum),
            float(seeds_y) + 10.0 * (random.randint(0, 4) - 2) * (time_val - feedback_state.time_sum)
        ] for (seeds_x, seeds_y) in params['voronoi_seeds']]

    voronoi_distances = np.inf * np.ones_like(x)
    for seed_x, seed_y in params['voronoi_seeds']:
        distance = np.sqrt((x - seed_x)**2 + (y - seed_y)**2)
        voronoi_distances = np.minimum(voronoi_distances, distance)

    voronoi_norm = voronoi_distances * voronoi_scale
    return voronoi_norm * 50 * voronoi_strength


