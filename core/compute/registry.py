from typing import Callable, Dict
from core.nd import Array

from core.patterns.sin_cos import apply_sin, apply_cos
from core.patterns.xor import apply as apply_xor
from core.patterns.cellular import apply as apply_cellular
from core.patterns.domain_warp import apply as apply_domain_warp
from core.patterns.polar import apply as apply_polar
from core.patterns.noise import apply as apply_noise
from core.patterns.abs_transform import apply as apply_abs
from core.patterns.power import apply as apply_power
from core.patterns.voronoi import apply as apply_voronoi
from core.patterns.reaction_diffusion import apply as apply_reaction_diffusion
from core.patterns.sinusoidal_field import apply as apply_sinusoidal_field
from core.patterns.sdf_shapes import apply as apply_sdf_shapes


OpFunc = Callable[[Array, Array, float, dict, dict], Array]


def get_registry() -> Dict[str, OpFunc]:
    return {
        'use_sin': apply_sin,
        'use_cos': apply_cos,
        'use_xor': apply_xor,
        'use_cellular': apply_cellular,
        'use_domain_warp': apply_domain_warp,
        'use_polar': apply_polar,
        'use_noise': apply_noise,
        'use_abs': apply_abs,
        'use_power': apply_power,
        'use_voronoi': apply_voronoi,
        'use_reaction_diffusion': apply_reaction_diffusion,
        'use_sinusoidal_field': apply_sinusoidal_field,
        'use_sdf_shapes': apply_sdf_shapes,
    }


