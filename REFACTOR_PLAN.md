## Refactor Plan: Split `math_function.py`

### Goals
- Improve readability, testability, and extensibility (adding new ops/patterns).
- Keep existing behavior and params, so saved presets remain compatible.
- Preserve the three public entrypoints and palette access used by `visualizer.py`.

### New module layout
- `core/nd.py`
  - Select array module: `xp = get_array_module()` and helpers like `to_cpu`.
- `core/color/palettes.py`
  - `PALETTES` (public) and `sample_palette(t, name, reverse)`.
- `core/color/space.py`
  - `rgb_to_hsv`, `hsv_to_rgb`, `apply_vibrance`, `enforce_min_variance`.
- `core/color/tone.py`
  - `auto_contrast`, `apply_gamma_contrast_brightness`.
- `core/feedback/state.py`
  - `FeedbackState` singleton with `.reset()` and fields `previous_frame`, `time_sum`.
- `core/feedback/compute.py`
  - `compute_feedback_values(...)` reading from the state singleton.
- `core/patterns/` (one file per op; pure functions)
  - `xor.py`, `sin_cos.py`, `cellular.py`, `domain_warp.py`, `polar.py`, `noise.py`, `abs_transform.py`, `power.py`, `voronoi.py`, `reaction_diffusion.py`, `sinusoidal_field.py`.
- `core/compute/registry.py`
  - Map op names → callables with signature `(x, y, time_val, params, context)`.
- `core/compute/compose.py`
  - `compute_function(...)` orchestrator: precompute shared values; apply enabled ops; run color mapping; update feedback state.
- `core/params.py`
  - `randomize_function_params()` returning `{**operations, **params}` with `function_order`.
- `core/rendering/image.py`
  - `generate_image_data(...)` that builds world coordinates and calls `compute_function(...)`.

### Public API (backward compatible)
- Keep `compute_function`, `randomize_function_params`, `generate_image_data` and palette access.
- Implement a shim in `math_function.py` that re-exports from the new modules, including an alias `_PALETTES` for compatibility.

### Migration steps
1. Add `core/nd.py` and use `xp`/`to_cpu` where CPU percentiles are needed.
2. Move color utils into `core/color/*`.
3. Extract feedback state and computations into `core/feedback/*`.
4. Extract each pattern into `core/patterns/*.py` with a uniform interface.
5. Build `core/compute/registry.py` mapping op names to functions.
6. Implement `core/compute/compose.py` by porting the original main loop and color mapping tail.
7. Move `randomize_function_params()` to `core/params.py` unchanged (preserve keys/ranges).
8. Move `generate_image_data()` to `core/rendering/image.py` unchanged.
9. Keep `math_function.py` as a shim that re-exports the API and `_PALETTES`.
10. Smoke test: launch app, randomize, palette cycle, verify feedback and reaction-diffusion; CPU/GPU paths OK.

### Notes
- Keep parameter names/structure identical to preserve existing saves.
- Use CPU percentile computations for stability (`to_cpu`).
- Maintain the feedback singleton semantics.
- Add type hints and docstrings in new modules.


