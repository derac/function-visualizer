"""Mathematical functions for visualization and computation with feedback loop."""

import random
import numpy as base_np  # For some operations that need pure numpy
from utils.hardware import get_array_module
from utils.logger import logger

# Get the appropriate array module
np = get_array_module()

# Global state for feedback loop
class FeedbackState:
    def __init__(self):
        self.previous_frame = None
        self.feedback_intensity = 0.7
        self.time_sum = 0.0
        
    def reset(self):
        """Reset the feedback state"""
        self.previous_frame = None
        self.time_sum = 0.0

feedback_state = FeedbackState()

def compute_function(x, y, time_val, params):
    """
    Compute a mathematical function for visualization based on parameters.
    
    Args:
        x: x-coordinates array
        y: y-coordinates array
        time_val: current time value for animation
        params: dictionary of function parameters
        
    Returns:
        RGB colors array
    """
    if params is None:
        params = randomize_function_params()
    
    # Add time-based modulation to all parameters
    wave1_mult = params['wave1_mult'] * (1 + 0.3 * np.sin(time_val * 0.15))
    domain_warp_strength = params['domain_warp_strength'] * (1 + 0.4 * np.sin(time_val * 0.25))
    
    # Ensure x and y are arrays
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    
    # Apply time-based translations
    time_offset_x = time_val * params['time_translate_x']
    time_offset_y = time_val * params['time_translate_y']
    
    # Apply wave translations with time evolution
    wave1_offset_x = time_offset_x + params['wave1_translate_x']
    wave1_offset_y = time_offset_y + params['wave1_translate_y']
    
    # Normalize with time-based scaling and translations
    x_normalized = (x - wave1_offset_x) / wave1_mult
    y_normalized = (y - wave1_offset_y) / wave1_mult
    
    # Dynamic wave components initialized as arrays
    wave1 = np.zeros_like(x)
    wave2 = np.zeros_like(x)
    
    if params['use_sin']:
        wave1 = np.abs(np.sin(x_normalized * params['wave1_freq'] + time_val * params['time_speed']))
    if params['use_cos']:
        wave2 = np.abs(np.cos(y_normalized * params['wave2_freq'] + time_val * params['time_speed']))
    
    combined = np.zeros_like(x, dtype=np.float32)
    
    # Apply functions in predetermined order
    operations = params.get('function_order', [])
    
    for op in operations:
        if not params.get(op, False):
            continue
        
        if op == 'use_xor':
            # Enhanced time-based translation and shape transformation
            time_shift_x = time_val * params.get('xor_translate_x', 0.5)
            time_shift_y = time_val * params.get('xor_translate_y', 0.3)
            morph_phase = time_val * params.get('xor_morph_speed', 0.2)
            
            # Apply smooth translation to coordinates
            trans_x = x + np.sin(time_shift_x) * params.get('xor_translate_range', 50)
            trans_y = y + np.cos(time_shift_y) * params.get('xor_translate_range', 50)
            
            # Subtle shape morphing through coordinate transformation
            morph_x = trans_x + np.sin(morph_phase + trans_y * 0.01) * (5 + 3 * np.sin(morph_phase * 0.7))
            morph_y = trans_y + np.cos(morph_phase + trans_x * 0.01) * (5 + 3 * np.cos(morph_phase * 0.5))
            
            # Add gentle rotation effect
            rot_angle = np.sin(time_val * 0.1) * 0.1  # Small rotation amount
            rot_x = morph_x * np.cos(rot_angle) - morph_y * np.sin(rot_angle)
            rot_y = morph_x * np.sin(rot_angle) + morph_y * np.cos(rot_angle)
            
            # Clean XOR with morphing; vary bitmask over time (exponent in [1,3])
            mask_exponent = int(np.clip((np.sin(time_val * 0.1) + 1.0) * 1.5, 1, 3))
            xor_mask = np.bitwise_xor(rot_x.astype(np.int32), rot_y.astype(np.int32)) & (1 << mask_exponent)

            # Gentle intensity modulation
            intensity = 0.8 + 0.2 * np.sin(time_val * 0.05 + (rot_x + rot_y) * 0.001)
            combined = combined + xor_mask * intensity * params['xor_strength'] * 20


        elif op == 'use_sin':
            combined = combined + wave1 * 200
        
        elif op == 'use_cos':
            combined = combined + wave2 * 200
        
        elif op == 'use_cellular':
            time_cellular = time_val * params['cellular_time_translate']
            grid_x = ((x + time_cellular * 5) / params['cellular_scale']).astype(int)
            grid_y = ((y + time_cellular * 3) / params['cellular_scale']).astype(int)
            cell_val = np.sin(grid_x * 0.1 + time_cellular) * np.cos(grid_y * 0.1 + time_cellular * 1.5)
            combined = combined + cell_val * 150
        
        elif op == 'use_domain_warp':
            animated_strength = domain_warp_strength * (1 + 0.3 * np.sin(time_val * params['domain_warp_time_factor']))
            time_phase = time_val * 0.5
            warped_x = (x + animated_strength * np.sin(y * 0.1 + time_phase + 
                                                    np.sin(time_phase * 2) * 0.5))/(3*np.sin(time_val)+4)
            warped_y = (y + animated_strength * np.cos(x * 0.1 + time_phase * 0.7 + 
                                                    np.sin(time_phase * 1.5) * 0.3))/(3*np.sin(time_val)+4)
            combined = combined + np.sin(warped_x) * np.cos(warped_y) * 25
        
        elif op == 'use_polar':
            # Polar coordinates computed in world space (x, y are already centered around 0)
            # Dynamic center orbits around the origin in coordinate units
            center_x = np.sin(time_val * params['polar_orbit_speed_x']) * params['polar_orbit_range']
            center_y = np.cos(time_val * params['polar_orbit_speed_y']) * params['polar_orbit_range']

            x_rel = x - center_x
            y_rel = y - center_y

            r = np.sqrt(x_rel**2 + y_rel**2) + 1e-8  # Avoid division by zero
            theta = np.arctan2(y_rel, x_rel) + time_val * params['polar_rotation_speed']

            # Apply polar transformation with time-based frequency modulation
            freq_mod = 1 + 0.3 * np.sin(time_val * 0.15)
            polar_wave = (
                np.sin(r * params['polar_freq_r'] * freq_mod + theta * params['polar_freq_theta'])
                * np.cos(theta * params['polar_theta_harmonics'] + time_val * params['polar_time_factor'])
            )

            # Add spiral motion effect
            spiral_angle = theta + r * params['polar_spiral_factor']
            spiral_wave = np.sin(spiral_angle * params['polar_spiral_freq'] + time_val * params['polar_spiral_speed'])

            combined = combined + (polar_wave + spiral_wave * 0.5) * 120 * params['polar_strength']
        
        elif op == 'use_noise':
            # Simplex-like noise approximation using sin/cos combinations
            time_noise = time_val * params['noise_time_speed']
            noise_scale = params['noise_scale']
            
            # Ensure x and y have compatible shapes
            if x.shape != y.shape:
                target_shape = np.broadcast(x, y).shape
                x = np.broadcast_to(x, target_shape)
                y = np.broadcast_to(y, target_shape)
            
            # Create multiple octaves of pseudo-noise
            noise_val = np.zeros_like(x)
            for i in range(params['noise_octaves']):
                octave_freq = noise_scale * (2 ** i)
                octave_amp = 0.5 ** i
                
                # Use hash-like approach with sine waves for noise
                octave_val = (np.sin(x * octave_freq + time_noise * (1 + i * 0.2)) + 
                             np.cos(y * octave_freq + time_noise * (1 + i * 0.3)) + 
                             np.sin((x + y) * octave_freq * 0.7 + time_noise * (1 + i * 0.1)) +
                             np.cos((x - y) * octave_freq * 0.8 + time_noise * (1 + i * 0.15)))
                
                noise_val += octave_val * octave_amp
            
            # Remap noise to 0-1 range
            noise_val = (noise_val + 4) / 8.0  # Normalize based on expected range
            combined = combined + noise_val * 100.0 * params['noise_strength']
        
        
        elif op == 'use_abs':
            # Absolute value transformations with time-based modulation
            time_abs = time_val * params['abs_time_speed']
            abs_wave1 = np.abs(np.sin(x * params['abs_freq_x'] * np.sin(time_abs/10)))
            abs_wave2 = np.abs(np.cos(y * params['abs_freq_y'] * np.cos(time_abs/10) * 0.7))
            
            # Create rich patterns through absolute value combinations
            abs_combo = abs_wave1 * abs_wave2 + np.abs(abs_wave1 - abs_wave2)
            
            # Add time-varying cross terms
            cross_term = np.abs(np.sin((x + y) * params['abs_freq_xy'] + time_abs * 1.5))
            combined = combined + (abs_combo + cross_term) * 120 * params['abs_strength']
        
        elif op == 'use_power':
            # Power transformations with time-evolving exponents
            time_power = time_val * params['power_time_speed']
            base = (np.sin(x * params['power_freq_x'] + time_power) + 
                   np.cos(y * params['power_freq_y'] + time_power * 0.8) + 2) / 2  # Ensure positive base
            
            # Dynamic power exponent with smooth transitions
            exp_base = params['power_exponent']
            exp_mod = exp_base * (1 + 0.4 * np.sin(time_power * params['power_exp_mod_freq']))
            
            # Apply power function with gradient safety
            power_val = np.power(base, exp_mod)
            power_val = np.nan_to_num(power_val, nan=0.0, posinf=1000.0, neginf=0.0)
            
            # Add additional power-based harmonics
            harmonic = np.power(
                (np.sin(x * params['power_freq_x'] * 2 + time_power * 1.5) + 
                 np.cos(y * params['power_freq_y'] * 2 + time_power * 2) + 2) / 2,
                exp_mod * 0.5
            )
            harmonic = np.nan_to_num(harmonic, nan=0.0, posinf=1000.0, neginf=0.0)
            
            combined = combined + (power_val + harmonic * 0.3) * 100 * params['power_strength']
        
        elif op == 'use_feedback':
            # Feedback integration from previous frame
            if feedback_state.previous_frame is not None:
                feedback_values = compute_feedback_values(x, y, time_val, params)
                combined = combined + feedback_values * 150
        
        elif op == 'use_voronoi':
            # Voronoi distance field - random points used as seeds
            num_points = params.get('voronoi_points', 8)
            voronoi_strength = params.get('voronoi_strength', 1.0)
            voronoi_scale = params.get('voronoi_scale', 1.0)
            
            # Generate random seed points if not already generated
            if 'voronoi_seeds' not in params:
                # Create random seed points within coordinate bounds as plain Python floats
                seeds_x = np.random.uniform(x.min(), x.max(), num_points)
                seeds_y = np.random.uniform(y.min(), y.max(), num_points)
                params['voronoi_seeds'] = [[float(sx), float(sy)] for sx, sy in zip(seeds_x.tolist(), seeds_y.tolist())]
            else:
                params['voronoi_seeds'] = [[
                        float(seeds_x) + 10.0 * (random.randint(0, 4) - 2) * (time_val - feedback_state.time_sum),
                        float(seeds_y) + 10.0 * (random.randint(0, 4) - 2) * (time_val - feedback_state.time_sum)
                    ] for (seeds_x, seeds_y) in params['voronoi_seeds']]
            
            # Calculate distance to nearest seed point
            voronoi_distances = np.inf * np.ones_like(x)
            
            for seed_x, seed_y in params['voronoi_seeds']:
                # Calculate Euclidean distance to this seed
                distance = np.sqrt((x - seed_x)**2 + (y - seed_y)**2)
                voronoi_distances = np.minimum(voronoi_distances, distance)
            
            # Normalize and scale the distance field
            voronoi_norm = voronoi_distances * voronoi_scale
            combined = combined + voronoi_norm * 50 * voronoi_strength

        elif op == 'use_reaction_diffusion':
            # Reaction-diffusion using Gray-Scott model with previous frame
            if feedback_state.previous_frame is not None:
                reaction_diffusion_values = compute_reaction_diffusion(x, y, time_val, params)
                combined = combined + reaction_diffusion_values * 100

        elif op == 'use_sinusoidal_field':
            # Enhanced Sinusoidal field curves with organic complexity
            time_liss = time_val * params['sinusoidal_time_speed']

            # Dynamic frequency modulation for breathing patterns
            freq_mod_time = time_val * params['sinusoidal_freq_mod_speed']
            a_freq_dynamic = params['sinusoidal_a_freq'] * (1 + params['sinusoidal_freq_mod_depth'] * np.sin(freq_mod_time))
            b_freq_dynamic = params['sinusoidal_b_freq'] * (1 + params['sinusoidal_freq_mod_depth'] * np.cos(freq_mod_time * 0.7))

            # Calculate phase with complex time evolution
            phase_x = time_liss + params['sinusoidal_phase']
            phase_y = time_liss * params['sinusoidal_phase_speed_ratio'] + params['sinusoidal_phase']
            phase_rotation = time_liss * params['sinusoidal_rotation_speed']

            # Independent axis scaling for rectangular patterns
            x_normalized = (x - x.mean()) / (params['sinusoidal_scale'] * params['sinusoidal_x_scale'])
            y_normalized = (y - y.mean()) / (params['sinusoidal_scale'] * params['sinusoidal_y_scale'])

            # Apply rotation to coordinates
            cos_rot = np.cos(phase_rotation)
            sin_rot = np.sin(phase_rotation)
            x_rot = x_normalized * cos_rot - y_normalized * sin_rot
            y_rot = x_normalized * sin_rot + y_normalized * cos_rot

            # Amplitude modulation for pulsing effect
            amp_mod = 1 + params['sinusoidal_amplitude_mod'] * np.sin(time_val * 0.15)

            # Base Sinusoidal field patterns
            sinusoidal_x = np.sin(x_rot * a_freq_dynamic + phase_x)
            sinusoidal_y = np.sin(y_rot * b_freq_dynamic + phase_y)

            # Multi-layer harmonic patterns
            sinusoidal_pattern = sinusoidal_x * sinusoidal_y

            # Add harmonic layers with decay
            for harmonic in range(2, params['sinusoidal_harmonics'] + 1):
                decay = params['sinusoidal_harmonic_decay'] ** (harmonic - 1)
                harmonic_amp = amp_mod * decay
                
                # Higher harmonics with frequency ratios
                harm_x_freq = a_freq_dynamic * harmonic
                harm_y_freq = b_freq_dynamic * harmonic
                harm_phase_x = phase_x * harmonic
                harm_phase_y = phase_y * harmonic
                
                # Cross-patterns between harmonics
                cross_freq = (a_freq_dynamic + b_freq_dynamic) * 0.5 * harmonic
                cross_phase = (phase_x + phase_y) * 0.5
                
                sinusoidal_pattern += (
                    harmonic_amp * (
                        np.sin(x_rot * harm_x_freq + harm_phase_x) * 
                        np.sin(y_rot * harm_y_freq + harm_phase_y) +
                        
                        np.sin((x_rot + y_rot) * cross_freq + cross_phase) * 0.3 +
                        np.cos((x_rot - y_rot) * cross_freq * 0.8 + cross_phase) * 0.2
                    )
                )

            # Add subtle noise for organic texture
            noise_layer = np.sin(x_rot * 0.01 + time_liss * 0.1) * np.cos(y_rot * 0.01 + time_liss * 0.15) * 0.1
            sinusoidal_combined = (sinusoidal_pattern + noise_layer) * params['sinusoidal_strength'] * amp_mod
            combined = combined + sinusoidal_combined * 100
    
    # Smooth color remapping using sigmoid-like functions
    # Normalize combined values and apply smooth transformation
    combined_norm = (combined - np.min(combined)) / (np.max(combined) - np.min(combined) + 1e-8)
    
    # Use smooth sigmoid remapping for smooth transitions
    smooth_factor = 3.0
    combined_smooth = 1.0 / (1.0 + np.exp(-smooth_factor * (combined_norm - 0.5)))
    
    # Apply power curve for additional color control
    adjusted = np.power(combined_smooth, params['color_power'] * 0.5 + 0.5)
    
    # Create smooth color flows with more organic variation
    time_factor = time_val * 0.08  # Slightly slower for smoother transitions
    time_warped = time_val * params.get('time_warp_factor', 1.0)
    
    # More nuanced hue generation with additional harmonic layers
    primary_hue = (time_factor + combined_smooth * 4.5 + time_warped * 0.25) % 6.0
    secondary_hue = (time_factor * 0.7 + combined_smooth * 2.8 + time_warped * 0.15 + 2.0) % 6.0
    tertiary_hue = (time_factor * 0.4 + combined_smooth * 1.2 + time_warped * 0.35 + 4.0) % 6.0
    
    # Reduced saturation for more pastel-like colors
    base_saturation = adjusted * params['color_saturation'] * 0.6  # Reduced from 1.0 to 0.6
    variance = np.sin(combined_smooth * 12.5) * 0.1 + 0.1
    c = base_saturation * (0.85 + variance)  # Subtle saturation variation
    
    # Generate RGB from hue with white-light mixing for toned colors
    def hue_to_rgb_soft(hue, intensity):
        # Softer color wheel with white-light blending
        segment = hue * params['color_hue_segments']
        rgb_phase = segment * 2 * np.pi
        
        red = intensity * params['color_red_mult'] * (1 + 0.3 * np.cos(rgb_phase)) * (0.8 + 0.2 * np.sin(secondary_hue))
        green = intensity * params['color_green_mult'] * (1 + 0.3 * np.cos(rgb_phase - 2.1)) * (0.8 + 0.2 * np.sin(tertiary_hue))
        blue = intensity * params['color_blue_mult']  * (1 + 0.3 * np.cos(rgb_phase + 2.1)) * (0.8 + 0.2 * np.sin(primary_hue))
        
        return red, green, blue
    
    red, green, blue = hue_to_rgb_soft(primary_hue, c)
    
    # More subtle time-based modulation with color washing effects
    modulation_factor = 0.15  # Much gentler modulation
    phase_red = params.get('color_phase_red', 0) * np.pi / 180
    phase_green = params.get('color_phase_green', 0) * np.pi / 180
    phase_blue = params.get('color_phase_blue', 0) * np.pi / 180
    
    # Create complex but subtle color variations
    slow_mod = np.sin(time_factor * 0.8 + combined_smooth * 2 * np.pi) * modulation_factor
    breathing = (np.sin(time_factor * 0.3) * 0.05 + 0.95)  # Gentle breathing effect
    
    # Apply phase-shifted color modulation with ambient light effects
    red_mod = np.sin(time_factor * 0.7 + phase_red) * modulation_factor * 0.6
    green_mod = np.sin(time_factor * 0.9 + phase_green) * modulation_factor * 0.8
    blue_mod = np.sin(time_factor * 1.1 + phase_blue) * modulation_factor * 0.7
    
    # Enhanced color multipliers with temperature variations
    warm_base = 1 + 0.2 * np.sin(time_factor * 0.15)  # Warm-to-cool shift
    cool_base = 1 + 0.2 * np.cos(time_factor * 0.15)
    
    red = np.clip(red * breathing * (1 + slow_mod * warm_base + red_mod) * params['color_red_mult'] * 0.85, 0, 0.9)
    green = np.clip(green * breathing * (1 + slow_mod * 0.9 + green_mod) * params['color_green_mult'] * 0.9, 0, 0.9)
    blue = np.clip(blue * breathing * (1 + slow_mod * cool_base + blue_mod) * params['color_blue_mult'] * 0.75, 0, 0.9)
    
    # Final smooth scaling to 8-bit values
    colors = np.stack([red, green, blue], axis=-1) * 255
    
    # Store frame for next iteration - save final colors after all operations
    final_colors = colors.astype(np.float32) / 255.0
    feedback_state.previous_frame = final_colors
    feedback_state.time_sum = time_val
    
    return colors.astype(np.float32)


def compute_feedback_values(x, y, time_val, params):
    """
    Compute feedback values from the previous frame for the feedback loop.
    
    Args:
        x: x-coordinates array
        y: y-coordinates array  
        time_val: current time value for animation
        params: function parameters
        
    Returns:
        Feedback values array to be added to current frame computation
    """
    if feedback_state.previous_frame is None:
        return np.zeros_like(x)
    
    # Extract previous frame dimensions
    prev_height, prev_width = feedback_state.previous_frame.shape[:2]
    
    # Create coordinate arrays for resampling
    # Need to map the current coordinates to the previous framebuffer coordinates
    aspect_ratio = prev_width / prev_height
    
    # Center coordinates on current coordinate system
    if len(x.shape) == 2:
        center_x = x.shape[0] // 2
        center_y = x.shape[1] // 2
    else:
        # Handle 1D arrays (broadcast scenario)
        center_x = len(x) // 2
        center_y = len(y) // 2
    
    # Scale previous frame to match current coordinate system
    x_range = x.max() - x.min()
    y_range = y.max() - y.min()
    
    # Normalize current coordinates to [0, 1] then scale to prev framebuffer size
    x_norm = (x - x.min()) / (x_range + 1e-8)
    y_norm = (y - y.min()) / (y_range + 1e-8)
    
    x_prev = x_norm * (prev_width - 1)
    y_prev = y_norm * (prev_height - 1)
    
    # Ensure coordinates are within bounds
    x_prev = np.clip(x_prev, 0, prev_width - 1)
    y_prev = np.clip(y_prev, 0, prev_height - 1)
    
    # Sample from previous frame using nearest neighbor for simplicity
    x_int = x_prev.astype(int)
    y_int = y_prev.astype(int)
    
    # Extract RGB values from previous frame
    feedback_rgb = feedback_state.previous_frame[y_int, x_int]
    
    # Convert to grayscale for feedback signal
    feedback_gray = 0.299 * feedback_rgb[..., 0] + 0.587 * feedback_rgb[..., 1] + 0.114 * feedback_rgb[..., 2]
    
    # Scale feedback signal based on parameters
    feedback_strength = params.get('feedback_strength', 0.7)
    feedback_decay = params.get('feedback_decay', 0.95)
    
    # Apply time-based feedback modulation
    feedback_mod = 1.0 + 0.1 * np.sin(time_val * params.get('feedback_mod_freq', 0.02))
    
    # Apply zoom transformation to feedback if enabled
    zoom_factor = 1.0 + params.get('feedback_zoom_amp', 0.05) * np.sin(time_val * params.get('feedback_zoom_freq', 0.1))
    
    # Apply panning motion to feedback
    pan_offset_x = params.get('feedback_pan_x_speed', 0) * time_val * params.get('feedback_pan_range', 50)
    pan_offset_y = params.get('feedback_pan_y_speed', 0) * time_val * params.get('feedback_pan_range', 50)
    
    # Apply color shifting
    color_shift = params.get('feedback_color_shift', 0.1)
    hue_shift = np.sin(time_val * 0.1) * color_shift
    
    # Calculate final feedback intensity with decay and modulation
    effective_feedback = (feedback_gray * feedback_strength * feedback_mod * 
                         (feedback_decay ** (time_val - feedback_state.time_sum)))
    
    return effective_feedback


def compute_reaction_diffusion(x, y, time_val, params):
    """
    Compute reaction-diffusion patterns using Gray-Scott model with previous frame.
    Enhanced to reduce strobing and color flickering.
    
    Args:
        x: x-coordinates array
        y: y-coordinates array
        time_val: current time value for animation
        params: function parameters
        
    Returns:
        Reaction-diffusion values array to be added to current frame computation
    """
    if feedback_state.previous_frame is None:
        return np.zeros_like(x)
    
    # Get dimensions and create coordinate mapping
    prev_height, prev_width = feedback_state.previous_frame.shape[:2]
    
    # Map current coordinates to previous framebuffer coordinates
    x_range = x.max() - x.min()
    y_range = y.max() - y.min()
    
    x_norm = (x - x.min()) / (x_range + 1e-8)
    y_norm = (y - y.min()) / (y_range + 1e-8)
    
    x_prev = np.clip(x_norm * (prev_width - 1), 0, prev_width - 1)
    y_prev = np.clip(y_norm * (prev_height - 1), 0, prev_height - 1)
    
    x_int = x_prev.astype(int)
    y_int = y_prev.astype(int)
    
    # Extract previous frame data
    current_frame = feedback_state.previous_frame[y_int, x_int]
    
    # Convert to grayscale for chemical concentrations
    # Reduce sensitivity by using slower saturation conversion
    A = (0.299 * current_frame[..., 0] + 0.587 * current_frame[..., 1] + 0.114 * current_frame[..., 2])
    A = np.clip(A * 0.85, 0.05, 0.95)  # Limit range to avoid extreme values
    
    # Create inhibitor concentration B with smoother transitions
    B = 1.0 - A
    # Add slight noise to prevent flat areas
    noise_scale = 0.05
    B += np.random.normal(0, noise_scale, B.shape) * 0.01  # Very subtle noise
    B = np.clip(B, 0.05, 0.95)
    
    # Use more stable parameters
    diffusion_a = params.get('reaction_diffusion_diffusion_a', 1.0)
    diffusion_b = params.get('reaction_diffusion_diffusion_b', 0.5)
    feed_rate = params.get('reaction_diffusion_feed_rate', 0.055)
    kill_rate = params.get('reaction_diffusion_kill_rate', 0.062)
    dt = params.get('reaction_diffusion_dt', 0.02)  # Smaller time step for stability
    
    # Drastically reduce time-based modulation to prevent strobing
    # Max change of ±1% over a full cycle
    time_factor = np.sin(time_val * 0.02) * 0.002  # Much slower and smaller changes
    feed_rate = np.clip(feed_rate + time_factor, 0.054, 0.056)
    kill_rate = np.clip(kill_rate + time_factor, 0.061, 0.063)
    
    # Finite difference gradients (Laplacian) with smoother boundary handling
    padded_a = np.pad(A, 1, mode='edge')
    padded_b = np.pad(B, 1, mode='edge')
    
    # Use more stable Laplacian calculation
    laplacian_a = (
        padded_a[2:, 1:-1] + padded_a[:-2, 1:-1] +
        padded_a[1:-1, 2:] + padded_a[1:-1, :-2] -
        4 * padded_a[1:-1, 1:-1]
    ) * 0.25  # Scale for stability
    
    laplacian_b = (
        padded_b[2:, 1:-1] + padded_b[:-2, 1:-1] +
        padded_b[1:-1, 2:] + padded_b[1:-1, :-2] -
        4 * padded_b[1:-1, 1:-1]
    ) * 0.25
    
    # Gray-Scott reaction equations with milder interaction
    reaction = A * B * B * 0.8  # Scale down reaction rate
    
    # Update concentrations with conservative changes
    dA = (diffusion_a * laplacian_a) - reaction + feed_rate * (1 - A)
    dB = (diffusion_b * laplacian_b) + reaction - (kill_rate + feed_rate) * B
    
    # Apply gradual changes
    new_A = A + dt * dA
    new_B = B + dt * dB
    
    # Clip to stable range with buffer
    new_A = np.clip(new_A, 0.1, 0.9)
    new_B = np.clip(new_B, 0.1, 0.9)
    
    # Calculate reaction-diffusion output with smooth transitions
    rd_output = new_A - new_B
    
    # Use percentile-based normalization to avoid outliers causing flashing
    # Compute more stable normalization
    percentiles = np.percentile(rd_output, [15, 85])  # Use inner 70% range
    rd_clipped = np.clip(rd_output, percentiles[0], percentiles[1])
    
    # Gentle mapping to visual range
    rd_normalized = (rd_clipped - percentiles[0]) / (percentiles[1] - percentiles[0] + 1e-8)
    rd_mapped = rd_normalized * 0.8 + 0.1  # Map to [0.1, 0.9]
    
    # Very subtle temporal modulation
    pattern_scale = params.get('reaction_diffusion_scale', 0.8) * 0.85  # Reduce intensity
    time_modulation = 1.0 + 0.05 * np.sin(time_val * 0.01)  # Much slower, smaller modulation
    
    final_output = rd_mapped * pattern_scale * time_modulation
    
    return final_output


def randomize_function_params():
    """Generate new random parameters for the mathematical function."""
    # Expanded operations list with more function types
    all_operations = ['use_sin', 'use_cos', 'use_xor', 'use_reaction_diffusion', 
                     'use_cellular', 'use_domain_warp', 'use_polar', 'use_sinusoidal_field',
                     'use_noise', 'use_abs', 'use_power', 'use_feedback', 'use_voronoi']
    
    # Create initial operations dict with deterministic/randomized selection
    operations = {}
    
    # Fill remaining operations (8-12 total active operations)
    remaining_ops = [op for op in all_operations if op not in operations]
    ops_to_select = random.randint(4, 6)
    additional_ops = random.sample(remaining_ops, k=ops_to_select)
    operations.update({op: True for op in additional_ops})
    # testing
    #operations = {'use_sinusoidal_field': True}#, 'use_abs':True}#, 'use_sin':True, 'use_cos':True}
    #operations.update({'use_reaction_diffusion':True})
    
    # Ensure all operations are in the dict
    for op in all_operations:
        if op not in operations:
            operations[op] = False
    
    # Create a deterministic order for applying operations
    enabled_ops = [op for op in all_operations if operations.get(op, False)]
    random.shuffle(enabled_ops)
    
    # Enhanced color mapping with harmonic ratios
    color_schemes = [
        # Nature-inspired schemes
        {'red': 1.0, 'green': 1.4, 'blue': 0.8},     # golden hour
        {'red': 0.8, 'green': 1.0, 'blue': 1.6},     # oceanic
        {'red': 1.5, 'green': 0.8, 'blue': 1.0},     # sunset
        {'red': 1.2, 'green': 1.2, 'blue': 1.2},     # moonlight
        {'red': 0.9, 'green': 1.3, 'blue': 0.7},     # forest
        {'red': 1.6, 'green': 0.9, 'blue': 1.3},     # aurora
        {'red': 1.3, 'green': 1.5, 'blue': 1.1},     # crystal
        {'red': 1.1, 'green': 0.7, 'blue': 1.5},     # nebula
    ]
    
    # Enhanced color parameters with phase shifts
    color_scheme = random.choice(color_schemes)
    #print(color_scheme)

    # Sophisticated parameter ranges for beautiful visuals
    params = {
        'wave1_freq': random.choice([0.618, 1.0, 1.618, 2.5, 3.14, 4.2, 5.8]),  # More frequency options
        'wave2_freq': random.choice([0.618, 1.0, 1.618, 2.5, 3.14, 4.2, 5.8]),
        'wave1_mult': random.uniform(50, 300),  # Broader range for larger patterns
        'wave1_translate_x': random.uniform(-100, 100),  # Translation parameters for wave 1
        'wave1_translate_y': random.uniform(-100, 100),

        'time_speed': random.uniform(0.2, 3.0),  # More balanced speed range
        'time_translate_x': random.uniform(-50, 50),  # Time-based translation speed
        'time_translate_y': random.uniform(-50, 50),
        'time_warp_factor': random.uniform(0.5, 2.0),  # Time warping for phase modulation

        'xor_strength': random.uniform(.7,1.3),  # Higher impact
        'xor_translate_x': random.uniform(0.2, 1.5),  # Time-based translation speed for XOR
        'xor_translate_y': random.uniform(0.1, 1.0),
        'xor_translate_range': random.uniform(20, 100),  # Translation range for XOR
        'xor_morph_speed': random.uniform(0.1, 0.5),  # Morphing speed for XOR shape

        'cellular_scale': random.uniform(1.0, 20.0),  # Better cellular resolution
        'cellular_time_translate': random.uniform(-2.0, 2.0),  # Cellular pattern translation over time

        'domain_warp_strength': random.uniform(15.0, 60.0),  # Stronger warping
        'domain_warp_time_factor': random.uniform(0.3, 2.0),  # How warping changes with time

        'color_hue_segments': random.uniform(1,2),
        'color_red_mult': color_scheme['red'],
        'color_green_mult': color_scheme['green'],
        'color_blue_mult': color_scheme['blue'],
        'color_phase_red': random.uniform(0, 360),    # Phase shifts for dynamic colors
        'color_phase_green': random.uniform(0, 360),
        'color_phase_blue': random.uniform(0, 360),
        'color_saturation': random.uniform(1.0, 2.0),  # Saturation boost
        'color_power': random.uniform(1.0, 1.5),       # Gamma-like adjustment

        'polar_strength': random.uniform(0.7, 1.3),   # Polar pattern strength
        'polar_freq_r': random.choice([0.01, 0.02, 0.05, 0.1]),  # Radial frequency
        'polar_freq_theta': random.choice([2, 3, 5, 8]),  # Angular frequency
        'polar_rotation_speed': random.uniform(-0.5, 0.5),  # Polar rotation speed
        'polar_orbit_speed_x': random.uniform(-0.3, 0.3),   # Center orbit speed X
        'polar_orbit_speed_y': random.uniform(-0.3, 0.3),   # Center orbit speed Y
        'polar_orbit_range': random.uniform(50, 150),    # Center orbit radius
        'polar_time_factor': random.uniform(0.2, 2.0),   # Time modulation for polar patterns
        'polar_theta_harmonics': random.uniform(1, 4),   # Angular harmonics
        'polar_spiral_factor': random.uniform(-0.005, 0.005),  # Spiral generation
        'polar_spiral_freq': random.uniform(2, 8),      # Spiral frequency
        'polar_spiral_speed': random.uniform(0.1, 1.5),  # Spiral animation speed

        'noise_strength': random.uniform(0.7, 1.3),   # Noise pattern strength
        'noise_scale': random.uniform(0.005, 0.02),   # Noise frequency scale
        'noise_time_speed': random.uniform(0.1, 1.0),  # Noise animation speed
        'noise_octaves': random.randint(3, 6),       # Noise complexity levels

        'abs_strength': random.uniform(0.5, 1.0),     # Absolute value strength
        'abs_freq_x': random.choice([0.01, 0.02, 0.05, 0.1]),  # X frequency
        'abs_freq_y': random.choice([0.01, 0.02, 0.05, 0.1]),  # Y frequency
        'abs_freq_xy': random.choice([0.01, 0.02, 0.05]),  # Combined frequency
        'abs_time_speed': random.uniform(0.2, 2.0),   # Animation speed

        'power_strength': random.uniform(0.5, 1.0),   # Power function strength
        # Single source of truth for power_exponent
        'power_exponent': random.uniform(0.5, 3.0),   # Power exponent
        'power_freq_x': random.choice([0.01, 0.02, 0.05, 0.1]),  # X frequency for power base
        'power_freq_y': random.choice([0.01, 0.02, 0.05, 0.1]),  # Y frequency for power base
        'power_time_speed': random.uniform(0.2, 1.8),  # Power animation speed
        'power_exp_mod_freq': random.uniform(0.1, 1.0),  # Exponent modulation frequency
        
        # Feedback loop parameters
        'feedback_strength': random.uniform(0.95, 0.99),  # How strongly feedback is mixed with new frame
        'feedback_decay': random.uniform(0.5, 1.0),  # How quickly old frames decay influence
        'feedback_zoom_speed': random.uniform(2.0, 5.0),  # Speed of zoom transformation on feedback
        'feedback_zoom_freq': random.uniform(0.05, 0.3),  # Frequency of zoom oscillation
        'feedback_zoom_amp': random.uniform(0.02, 0.15),  # Amplitude of zoom oscillation
        'feedback_rotation_speed': random.uniform(-0.1, 0.1),  # Rotation speed of feedback frame
        'feedback_pan_x_speed': random.uniform(0, 0.5),  # Horizontal panning speed
        'feedback_pan_y_speed': random.uniform(0, 0.5),  # Vertical panning speed
        'feedback_pan_range': random.uniform(10, 80),  # Maximum panning distance
        'feedback_mod_freq': random.uniform(0.02, 0.1),  # Modulation frequency
        'feedback_color_shift': random.uniform(-0.1, 0.1),  # Color shift strength
        
        # Voronoi/cellular distance field parameters
        'voronoi_points': random.randint(7, 12),  # Number of Voronoi seed points
        'voronoi_strength': random.uniform(0.4, 0.6),  # Voronoi pattern strength
        'voronoi_scale': random.uniform(0.01, 0.1),  # Distance scaling factor
        
        # Sinusoidal field patterns - complex 2D field generation
        'sinusoidal_a_freq': random.choice([1.618, 2.414, 3.236, 4.236, 2.718, 3.141, 1.414]),  # Golden ratio & mathematical constants
        'sinusoidal_b_freq': random.choice([2.618, 3.414, 4.618, 5.236, 4.442, 2.449, 1.732]), # Harmonic ratios
        # Provide a real range rather than a single value
        'sinusoidal_phase': random.uniform(0.0, 4 * np.pi),     # Extended phase range
        'sinusoidal_strength': random.uniform(0.7, 1.0),      # Stronger influence range
        'sinusoidal_time_speed': random.uniform(0.3, 1.0),   # Variable animation speeds
        'sinusoidal_phase_speed_ratio': random.uniform(0.3, 3.0), # Asymmetric phase evolution
        'sinusoidal_scale': random.uniform(25.0, 50.0),        # Larger pattern range
        'sinusoidal_freq_mod_depth': random.uniform(0.1, 0.4), # Frequency modulation depth
        'sinusoidal_freq_mod_speed': random.uniform(0.05, 0.3), # Frequency modulation speed
        'sinusoidal_amplitude_mod': random.uniform(0.2, 0.8),# Amplitude modulation
        'sinusoidal_harmonics': random.randint(2, 5),        # Additional harmonic layers
        'sinusoidal_harmonic_decay': random.uniform(0.3, 0.7), # Harmonic intensity decay
        'sinusoidal_x_scale': random.uniform(0.5, 2.0),      # Independent X-axis scaling
        'sinusoidal_y_scale': random.uniform(0.5, 2.0),      # Independent Y-axis scaling
        'sinusoidal_rotation_speed': random.uniform(-0.2, 0.2), # Pattern rotation
        
        # Reaction-diffusion parameters - more stable values to prevent strobing
        'reaction_diffusion_diffusion_a': 1.0,  # Fixed stable value
        'reaction_diffusion_diffusion_b': 0.5,  # Fixed stable value
        'reaction_diffusion_feed_rate': 0.055,  # Fixed for stable patterns
        'reaction_diffusion_kill_rate': 0.062,  # Fixed for stable patterns
        'reaction_diffusion_dt': 0.02,          # Small fixed step for stability
        'reaction_diffusion_scale': 0.7,        # Lower intensity for smooth patterns
        
        'function_order': enabled_ops,  # Store the order for consistent application
    }

    print(params['function_order'])
    #print(operations)
    #print(params)

    return {**operations, **params}


def generate_image_data(width, height, time_val, params, full_width=None, full_height=None):
    """
    Generate image data for visualization.
    
    Args:
        width: sample width (may be reduced for speed)
        height: sample height (may be reduced for speed)
        time_val: current time value for animation
        params: function parameters
        full_width: actual display width (default: width)
        full_height: actual display height (default: height)
        
    Returns:
        RGB image array
    """
    if full_width is None:
        full_width = width
    if full_height is None:
        full_height = height
    
    # Use a world coordinate system with constant units-per-pixel so resizing reveals more area
    # Choose baseline so previous visuals are roughly preserved at default 640x480
    baseline_width = 640.0
    baseline_range = 400.0  # previous coord_range
    units_per_pixel = baseline_range / baseline_width  # constant scale across sizes

    # Compute world extents based on the actual viewport size (full_width/height)
    x_world_range = units_per_pixel * max(full_width, 1)
    y_world_range = units_per_pixel * max(full_height, 1)

    # Create coordinate arrays spanning the full world extents but sampled at reduced resolution
    x = np.linspace(-x_world_range / 2.0, x_world_range / 2.0, width)[:, None]
    y = np.linspace(-y_world_range / 2.0, y_world_range / 2.0, height)[None, :]
    
    colors = compute_function(x, y, time_val, params)
    return np.transpose(colors, (1, 0, 2))