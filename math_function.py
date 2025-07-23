"""Mathematical functions for visualization and computation with feedback loop."""

import random
import numpy as base_np  # For some operations that need pure numpy
from utils.hardware import get_array_module

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
            
            # Clean XOR with morphing
            xor_mask = np.bitwise_xor(rot_x.astype(np.int32), rot_y.astype(np.int32)) & 4

            # Gentle intensity modulation
            intensity = 0.8 + 0.2 * np.sin(time_val * 0.05 + (rot_x + rot_y) * 0.001)
            combined = combined + xor_mask * intensity * params['xor_strength']


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
                                                    np.sin(time_phase * 2) * 0.5))/(3*np.sin(time_val/10)+4)
            warped_y = (y + animated_strength * np.cos(x * 0.1 + time_phase * 0.7 + 
                                                    np.sin(time_phase * 1.5) * 0.3))/(3*np.sin(time_val/10)+4)
            combined = combined + np.sin(warped_x) * np.cos(warped_y) * 50
        
        elif op == 'use_polar':
            # Calculate polar coordinates with time evolution
            width = x.shape[0]  # Assuming x is a 2D array with width as first dimension
            height = x.shape[1] if len(x.shape) > 1 else y.shape[0]  # Handle both 2D and separate arrays
            
            center_x = width / 2 + np.sin(time_val * params['polar_orbit_speed_x']) * params['polar_orbit_range']
            center_y = height / 2 + np.cos(time_val * params['polar_orbit_speed_y']) * params['polar_orbit_range']
            
            x_rel = x - center_x
            y_rel = y - center_y
            
            r = np.sqrt(x_rel**2 + y_rel**2) + 1e-8  # Add small value to avoid division by zero
            theta = np.arctan2(y_rel, x_rel) + time_val * params['polar_rotation_speed']
            
            # Apply polar transformation with time-based frequency modulation
            freq_mod = 1 + 0.3 * np.sin(time_val * 0.15)
            polar_wave = np.sin(r * params['polar_freq_r'] * freq_mod + theta * params['polar_freq_theta']) * \
                        np.cos(theta * params['polar_theta_harmonics'] + time_val * params['polar_time_factor'])
            
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
            combined = combined + noise_val * 200.0 * params['noise_strength']
        
        
        elif op == 'use_abs':
            # Absolute value transformations with time-based modulation
            time_abs = time_val * params['abs_time_speed']
            abs_wave1 = np.abs(np.sin(x * params['abs_freq_x'] + time_abs))
            abs_wave2 = np.abs(np.cos(y * params['abs_freq_y'] + time_abs * 0.7))
            
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
                # Create random seed points within coordinate bounds
                x_bounds = x.max() - x.min()
                y_bounds = y.max() - y.min()
                seeds_x = np.random.uniform(x.min(), x.max(), num_points)
                seeds_y = np.random.uniform(y.min(), y.max(), num_points)
                params['voronoi_seeds'] = list(zip(seeds_x, seeds_y))
            else:
                params['voronoi_seeds'] = [[
                        seeds_x + 10*(random.randint(0,4)-2)*(time_val - feedback_state.time_sum),
                        seeds_y + 10*(random.randint(0,4)-2)*(time_val - feedback_state.time_sum)
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
    
    # Smooth color remapping using sigmoid-like functions
    # Normalize combined values and apply smooth transformation
    combined_norm = (combined - np.min(combined)) / (np.max(combined) - np.min(combined) + 1e-8)
    
    # Use smooth sigmoid remapping for smooth transitions
    smooth_factor = 3.0
    combined_smooth = 1.0 / (1.0 + np.exp(-smooth_factor * (combined_norm - 0.5)))
    
    # Apply power curve for additional color control
    adjusted = np.power(combined_smooth, params['color_power'] * 0.5 + 0.5)
    
    # Create smooth color flows using multiple overlapping waves
    time_factor = time_val * 0.1
    time_warped = time_val * params.get('time_warp_factor', 1.0)
    
    # Create continuous color gradients using smooth trigonometric functions
    base_hue = (time_factor + combined_smooth * 6.0 + time_warped * 0.3) % 6.0
    
    # Generate RGB from hue using smooth 6-segment color wheel
    c = adjusted * params['color_saturation']
    x = c * (1 - np.abs(np.mod(base_hue, 2) - 1))
    
    # Smooth RGB transitions
    red = np.where(base_hue < 1, c, np.where(base_hue < 2, x, np.where(base_hue < 4, 0, np.where(base_hue < 5, x, c))))
    green = np.where(base_hue < 1, x, np.where(base_hue < 3, c, np.where(base_hue < 4, x, np.where(base_hue < 5, 0, 0))))
    blue = np.where(base_hue < 2, 0, np.where(base_hue < 3, x, np.where(base_hue < 5, c, x)))
    
    # Add enhanced time-based modulation using phase parameters
    modulation_factor = 0.15
    phase_red = params.get('color_phase_red', 0) * np.pi / 180
    phase_green = params.get('color_phase_green', 0) * np.pi / 180
    phase_blue = params.get('color_phase_blue', 0) * np.pi / 180
    
    mod_wave = np.sin(time_factor * 2 + combined_smooth * 4 * np.pi) * modulation_factor
    
    # Apply phase-shifted color modulation
    red_mod = np.sin(time_factor * 1.7 + phase_red) * modulation_factor
    green_mod = np.sin(time_factor * 1.9 + phase_green) * modulation_factor * 0.8
    blue_mod = np.sin(time_factor * 2.1 + phase_blue) * modulation_factor * 1.2
    
    red = np.clip(red * (1 + mod_wave + red_mod) * params['color_red_mult'], 0, 1)
    green = np.clip(green * (1 + mod_wave * 0.8 + green_mod) * params['color_green_mult'], 0, 1)
    blue = np.clip(blue * (1 + mod_wave * 1.2 + blue_mod) * params['color_blue_mult'], 0, 1)
    
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


def randomize_function_params():
    """Generate new random parameters for the mathematical function."""
    # Expanded operations list with more function types
    all_operations = ['use_sin', 'use_cos', 'use_xor', 
                     'use_cellular', 'use_domain_warp', 'use_polar',
                     'use_noise', 'use_abs', 'use_power', 'use_feedback', 'use_voronoi']
    
    # Create initial operations dict with deterministic/randomized selection
    operations = {}
    
    # Fill remaining operations (8-12 total active operations)
    remaining_ops = [op for op in all_operations if op not in operations]
    ops_to_select = random.randint(4, 6)
    additional_ops = random.sample(remaining_ops, k=ops_to_select)
    operations.update({op: True for op in additional_ops})
    # testing
    #operations = {'use_voronoi': True}#, 'use_abs':True}#, 'use_sin':True, 'use_cos':True}
    #operations.update({'use_feedback':True})
    
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

    # Sophisticated parameter ranges for beautiful visuals
    params = {
        'wave1_freq': random.choice([0.618, 1.0, 1.618, 2.5, 3.14, 4.2, 5.8]),  # More frequency options
        'wave2_freq': random.choice([0.618, 1.0, 1.618, 2.5, 3.14, 4.2, 5.8]),
        'wave1_mult': random.uniform(50, 300),  # Broader range for larger patterns
        'wave2_mult': random.uniform(50, 300),
        'wave1_translate_x': random.uniform(-100, 100),  # Translation parameters for wave 1
        'wave1_translate_y': random.uniform(-100, 100),
        'wave2_translate_x': random.uniform(-100, 100),  # Translation parameters for wave 2
        'wave2_translate_y': random.uniform(-100, 100),

        'time_speed': random.uniform(0.2, 3.0),  # More balanced speed range
        'time_translate_x': random.uniform(-50, 50),  # Time-based translation speed
        'time_translate_y': random.uniform(-50, 50),
        'time_warp_factor': random.uniform(0.5, 2.0),  # Time warping for phase modulation

        'xor_strength': random.uniform(1.0, 10.0),  # Higher impact
        'xor_translate_x': random.uniform(0.2, 1.5),  # Time-based translation speed for XOR
        'xor_translate_y': random.uniform(0.1, 1.0),
        'xor_translate_range': random.uniform(20, 100),  # Translation range for XOR
        'xor_morph_speed': random.uniform(0.1, 0.5),  # Morphing speed for XOR shape

        'cellular_scale': random.uniform(1.0, 20.0),  # Better cellular resolution
        'cellular_time_translate': random.uniform(-2.0, 2.0),  # Cellular pattern translation over time

        'domain_warp_strength': random.uniform(15.0, 30.0),  # Stronger warping
        'domain_warp_time_factor': random.uniform(0.3, 2.0),  # How warping changes with time

        'color_red_mult': color_scheme['red'],
        'color_green_mult': color_scheme['green'],
        'color_blue_mult': color_scheme['blue'],
        'color_phase_red': random.uniform(0, 360),    # Phase shifts for dynamic colors
        'color_phase_green': random.uniform(0, 360),
        'color_phase_blue': random.uniform(0, 360),
        'color_saturation': random.uniform(0.7, 1.5),  # Saturation boost
        'color_power': random.uniform(0.8, 1.4),       # Gamma-like adjustment

        'polar_strength': random.uniform(0.4, 2.2),   # Polar pattern strength
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

        'noise_strength': random.uniform(0.3, 2.0),   # Noise pattern strength
        'noise_scale': random.uniform(0.005, 0.02),   # Noise frequency scale
        'noise_time_speed': random.uniform(0.1, 1.0),  # Noise animation speed
        'noise_octaves': random.randint(3, 6),       # Noise complexity levels

        'abs_strength': random.uniform(0.5, 2.5),     # Absolute value strength
        'abs_freq_x': random.choice([0.01, 0.02, 0.05, 0.1]),  # X frequency
        'abs_freq_y': random.choice([0.01, 0.02, 0.05, 0.1]),  # Y frequency
        'abs_freq_xy': random.choice([0.01, 0.02, 0.05]),  # Combined frequency
        'abs_time_speed': random.uniform(0.2, 2.0),   # Animation speed

        'power_strength': random.uniform(0.4, 2.0),   # Power function strength
        'power_exponent': random.uniform(0.5, 6.0),   # Power exponent
        'power_freq_x': random.choice([0.01, 0.02, 0.05, 0.1]),  # X frequency for power base
        'power_freq_y': random.choice([0.01, 0.02, 0.05, 0.1]),  # Y frequency for power base
        'power_time_speed': random.uniform(0.2, 1.8),  # Power animation speed
        'power_exp_mod_freq': random.uniform(0.1, 1.0),  # Exponent modulation frequency
        'power_exponent': random.uniform(0.5, 6.0),  # Wider range
        
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
        
        'function_order': enabled_ops,  # Store the order for consistent application
    }

    print(params['function_order'])

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
    
    # Calculate step sizes to sample within full coordinate system
    step_x = max(1, full_width) / max(1, width)
    step_y = max(1, full_height) / max(1, height)
    
    # Create coordinate arrays that span the full viewport
    x_start = step_x * 0.5  # Center samples in pixels
    x_end = full_width - x_start
    y_start = step_y * 0.5
    y_end = full_height - y_start
    
    x = np.arange(width)[:, None] * step_x + x_start
    y = np.arange(height)[None, :] * step_y + y_start
    
    colors = compute_function(x, y, time_val, params)
    return np.transpose(colors, (1, 0, 2))