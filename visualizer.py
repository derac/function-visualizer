import tkinter as tk
from tkinter import ttk
import time
import threading
from PIL import Image, ImageTk

try:
    import cupy as np
    CUPY_AVAILABLE = True
except ImportError:
    import numpy as np
    CUPY_AVAILABLE = False


class XORVisualizer:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("XOR/Modulus Function Visualizer")
        self.root.geometry("800x600")
        
        self.using_cupy = CUPY_AVAILABLE
        self.running = False
        self.time_val = 0.0
        self.time_step = 0.05
        self.width = 640
        self.height = 480
        self.frame_time_ms = 0.0
        
        # Random parameters for function generation
        self.random_params = None
        self.randomize_function_params()
        
        self.setup_ui()
        self.setup_bindings()
        
    def setup_ui(self):
        self.toolbar = ttk.Frame(self.root)
        self.toolbar.pack(side=tk.TOP, fill=tk.X)
        
        self.status_label = ttk.Label(self.toolbar, text="GPU: " + ("CUDA" if self.using_cupy else "CPU"), 
                                     relief=tk.SUNKEN)
        self.status_label.pack(side=tk.LEFT, padx=5, pady=2)
        
        self.frame_time_label = ttk.Label(self.toolbar, text="Frame: 0.0ms", 
                                         relief=tk.SUNKEN)
        self.frame_time_label.pack(side=tk.LEFT, padx=5, pady=2)
        
        self.start_btn = ttk.Button(self.toolbar, text="Start", command=self.start_animation)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = ttk.Button(self.toolbar, text="Stop", command=self.stop_animation)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        self.randomize_btn = ttk.Button(self.toolbar, text="Randomize", command=self.randomize_function_params)
        self.randomize_btn.pack(side=tk.LEFT, padx=5)
        
        # Time step control
        self.time_step_label = ttk.Label(self.toolbar, text="Time Step:", relief=tk.SUNKEN)
        self.time_step_label.pack(side=tk.LEFT, padx=5, pady=2)
        
        self.time_step_slider = ttk.Scale(self.toolbar, from_=0.0, to=0.2, 
                                         command=self.update_time_step, orient=tk.HORIZONTAL, length=100)
        self.time_step_slider.set(self.time_step)
        self.time_step_slider.pack(side=tk.LEFT, padx=5)
        
        self.time_step_value = ttk.Label(self.toolbar, text=f"{self.time_step:.3f}", relief=tk.SUNKEN, width=6)
        self.time_step_value.pack(side=tk.LEFT, padx=2)
        
        self.canvas = tk.Canvas(self.root, width=self.width, height=self.height, bg='black', highlightthickness=0)
        self.canvas.pack(expand=True, fill=tk.BOTH, padx=0, pady=0)
        
    def setup_bindings(self):
        self.root.bind('<Configure>', self.on_resize)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
            
    def on_resize(self, event):
        # Let the canvas automatically resize via pack(expand=True, fill=tk.BOTH)
        pass
                
    def compute_function(self, x, y, time_val):
        if self.random_params is None:
            self.randomize_function_params()
        
        params = self.random_params
        
        # Add time-based modulation to all parameters
        time_mod = np.sin(time_val * 0.2) * 0.3 + np.cos(time_val * 0.15) * 0.2
        wave1_mult = params['wave1_mult'] * (1 + 0.3 * np.sin(time_val * 0.15))
        wave2_mult = params['wave2_mult'] * (1 + 0.3 * np.cos(time_val * 0.12))
        domain_warp_strength = params['domain_warp_strength'] * (1 + 0.4 * np.sin(time_val * 0.25))
        
        # Ensure x and y are arrays
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        
        # Apply time-based translations
        time_offset_x = time_val * params['time_translate_x']
        time_offset_y = time_val * params['time_translate_y']
        time_warp = time_val * params['time_warp_factor']
        
        # Apply wave translations with time evolution
        wave1_offset_x = time_offset_x + params['wave1_translate_x']
        wave1_offset_y = time_offset_y + params['wave1_translate_y']
        wave2_offset_x = time_offset_x + params['wave2_translate_x']
        wave2_offset_y = time_offset_y + params['wave2_translate_y']
        
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
                xor_mask = np.bitwise_xor(rot_x.astype(np.int32), rot_y.astype(np.int32)) & int(50 * (np.sin(time_val)+1))

                if params.get('use_mod', False):
                    # Apply XOR with mod using morphed coordinates
                    
                    mod_factor = ((rot_x + rot_y + int(time_val * params['mod_factor'])) % 255) / 255.0
                    # Smooth modulation of the modulation factor
                    mod_smooth = 0.5 + 0.5 * np.sin(time_val * 0.15 + mod_factor * np.pi)
                    combined = combined + xor_mask * mod_factor * mod_smooth * params['xor_strength']
                else:
                    # Gentle intensity modulation
                    intensity = 0.8 + 0.2 * np.sin(time_val * 0.05 + (rot_x + rot_y) * 0.001)
                    combined = combined + xor_mask * intensity * params['xor_strength']
            
            elif op == 'use_sin':
                combined = combined + wave1 * 200
            
            elif op == 'use_cos':
                combined = combined + wave2 * 200
            
            elif op == 'use_product' and params['use_sin'] and params['use_cos']:
                combined = combined + wave1 * wave2 * 150
            
            elif op == 'use_addition':
                if params['use_sin']:
                    combined = combined + wave1 * 150
                if params['use_cos']:
                    combined = combined + wave2 * 150
            
            elif op == 'use_cellular':
                time_cellular = time_val * params['cellular_time_translate']
                grid_x = ((x + time_cellular * 5) / params['cellular_scale']).astype(int)
                grid_y = ((y + time_cellular * 3) / params['cellular_scale']).astype(int)
                cell_val = np.sin(grid_x * 0.1 + time_cellular) * np.cos(grid_y * 0.1 + time_cellular * 1.5)
                combined = combined + cell_val * 150
            
            elif op == 'use_domain_warp':
                animated_strength = domain_warp_strength * (1 + 0.3 * np.sin(time_val * params['domain_warp_time_factor']))
                time_phase = time_val * 0.5
                warped_x = x + animated_strength * np.sin(y * 0.1 + time_phase + 
                                                        np.sin(time_phase * 2) * 0.5)
                warped_y = y + animated_strength * np.cos(x * 0.1 + time_phase * 0.7 + 
                                                        np.sin(time_phase * 1.5) * 0.3)
                combined = combined + np.sin(warped_x) * np.cos(warped_y) * 100
            
            elif op == 'use_tan':
                time_tan = time_val * params['tan_time_speed']
                tan_mult = params['tan_mult'] * (1 + 0.2 * np.sin(time_val * 0.1))
                
                # Apply tan with domain stretching and modulation
                x_tan = (x / tan_mult + time_tan) % (2 * np.pi)
                y_tan = (y / tan_mult + time_tan * 0.7) % (2 * np.pi)
                
                # Smooth tan function with safe handling
                tan_x = np.tan(x_tan * params['tan_freq_x'])
                tan_y = np.tan(y_tan * params['tan_freq_y'])
                
                # Clamp extreme values for smooth visualization
                tan_x = np.clip(tan_x, -3, 3)
                tan_y = np.clip(tan_y, -3, 3)
                
                combined = combined + tan_x * tan_y * 80 * params['tan_strength']
            
            elif op == 'use_polar':
                # Calculate polar coordinates with time evolution
                center_x = self.width / 2 + np.sin(time_val * params['polar_orbit_speed_x']) * params['polar_orbit_range']
                center_y = self.height / 2 + np.cos(time_val * params['polar_orbit_speed_y']) * params['polar_orbit_range']
                
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
                noise_val = (noise_val + 4) / 8  # Normalize based on expected range
                combined = combined + noise_val * 200 * params['noise_strength']
            
            elif op == 'use_voronoi':
                time_voronoi = time_val * params['voronoi_time_speed']
                cell_scale = params['voronoi_cell_scale']
                
                # Dynamic grid with cell drift
                grid_x = ((x + time_voronoi * params['voronoi_drift_x']) / cell_scale).astype(int)
                grid_y = ((y + time_voronoi * params['voronoi_drift_y']) / cell_scale).astype(int)
                
                # Generate cell distances with time evolution
                offset_x = np.sin(grid_x * 0.1 + time_voronoi * 0.5) * cell_scale
                offset_y = np.cos(grid_y * 0.15 + time_voronoi * 0.7) * cell_scale
                
                # Calculate 3D distance to nearby points for smooth patterns
                dist1 = np.sqrt(((x - (offset_x + grid_x * cell_scale)) % self.width) ** 2 + 
                              ((y - (offset_y + grid_y * cell_scale)) % self.height) ** 2)
                
                dist2 = np.sqrt(((x - (offset_x + (grid_x + 1) * cell_scale)) % self.width) ** 2 + 
                              ((y - (offset_y + grid_y * cell_scale)) % self.height) ** 2)
                
                # Create cell boundaries using distance differences
                cell_pattern = np.abs(dist1 - dist2) / cell_scale
                cell_pattern = np.clip(cell_pattern, 0, 1)
                
                # Add cellular detail with time modulation
                detail = np.sin(cell_pattern * np.pi * params['voronoi_freq']) * np.cos(time_voronoi * 2)
                combined = combined + (cell_pattern * 100 + detail * 50) * params['voronoi_strength']
            
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
        phase_shift = combined_smooth * 2 * np.pi
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
        return colors.astype(np.uint8)
        
        
    def generate_image(self):
        x = np.arange(self.width)[:, None] * np.ones((1, self.height), dtype=np.float32)
        y = np.arange(self.height)[None, :] * np.ones((self.width, 1), dtype=np.float32)
        
        colors = self.compute_function(x, y, self.time_val)
        img_array = np.transpose(colors, (1, 0, 2)).get()
            
        img = Image.fromarray(img_array, 'RGB')
        return ImageTk.PhotoImage(img)
        
    def update_time_step(self, value):
        self.time_step = float(value)
        self.time_step_label.config(text=f"{self.time_step:.3f}")
    
    def update_display(self):
        if self.running:
            self.time_val += self.time_step
            try:
                start_time = time.time()
                
                actual_width = self.canvas.winfo_width()
                actual_height = self.canvas.winfo_height()
                
                if actual_width > 0 and actual_height > 0:
                    self.width = actual_width
                    self.height = actual_height
                        
                photo = self.generate_image()
                
                end_time = time.time()
                self.frame_time_ms = (end_time - start_time) * 1000
                self.frame_time_label.config(text=f"Frame: {self.frame_time_ms:.1f}ms")
                
                self.canvas.delete("all")
                self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
                self.canvas.image = photo
            except Exception as e:
                print(f"Error updating display: {e}")
                
            self.root.after(50, self.update_display)
            
    def start_animation(self):
        if not self.running:
            self.running = True
            self.update_display()
            
    def stop_animation(self):
        self.running = False
        
    def randomize_function_params(self):
        """Generate new random parameters for the mathematical function."""
        import random
        
        # Expanded operations list with more function types
        all_operations = ['use_sin', 'use_cos', 'use_tan', 'use_xor', 'use_mod', 
                         'use_product', 'use_addition', 
                         'use_cellular', 'use_domain_warp', 'use_polar',
                         'use_noise', 'use_voronoi', 'use_abs', 'use_power']
        
        # Create initial operations dict with deterministic/randomized selection
        operations = {}
        
        # Forced interesting combinations
        interesting_combinations = [
            ['use_sin', 'use_cos', 'use_mod', 'use_xor'],
            ['use_fractal', 'use_domain_warp'],
            ['use_cellular', 'use_voronoi'],
        ]
        
        # Start with an interesting combination (50% chance)
        if random.random() > 0.5:
            combo = random.choice(interesting_combinations)
            operations.update({op: True for op in combo})
        
        # Fill remaining operations (8-12 total active operations)
        remaining_ops = [op for op in all_operations if op not in operations]
        ops_to_select = random.randint(
            max(0, 8 - len(operations)),
            max(0, 12 - len(operations))
        )
        additional_ops = random.sample(remaining_ops, k=ops_to_select)
        operations.update({op: True for op in additional_ops})
        
        # Ensure all operations are in the dict
        for op in all_operations:
            if op not in operations:
                operations[op] = False
        
        # Create a deterministic order for applying operations
        enabled_ops = [op for op in all_operations if operations.get(op, False)]
        random.shuffle(enabled_ops)
        
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
            'mod_factor': random.uniform(20, 800),  # Extended range
            'xor_strength': random.uniform(1.0, 10.0),  # Higher impact
            'xor_translate_x': random.uniform(0.2, 1.5),  # Time-based translation speed for XOR
            'xor_translate_y': random.uniform(0.1, 1.0),
            'xor_translate_range': random.uniform(20, 100),  # Translation range for XOR
            'xor_morph_speed': random.uniform(0.1, 0.5),  # Morphing speed for XOR shape
            'time_speed': random.uniform(0.2, 3.0),  # More balanced speed range
            'time_translate_x': random.uniform(-50, 50),  # Time-based translation speed
            'time_translate_y': random.uniform(-50, 50),
            'time_warp_factor': random.uniform(0.5, 2.0),  # Time warping for phase modulation
            'power_exponent': random.uniform(0.5, 6.0),  # Wider range
            'cellular_scale': random.uniform(1.0, 20.0),  # Better cellular resolution
            'cellular_time_translate': random.uniform(-2.0, 2.0),  # Cellular pattern translation over time
            'domain_warp_strength': random.uniform(5.0, 30.0),  # Stronger warping
            'domain_warp_time_factor': random.uniform(0.3, 2.0),  # How warping changes with time
            'function_order': enabled_ops  # Store the order for consistent application
        }
        
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
        params.update({
            'color_red_mult': color_scheme['red'],
            'color_green_mult': color_scheme['green'],
            'color_blue_mult': color_scheme['blue'],
            'color_phase_red': random.uniform(0, 360),    # Phase shifts for dynamic colors
            'color_phase_green': random.uniform(0, 360),
            'color_phase_blue': random.uniform(0, 360),
            'color_saturation': random.uniform(0.7, 1.5),  # Saturation boost
            'color_power': random.uniform(0.8, 1.4),       # Gamma-like adjustment
            
            # New operation parameters
            'tan_mult': random.uniform(30, 180),         # Tan domain scale
            'tan_freq_x': random.choice([0.5, 1.0, 2.0, 3.0, 5.0]),
            'tan_freq_y': random.choice([0.5, 1.0, 2.0, 3.0, 5.0]),
            'tan_strength': random.uniform(0.3, 2.5),     # Tan output strength
            'tan_time_speed': random.uniform(0.1, 1.8),   # Tan animation speed

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

            'voronoi_strength': random.uniform(0.5, 2.5),  # Voronoi pattern strength
            'voronoi_cell_scale': random.uniform(15, 60),  # Cell size scale
            'voronoi_time_speed': random.uniform(0.05, 0.5),  # Cell motion speed
            'voronoi_drift_x': random.uniform(5, 20),     # X direction drift
            'voronoi_drift_y': random.uniform(5, 20),     # Y direction drift
            'voronoi_freq': random.uniform(1.0, 5.0),     # Cell detail frequency

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
        })
        
        self.random_params = {**operations, **params}
    
    def on_closing(self):
        self.stop_animation()
        self.root.destroy()
        
    def run(self):
        self.start_animation()
        self.root.mainloop()


if __name__ == "__main__":
    visualizer = XORVisualizer()
    visualizer.run()