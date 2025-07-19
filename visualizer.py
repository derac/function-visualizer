import tkinter as tk
from tkinter import ttk
import numpy as np
import time
import threading
from PIL import Image, ImageTk

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
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
        
        self.toggle_btn = ttk.Button(self.toolbar, text="Toggle GPU/CPU", command=self.toggle_compute_method)
        self.toggle_btn.pack(side=tk.LEFT, padx=5)
        
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
        
    def toggle_compute_method(self):
        if CUPY_AVAILABLE:
            self.using_cupy = not self.using_cupy
            self.status_label.config(text="GPU: " + ("CUDA" if self.using_cupy else "CPU"))
        else:
            self.status_label.config(text="GPU: Not Available")
            
    def on_resize(self, event):
        # Let the canvas automatically resize via pack(expand=True, fill=tk.BOTH)
        pass
                
    def compute_function(self, x, y, time_val):
        if self.using_cupy and CUPY_AVAILABLE:
            return self.compute_function_cupy(x, y, time_val)
        else:
            return self.compute_function_numpy(x, y, time_val)
            
    def compute_function_numpy(self, x, y, time_val):
        if self.random_params is None:
            self.randomize_function_params()
        
        params = self.random_params
        
        # Add time-based modulation to all parameters
        time_mod = np.sin(time_val * 0.2) * 0.3 + np.cos(time_val * 0.15) * 0.2
        wave1_mult = params['wave1_mult'] * (1 + 0.3 * np.sin(time_val * 0.15))
        wave2_mult = params['wave2_mult'] * (1 + 0.3 * np.cos(time_val * 0.12))
        domain_warp_strength = params['domain_warp_strength'] * (1 + 0.4 * np.sin(time_val * 0.25))
        fractal_iterations = params['fractal_iterations'] * (1 + 0.2 * np.sin(time_val * 0.1))
        
        # Ensure x and y are arrays
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        
        # Normalize with time-based scaling
        x_normalized = x / wave1_mult
        y_normalized = y / wave2_mult
        
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
                if params.get('use_mod', False):
                    xor_mask = np.bitwise_xor(x.astype(np.int32), y.astype(np.int32)) & 0xFF
                    mod_factor = ((x + y + int(time_val * params['mod_factor'])) % 256) / 255.0
                    combined = combined + xor_mask * mod_factor * params['xor_strength']
                else:
                    xor_mask = np.bitwise_xor(x.astype(np.int32), y.astype(np.int32)) & 0xFF
                    combined = combined + xor_mask / 2.0
            
            elif op == 'use_sin':
                combined = combined + wave1 * 200
            
            elif op == 'use_cos':
                combined = combined + wave2 * 200
            
            elif op == 'use_fractal':
                radius = np.sqrt(x**2 + y**2)
                combined = combined + radius * np.sin(radius * fractal_iterations + time_val * 0.3)
            
            elif op == 'use_product' and params['use_sin'] and params['use_cos']:
                combined = combined + wave1 * wave2 * 150
            
            elif op == 'use_addition':
                if params['use_sin']:
                    combined = combined + wave1 * 150
                if params['use_cos']:
                    combined = combined + wave2 * 150
            
            elif op == 'use_cellular':
                grid_x = (x / params['cellular_scale']).astype(int)
                grid_y = (y / params['cellular_scale']).astype(int)
                cell_val = np.sin(grid_x * 0.1) * np.cos(grid_y * 0.1)
                combined = combined + cell_val * 150
            
            elif op == 'use_domain_warp':
                animated_strength = domain_warp_strength * (1 + 0.3 * np.sin(time_val * 0.8))
                warped_x = x + animated_strength * np.sin(y * 0.1 + time_val * 0.5)
                warped_y = y + animated_strength * np.cos(x * 0.1 + time_val * 0.5)
                combined = combined + np.sin(warped_x) * np.cos(warped_y) * 100
        
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
        
        # Create continuous color gradients using smooth trigonometric functions
        base_hue = (time_factor + combined_smooth * 6.0) % 6.0
        
        # Generate RGB from hue using smooth 6-segment color wheel
        c = adjusted * params['color_saturation']
        x = c * (1 - np.abs(np.mod(base_hue, 2) - 1))
        
        # Smooth RGB transitions
        red = np.where(base_hue < 1, c, np.where(base_hue < 2, x, np.where(base_hue < 4, 0, np.where(base_hue < 5, x, c))))
        green = np.where(base_hue < 1, x, np.where(base_hue < 3, c, np.where(base_hue < 4, x, np.where(base_hue < 5, 0, 0))))
        blue = np.where(base_hue < 2, 0, np.where(base_hue < 3, x, np.where(base_hue < 5, c, x)))
        
        # Add subtle modulation for dynamic color evolution
        modulation_factor = 0.1
        mod_wave = np.sin(time_factor + combined_smooth * 4 * np.pi) * modulation_factor
        
        red = np.clip(red * (1 + mod_wave) * params['color_red_mult'], 0, 1)
        green = np.clip(green * (1 + mod_wave * 0.8) * params['color_green_mult'], 0, 1)
        blue = np.clip(blue * (1 + mod_wave * 1.2) * params['color_blue_mult'], 0, 1)
        
        # Final smooth scaling to 8-bit values
        colors = np.stack([red, green, blue], axis=-1) * 255
        return colors.astype(np.uint8)
        
    def compute_function_cupy(self, x, y, time_val):
        if self.random_params is None:
            self.randomize_function_params()
        
        params = self.random_params
        
        # Add time-based modulation to all parameters
        time_mod = cp.sin(time_val * 0.2) * 0.3 + cp.cos(time_val * 0.15) * 0.2
        wave1_mult = params['wave1_mult'] * (1 + 0.3 * cp.sin(time_val * 0.15))
        wave2_mult = params['wave2_mult'] * (1 + 0.3 * cp.cos(time_val * 0.12))
        domain_warp_strength = params['domain_warp_strength'] * (1 + 0.4 * cp.sin(time_val * 0.25))
        fractal_iterations = params['fractal_iterations'] * (1 + 0.2 * cp.sin(time_val * 0.1))
        
        # Ensure x and y are cupy arrays
        x = cp.asarray(x, dtype=cp.float32)
        y = cp.asarray(y, dtype=cp.float32)
        
        # Normalize with time-based scaling
        x_normalized = x / wave1_mult
        y_normalized = y / wave2_mult
        
        # Dynamic wave components initialized as arrays
        wave1 = cp.zeros_like(x)
        wave2 = cp.zeros_like(x)
        
        if params['use_sin']:
            wave1 = cp.abs(cp.sin(x_normalized * params['wave1_freq'] + time_val * params['time_speed']))
        if params['use_cos']:
            wave2 = cp.abs(cp.cos(y_normalized * params['wave2_freq'] + time_val * params['time_speed']))
        
        combined = cp.zeros_like(x, dtype=cp.float32)
        
        # Apply functions in predetermined order
        operations = params.get('function_order', [])
        
        for op in operations:
            if not params.get(op, False):
                continue
            
            if op == 'use_xor':
                if params.get('use_mod', False):
                    xor_mask = cp.bitwise_xor(x.astype(cp.int32), y.astype(cp.int32)) & 0xFF
                    mod_factor = ((x + y + int(time_val * params['mod_factor'])) % 256) / 255.0
                    combined = combined + xor_mask * mod_factor * params['xor_strength']
                else:
                    xor_mask = cp.bitwise_xor(x.astype(cp.int32), y.astype(cp.int32)) & 0xFF
                    combined = combined + xor_mask / 2.0
            
            elif op == 'use_sin':
                combined = combined + wave1 * 200
            
            elif op == 'use_cos':
                combined = combined + wave2 * 200
            
            elif op == 'use_fractal':
                radius = cp.sqrt(x**2 + y**2)
                combined = combined + radius * cp.sin(radius * fractal_iterations)
            
            elif op == 'use_product' and params['use_sin'] and params['use_cos']:
                combined = combined + wave1 * wave2 * 150
            
            elif op == 'use_addition':
                if params['use_sin']:
                    combined = combined + wave1 * 150
                if params['use_cos']:
                    combined = combined + wave2 * 150
            
            elif op == 'use_cellular':
                grid_x = (x / params['cellular_scale']).astype(int)
                grid_y = (y / params['cellular_scale']).astype(int)
                cell_val = cp.sin(grid_x * 0.1) * cp.cos(grid_y * 0.1)
                combined = combined + cell_val * 150
            
            elif op == 'use_domain_warp':
                warped_x = x + domain_warp_strength * cp.sin(y * 0.1)
                warped_y = y + domain_warp_strength * cp.cos(x * 0.1)
                combined = combined + cp.sin(warped_x) * cp.cos(warped_y) * 100
        
        # Smooth color remapping using sigmoid-like functions
        # Normalize combined values and apply smooth transformation
        min_val = cp.min(combined)
        max_val = cp.max(combined)
        combined_norm = (combined - min_val) / (max_val - min_val + 1e-8)
        
        # Use smooth sigmoid remapping for smooth transitions
        smooth_factor = 3.0
        combined_smooth = 1.0 / (1.0 + cp.exp(-smooth_factor * (combined_norm - 0.5)))
        
        # Apply power curve for additional color control
        adjusted = cp.power(combined_smooth, params['color_power'] * 0.5 + 0.5)
        
        # Create smooth color flows using multiple overlapping waves
        time_factor = time_val * 0.1
        
        # Create continuous color gradients using smooth trigonometric functions
        base_hue = (time_factor + combined_smooth * 6.0) % 6.0
        
        # Generate RGB from hue using smooth 6-segment color wheel
        c = adjusted * params['color_saturation']
        x = c * (1 - cp.abs(cp.mod(base_hue, 2) - 1))
        
        # Smooth RGB transitions using vectorized operations
        red = cp.where(base_hue < 1, c, 
              cp.where(base_hue < 2, x, 
              cp.where(base_hue < 4, 0, 
              cp.where(base_hue < 5, x, c))))
        
        green = cp.where(base_hue < 1, x, 
                cp.where(base_hue < 3, c, 
                cp.where(base_hue < 4, x, 
                cp.where(base_hue < 5, 0, 0))))
        
        blue = cp.where(base_hue < 2, 0,
               cp.where(base_hue < 3, x,
               cp.where(base_hue < 5, c, x)))
        
        # Add subtle modulation for dynamic color evolution
        modulation_factor = 0.1
        mod_wave = cp.sin(time_factor + combined_smooth * 4 * cp.pi) * modulation_factor
        
        red = cp.clip(red * (1 + mod_wave) * params['color_red_mult'], 0, 1)
        green = cp.clip(green * (1 + mod_wave * 0.8) * params['color_green_mult'], 0, 1)
        blue = cp.clip(blue * (1 + mod_wave * 1.2) * params['color_blue_mult'], 0, 1)
        
        # Final smooth scaling to 8-bit values
        colors = cp.stack([red, green, blue], axis=-1) * 255
        return colors.astype(cp.uint8)
        
    def generate_image(self):
        if self.using_cupy and CUPY_AVAILABLE:
            x = cp.arange(self.width)[:, None] * cp.ones((1, self.height), dtype=cp.float32)
            y = cp.arange(self.height)[None, :] * cp.ones((self.width, 1), dtype=cp.float32)
            
            colors = self.compute_function_cupy(x, y, self.time_val)
            img_array = cp.transpose(colors, (1, 0, 2)).get()
        else:
            x = np.arange(self.width)[:, None] * np.ones((1, self.height), dtype=np.float32)
            y = np.arange(self.height)[None, :] * np.ones((self.width, 1), dtype=np.float32)
            
            colors = self.compute_function_numpy(x, y, self.time_val)
            img_array = np.transpose(colors, (1, 0, 2))
            
        img = Image.fromarray(img_array, 'RGB')
        return ImageTk.PhotoImage(img)
        
    def update_time_step(self, value):
        self.time_step = float(value)
        self.time_step_label.config(text=f"{self.time_step:.3f}")
    
    def update_display(self):
        if self.running:
            self.time_val += self.time_step
            try:
                actual_width = self.canvas.winfo_width()
                actual_height = self.canvas.winfo_height()
                
                if actual_width > 0 and actual_height > 0:
                    self.width = actual_width
                    self.height = actual_height
                        
                photo = self.generate_image()
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
                         'use_product', 'use_addition', 'use_fractal', 
                         'use_cellular', 'use_domain_warp', 'use_polar',
                         'use_noise', 'use_voronoi', 'use_abs', 'use_power']
        
        # Create initial operations dict with deterministic/randomized selection
        operations = {}
        
        # Forced interesting combinations
        interesting_combinations = [
            ['use_sin', 'use_cos', 'use_mod', 'use_xor'],
            ['use_fractal', 'use_domain_warp'],
            ['use_cellular', 'use_voronoi'],
            ['use_polar', 'use_noise']
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
            'mod_factor': random.uniform(20, 800),  # Extended range
            'xor_strength': random.uniform(1.0, 10.0),  # Higher impact
            'time_speed': random.uniform(0.2, 3.0),  # More balanced speed range
            'power_exponent': random.uniform(0.5, 6.0),  # Wider range
            'fractal_iterations': random.randint(4, 15),  # More detailed fractals
            'cellular_scale': random.uniform(1.0, 20.0),  # Better cellular resolution
            'domain_warp_strength': random.uniform(5.0, 30.0),  # Stronger warping
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
        })
        
        self.random_params = {**operations, **params}
    
    def on_closing(self):
        self.stop_animation()
        self.root.destroy()
        
    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    visualizer = XORVisualizer()
    visualizer.run()