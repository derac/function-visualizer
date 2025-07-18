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
        
        # Normalize with variation
        x_normalized = x / params['wave1_mult']
        y_normalized = y / params['wave2_mult']
        
        # Dynamic wave components
        wave1 = 0
        wave2 = 0
        
        if params['use_sin']:
            wave1 = np.abs(np.sin(x_normalized + time_val * params['time_speed']))
        if params['use_cos']:
            wave2 = np.abs(np.cos(y_normalized + time_val * params['time_speed'] * params['wave2_freq']))
        
        # Dynamic operations
        combined = 0
        
        if params['use_xor']:
            xor_mask = np.bitwise_xor(x, y) & 0xFF
            if params['use_mod']:
                mod_factor = ((x + y + int(time_val * params['mod_factor'])) % 256) / 255.0
                combined += xor_mask * mod_factor * params['xor_strength']
            else:
                combined += xor_mask / 2.0
        
        if params['use_product'] and params['use_sin'] and params['use_cos']:
            combined += wave1 * wave2 * 255
        elif params['use_addition']:
            if params['use_sin']:
                combined += wave1 * 255
            if params['use_cos']:
                combined += wave2 * 255
        
        combined = np.clip(combined, 0, 255)
        
        # Dynamic color mapping
        red = (combined * params['color_red_mult']).astype(np.uint8)
        green = ((combined * params['color_green_mult'] + 127) % 256).astype(np.uint8)
        blue = ((combined * params['color_blue_mult'] + 200) % 256).astype(np.uint8)
        
        return np.stack([red, green, blue], axis=-1)
        
    def compute_function_cupy(self, x, y, time_val):
        if self.random_params is None:
            self.randomize_function_params()
        
        params = self.random_params
        
        # Normalize with variation
        x_normalized = x / params['wave1_mult']
        y_normalized = y / params['wave2_mult']
        
        # Dynamic wave components
        wave1 = 0
        wave2 = 0
        
        if params['use_sin']:
            wave1 = cp.abs(cp.sin(x_normalized + time_val * params['time_speed']))
        if params['use_cos']:
            wave2 = cp.abs(cp.cos(y_normalized + time_val * params['time_speed'] * params['wave2_freq']))
        
        # Dynamic operations
        combined = 0
        
        if params['use_xor']:
            xor_mask = cp.bitwise_xor(x.astype(cp.int32), y.astype(cp.int32)) & 0xFF
            if params['use_mod']:
                mod_factor = ((x + y + int(time_val * params['mod_factor'])) % 256) / 255.0
                combined += xor_mask * mod_factor * params['xor_strength']
            else:
                combined += xor_mask / 2.0
        
        if params['use_product'] and params['use_sin'] and params['use_cos']:
            combined += wave1 * wave2 * 255
        elif params['use_addition']:
            if params['use_sin']:
                combined += wave1 * 255
            if params['use_cos']:
                combined += wave2 * 255
        
        combined = cp.clip(combined, 0, 255)
        
        # Dynamic color mapping
        red = (combined * params['color_red_mult']).astype(cp.uint8)
        green = ((combined * params['color_green_mult'] + 127) % 256).astype(cp.uint8)
        blue = ((combined * params['color_blue_mult'] + 200) % 256).astype(cp.uint8)
        
        return cp.stack([red, green, blue], axis=-1)
        
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
        
    def update_display(self):
        if self.running:
            self.time_val += 0.05
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
        
        # Random operation selection probabilities
        operations = {
            'use_xor': random.random() > 0.3,
            'use_mod': random.random() > 0.2,
            'use_sin': random.random() > 0.3,
            'use_cos': random.random() > 0.3,
            'use_product': random.random() > 0.4,
            'use_addition': random.random() > 0.4
        }
        
        # Random scaling factors
        params = {
            'wave1_freq': random.uniform(0.1, 2.0),
            'wave2_freq': random.uniform(0.1, 2.0),
            'wave1_mult': random.uniform(8, 30),
            'wave2_mult': random.uniform(8, 30),
            'mod_factor': random.uniform(20, 200),
            'xor_strength': random.uniform(0.3, 2.0),
            'color_red_mult': random.uniform(0.8, 1.2),
            'color_green_mult': random.uniform(0.3, 0.8),
            'color_blue_mult': random.uniform(0.1, 0.5),
            'time_speed': random.uniform(0.3, 2.0)
        }
        
        self.random_params = {**operations, **params}
    
    def on_closing(self):
        self.stop_animation()
        self.root.destroy()
        
    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    visualizer = XORVisualizer()
    visualizer.run()