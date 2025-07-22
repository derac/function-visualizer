import tkinter as tk
import time
import threading
from PIL import Image, ImageTk
from utils.hardware import get_array_module, CUPY_AVAILABLE
from math_function import compute_function, randomize_function_params, generate_image_data
from ui.visualizer_ui import VisualizerUI

# Get the appropriate array module
np = get_array_module()


class Visualizer:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Visualizer")
        self.root.geometry("800x600")
        
        self.using_cupy = CUPY_AVAILABLE
        self.running = False
        self.time_val = 0.0
        self.time_step = 0.05
        self.brightness = 1.0
        self.width = 640
        self.height = 480
        self.frame_time_ms = 0.0
        self.visual_fidelity = 100.0  # Percentage scale: 100% = full resolution
        
        # Random parameters for function generation
        self.random_params = None
        self.randomize_function_params()
        
        self.setup_ui()
        self.setup_bindings()
        
    def setup_ui(self):
        # Create the UI components
        self.ui = VisualizerUI(
            self.root, 
            self.width, 
            self.height, 
            self.time_step, 
            self.brightness,
            self.visual_fidelity,
            self.randomize_function_params,
            self.generate_image_wrapper,
            self.update_time_step,
            self.update_brightness,
            self.update_visual_fidelity
        )
        
    def setup_bindings(self):
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def generate_image_wrapper(self, width, height, time_val):
        """Wrapper for the generate_image method to use as a callback"""
        # Make sure width and height are updated
        self.width = width
        self.height = height
        return self.generate_image()
                
    def compute_function(self, x, y, time_val):
        if self.random_params is None:
            self.randomize_function_params()
        
        return compute_function(x, y, time_val, self.random_params)
        
        
    def generate_image(self):
        # Use full viewport dimensions
        full_width = self.width
        full_height = self.height
        
        # Calculate reduced sample count based on fidelity
        scale_factor = self.visual_fidelity / 100.0
        sample_width = max(1, int(full_width * scale_factor))
        sample_height = max(1, int(full_height * scale_factor))
        
        # Generate image at reduced resolution but with full coordinate system
        img_array = generate_image_data(sample_width, sample_height, self.time_val, self.random_params, full_width, full_height).get()
        
        # Apply brightness adjustment
        img_array = img_array.astype(np.float32) * self.brightness
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
            
        # Create PIL image from array and stretch to viewport
        img = Image.fromarray(img_array, 'RGB')
        if sample_width != full_width or sample_height != full_height:
            img = img.resize((full_width, full_height), Image.Resampling.NEAREST)
        
        return ImageTk.PhotoImage(img)
        
    def update_time_step(self, value):
        self.time_step = float(value)
        
    def update_brightness(self, value):
        self.brightness = float(value)
        
    def update_visual_fidelity(self, value):
        self.visual_fidelity = float(value)
    
    def update_display(self):
        if self.running:
            self.time_val += self.time_step
            
            # Update the display using the UI component
            updated_vals = self.ui.update_display(self.time_val, self.running)
            self.time_val, self.width, self.height = updated_vals
                
            self.root.after(50, self.update_display)
            
    def start_animation(self):
        if not self.running:
            self.running = True
            self.update_display()
            
    def stop_animation(self):
        self.running = False
        
    def randomize_function_params(self):
        """Generate new random parameters for the mathematical function."""
        self.random_params = randomize_function_params()
    
    def on_closing(self):
        self.stop_animation()
        self.root.destroy()
        
    def run(self):
        self.start_animation()
        self.root.mainloop()


if __name__ == "__main__":
    visualizer = Visualizer()
    visualizer.run()