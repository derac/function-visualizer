import tkinter as tk
from tkinter import ttk
import time
import threading
from PIL import Image, ImageTk
from utils.hardware import get_array_module, get_hardware_info, CUPY_AVAILABLE
from math_function import compute_function, randomize_function_params, generate_image_data

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
        
        self.status_label = ttk.Label(self.toolbar, text="GPU: " + get_hardware_info(), 
                                     relief=tk.SUNKEN)
        self.status_label.pack(side=tk.LEFT, padx=5, pady=2)
        
        self.frame_time_label = ttk.Label(self.toolbar, text="Frame: 0.0ms", 
                                         relief=tk.SUNKEN)
        self.frame_time_label.pack(side=tk.LEFT, padx=5, pady=2)
        
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
        
        return compute_function(x, y, time_val, self.random_params)
        
        
    def generate_image(self):
        img_array = generate_image_data(self.width, self.height, self.time_val, self.random_params).get()
            
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