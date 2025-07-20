import tkinter as tk
from tkinter import ttk
import time
from PIL import Image, ImageTk
from utils.hardware import get_hardware_info, CUPY_AVAILABLE


class VisualizerUI:
    def __init__(self, root, width, height, time_step, randomize_callback, generate_image_callback, update_time_step_callback=None):
        """Initialize the UI components for the visualizer
        
        Args:
            root: The tkinter root window
            width: Initial canvas width
            height: Initial canvas height
            time_step: Initial time step value
            randomize_callback: Function to call when randomize button is clicked
            generate_image_callback: Function to generate image data for display
            update_time_step_callback: Function to call when time step slider is moved
        """
        self.root = root
        self.width = width
        self.height = height
        self.time_step = time_step
        self.randomize_callback = randomize_callback
        self.generate_image_callback = generate_image_callback
        self.update_time_step_callback = update_time_step_callback
        self.frame_time_ms = 0.0

        self.setup_ui()
        self.setup_bindings()
        
    def setup_ui(self):
        """Create and arrange all UI components"""
        self.toolbar = ttk.Frame(self.root)
        self.toolbar.pack(side=tk.TOP, fill=tk.X)
        
        self.status_label = ttk.Label(self.toolbar, text="GPU: " + get_hardware_info(), 
                                     relief=tk.SUNKEN)
        self.status_label.pack(side=tk.LEFT, padx=5, pady=2)
        
        self.frame_time_label = ttk.Label(self.toolbar, text="Frame: 0.0ms", 
                                         relief=tk.SUNKEN)
        self.frame_time_label.pack(side=tk.LEFT, padx=5, pady=2)
        
        self.randomize_btn = ttk.Button(self.toolbar, text="Randomize", command=self.randomize_callback)
        self.randomize_btn.pack(side=tk.LEFT, padx=5)
        
        # Time step control
        self.time_step_label = ttk.Label(self.toolbar, text="Time Step:", relief=tk.SUNKEN)
        self.time_step_label.pack(side=tk.LEFT, padx=5, pady=2)
        
        self.time_step_slider = ttk.Scale(self.toolbar, from_=0.0, to=0.2, 
                                         command=lambda v: [setattr(self, 'time_step', float(v)), 
                                                          self.time_step_label.config(text=f"{float(v):.3f}"),
                                                          self.update_time_step_callback(float(v)) if self.update_time_step_callback else None], 
                                         orient=tk.HORIZONTAL, length=100)
        self.time_step_slider.set(self.time_step)
        self.time_step_slider.pack(side=tk.LEFT, padx=5)
        
        self.time_step_value = ttk.Label(self.toolbar, text=f"{self.time_step:.3f}", relief=tk.SUNKEN, width=6)
        self.time_step_value.pack(side=tk.LEFT, padx=2)
        
        self.canvas = tk.Canvas(self.root, width=self.width, height=self.height, bg='black', highlightthickness=0)
        self.canvas.pack(expand=True, fill=tk.BOTH, padx=0, pady=0)
        
    def setup_bindings(self):
        """Set up event bindings for the UI"""
        self.root.bind('<Configure>', self.on_resize)
        
    def on_resize(self, event):
        """Handle window resize events"""
        # Let the canvas automatically resize via pack(expand=True, fill=tk.BOTH)
        pass
    
    def update_time_step(self, value):
        """Update the time step value when the slider is moved"""
        self.time_step = float(value)
        # Update the time step value label
        try:
            self.time_step_value.config(text=f"{self.time_step:.3f}")
        except AttributeError:
            # Handle case where the attribute might not be set yet
            pass
        
        if self.update_time_step_callback:
            self.update_time_step_callback(value)
    
    def update_display(self, time_val, running):
        """Update the display with a new generated image
        
        Args:
            time_val: Current time value
            running: Boolean indicating if animation is running
            
        Returns:
            tuple: (updated_time_val, actual_width, actual_height)
        """
        if running:
            try:
                start_time = time.time()
                
                actual_width = self.canvas.winfo_width()
                actual_height = self.canvas.winfo_height()
                
                if actual_width > 0 and actual_height > 0:
                    self.width = actual_width
                    self.height = actual_height
                        
                photo = self.generate_image_callback(self.width, self.height, time_val)
                
                end_time = time.time()
                self.frame_time_ms = (end_time - start_time) * 1000
                self.frame_time_label.config(text=f"Frame: {self.frame_time_ms:.1f}ms")
                
                self.canvas.delete("all")
                self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
                self.canvas.image = photo
                
                return (time_val, actual_width, actual_height)
            except Exception as e:
                print(f"Error updating display: {e}")
                
        return (time_val, self.width, self.height)