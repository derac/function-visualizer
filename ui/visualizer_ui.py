import tkinter as tk
from tkinter import ttk
import time
from PIL import Image, ImageTk
from utils.hardware import get_hardware_info, CUPY_AVAILABLE


class VisualizerUI:
    def __init__(self, root, width, height, time_step, brightness, visual_fidelity, randomize_callback, generate_image_callback, update_time_step_callback=None, update_brightness_callback=None, update_visual_fidelity_callback=None):
        """Initialize the UI components for the visualizer
        
        Args:
            root: The tkinter root window
            width: Initial canvas width
            height: Initial canvas height
            time_step: Initial time step value
            brightness: Initial brightness value (0.1-2.0)
            randomize_callback: Function to call when randomize button is clicked
            generate_image_callback: Function to generate image data for display
            update_time_step_callback: Function to call when time step slider is moved
            update_brightness_callback: Function to call when brightness slider is moved
        """
        self.root = root
        self.width = width
        self.height = height
        self.time_step = time_step
        self.brightness = brightness
        self.visual_fidelity = visual_fidelity
        self.randomize_callback = randomize_callback
        self.generate_image_callback = generate_image_callback
        self.update_time_step_callback = update_time_step_callback
        self.update_brightness_callback = update_brightness_callback
        self.update_visual_fidelity_callback = update_visual_fidelity_callback
        self.frame_time_ms = 0.0

        self.setup_ui()
        self.setup_bindings()
        
    def setup_ui(self):
        """Create and arrange all UI components"""
        self.root.title(f"Function Visualizer - GPU: {get_hardware_info()} | Frame: 0.0ms")
        
        self.toolbar = ttk.Frame(self.root)
        self.toolbar.pack(side=tk.TOP, fill=tk.X)
        
        self.randomize_btn = ttk.Button(self.toolbar, text="Randomize (R)", command=self.randomize_callback)
        self.randomize_btn.pack(side=tk.LEFT, padx=5)
        
        self.fullscreen_btn = ttk.Button(self.toolbar, text="Fullscreen (F11)", command=self.toggle_fullscreen)
        self.fullscreen_btn.pack(side=tk.LEFT, padx=5)
        
        # Time step control
        self.time_step_label = ttk.Label(self.toolbar, text="Time Step:", relief=tk.SUNKEN)
        self.time_step_label.pack(side=tk.LEFT, padx=5, pady=2)
        
        self.time_step_slider = ttk.Scale(self.toolbar, from_=0.0, to=0.2, 
                                         command=lambda v: [setattr(self, 'time_step', float(v)), 
                                                          self.time_step_label.config(text=f"Time Step: {float(v):.3f}"),
                                                          self.update_time_step_callback(float(v)) if self.update_time_step_callback else None], 
                                         orient=tk.HORIZONTAL, length=100)
        self.time_step_slider.set(self.time_step)
        self.time_step_slider.pack(side=tk.LEFT, padx=5)

        # Brightness control
        self.brightness_label = ttk.Label(self.toolbar, text=f"Brightness: {float(self.brightness):.2f}", relief=tk.SUNKEN)
        self.brightness_label.pack(side=tk.LEFT, padx=5, pady=2)
        
        self.brightness_slider = ttk.Scale(self.toolbar, from_=0.1, to=2.0, 
                                         command=lambda v: [setattr(self, 'brightness', float(v)), 
                                                          self.brightness_label.config(text=f"Brightness: {float(v):.2f}"),
                                                          self.update_brightness_callback(float(v)) if self.update_brightness_callback else None], 
                                         orient=tk.HORIZONTAL, length=100)
        self.brightness_slider.set(self.brightness)
        self.brightness_slider.pack(side=tk.LEFT, padx=5)

        # Visual fidelity control
        self.fidelity_label = ttk.Label(self.toolbar, text=f"Fidelity: {int(self.visual_fidelity)}%", relief=tk.SUNKEN)
        self.fidelity_label.pack(side=tk.LEFT, padx=5, pady=2)
        
        self.fidelity_slider = ttk.Scale(self.toolbar, from_=5, to=100, 
                                       command=lambda v: [setattr(self, 'visual_fidelity', float(v)), 
                                                        self.fidelity_label.config(text=f"Fidelity: {int(float(v))}%"),
                                                        self.update_visual_fidelity_callback(float(v)) if self.update_visual_fidelity_callback else None], 
                                       orient=tk.HORIZONTAL, length=100)
        self.fidelity_slider.set(self.visual_fidelity)
        self.fidelity_slider.pack(side=tk.LEFT, padx=5)
    
        
        self.canvas = tk.Canvas(self.root, width=self.width, height=self.height, bg='black', highlightthickness=0)
        self.canvas.pack(expand=True, fill=tk.BOTH, padx=0, pady=0)
        
    def setup_bindings(self):
        """Set up event bindings for the UI"""
        self.root.bind('<Configure>', self.on_resize)
        self.root.bind('<F11>', lambda e: self.toggle_fullscreen())
        self.root.bind('<r>', lambda e: self.randomize_callback())
        self.root.bind('<R>', lambda e: self.randomize_callback())
        
    def on_resize(self, event):
        """Handle window resize events"""
        # Let the canvas automatically resize via pack(expand=True, fill=tk.BOTH)
        pass
        
    def toggle_fullscreen(self):
        """Toggle fullscreen mode"""
        self.root.attributes("-fullscreen", not self.root.attributes("-fullscreen"))
    
    def update_time_step(self, value):
        """Update the time step value when the slider is moved"""
        self.time_step = float(value)
        
        if self.update_time_step_callback:
            self.update_time_step_callback(value)

    def update_brightness(self, value):
        """Update the brightness value when the slider is moved"""
        self.brightness = float(value)
        
        if self.update_brightness_callback:
            self.update_brightness_callback(value)
            
    def update_visual_fidelity(self, value):
        """Update the visual fidelity value when the slider is moved"""
        self.visual_fidelity = float(value)
        
        if self.update_visual_fidelity_callback:
            self.update_visual_fidelity_callback(value)
    
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
                self.root.title(f"Function Visualizer - GPU: {get_hardware_info()} | Frame: {self.frame_time_ms:.1f}ms")
                
                self.canvas.delete("all")
                self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
                self.canvas.image = photo
                
                return (time_val, actual_width, actual_height)
            except Exception as e:
                print(f"Error updating display: {e}")
                
        return (time_val, self.width, self.height)