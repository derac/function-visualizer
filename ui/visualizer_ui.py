import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import time
from PIL import Image, ImageTk
from utils.hardware import get_hardware_info, CUPY_AVAILABLE
from utils.save_manager import save_manager
from utils.performance import performance_monitor
from config import config
from utils.logger import logger


class VisualizerUI:
    def __init__(self, root, width, height, time_step, visual_fidelity, randomize_callback, generate_image_callback, update_time_step_callback=None, update_visual_fidelity_callback=None, save_callback=None, load_callback=None, toggle_color_mode_callback=None, cycle_palette_callback=None):
        """Initialize the UI components for the visualizer
        
        Args:
            root: The tkinter root window
            width: Initial canvas width
            height: Initial canvas height
            time_step: Initial time step value
            randomize_callback: Function to call when randomize button is clicked
            generate_image_callback: Function to generate image data for display
            update_time_step_callback: Function to call when time step slider is moved
            save_callback: Function to call when save button is clicked
            load_callback: Function to call when load button is clicked
        """
        self.root = root
        self.width = width
        self.height = height
        self.time_step = time_step
        self.visual_fidelity = visual_fidelity
        self.randomize_callback = randomize_callback
        self.generate_image_callback = generate_image_callback
        self.update_time_step_callback = update_time_step_callback
        self.update_visual_fidelity_callback = update_visual_fidelity_callback
        self.save_callback = save_callback
        self.load_callback = load_callback
        self.toggle_color_mode_callback = toggle_color_mode_callback
        self.cycle_palette_callback = cycle_palette_callback
        self.frame_time_ms = 0.0
        self.performance_stats = {}
        self.show_performance = config.get('ui.show_fps', True)

        self.setup_ui()
        self.setup_bindings()
        
    def setup_ui(self):
        """Create and arrange all UI components"""
        self.root.title(f"Function Visualizer - GPU: {get_hardware_info()} | Frame: 0.0ms")
        
        self.toolbar = ttk.Frame(self.root)
        self.toolbar.pack(side=tk.TOP, fill=tk.X)
        
        # Control buttons with better labels
        self.randomize_btn = ttk.Button(self.toolbar, text="🎲 Random", command=self.randomize_callback)
        self.randomize_btn.pack(side=tk.LEFT, padx=5)
        
        self.save_btn = ttk.Button(self.toolbar, text="💾 Save", command=self.save_visualization)
        self.save_btn.pack(side=tk.LEFT, padx=5)
        
        self.load_btn = ttk.Button(self.toolbar, text="📂 Load", command=self.load_visualization)
        self.load_btn.pack(side=tk.LEFT, padx=5)
        
        self.fullscreen_btn = ttk.Button(self.toolbar, text="⛶ Full", command=self.toggle_fullscreen)
        self.fullscreen_btn.pack(side=tk.LEFT, padx=5)
        
        # Help button
        self.help_btn = ttk.Button(self.toolbar, text="❓ Help", command=self.show_help)
        self.help_btn.pack(side=tk.LEFT, padx=5)
        
        
        # Time step control
        self.time_step_label = ttk.Label(self.toolbar, text="⏱️ Speed:", relief=tk.SUNKEN)
        self.time_step_label.pack(side=tk.LEFT, padx=5, pady=2)
        
        self.time_step_slider = ttk.Scale(self.toolbar, from_=0.0, to=0.2, 
                                         command=lambda v: [setattr(self, 'time_step', float(v)), 
                                                          self.time_step_label.config(text=f"⏱️ Speed: {float(v):.3f}"),
                                                          self.update_time_step_callback(float(v)) if self.update_time_step_callback else None], 
                                         orient=tk.HORIZONTAL, length=100)
        self.time_step_slider.set(self.time_step)
        self.time_step_slider.pack(side=tk.LEFT, padx=5)


        # Auto scaling toggle
        self.auto_scaling_var = tk.BooleanVar(value=config.get('performance.auto_scaling', True))
        self.auto_scaling_check = ttk.Checkbutton(self.toolbar, text="🔄 Auto", 
                                                 variable=self.auto_scaling_var, 
                                                 command=self.toggle_auto_scaling)
        self.auto_scaling_check.pack(side=tk.LEFT, padx=5)
        
        # Visual fidelity control
        self.fidelity_label = ttk.Label(self.toolbar, text=f"📏 Quality: {int(self.visual_fidelity)}%", relief=tk.SUNKEN)
        self.fidelity_label.pack(side=tk.LEFT, padx=5, pady=2)
        
        self.fidelity_slider = ttk.Scale(self.toolbar, from_=5, to=100, 
                                       command=lambda v: [setattr(self, 'visual_fidelity', float(v)), 
                                                        self.fidelity_label.config(text=f"📏 Quality: {int(float(v))}%"),
                                                        self.update_visual_fidelity_callback(float(v)) if self.update_visual_fidelity_callback else None], 
                                       orient=tk.HORIZONTAL, length=100)
        self.fidelity_slider.set(self.visual_fidelity)
        self.fidelity_slider.pack(side=tk.LEFT, padx=5)
        
        # Set initial state of fidelity slider based on auto scaling
        if self.auto_scaling_var.get():
            self.fidelity_slider.config(state='disabled')
        
        # Performance display
        if self.show_performance:
            self.performance_label = ttk.Label(self.toolbar, text="FPS: 0.0", relief=tk.SUNKEN)
            self.performance_label.pack(side=tk.RIGHT, padx=5, pady=2)
    
    
        self.canvas = tk.Canvas(self.root, width=self.width, height=self.height, bg='black', highlightthickness=0)
        self.canvas.pack(expand=True, fill=tk.BOTH, padx=0, pady=0)
        
    def setup_bindings(self):
        """Set up event bindings for the UI"""
        self.root.bind('<Configure>', self.on_resize)
        self.root.bind('<f>', lambda e: self.toggle_fullscreen())
        self.root.bind('<F>', lambda e: self.toggle_fullscreen())
        self.root.bind('<r>', lambda e: self.randomize_callback())
        self.root.bind('<R>', lambda e: self.randomize_callback())
        self.root.bind('<s>', lambda e: self.save_visualization())
        self.root.bind('<S>', lambda e: self.save_visualization())
        self.root.bind('<l>', lambda e: self.load_visualization())
        self.root.bind('<L>', lambda e: self.load_visualization())
        self.root.bind('<h>', lambda e: self.toggle_toolbar())
        self.root.bind('<H>', lambda e: self.toggle_toolbar())
        self.root.bind('<p>', lambda e: self.toggle_performance_display())
        self.root.bind('<P>', lambda e: self.toggle_performance_display())
        self.root.bind('<F1>', lambda e: self.show_help())
        # Color controls
        self.root.bind('<c>', lambda e: self.toggle_color_mode_callback() if self.toggle_color_mode_callback else None)
        self.root.bind('<C>', lambda e: self.toggle_color_mode_callback() if self.toggle_color_mode_callback else None)
        self.root.bind('<bracketright>', lambda e: self.cycle_palette_callback(1) if self.cycle_palette_callback else None)
        self.root.bind('<bracketleft>', lambda e: self.cycle_palette_callback(-1) if self.cycle_palette_callback else None)
        
    def on_resize(self, event):
        """Handle window resize events"""
        # Let the canvas automatically resize via pack(expand=True, fill=tk.BOTH)
        pass
        
    def toggle_fullscreen(self):
        """Toggle fullscreen mode"""
        self.root.attributes("-fullscreen", not self.root.attributes("-fullscreen"))
    
    def toggle_toolbar(self):
        """Toggle toolbar visibility"""
        if self.toolbar.winfo_viewable():
            self.toolbar.pack_forget()
        else:
            self.toolbar.pack(side=tk.TOP, fill=tk.X, before=self.canvas)
    
    def update_time_step(self, value):
        """Update the time step value when the slider is moved"""
        self.time_step = float(value)
        
        if self.update_time_step_callback:
            self.update_time_step_callback(value)

            
    def update_visual_fidelity(self, value):
        """Update the visual fidelity value when the slider is moved"""
        self.visual_fidelity = float(value)
        
        if self.update_visual_fidelity_callback:
            self.update_visual_fidelity_callback(value)
    
    def toggle_auto_scaling(self):
        """Toggle auto scaling functionality"""
        auto_scaling = self.auto_scaling_var.get()
        config.set('performance.auto_scaling', auto_scaling)
        logger.info(f"Auto scaling {'enabled' if auto_scaling else 'disabled'}")
        
        # Update the fidelity slider state
        if auto_scaling:
            self.fidelity_slider.config(state='disabled')
        else:
            self.fidelity_slider.config(state='normal')
    
    def update_fidelity_slider(self, value):
        """Update the fidelity slider position and label"""
        self.visual_fidelity = float(value)
        self.fidelity_slider.set(value)
        
        # Update label with visual feedback
        self.fidelity_label.config(text=f"📏 Quality: {int(float(value))}%")
        
        # Temporarily highlight the label to show auto adjustment
        if self.auto_scaling_var.get():
            original_bg = self.fidelity_label.cget('background')
            self.fidelity_label.config(background='yellow')
            self.root.after(500, lambda: self.fidelity_label.config(background=original_bg))
    
    def show_help(self):
        """Show help dialog with controls and shortcuts"""
        help_text = """
🎨 Function Visualizer - Controls & Shortcuts

📋 Controls:
• 🎲 Random: Generate new random patterns
• 💾 Save: Save current visualization parameters
• 📂 Load: Load saved visualization parameters
• ⛶ Full: Toggle fullscreen mode
• ❓ Help: Show this help dialog

        🎛️ Sliders:
        • ⏱️ Speed: Animation speed (0.0-0.2)
        • 📏 Quality: Rendering quality (5%-100%)
        • 🎨 Color: Press C to toggle color mode; [ and ] to change palette

⚙️ Settings:
• 🔄 Auto: Enable/disable automatic quality adjustment

⌨️ Keyboard Shortcuts:
• R: Randomize patterns
• S: Save parameters
• L: Load parameters
• F: Toggle fullscreen
• H: Hide/show toolbar
• P: Toggle performance display
        • C: Toggle color mode (harmonic/palette)
        • [: Previous palette, ]: Next palette (palette mode)
• F1: Show help

💡 Tips:
• When Auto is enabled, quality adjusts automatically for performance
• When Auto is disabled, you can manually control quality
• Save interesting patterns to load them later
• Use fullscreen for immersive viewing
        """
        messagebox.showinfo("Help - Function Visualizer", help_text)
    
    def save_visualization(self):
        """Save current visualization parameters."""
        try:
            if self.save_callback:
                self.save_callback()
            else:
                messagebox.showinfo("Save", "Save functionality not implemented")
        except Exception as e:
            logger.error(f"Error saving visualization: {e}")
            messagebox.showerror("Error", f"Failed to save: {e}")
    
    def load_visualization(self):
        """Load visualization parameters using file picker."""
        try:
            # Get the saves directory path
            from pathlib import Path
            saves_dir = Path(config.get('saving.save_directory', 'saves')) / 'parameters'
            
            if not saves_dir.exists():
                messagebox.showwarning("No Saves", "No saved parameters found. Save some parameters first.")
                return
            
            # Open file dialog to select a parameter file
            filename = filedialog.askopenfilename(
                title="Load Visualization Parameters",
                initialdir=str(saves_dir),
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                defaultextension=".json"
            )
            
            if filename:
                # Extract just the filename without path and extension
                from pathlib import Path
                param_name = Path(filename).stem
                
                success = self.load_callback(param_name)
                if not success:
                    messagebox.showerror("Error", f"Failed to load parameters: {param_name}")
        except Exception as e:
            logger.error(f"Error loading visualization: {e}")
            messagebox.showerror("Error", f"Failed to load: {e}")
    
    def toggle_performance_display(self):
        """Toggle performance display visibility."""
        if hasattr(self, 'performance_label'):
            if self.performance_label.winfo_viewable():
                self.performance_label.pack_forget()
            else:
                self.performance_label.pack(side=tk.RIGHT, padx=5, pady=2)
    
    def update_performance_display(self, fps, frame_time_ms):
        """Update performance display with current metrics."""
        if hasattr(self, 'performance_label') and self.show_performance:
            self.performance_label.config(text=f"FPS: {fps:.1f} | {frame_time_ms:.1f}ms")
    
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
                # Start performance monitoring
                start_time = performance_monitor.start_frame()
                
                actual_width = self.canvas.winfo_width()
                actual_height = self.canvas.winfo_height()
                
                if actual_width > 0 and actual_height > 0:
                    self.width = actual_width
                    self.height = actual_height
                        
                photo = self.generate_image_callback(self.width, self.height, time_val)
                
                # End performance monitoring
                frame_time_ms, fps = performance_monitor.end_frame(start_time)
                self.frame_time_ms = frame_time_ms
                
                # Update performance display
                self.update_performance_display(fps, frame_time_ms)
                
                # Update window title with performance info
                hardware_info = get_hardware_info()
                self.root.title(f"Function Visualizer - {hardware_info} | Frame: {frame_time_ms:.1f}ms | FPS: {fps:.1f}")
                
                self.canvas.delete("all")
                self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
                self.canvas.image = photo
                
                return (time_val, actual_width, actual_height)
            except Exception as e:
                logger.error(f"Error updating display: {e}")
                print(f"Error updating display: {e}")
                
        return (time_val, self.width, self.height)