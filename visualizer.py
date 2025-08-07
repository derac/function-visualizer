import tkinter as tk
from tkinter import filedialog
import time
import threading
from PIL import Image, ImageTk
from utils.hardware import get_array_module, CUPY_AVAILABLE
from math_function import compute_function, randomize_function_params, generate_image_data
from ui.visualizer_ui import VisualizerUI
from config import config
from utils.logger import logger
from utils.save_manager import save_manager
from utils.performance import performance_monitor, performance_optimizer

# Get the appropriate array module
np = get_array_module()


class Visualizer:
    def __init__(self):
        # Load configuration
        window_config = config.get_window_config()
        viz_config = config.get_visualization_config()
        
        self.root = tk.Tk()
        self.root.title(window_config.get('title', 'Function Visualizer'))
        self.root.geometry(f"{window_config.get('width', 800)}x{window_config.get('height', 600)}")
        
        self.using_cupy = CUPY_AVAILABLE
        self.running = False
        self.time_val = 0.0
        self.time_step = viz_config.get('default_time_step', 0.05)
        self.width = 640
        self.height = 480
        self.frame_time_ms = 0.0
        self.visual_fidelity = viz_config.get('default_visual_fidelity', 100.0)
        
        # Random parameters for function generation
        self.random_params = None
        self.randomize_function_params()
        
        # Performance tracking
        self.last_auto_save = 0
        self.auto_save_interval = config.get('saving.auto_save_interval', 0)
        
        logger.info("Visualizer initialized")
        self.setup_ui()
        self.setup_bindings()
        
    def setup_ui(self):
        # Create the UI components
        self.ui = VisualizerUI(
            self.root, 
            self.width, 
            self.height, 
            self.time_step, 
            self.visual_fidelity,
            self.randomize_function_params,
            self.generate_image_wrapper,
            self.update_time_step,
            self.update_visual_fidelity,
            self.save_current_state,
            self.load_saved_state,
            self.toggle_color_mode,
            self.cycle_palette
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
        img_array = generate_image_data(sample_width, sample_height, self.time_val, self.random_params, full_width, full_height)
        
        # Convert GPU array to numpy if needed
        if hasattr(img_array, 'get'):
            img_array = img_array.get()
        
        # Convert to 8-bit without brightness scaling
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
            
        # Create PIL image from array and stretch to viewport
        img = Image.fromarray(img_array, 'RGB')
        if sample_width != full_width or sample_height != full_height:
            img = img.resize((full_width, full_height), Image.Resampling.NEAREST)
        
        return ImageTk.PhotoImage(img)
        
    def update_time_step(self, value):
        self.time_step = float(value)
        
    def update_visual_fidelity(self, value):
        self.visual_fidelity = float(value)
    
    def save_current_state(self):
        """Save current visualization state."""
        try:
            # Open Save As dialog with default directory and filename
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            default_name = f"params_{timestamp}.json"
            default_dir = save_manager.params_dir

            filename = filedialog.asksaveasfilename(
                title="Save Visualization Parameters",
                initialdir=str(default_dir),
                initialfile=default_name,
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )

            if not filename:
                return False

            save_path = save_manager.save_parameters_to_path(self.random_params, filename)
            if save_path:
                logger.info(f"State saved to {save_path}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
            return False
    
    def load_saved_state(self, filename=None):
        """Load saved visualization state."""
        try:
            if filename is None:
                # Get list of saved parameters
                saved_params = save_manager.get_saved_parameters_list()
                if not saved_params:
                    logger.warning("No saved states found")
                    return False
                
                # For now, load the most recent one
                # In a full implementation, you'd show a dialog to select which one
                most_recent = saved_params[-1]
                loaded_params = save_manager.load_parameters(most_recent)
                
                if loaded_params:
                    self.random_params = loaded_params
                    logger.info(f"Loaded state: {most_recent}")
                    return True
                return False
            else:
                # Load specific file
                loaded_params = save_manager.load_parameters(filename)
                
                if loaded_params:
                    self.random_params = loaded_params
                    logger.info(f"Loaded state: {filename}")
                    return True
                return False
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            return False
    
    def update_display(self):
        if self.running:
            self.time_val += self.time_step
            
            # Check for auto-save
            if self.auto_save_interval > 0:
                current_time = time.time()
                if current_time - self.last_auto_save > self.auto_save_interval:
                    save_manager.auto_save(self.random_params)
                    self.last_auto_save = current_time
            
            # Check for performance optimization (only if auto scaling is enabled)
            auto_scaling = config.get('performance.auto_scaling', True)
            if auto_scaling:
                should_adjust, new_fidelity = performance_monitor.should_adjust_fidelity(self.visual_fidelity)
                if should_adjust:
                    self.visual_fidelity = new_fidelity
                    self.ui.update_fidelity_slider(new_fidelity)
            
            # Update the display using the UI component
            updated_vals = self.ui.update_display(self.time_val, self.running)
            self.time_val, self.width, self.height = updated_vals
                
            update_interval = config.get('visualization.update_interval_ms', 50)
            self.root.after(update_interval, self.update_display)
            
    def start_animation(self):
        if not self.running:
            self.running = True
            self.update_display()
            
    def stop_animation(self):
        self.running = False
        
    def randomize_function_params(self):
        """Generate new random parameters for the mathematical function."""
        self.random_params = randomize_function_params()
        logger.log_function_params(self.random_params)

    def toggle_color_mode(self):
        if not self.random_params:
            return
        mode = self.random_params.get('color_mode', 'harmonic')
        self.random_params['color_mode'] = 'palette' if mode != 'palette' else 'harmonic'
        logger.info(f"Color mode: {self.random_params['color_mode']}")

    def cycle_palette(self, step):
        if not self.random_params:
            return
        # Palettes defined in math_function._PALETTES
        try:
            from math_function import _PALETTES  # type: ignore
        except Exception:
            return
        names = list(_PALETTES.keys())
        current = self.random_params.get('palette_name', names[0])
        if current not in names:
            current = names[0]
        idx = (names.index(current) + (1 if step >= 0 else -1)) % len(names)
        self.random_params['palette_name'] = names[idx]
        logger.info(f"Palette: {self.random_params['palette_name']}")
    
    def on_closing(self):
        """Handle application closing."""
        try:
            self.stop_animation()
            logger.info("Application closing")
            self.root.destroy()
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
        
    def run(self):
        """Start the application."""
        try:
            logger.info("Starting visualizer application")
            self.start_animation()
            self.root.mainloop()
        except Exception as e:
            logger.error(f"Application error: {e}")
            raise


if __name__ == "__main__":
    visualizer = Visualizer()
    visualizer.run()