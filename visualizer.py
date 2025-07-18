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
        if event.widget == self.root:
            toolbar_height = self.toolbar.winfo_height()
            new_width = event.width - 20
            new_height = event.height - toolbar_height - 20
            if new_width > 0 and new_height > 0:
                self.width = new_width
                self.height = new_height
                self.canvas.config(width=self.width, height=self.height)
                
    def compute_function(self, x, y, time_val):
        if self.using_cupy and CUPY_AVAILABLE:
            return self.compute_function_cupy(x, y, time_val)
        else:
            return self.compute_function_numpy(x, y, time_val)
            
    def compute_function_numpy(self, x, y, time_val):
        x_normalized = x / 10.0
        y_normalized = y / 10.0
        
        wave1 = np.abs(np.sin(x_normalized + time_val * 0.5))
        wave2 = np.abs(np.cos(y_normalized + time_val * 0.3))
        
        xor_mask = np.bitwise_xor(x, y) & 0xFF
        mod_factor = ((x + y + int(time_val * 10)) % 256) / 255.0
        
        combined = np.clip((xor_mask * mod_factor + wave1 * wave2 * 255), 0, 255)
        
        red = combined.astype(np.uint8)
        green = ((combined * 0.5 + 127) % 256).astype(np.uint8)
        blue = ((combined * 0.3 + 200) % 256).astype(np.uint8)
        
        return np.stack([red, green, blue], axis=-1)
        
    def compute_function_cupy(self, x, y, time_val):
        x_norm = x / 10.0
        y_norm = y / 10.0
        
        wave1 = cp.abs(cp.sin(x_norm + time_val * 0.5))
        wave2 = cp.abs(cp.cos(y_norm + time_val * 0.3))
        
        xor_mask = cp.bitwise_xor(x.astype(cp.int32), y.astype(cp.int32)) & 0xFF
        mod_factor = ((x + y + int(time_val * 10)) % 256) / 255.0
        
        combined = cp.clip((xor_mask * mod_factor + wave1 * wave2 * 255), 0, 255)
        
        red = combined.astype(cp.uint8)
        green = ((combined * 0.5 + 127) % 256).astype(cp.uint8)
        blue = ((combined * 0.3 + 200) % 256).astype(cp.uint8)
        
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
                    if actual_width != self.width or actual_height != self.height:
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
        
    def on_closing(self):
        self.stop_animation()
        self.root.destroy()
        
    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    visualizer = XORVisualizer()
    visualizer.run()