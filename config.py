"""Configuration management for the function visualizer."""

import json
import os
from pathlib import Path

class Config:
    """Configuration manager for the visualizer application."""
    
    def __init__(self, config_file="config.json"):
        self.config_file = Path(config_file)
        self.default_config = {
            "window": {
                "width": 800,
                "height": 600,
                "title": "Function Visualizer"
            },
            "visualization": {
                "default_time_step": 0.05,
                "default_visual_fidelity": 100.0,
                "frame_rate": 20,  # FPS
                "update_interval_ms": 50
            },
            "performance": {
                "enable_gpu": True,
                "max_resolution": 1920,
                "min_resolution": 320,
                "auto_adjust_fidelity": True,
                "auto_scaling": True
            },
            "ui": {
                "show_toolbar": True,
                "show_fps": True,
                "show_hardware_info": True,
                "theme": "default"
            },
            "saving": {
                "auto_save_interval": 0,  # 0 = disabled
                "save_directory": "saves",
                "image_format": "PNG",
                "video_format": "MP4"
            },
            "logging": {
                "level": "INFO",
                "log_to_file": True,
                "log_file": "visualizer.log"
            }
        }
        self.config = self.load_config()
    
    def load_config(self):
        """Load configuration from file or create default."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    loaded_config = json.load(f)
                # Merge with defaults to ensure all keys exist
                return self._merge_configs(self.default_config, loaded_config)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error loading config: {e}. Using defaults.")
                return self.default_config.copy()
        else:
            self.save_config(self.default_config)
            return self.default_config.copy()
    
    def _merge_configs(self, default, loaded):
        """Recursively merge loaded config with defaults."""
        result = default.copy()
        for key, value in loaded.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        return result
    
    def save_config(self, config=None):
        """Save current configuration to file."""
        if config is None:
            config = self.config
        
        try:
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
        except IOError as e:
            print(f"Error saving config: {e}")
    
    def get(self, key_path, default=None):
        """Get configuration value using dot notation (e.g., 'window.width')."""
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key_path, value):
        """Set configuration value using dot notation."""
        keys = key_path.split('.')
        config = self.config
        
        # Navigate to the parent of the target key
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        # Set the value
        config[keys[-1]] = value
        self.save_config()
    
    def reset_to_defaults(self):
        """Reset configuration to default values."""
        self.config = self.default_config.copy()
        self.save_config()
    
    def get_window_config(self):
        """Get window configuration."""
        return self.get('window', {})
    
    def get_visualization_config(self):
        """Get visualization configuration."""
        return self.get('visualization', {})
    
    def get_performance_config(self):
        """Get performance configuration."""
        return self.get('performance', {})
    
    def get_ui_config(self):
        """Get UI configuration."""
        return self.get('ui', {})
    
    def get_saving_config(self):
        """Get saving configuration."""
        return self.get('saving', {})
    
    def get_logging_config(self):
        """Get logging configuration."""
        return self.get('logging', {})

# Global configuration instance
config = Config()
