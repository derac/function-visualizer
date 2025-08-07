"""Save and load functionality for visualizations."""

import json
import pickle
from pathlib import Path
from datetime import datetime
from utils.hardware import get_array_module
from config import config
from utils.logger import logger

# Get the appropriate array module (numpy or cupy)
np = get_array_module()

class SaveManager:
    """Manages saving and loading of visualizations and parameters."""
    
    def __init__(self):
        self.save_dir = Path(config.get('saving.save_directory', 'saves'))
        self.save_dir.mkdir(exist_ok=True)
        
        # Create parameters subdirectory only
        self.params_dir = self.save_dir / 'parameters'
        self.params_dir.mkdir(exist_ok=True)
    
    def save_parameters(self, params, name=None):
        """Save function parameters to file."""
        if name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            name = f"params_{timestamp}"
        
        file_path = self.params_dir / f"{name}.json"
        
        try:
            # Convert numpy arrays to lists for JSON serialization
            serializable_params = self._prepare_params_for_saving(params)
            
            with open(file_path, 'w') as f:
                json.dump(serializable_params, f, indent=2)
            
            logger.info(f"Parameters saved to {file_path}")
            return str(file_path)
        except Exception as e:
            logger.error(f"Failed to save parameters: {e}")
            return None

    def save_parameters_to_path(self, params, file_path):
        """Save function parameters directly to a provided file path."""
        try:
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            serializable_params = self._prepare_params_for_saving(params)
            with open(file_path, 'w') as f:
                json.dump(serializable_params, f, indent=2)
            logger.info(f"Parameters saved to {file_path}")
            return str(file_path)
        except Exception as e:
            logger.error(f"Failed to save parameters: {e}")
            return None
    
    def load_parameters(self, name):
        """Load function parameters from file."""
        file_path = self.params_dir / f"{name}.json"
        
        if not file_path.exists():
            logger.error(f"Parameter file not found: {file_path}")
            return None
        
        try:
            with open(file_path, 'r') as f:
                params = json.load(f)
            
            # Convert lists back to numpy arrays where needed
            params = self._prepare_params_for_loading(params)
            
            logger.info(f"Parameters loaded from {file_path}")
            return params
        except Exception as e:
            logger.error(f"Failed to load parameters: {e}")
            return None
    
    
    def get_saved_parameters_list(self):
        """Get list of all saved parameter files."""
        param_files = list(self.params_dir.glob("*.json"))
        return [f.stem for f in param_files]
    
    def delete_saved_item(self, name, item_type='parameters'):
        """Delete a saved item."""
        if item_type != 'parameters':
            logger.error(f"Unknown item type: {item_type}")
            return False
        file_path = self.params_dir / f"{name}.json"
        
        try:
            if file_path.exists():
                file_path.unlink()
                logger.info(f"Deleted {item_type}: {name}")
                return True
            else:
                logger.warning(f"File not found: {file_path}")
                return False
        except Exception as e:
            logger.error(f"Failed to delete {item_type}: {e}")
            return False
    
    def _prepare_params_for_saving(self, params):
        """Convert parameters to JSON-serializable format recursively."""
        def convert(value):
            # Arrays (numpy/cupy)
            if hasattr(value, 'tolist'):
                return value.tolist()
            # Numpy/cupy scalars
            if hasattr(value, 'item') and hasattr(value, 'dtype'):
                return value.item()
            # Dicts
            if isinstance(value, dict):
                return {str(k): convert(v) for k, v in value.items()}
            # Lists/Tuples/Sets
            if isinstance(value, (list, tuple, set)):
                return [convert(v) for v in value]
            # Basic types
            if isinstance(value, (str, int, float, bool)) or value is None:
                return value
            # Fallback to string
            return str(value)

        return {str(k): convert(v) for k, v in params.items()}
    
    def _prepare_params_for_loading(self, params):
        """Convert loaded parameters back to proper format."""
        converted = {}
        
        for key, value in params.items():
            if isinstance(value, list) and key in ['voronoi_seeds']:
                # Convert list of coordinates back to proper format
                converted[key] = value
            elif isinstance(value, (int, float, bool, str)):
                converted[key] = value
            else:
                converted[key] = value
        
        return converted
    
    def auto_save(self, params):
        """Auto-save functionality based on configuration (parameters only)."""
        auto_save_interval = config.get('saving.auto_save_interval', 0)
        
        if auto_save_interval > 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save parameters only
            self.save_parameters(params, f"auto_params_{timestamp}")
    
    def export_preset(self, params, name, description=""):
        """Export parameters as a preset with metadata."""
        preset = {
            'name': name,
            'description': description,
            'created': datetime.now().isoformat(),
            'parameters': self._prepare_params_for_saving(params)
        }
        
        file_path = self.params_dir / f"preset_{name}.json"
        
        try:
            with open(file_path, 'w') as f:
                json.dump(preset, f, indent=2)
            
            logger.info(f"Preset saved: {name}")
            return str(file_path)
        except Exception as e:
            logger.error(f"Failed to save preset: {e}")
            return None
    
    def import_preset(self, name):
        """Import a preset and return its parameters."""
        file_path = self.params_dir / f"preset_{name}.json"
        
        if not file_path.exists():
            logger.error(f"Preset not found: {file_path}")
            return None
        
        try:
            with open(file_path, 'r') as f:
                preset = json.load(f)
            
            params = self._prepare_params_for_loading(preset['parameters'])
            logger.info(f"Preset loaded: {preset.get('name', name)}")
            return params
        except Exception as e:
            logger.error(f"Failed to load preset: {e}")
            return None

# Global save manager instance
save_manager = SaveManager()
