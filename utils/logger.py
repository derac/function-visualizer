"""Logging utilities for the function visualizer."""

import logging
import sys
from pathlib import Path
from datetime import datetime
from config import config

class VisualizerLogger:
    """Custom logger for the visualizer application."""
    
    def __init__(self, name="visualizer"):
        self.logger = logging.getLogger(name)
        self.setup_logger()
    
    def setup_logger(self):
        """Setup the logger with file and console handlers."""
        # Clear any existing handlers
        self.logger.handlers.clear()
        
        # Get logging configuration
        log_config = config.get_logging_config()
        log_level = getattr(logging, log_config.get('level', 'INFO').upper())
        self.logger.setLevel(log_level)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler (if enabled)
        if log_config.get('log_to_file', True):
            log_file = log_config.get('log_file', 'visualizer.log')
            log_path = Path(log_file)
            
            # Create log directory if it doesn't exist
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_path)
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def debug(self, message):
        """Log debug message."""
        self.logger.debug(message)
    
    def info(self, message):
        """Log info message."""
        self.logger.info(message)
    
    def warning(self, message):
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message):
        """Log error message."""
        self.logger.error(message)
    
    def critical(self, message):
        """Log critical message."""
        self.logger.critical(message)
    
    def log_performance(self, frame_time_ms, fps, resolution):
        """Log performance metrics."""
        self.debug(f"Frame: {frame_time_ms:.1f}ms, FPS: {fps:.1f}, Resolution: {resolution}")
    
    def log_function_params(self, params):
        """Log function parameters for debugging."""
        enabled_ops = [op for op, enabled in params.items() if enabled and op.startswith('use_')]
        self.info(f"Function parameters loaded - Enabled operations: {enabled_ops}")

# Global logger instance
logger = VisualizerLogger()
