"""Performance monitoring and optimization utilities."""

import time
import threading
from collections import deque
from config import config
from utils.logger import logger

class PerformanceMonitor:
    """Monitors and optimizes application performance."""
    
    def __init__(self, max_history=100):
        self.max_history = max_history
        self.frame_times = deque(maxlen=max_history)
        self.fps_history = deque(maxlen=max_history)
        
        # Performance thresholds
        self.target_fps = config.get('visualization.frame_rate', 20)
        self.min_fps = self.target_fps * 0.8
        self.max_fps = self.target_fps * 1.2
        
        # Auto-adjustment settings
        self.auto_adjust = config.get('performance.auto_adjust_fidelity', True)
        self.min_fidelity = 5.0
        self.max_fidelity = 100.0
        self.fidelity_step = 5.0
        
        # Current performance state
        self.current_fps = 0.0
        self.avg_frame_time = 0.0
        self.performance_warnings = []
        
        # Thread safety
        self.lock = threading.Lock()
    
    def start_frame(self):
        """Start timing a frame."""
        return time.time()
    
    def end_frame(self, start_time):
        """End timing a frame and record metrics."""
        frame_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        with self.lock:
            self.frame_times.append(frame_time)
            
            # Calculate current FPS
            if frame_time > 0:
                self.current_fps = 1000.0 / frame_time
                self.fps_history.append(self.current_fps)
            
            # Calculate average frame time
            if self.frame_times:
                self.avg_frame_time = sum(self.frame_times) / len(self.frame_times)
            
        
        # Log performance metrics
        logger.log_performance(frame_time, self.current_fps, f"{self.avg_frame_time:.1f}ms avg")
        
        return frame_time, self.current_fps
    
    def get_performance_stats(self):
        """Get current performance statistics."""
        with self.lock:
            stats = {
                'current_fps': self.current_fps,
                'avg_fps': sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0,
                'avg_frame_time': self.avg_frame_time,
                'min_frame_time': min(self.frame_times) if self.frame_times else 0,
                'max_frame_time': max(self.frame_times) if self.frame_times else 0,
                'frame_count': len(self.frame_times),
            }
        return stats
    
    def should_adjust_fidelity(self, current_fidelity):
        """Determine if fidelity should be adjusted based on performance."""
        if not self.auto_adjust:
            return False, current_fidelity
        
        with self.lock:
            if not self.fps_history:
                return False, current_fidelity
            
            recent_fps = sum(list(self.fps_history)[-10:]) / min(10, len(self.fps_history))
            
            # Performance is too low - reduce fidelity
            if recent_fps < self.min_fps and current_fidelity > self.min_fidelity:
                new_fidelity = max(self.min_fidelity, current_fidelity - self.fidelity_step)
                logger.warning(f"Performance low ({recent_fps:.1f} FPS), reducing fidelity to {new_fidelity}%")
                return True, new_fidelity
            
            # Performance is good - increase fidelity
            elif recent_fps > self.max_fps and current_fidelity < self.max_fidelity:
                new_fidelity = min(self.max_fidelity, current_fidelity + self.fidelity_step)
                logger.info(f"Performance good ({recent_fps:.1f} FPS), increasing fidelity to {new_fidelity}%")
                return True, new_fidelity
        
        return False, current_fidelity
    
    def get_performance_warnings(self):
        """Get list of performance warnings."""
        warnings = []
        
        with self.lock:
            if self.fps_history:
                recent_fps = sum(list(self.fps_history)[-10:]) / min(10, len(self.fps_history))
                
                if recent_fps < self.min_fps:
                    warnings.append(f"Low FPS: {recent_fps:.1f} (target: {self.target_fps})")
                
                if self.avg_frame_time > 1000 / self.target_fps:
                    warnings.append(f"High frame time: {self.avg_frame_time:.1f}ms")
        
        return warnings
    
    def reset_stats(self):
        """Reset performance statistics."""
        with self.lock:
            self.frame_times.clear()
            self.fps_history.clear()
            self.performance_warnings.clear()
    
    def get_optimal_resolution(self, current_width, current_height):
        """Calculate optimal resolution based on performance."""
        max_resolution = config.get('performance.max_resolution', 1920)
        min_resolution = config.get('performance.min_resolution', 320)
        
        # If performance is poor, reduce resolution
        if self.current_fps < self.min_fps:
            scale_factor = 0.8
            new_width = max(min_resolution, int(current_width * scale_factor))
            new_height = max(min_resolution, int(current_height * scale_factor))
            return new_width, new_height
        
        # If performance is excellent, increase resolution
        elif self.current_fps > self.max_fps:
            scale_factor = 1.2
            new_width = min(max_resolution, int(current_width * scale_factor))
            new_height = min(max_resolution, int(current_height * scale_factor))
            return new_width, new_height
        
        return current_width, current_height

class PerformanceOptimizer:
    """Optimizes performance based on monitoring data."""
    
    def __init__(self, monitor):
        self.monitor = monitor
        self.optimization_history = []
    
    def optimize_settings(self, current_settings):
        """Optimize settings based on performance data."""
        optimized = current_settings.copy()
        
        # Get performance stats
        stats = self.monitor.get_performance_stats()
        warnings = self.monitor.get_performance_warnings()
        
        # Apply optimizations based on warnings
        for warning in warnings:
            if "Low FPS" in warning:
                # Reduce visual fidelity
                if 'visual_fidelity' in optimized:
                    optimized['visual_fidelity'] = max(5.0, optimized['visual_fidelity'] - 10.0)
                
                # Reduce time step for smoother animation
                if 'time_step' in optimized:
                    optimized['time_step'] = min(0.2, optimized['time_step'] + 0.01)
        
        # Record optimization
        if optimized != current_settings:
            self.optimization_history.append({
                'timestamp': time.time(),
                'old_settings': current_settings,
                'new_settings': optimized,
                'reason': warnings
            })
        
        return optimized
    
    def get_optimization_history(self):
        """Get history of optimizations made."""
        return self.optimization_history.copy()

# Global performance monitor
performance_monitor = PerformanceMonitor()
performance_optimizer = PerformanceOptimizer(performance_monitor)
