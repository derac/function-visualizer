# Function Visualizer

A real-time mathematical function visualizer that creates beautiful, animated patterns using complex mathematical operations. The application supports GPU acceleration and includes advanced features for performance monitoring, saving/loading, and customization.

## Features

### 🎨 Visualization
- **13 Mathematical Operations**: Sine/cosine waves, XOR patterns, cellular automata, domain warping, polar transformations, noise generation, and more
- **Real-time Animation**: Smooth, time-based parameter evolution
- **GPU Acceleration**: CuPy support for faster computation when available
- **Visual Fidelity Control**: Adjustable resolution scaling for performance optimization
- **Color Evolution**: Sophisticated color mapping with harmonic layers and temperature variations

### 💾 Save/Load System
- **Parameter Saving**: Save your favorite visualizations with timestamps
- **Preset System**: Export and import parameter presets with metadata
- **Auto-save**: Configurable automatic saving at specified intervals
- **Image Export**: Save current visualizations as PNG/JPG images
- **Video Frame Export**: Export frames for video creation

### 📊 Performance Monitoring
- **Real-time FPS Display**: Monitor frame rate and performance
- **Auto-optimization**: Automatic fidelity adjustment based on performance
- **Memory Tracking**: Monitor memory usage (requires psutil)
- **Performance Warnings**: Alerts for low FPS or high memory usage
- **Optimization History**: Track performance optimizations made

### ⚙️ Configuration
- **JSON Configuration**: Centralized settings management
- **Customizable UI**: Show/hide performance display, toolbar
- **Hardware Detection**: Automatic GPU/CPU selection
- **Logging System**: Comprehensive logging with file output

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd function-visualizer
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Optional GPU acceleration**:
   ```bash
   pip install cupy-cuda11x  # Replace with your CUDA version
   ```

## Usage

### Basic Usage
```bash
python visualizer.py
```

### Controls
- **R**: Randomize parameters (new pattern)
- **S**: Save current state
- **L**: Load saved state
- **F**: Toggle fullscreen
- **H**: Toggle toolbar visibility
- **P**: Toggle performance display

### Sliders
- **Step**: Animation speed (0.0-0.2)
- **Scale**: Visual fidelity/performance (5-100%)

## Configuration

The application uses `config.json` for settings. Key configuration options:

```json
{
  "window": {
    "width": 800,
    "height": 600,
    "title": "Function Visualizer"
  },
  "visualization": {
    "default_time_step": 0.05,
    "default_visual_fidelity": 100.0,
    "frame_rate": 20
  },
  "performance": {
    "enable_gpu": true,
    "auto_adjust_fidelity": true
  },
  "saving": {
    "auto_save_interval": 0,
    "save_directory": "saves"
  }
}
```

## Project Structure

```
function-visualizer/
├── visualizer.py          # Main application
├── math_function.py       # Mathematical computations
├── config.py             # Configuration management
├── requirements.txt       # Dependencies
├── README.md             # This file
├── ui/
│   └── visualizer_ui.py  # User interface
├── utils/
│   ├── hardware.py       # GPU/CPU detection
│   ├── logger.py         # Logging system
│   ├── save_manager.py   # Save/load functionality
│   └── performance.py    # Performance monitoring
└── saves/                # Saved files (auto-created)
    ├── parameters/       # Saved parameters
    ├── images/          # Exported images
    └── videos/          # Video frames
```

## Mathematical Operations

The visualizer supports 13 different mathematical operations:

1. **Sine Waves** (`use_sin`): Basic sine wave patterns
2. **Cosine Waves** (`use_cos`): Basic cosine wave patterns
3. **XOR Patterns** (`use_xor`): Bitwise XOR with morphing
4. **Cellular Automata** (`use_cellular`): Grid-based patterns
5. **Domain Warping** (`use_domain_warp`): Coordinate transformation
6. **Polar Transformations** (`use_polar`): Radial coordinate patterns
7. **Noise Generation** (`use_noise`): Multi-octave pseudo-noise
8. **Absolute Value** (`use_abs`): Absolute value transformations
9. **Power Functions** (`use_power`): Exponential transformations
10. **Feedback Loops** (`use_feedback`): Frame-to-frame feedback
11. **Voronoi Diagrams** (`use_voronoi`): Distance field patterns
12. **Reaction-Diffusion** (`use_reaction_diffusion`): Gray-Scott model
13. **Sinusoidal Fields** (`use_sinusoidal_field`): Complex 2D field patterns

## Performance Tips

- **GPU Acceleration**: Install CuPy for significant performance improvement
- **Visual Fidelity**: Lower the scale slider for better performance
- **Auto-optimization**: Enable auto-adjust fidelity in config
- **Memory Monitoring**: Install psutil for memory tracking

## Troubleshooting

### Common Issues

1. **Low FPS**: Reduce visual fidelity or enable auto-optimization
2. **High Memory Usage**: Lower resolution or restart application
3. **GPU Not Detected**: Ensure CuPy is properly installed for your CUDA version
4. **Save/Load Errors**: Check file permissions in saves directory

### Logging

The application logs to both console and file (`visualizer.log`). Check logs for detailed error information.

## Development

### Adding New Mathematical Operations

1. Add operation to `math_function.py` in the `compute_function` method
2. Add parameters to `randomize_function_params()`
3. Update the operations list in the randomization function

### Extending the UI

1. Modify `ui/visualizer_ui.py` for new controls
2. Update the main visualizer class for new callbacks
3. Add configuration options in `config.py`

## License

This project is open source. Feel free to contribute improvements!

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Acknowledgments

- PIL/Pillow for image processing
- NumPy for array operations
- CuPy for GPU acceleration
- Tkinter for the user interface
