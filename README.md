# PyAudioLocalization

[![GitHub license](https://img.shields.io/github/license/zeynelacikgoez/PyAudioLocalization)](https://github.com/zeynelacikgoez/PyAudioLocalization/blob/main/LICENSE)
[![GitHub issues](https://img.shields.io/github/issues/zeynelacikgoez/PyAudioLocalization)](https://github.com/zeynelacikgoez/PyAudioLocalization/issues)
[![GitHub stars](https://img.shields.io/github/stars/zeynelacikgoez/PyAudioLocalization)](https://github.com/zeynelacikgoez/PyAudioLocalization/stargazers)

## Project Description

**PyAudioLocalization** is a Python framework for sound source localization and advanced audio signal processing. The framework supports both simulations and real-world audio data, combining techniques such as Time Difference of Arrival (TDOA), cross-correlation, noise reduction, and optimization (Least Squares and Differential Evolution) to estimate the position of a sound source. It also simulates multipath propagation—including reflections and attenuation—based on material properties defined in `materials.py`.

## Features

- **Signal Generation:** Generate sine waves, noise, chirps, and simulated speech signals.
- **Cross-Correlation:** Estimate time delays between microphones using PHAT cross-correlation.
- **Source Localization:** Determine the position of a sound source via optimization methods (Least Squares, Differential Evolution) based on TDOA.
- **Noise Reduction:** Apply Butterworth, FIR, or Wiener filtering to audio signals.
- **Multipath Simulation:** Simulate multipath propagation with reflections and attenuation.
- **Clustering & Analysis:** Use clustering algorithms (KMeans, DBSCAN) to initialize and analyze source positions.
- **Visualization:** Plot heatmaps and 3D graphs for cross-correlation and TDOA analysis.
- **Modular Structure:** Clean, modular code organization for easier maintenance and extension.
- **Extensible Material Definitions:** Easily add or adjust material properties in `materials.py`.

## Project Structure

```
.
├── main.py                 # Main script for calibration, simulation, and localization
├── calibration.py          # Functions for microphone calibration
├── plotting.py             # Visualization functions (heatmaps and 3D plots)
├── signal_processing.py    # Signal processing functions (generation, normalization, filtering, etc.)
├── utils.py                # Utility functions (cross-correlation, TDOA, optimization, etc.)
├── materials.py            # Material properties and definitions
└── requirements.txt        # Project dependencies
```

## Installation

### Prerequisites

Make sure you have Python 3.7 or higher installed. The following Python packages are required:

- `numpy`
- `scipy`
- `matplotlib`
- `scikit-learn`
- `soundfile`
- `resampy`

### Installing Dependencies

Clone the repository and install the necessary packages:

```bash
git clone https://github.com/zeynelacikgoez/PyAudioLocalization.git
cd PyAudioLocalization
pip install -r requirements.txt
```

## Usage

The main script `main.py` runs the complete workflow for calibration and sound source localization. The framework can operate with simulated signals or real audio data.

### Example 1: Simulating a Sound Source

```python
import numpy as np
from main import localize_sound_source
from materials import material_properties

# Configuration for simulation
config = {
    "fs": 44100,
    "duration": 1.0,
    "celsius": 20,
    "humidity": 50,
    "mic_positions": [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ],
    "source_position": [0.5, 0.5, 0.5],
    "signal_type": "sine",
    "freq": 1000,
    "reflective_planes": [
        {'plane': [1, 0, 0, -5], 'material': 'wood'},
        {'plane': [0, 1, 0, -5], 'material': 'metal'},
        {'plane': [0, 0, 1, -5], 'material': 'wood'}
    ],
    "calibration": {
        "signal_type": "chirp",
        "freq_start": 500,
        "freq_end": 5000,
        "attenuation_factor": 1.0,
        "noise_level": 0.01
    },
    "localization": {
        "max_reflections": 3,
        "filter_method": "butterworth",
        "absorption_threshold": 0.01,
        "analyze_correlation": True,
        "visualize_correlation": True,
        "clustering_method": "kmeans",
        "clustering_eps": 0.001,
        "clustering_min_samples": 2,
        "max_expected_delay": 0.05  # in seconds
    }
}

results = localize_sound_source(config, use_simulation=True, show_plots=True)
print("Estimated Position of Sound Source:", results['estimated_position'])
```

### Example 2: Processing Real Audio Data

```python
import numpy as np
from main import localize_sound_source
from materials import material_properties

audio_files = ['mic1.wav', 'mic2.wav', 'mic3.wav', 'mic4.wav']
config = {
    "fs": 44100,
    "celsius": 20,
    "humidity": 50,
    "mic_positions": [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ],
    "localization": {
        "max_reflections": 3,
        "filter_method": "butterworth",
        "absorption_threshold": 0.01,
        "analyze_correlation": True,
        "visualize_correlation": True,
        "clustering_method": "kmeans",
        "clustering_eps": 0.001,
        "clustering_min_samples": 2,
        "max_expected_delay": 0.05
    }
}

results = localize_sound_source(config, use_simulation=False, audio_files=audio_files, show_plots=True)
print("Estimated Position of Sound Source:", results['estimated_position'])
```

## Function Overview

- **`speed_of_sound(temperature, humidity, pressure)`**  
  Computes the speed of sound based on temperature, humidity, and atmospheric pressure.

- **`reflect_point_across_plane(point, plane)`**  
  Reflects a point across a given plane (used for generating image sources).

- **`distance(point1, point2)`**  
  Calculates the Euclidean distance between two points in 3D space.

- **`generate_image_sources_iterative(...)`**  
  Iteratively generates image sources based on reflections and attenuation.

- **`phat_correlation(sig1, sig2)`**  
  Computes the PHAT cross-correlation between two signals.

- **`get_time_delays_phat(...)`**  
  Estimates time delays between signals using PHAT-based cross-correlation.

- **`synchronize_signals_improved(...)`**  
  Synchronizes multiple signals using cross-correlation and optional cubic spline interpolation.

- **`heuristic_initialization_adaptive(...)`**  
  Generates initial source position guesses based on TDOA estimates and clustering.

- **`localize_sound_source(config, ...)`**  
  Main function that combines calibration, signal synchronization, TDOA estimation, and localization.

## Customizing Materials

To add new materials or modify existing ones, update the `materials.py` file. For example:

```python
material_properties = {
    'air': {
        'absorption': 0.01,
        'freq': 0.1
    },
    'wood': {
        'absorption': 0.05,
        'freq': 0.8
    },
    'metal': {
        'absorption': 0.1,
        'freq': 0.6
    },
    'glass': {
        'absorption': 0.07,
        'freq': 0.5
    }
    # Additional materials can be added here
}
```

## Logging and Error Handling

The framework uses extensive logging to provide feedback during processing. You can set the logging level (e.g., `INFO` or `DEBUG`) as follows:

```python
import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
```

## License

This project is licensed under the MIT License. See the [LICENSE](https://github.com/zeynelacikgoez/PyAudioLocalization/blob/main/LICENSE) file for details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request with any improvements, bug fixes, or new features.
