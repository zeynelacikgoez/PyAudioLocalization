# PyAudioLocalization

[![GitHub license](https://img.shields.io/github/license/zeynelacikgoez/PyAudioLocalization)](https://github.com/zeynelacikgoez/PyAudioLocalization/blob/main/LICENSE)
[![GitHub issues](https://img.shields.io/github/issues/zeynelacikgoez/PyAudioLocalization)](https://github.com/zeynelacikgoez/PyAudioLocalization/issues)
[![GitHub stars](https://img.shields.io/github/stars/zeynelacikgoez/PyAudioLocalization)](https://github.com/zeynelacikgoez/PyAudioLocalization/stargazers)

## Project Description

**PyAudioLocalization** is a Python-based framework designed for sound source localization and advanced audio signal processing. It supports both simulations and real-world audio data for analyzing acoustic signals, performing noise filtering, and estimating sound source positions. The framework utilizes Time Difference of Arrival (TDOA) and cross-correlation techniques for localization, while also providing noise reduction and multi-path propagation simulation.

## Features

- **Signal Generation**: Generate sine waves, noise, chirps, and speech signals.
- **Cross-Correlation**: Estimate time delays between microphones using PHAT.
- **Source Localization**: Estimate sound source position using Least Squares and Differential Evolution optimization.
- **Noise Reduction**: Apply Butterworth or Wiener filtering to audio signals.
- **Multi-path Simulation**: Simulate multi-path sound propagation, including reflections and attenuation.
- **Clustering & Analysis**: Perform cluster analysis on signal correlations and visualize results.
- **Visualization**: Plot heatmaps and 3D graphs for cross-correlation and TDOA analysis.
- **Modular Project Structure**: Enhanced maintainability and extensibility through modular code organization.
- **Extensible Material Definitions**: Easily add or adjust materials for simulating reflections.

## Project Structure

```
project/
│
├── main.py
├── signal_processing.py
├── plotting.py
├── utils.py
├── materials.py
└── requirements.txt
```

- **main.py**: Main script that controls the workflow.
- **signal_processing.py**: Functions for signal processing (generation, normalization, compression, resampling).
- **plotting.py**: Visualization functions (heatmaps, 3D plots).
- **utils.py**: Utility functions (audio file reading, distance calculations, attenuation, etc.).
- **materials.py**: Definitions and properties of materials.
- **requirements.txt**: List of required Python libraries.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/zeynelacikgoez/PyAudioLocalization.git
   ```
2. **Navigate to the project directory:**
   ```bash
   cd PyAudioLocalization
   ```
3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Required Libraries

- `numpy`
- `scipy`
- `matplotlib`
- `soundfile`
- `scikit-learn`
- `resampy`

## Usage

### Basic Usage

The main entry point is the `lokalisieren_schallquelle` function, which can either simulate or process real audio data for sound source localization.

#### Simulating Sound Sources

```python
import numpy as np
from main import lokalisieren_schallquelle
from materials import material_properties

result = lokalisieren_schallquelle(
    temperatur=20,
    luftfeuchtigkeit=50,
    fs=44100,
    mic_positions=np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ]),
    source_position=np.array([0.5, 0.5, 0.5]),
    duration=1.0,
    signal_type='sine',
    freq=1000,
    reflektierende_ebenen=[
        {'plane': np.array([1, 0, 0, -5]), 'material': 'wood'},
        {'plane': np.array([0, 1, 0, -5]), 'material': 'metal'},
        {'plane': np.array([0, 0, 1, -5]), 'material': 'wood'}
    ],
    material_properties=material_properties,
    filter_method='butterworth',
    use_simulation=True,
    absorption_threshold=0.01,
    analyze_correlation=True,
    visualize_correlation=True,
    clustering_method='kmeans',
    clustering_eps=0.001,
    clustering_min_samples=2,
    max_reflections=3,
    show_plots=True
)

print("Estimated Position of Sound Source:", result['geschätzte_position'])
```

#### Processing Real Audio Data

```python
import numpy as np
from main import lokalisieren_schallquelle
from materials import material_properties

audiodateien = ['mic1.wav', 'mic2.wav', 'mic3.wav', 'mic4.wav']
result = lokalisieren_schallquelle(
    temperatur=20,
    luftfeuchtigkeit=50,
    fs=44100,
    mic_positions=np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ]),
    use_simulation=False,
    audiodateien=audiodateien,
    material_properties=material_properties,
    filter_method='butterworth',
    absorption_threshold=0.01,
    analyze_correlation=True,
    visualize_correlation=True,
    clustering_method='kmeans',
    clustering_eps=0.001,
    clustering_min_samples=2,
    max_reflections=3,
    show_plots=True
)

print("Estimated Position of Sound Source:", result['geschätzte_position'])
```

## Functions Overview

- **`schallgeschwindigkeit(temp, feuchte, druck=101.325)`**: Computes the speed of sound based on temperature, humidity, and atmospheric pressure.
- **`reflect_point_across_plane(point, plane)`**: Reflects a point across a given plane.
- **`distance(point1, point2)`**: Computes the Euclidean distance between two points in 3D space.
- **`generate_image_sources_iterative(source, planes, max_order, frequency, material_properties, mic_positions, absorption_threshold=0.01)`**: Generates image sources considering reflection and absorption.
- **`normalize_signal(signal)`**: Normalizes a signal to the range [-1, 1].
- **`dynamic_range_compression(signal, threshold=0.8, epsilon=1e-8)`**: Applies dynamic range compression to a signal.
- **`generate_signal(signal_type, fs, duration, freq)`**: Generates different types of signals (sine, noise, chirp, or speech).
- **`get_time_delays_phat(sig1, sig2, fs, num_peaks=1)`**: Estimates the time delay between two signals using PHAT cross-correlation.
- **`lokalisieren_schallquelle(...)`**: Main function to localize sound sources either through simulation or real-world audio data.

## Examples

### Example 1: Simulating a Sound Source

```python
import numpy as np
from main import lokalisieren_schallquelle
from materials import material_properties

result = lokalisieren_schallquelle(
    temperatur=25,
    luftfeuchtigkeit=40,
    fs=48000,
    mic_positions=np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0]
    ]),
    source_position=np.array([0.5, 0.5, 0.5]),
    duration=1.0,
    signal_type='chirp',
    freq=1500,
    reflektierende_ebenen=[
        {'plane': np.array([1, 0, 0, -5]), 'material': 'wood'},
        {'plane': np.array([0, 1, 0, -5]), 'material': 'metal'},
        {'plane': np.array([0, 0, 1, -5]), 'material': 'wood'}
    ],
    material_properties=material_properties,
    filter_method='butterworth',
    use_simulation=True,
    absorption_threshold=0.01,
    analyze_correlation=True,
    visualize_correlation=True,
    clustering_method='kmeans',
    clustering_eps=0.001,
    clustering_min_samples=2,
    max_reflections=3,
    show_plots=True
)

print("Estimated Position of Sound Source:", result['geschätzte_position'])
```

### Example 2: Processing Real Audio Data

```python
import numpy as np
from main import lokalisieren_schallquelle
from materials import material_properties

audiodateien = ['mic1.wav', 'mic2.wav', 'mic3.wav']
result = lokalisieren_schallquelle(
    temperatur=22,
    luftfeuchtigkeit=55,
    fs=44100,
    mic_positions=np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0]
    ]),
    use_simulation=False,
    audiodateien=audiodateien,
    material_properties=material_properties,
    filter_method='wiener',
    absorption_threshold=0.01,
    analyze_correlation=True,
    visualize_correlation=True,
    clustering_method='dbscan',
    clustering_eps=0.002,
    clustering_min_samples=3,
    max_reflections=2,
    show_plots=True
)

print("Estimated Position of Sound Source:", result['geschätzte_position'])
```

## Adding Materials

To add new materials or adjust existing ones, edit the `materials.py` file. Material properties must be defined as a nested dictionary, where each material name maps to another dictionary containing the `absorption` and `freq` properties.

```python
# materials.py

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

## Error Handling and Logging

The framework utilizes extensive logging to inform users about progress and any potential errors. Ensure that the logging level is appropriately configured to receive detailed information.

```python
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
```

## License

This project is licensed under the MIT License. See the [LICENSE](https://github.com/zeynelacikgoez/PyAudioLocalization/blob/main/LICENSE) file for details.

## Contributing

Contributions are welcome! Please open an issue or a pull request to suggest improvements, report bugs, or add new features.
