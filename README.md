# PyAudioLocalization

[![GitHub license](https://img.shields.io/github/license/zeynelacikgoez/PyAudioLocalization)](https://github.com/zeynelacikgoez/PyAudioLocalization/blob/main/LICENSE)
[![GitHub issues](https://img.shields.io/github/issues/zeynelacikgoez/PyAudioLocalization)](https://github.com/zeynelacikgoez/PyAudioLocalization/issues)
[![GitHub stars](https://img.shields.io/github/stars/zeynelacikgoez/PyAudioLocalization)](https://github.com/zeynelacikgoez/PyAudioLocalization/stargazers)

## Project Description

**PyAudioLocalization** is a Python-based framework designed for sound source localization and advanced audio signal processing. It supports both simulations and real-world audio data for analyzing acoustic signals, performing noise filtering, and estimating sound source positions. The framework utilizes time difference of arrival (TDOA) and cross-correlation techniques for localization, while also providing noise reduction and multi-path propagation simulation.

## Features
- **Signal Generation**: Generate sine waves, noise, chirps, and speech signals.
- **Cross-Correlation**: Estimate time delays between microphones using PHAT.
- **Source Localization**: Estimate sound source position using least squares and differential evolution optimization.
- **Noise Reduction**: Apply Butterworth or Wiener filtering to audio signals.
- **Multi-path Simulation**: Simulate multi-path sound propagation, including reflections and attenuation.
- **Clustering & Analysis**: Perform cluster analysis on signal correlations and visualize results.
- **Visualization**: Plot heatmaps and 3D graphs for cross-correlation and TDOA analysis.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/zeynelacikgoez/PyAudioLocalization.git
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Required Libraries
- `numpy`
- `scipy`
- `matplotlib`
- `soundfile`
- `logging`
- `sklearn`
- `statsmodels`
- `resampy`

## Usage

### Basic Usage
The main entry point is the `lokalisieren_schallquelle` function, which can either simulate or process real audio data for sound source localization. 

#### Simulating Sound Sources

```python
from signal_localization import lokalisieren_schallquelle

result = lokalisieren_schallquelle(
    temperatur=20,
    luftfeuchtigkeit=50,
    fs=44100,
    mic_positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
    source_position=np.array([0.5, 0.5, 0.5]),
    signal_type='sine',
    freq=1000,
    use_simulation=True
)
print(result['geschätzte_position'])
```

#### Processing Real Audio Data

```python
audiodateien = ['mic1.wav', 'mic2.wav', 'mic3.wav', 'mic4.wav']
result = lokalisieren_schallquelle(
    use_simulation=False,
    audiodateien=audiodateien,
    fs=44100
)
print(result['geschätzte_position'])
```

## Functions Overview

- **`schallgeschwindigkeit(temp, feuchte, druck=101.325)`**: Computes the speed of sound based on temperature, humidity, and atmospheric pressure.
- **`reflect_point_across_plane(point, plane)`**: Reflects a point across a given plane.
- **`distance(point1, point2)`**: Computes the Euclidean distance between two points in 3D space.
- **`generate_image_sources_iterative(source, planes, max_order, frequency, ...)`**: Generates image sources considering reflection and absorption.
- **`normalize_signal(signal)`**: Normalizes a signal to the range [-1, 1].
- **`dynamic_range_compression(signal, threshold=0.8)`**: Applies dynamic range compression to a signal.
- **`generate_signal(signal_type, fs, duration, freq)`**: Generates different types of signals (sine, noise, chirp, or speech).
- **`get_time_delays_phat(sig1, sig2, fs)`**: Estimates the time delay between two signals using PHAT cross-correlation.
- **`lokalisieren_schallquelle(...)`**: Main function to localize sound sources either through simulation or real-world audio data.

## Examples

### Example 1: Simulating a Sound Source
```python
from signal_localization import lokalisieren_schallquelle

result = lokalisieren_schallquelle(
    temperatur=25,
    luftfeuchtigkeit=40,
    fs=48000,
    mic_positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
    source_position=np.array([0.5, 0.5, 0.5]),
    signal_type='chirp',
    freq=1500,
    use_simulation=True
)
```

### Example 2: Processing Real Audio Data
```python
from signal_localization import lokalisieren_schallquelle

audiodateien = ['mic1.wav', 'mic2.wav', 'mic3.wav']
result = lokalisieren_schallquelle(
    use_simulation=False,
    audiodateien=audiodateien,
    fs=44100,
    max_distance=5
)
```

## License
This project is licensed under the MIT License. See the [LICENSE](https://github.com/zeynelacikgoez/PyAudioLocalization/blob/main/LICENSE) file for details.
