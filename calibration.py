######### calibration.py #########

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate, correlation_lags, chirp, get_window
import logging
from signal_processing import normalize_signal, dynamic_range_compression, fractional_delay
from utils import speed_of_sound

def generate_calibration_signal(fs, duration=1.0, signal_type='chirp', freq_start=500, freq_end=5000):
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    if signal_type == 'chirp':
        calib_signal = chirp(t, f0=freq_start, f1=freq_end, t1=duration, method='linear')
    elif signal_type == 'impulse':
        calib_signal = np.zeros_like(t)
        calib_signal[0] = 1.0
    else:
        raise ValueError("Unsupported calibration signal type. Use 'chirp' or 'impulse'.")
    calib_signal = normalize_signal(calib_signal)
    calib_signal = dynamic_range_compression(calib_signal)
    return calib_signal

def simulate_calibration_recording(calib_signal, mic_positions, source_position, fs, c, attenuation_factor=1.0, noise_level=0.01, freq=None, material_properties=None):
    from utils import calculate_attenuation
    if freq is None:
        freq = 1000
    if material_properties is None:
        from materials import material_properties as default_materials
        material_properties = default_materials

    recordings = []
    for mic_pos in mic_positions:
        distance_val = np.linalg.norm(np.array(source_position) - np.array(mic_pos))
        time_delay = distance_val / c
        attenuation = attenuation_factor * calculate_attenuation(distance_val, 'air', freq, material_properties)
        delayed_signal = fractional_delay(calib_signal, time_delay, fs)
        recorded_signal = delayed_signal * attenuation
        recorded_signal += np.random.normal(0, noise_level, size=recorded_signal.shape)
        recordings.append(recorded_signal)
    return recordings

def analyze_calibration(recorded_signals, calib_signal, fs):
    results = []
    for rec in recorded_signals:
        corr = correlate(rec, calib_signal, mode='full')
        lags = correlation_lags(len(rec), len(calib_signal), mode='full')
        lag = lags[np.argmax(np.abs(corr))]
        delay_estimate = lag / fs
        amplitude_estimate = np.max(np.abs(corr))
        results.append({'delay': delay_estimate, 'amplitude': amplitude_estimate})
    return results

def plot_calibration_results(results):
    delays = [res['delay'] for res in results]
    amplitudes = [res['amplitude'] for res in results]
    
    fig, ax1 = plt.subplots(figsize=(8, 5))
    indices = np.arange(len(results))
    
    ax1.bar(indices, delays, color='skyblue', alpha=0.7, label='Delay (s)')
    ax1.set_xlabel('Microphone Index')
    ax1.set_ylabel('Delay (s)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    
    ax2 = ax1.twinx()
    ax2.plot(indices, amplitudes, 'r-o', label='Amplitude')
    ax2.set_ylabel('Cross-correlation Amplitude', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    
    plt.title('Calibration Results per Microphone')
    fig.tight_layout()
    plt.show()

def run_calibration(config):
    fs = config["fs"]
    duration = config["duration"]
    source_position = config["source_position"]
    mic_positions = config["mic_positions"]
    c = speed_of_sound(config["celsius"], config["humidity"])
    
    cal_config = config["calibration"]
    calib_signal = generate_calibration_signal(
        fs,
        duration,
        signal_type=cal_config.get("signal_type", "chirp"),
        freq_start=cal_config.get("freq_start", 500),
        freq_end=cal_config.get("freq_end", 5000)
    )
    logging.info("Calibration signal generated.")
    
    recorded_signals = simulate_calibration_recording(
        calib_signal,
        mic_positions,
        source_position,
        fs,
        c,
        attenuation_factor=cal_config.get("attenuation_factor", 1.0),
        noise_level=cal_config.get("noise_level", 0.01)
    )
    logging.info("Simulated calibration recordings created.")
    
    results = analyze_calibration(recorded_signals, calib_signal, fs)
    logging.info("Calibration analysis completed.")
    
    return results, calib_signal, recorded_signals

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
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
        "calibration": {
            "signal_type": "chirp",
            "freq_start": 500,
            "freq_end": 5000,
            "attenuation_factor": 1.0,
            "noise_level": 0.01
        }
    }
    
    results, calib_signal, recorded_signals = run_calibration(config)
    for idx, res in enumerate(results):
        logging.info(f"Microphone {idx+1}: Delay = {res['delay']:.6f} s, Amplitude = {res['amplitude']:.3f}")
    plot_calibration_results(results)
