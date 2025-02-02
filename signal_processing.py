######### signal_processing.py #########

import numpy as np
from scipy.signal import butter, filtfilt, wiener, chirp, get_window, firwin
from scipy.fft import fft, ifft
from scipy.interpolate import CubicSpline
from scipy.stats import norm
import resampy
import logging

def generate_pink_noise(fs: float, duration: float) -> np.ndarray:
    num_samples = int(fs * duration)
    white = np.random.randn(num_samples)
    fft_white = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(num_samples, d=1./fs)
    scaling = np.ones_like(freqs)
    scaling[1:] = 1 / np.sqrt(freqs[1:])
    scaling[0] = 0
    fft_pink = fft_white * scaling
    pink = np.fft.irfft(fft_pink, n=num_samples)
    pink = normalize_signal(pink)
    pink = dynamic_range_compression(pink)
    return pink

def generate_signal(signal_type: str, fs: float, duration: float, freq: float) -> np.ndarray:
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    if signal_type == 'sine':
        return np.sin(2 * np.pi * freq * t)
    elif signal_type == 'noise':
        return np.random.normal(0, 1, size=t.shape)
    elif signal_type == 'chirp':
        return chirp(t, f0=freq, f1=freq * 5, t1=duration, method='linear')
    elif signal_type == 'speech':
        return generate_realistic_speech(fs, duration)
    else:
        raise ValueError("Unknown signal type. Available types: 'sine', 'noise', 'chirp', 'speech'")

def generate_realistic_speech(fs: float, duration: float) -> np.ndarray:
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    F1, F2, F3 = 800, 1150, 2900
    A1, A2, A3 = 1.0, 0.8, 0.5
    phi1, phi2, phi3 = 0, np.pi / 4, np.pi / 2
    window = get_window('hann', len(t))
    s_formant = (
        A1 * np.sin(2 * np.pi * F1 * t + phi1) +
        A2 * np.sin(2 * np.pi * F2 * t + phi2) +
        A3 * np.sin(2 * np.pi * F3 * t + phi3)
    ) * window

    num_transients = int(duration * 5)
    transient_duration = 0.01
    transient_samples = int(transient_duration * fs)

    s_transient = np.zeros_like(t)
    for _ in range(num_transients):
        start_idx = np.random.randint(0, len(t) - transient_samples)
        transient = np.random.normal(0, 1, transient_samples) * np.hanning(transient_samples)
        s_transient[start_idx:start_idx + transient_samples] += transient

    s_pink = generate_pink_noise(fs, duration) * 0.05
    s = s_formant + s_transient + s_pink
    s = normalize_signal(s)
    s = dynamic_range_compression(s)
    return s

def fractional_delay(signal: np.ndarray, delay: float, fs: float) -> np.ndarray:
    N = len(signal)
    padded_length = 2 * N
    SIGNAL = np.fft.fft(signal, n=padded_length)
    freqs = np.fft.fftfreq(padded_length, d=1./fs)
    phase_shift = np.exp(-1j * 2 * np.pi * freqs * delay)
    delayed_signal = np.fft.ifft(SIGNAL * phase_shift)
    delayed_signal = delayed_signal.real[:N]
    window = get_window('hann', N)
    fade_length = int(0.01 * N)
    window_full = np.ones(N)
    window_full[:fade_length] *= np.linspace(0, 1, fade_length)
    window_full[-fade_length:] *= np.linspace(1, 0, fade_length)
    delayed_signal *= window_full
    return delayed_signal

def normalize_signal(signal: np.ndarray) -> np.ndarray:
    max_val = np.max(np.abs(signal))
    if max_val == 0:
        return signal
    return signal / max_val

def dynamic_range_compression(signal: np.ndarray, threshold: float = 0.8, epsilon: float = 1e-8) -> np.ndarray:
    normalized_signal = normalize_signal(signal)
    compressed_signal = np.sign(normalized_signal) * np.log1p(np.abs(normalized_signal) / threshold + epsilon)
    max_val = np.max(np.abs(compressed_signal))
    if max_val > 0:
        compressed_signal /= max_val
    return compressed_signal

def dynamic_range_compression_soft_clip(signal: np.ndarray, threshold: float = 0.8) -> np.ndarray:
    signal = normalize_signal(signal)
    compressed_signal = np.where(
        np.abs(signal) > threshold,
        np.sign(signal) * (threshold + (np.abs(signal) - threshold) * 0.5),
        signal
    )
    return compressed_signal

def resample_audio(data: np.ndarray, original_fs: float, target_fs: float) -> np.ndarray:
    resampled = resampy.resample(data, original_fs, target_fs, filter='kaiser_best')
    return resampled

def noise_reduction(signal: np.ndarray,
                    fs: float,
                    method: str = 'butterworth',
                    lowcut: float = 300,
                    highcut: float = 3400,
                    filter_order: int = 101) -> np.ndarray:
    """
    Reduce noise in a signal using the specified method.
    
    Verfügbare Methoden:
      - 'butterworth': IIR-Filter (Butterworth)
      - 'fir': FIR-Filter mit linearer Phase (verwendet firwin + filtfilt)
      - 'wiener': Wiener-Filterung
    """
    nyquist = 0.5 * fs
    if method == 'butterworth':
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(5, [low, high], btype='band')
        return filtfilt(b, a, signal)
    elif method == 'fir':
        low = lowcut / nyquist
        high = highcut / nyquist
        taps = firwin(filter_order, [low, high], pass_zero=False)
        # filtfilt anwenden, um Phasenverzerrungen zu vermeiden
        return filtfilt(taps, [1.0], signal)
    elif method == 'wiener':
        return wiener(signal)
    else:
        raise ValueError("Unknown filter method. Available methods: 'butterworth', 'fir', 'wiener'")
