import numpy as np
from scipy.signal import butter, filtfilt, wiener, chirp, get_window
import resampy
import logging

def generate_signal(signal_type, fs, duration, freq):
    """
    Generiert ein Signal basierend auf dem angegebenen Typ.

    :param signal_type: Typ des Signals ('sine', 'noise', 'chirp', 'speech')
    :param fs: Abtastrate in Hz
    :param duration: Dauer des Signals in Sekunden
    :param freq: Frequenz des Signals in Hz (für 'sine' und 'chirp')
    :return: Generiertes Signal als numpy-Array
    """
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
        raise ValueError("Unbekannter Signaltyp. Verfügbare Typen: 'sine', 'noise', 'chirp', 'speech'")

def generate_pink_noise(fs, duration):
    """
    Generiert pinkes Rauschen mittels Fourier-Transformation.

    :param fs: Abtastrate in Hz
    :param duration: Dauer des Rauschens in Sekunden
    :return: Pinkes Rauschen als numpy-Array
    """
    num_samples = int(fs * duration)
    white = np.random.randn(num_samples)
    fft_white = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(num_samples, d=1./fs)
    scaling = np.ones_like(freqs)
    scaling[1:] = 1 / np.sqrt(freqs[1:])
    fft_pink = fft_white * scaling
    pink = np.fft.irfft(fft_pink, n=num_samples)
    pink = normalize_signal(pink)
    pink = dynamic_range_compression(pink)
    return pink

def generate_realistic_speech(fs, duration):
    """
    Generiert ein realistisches Sprachsignal basierend auf Formanten und transienten Komponenten.

    :param fs: Abtastrate in Hz
    :param duration: Dauer des Signals in Sekunden
    :return: Realistisches Sprachsignal als numpy-Array
    """
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)

    # Formantenfrequenzen (typische Werte für einen Vokal, z.B. [A])
    F1 = 800  # Hz
    F2 = 1150  # Hz
    F3 = 2900  # Hz

    # Amplituden der Formanten
    A1 = 1.0
    A2 = 0.8
    A3 = 0.5

    # Phasen
    phi1 = 0
    phi2 = np.pi / 4
    phi3 = np.pi / 2

    # Amplitudenhüllkurve (z.B. Hanning-Fenster für natürliche Lautstärkeveränderung)
    window = get_window('hann', len(t))

    # Formantenbestandteile
    s_formant = (
        A1 * np.sin(2 * np.pi * F1 * t + phi1) +
        A2 * np.sin(2 * np.pi * F2 * t + phi2) +
        A3 * np.sin(2 * np.pi * F3 * t + phi3)
    ) * window

    # Transiente Komponenten (z.B. kurze Rauschimpulse für Konsonanten)
    num_transients = int(duration * 5)  # Anzahl der transienten Ereignisse pro Sekunde
    transient_duration = 0.01  # Dauer jedes transienten Ereignisses in Sekunden
    transient_samples = int(transient_duration * fs)

    s_transient = np.zeros_like(t)
    for _ in range(num_transients):
        start_idx = np.random.randint(0, len(t) - transient_samples)
        transient = np.random.normal(0, 1, transient_samples) * np.hanning(transient_samples)
        s_transient[start_idx:start_idx + transient_samples] += transient

    # Rauschkomponente (Pink Noise)
    s_pink = generate_pink_noise(fs, duration) * 0.05  # Skalierung für realistische Lautstärke

    # Gesamtes Sprachsignal
    s = s_formant + s_transient + s_pink

    # Normalisierung und dynamische Bereichskompression
    s = normalize_signal(s)
    s = dynamic_range_compression(s)

    return s

def normalize_signal(signal):
    """
    Normalisiert das Signal auf den Bereich [-1, 1], ohne Division durch Null.

    :param signal: Eingabesignal als numpy-Array
    :return: Normalisiertes Signal als numpy-Array
    """
    max_val = np.max(np.abs(signal))
    if max_val == 0:
        return signal
    return signal / max_val

def dynamic_range_compression(signal, threshold=0.8, epsilon=1e-8):
    """
    Wendet eine logarithmische dynamische Bereichskompression an, um die Signalamplitude zu steuern.

    :param signal: Eingabesignal als numpy-Array
    :param threshold: Schwellenwert für die Kompression (zwischen 0 und 1)
    :param epsilon: Kleiner Wert zur Vermeidung von log(0)
    :return: Komprimiertes Signal als numpy-Array
    """
    normalized_signal = normalize_signal(signal)
    compressed_signal = np.sign(normalized_signal) * np.log1p(np.abs(normalized_signal) / threshold + epsilon)
    max_val = np.max(np.abs(compressed_signal))
    if max_val > 0:
        compressed_signal /= max_val
    return compressed_signal

def dynamic_range_compression_soft_clip(signal, threshold=0.8):
    """
    Wendet eine weiche Clipping-Kompression an, um die Signalamplitude zu steuern.

    :param signal: Eingabesignal als numpy-Array
    :param threshold: Schwellenwert für die Kompression (zwischen 0 und 1)
    :return: Komprimiertes Signal als numpy-Array
    """
    signal = normalize_signal(signal)
    compressed_signal = np.where(np.abs(signal) > threshold,
                                 np.sign(signal) * (threshold + (np.abs(signal) - threshold) * 0.5),
                                 signal)
    return compressed_signal

def fractional_delay(signal, delay, fs):
    """
    Verzögert das Signal um eine nicht-ganzzahlige Anzahl von Samples mittels Fourier-basierter Phase-Verschiebung.

    :param signal: Eingabesignal als numpy-Array
    :param delay: Verzögerung in Sekunden
    :param fs: Abtastrate in Hz
    :return: Verzögertes Signal
    """
    N = len(signal)
    padded_length = 2 * N
    SIGNAL = np.fft.fft(signal, n=padded_length)
    freqs = np.fft.fftfreq(padded_length, d=1./fs)
    phase_shift = np.exp(-1j * 2 * np.pi * freqs * delay)
    delayed_signal = np.fft.ifft(SIGNAL * phase_shift)
    delayed_signal = delayed_signal.real[:N]
    window = get_window('hann', N)
    delayed_signal *= window
    return delayed_signal

def resample_audio(data, original_fs, target_fs):
    """
    Resampelt das Audiosignal von original_fs auf target_fs unter Verwendung von resampy.

    :param data: Eingabesignal als numpy-Array
    :param original_fs: Originale Abtastrate in Hz
    :param target_fs: Ziel-Abtastrate in Hz
    :return: Resampeltes Signal als numpy-Array
    """
    resampled = resampy.resample(data, original_fs, target_fs, filter='kaiser_best')
    return resampled

def rauschunterdrueckung(signal, fs, method='butterworth'):
    """
    Wendet Rauschunterdrückung auf das Signal an.

    :param signal: Eingabesignal als numpy-Array
    :param fs: Abtastrate in Hz
    :param method: Methode zur Rauschunterdrückung ('butterworth', 'wiener')
    :return: Gefiltertes Signal als numpy-Array
    """
    if method == 'butterworth':
        nyquist = 0.5 * fs
        low = 300 / nyquist
        high = 3400 / nyquist
        b, a = butter(5, [low, high], btype='band')
        return filtfilt(b, a, signal)
    elif method == 'wiener':
        return wiener(signal)
    else:
        raise ValueError("Unbekannte Filtermethode. Verfügbare Methoden: 'butterworth', 'wiener'")
