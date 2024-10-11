import numpy as np
from scipy.signal import butter, filtfilt, wiener, chirp, find_peaks, correlate, correlation_lags, get_window
from scipy.optimize import least_squares, differential_evolution
from scipy.interpolate import CubicSpline
from scipy.fft import fft, ifft
from scipy.stats import norm
import soundfile as sf
import matplotlib.pyplot as plt
import logging
from mpl_toolkits.mplot3d import Axes3D
import time
from math import gcd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from statsmodels.stats.multitest import fdrcorrection
import os
import resampy  # Für hochwertiges Resampling

# Konfigurieren des Loggings
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def schallgeschwindigkeit(temp, feuchte, druck=101.325):
    """
    Berechnet die Schallgeschwindigkeit in Luft unter Berücksichtigung von Temperatur, Luftfeuchtigkeit und Druck.
    Genauere Formel basierend auf der empirischen Formel:
    c = 331 + 0.6 * T + 0.0124 * RH + 0.0006 * (P / 1000)
    wobei P der Druck in kPa ist.

    :param temp: Temperatur in °C
    :param feuchte: Luftfeuchtigkeit in %
    :param druck: Atmosphärischer Druck in kPa (optional, Standard: 101.325 kPa)
    :return: Schallgeschwindigkeit in m/s
    """
    return 331 + 0.6 * temp + 0.0124 * feuchte + 0.0006 * (druck / 1000)


def reflect_point_across_plane(point, plane):
    """
    Berechnet den Bildpunkt einer gegebenen Punktposition über eine Ebene.

    :param point: Ursprüngliche Punktposition als numpy-Array [x, y, z]
    :param plane: Ebene definiert durch [a, b, c, d] für ax + by + cz + d = 0
    :return: Bildpunkt als numpy-Array [x', y', z']
    """
    x_s, y_s, z_s = point
    a, b, c, d = plane
    denominator = a**2 + b**2 + c**2
    if denominator == 0:
        raise ValueError("Ungültige Ebene mit a^2 + b^2 + c^2 = 0.")
    factor = 2 * (a * x_s + b * y_s + c * z_s + d) / denominator
    x_prime = x_s - a * factor
    y_prime = y_s - b * factor
    z_prime = z_s - c * factor
    return np.array([x_prime, y_prime, z_prime])


def distance(point1, point2):
    """
    Berechnet die euklidische Entfernung zwischen zwei Punkten im 3D-Raum.

    :param point1: Erster Punkt als numpy-Array [x, y, z]
    :param point2: Zweiter Punkt als numpy-Array [x, y, z]
    :return: Entfernung als float
    """
    return np.linalg.norm(point1 - point2)


def generate_image_sources_iterative(source, planes, max_order, frequency, material_properties, mic_positions, absorption_threshold=0.01):
    """
    Iterativ generierte Bildquellen bis zur maximalen Reflexionsordnung, unter Berücksichtigung von Absorptionsschwellen.

    :param source: Ursprüngliche Schallquelle als numpy-Array [x, y, z]
    :param planes: Liste von Ebenen, jede definiert durch {'plane': [a, b, c, d], 'material': 'material_name'}
    :param max_order: Maximale Reflexionsordnung
    :param frequency: Frequenz des Signals in Hz
    :param material_properties: Dictionary mit materialabhängigen Dämpfungsfaktoren
    :param mic_positions: Array der Mikrofonpositionen (nx3)
    :param absorption_threshold: Minimale Abschwächung, um eine Bildquelle zu berücksichtigen
    :return: Liste von Bildquellen als Dictionary {'source': [x', y', z'], 'material': 'material_name'}
    """
    image_sources = []
    current_sources = [source]
    seen_sources = set()
    source_tuple = tuple(np.round(source, decimals=6))
    seen_sources.add(source_tuple)

    for order in range(1, max_order + 1):
        new_sources = []
        for src in current_sources:
            for plane in planes:
                image = reflect_point_across_plane(src, plane['plane'])
                image_tuple = tuple(np.round(image, decimals=6))  # Runden zur Vermeidung von Float-Duplikaten
                if image_tuple not in seen_sources:
                    # Berechne die Abschwächung für die Bildquelle basierend auf Entfernung zu jedem Mikrofon
                    attenuations = [calculate_attenuation(distance(image, mic_pos), plane.get('material', 'air'), frequency, material_properties) for mic_pos in mic_positions]
                    # Überprüfen, ob mindestens eine Abschwächung über dem Schwellenwert liegt
                    if any(att > absorption_threshold for att in attenuations):
                        seen_sources.add(image_tuple)
                        image_sources.append({'source': image, 'material': plane.get('material', 'air')})
                        new_sources.append(image)
        current_sources = new_sources
        if not current_sources:
            break  # Keine neuen Bildquellen generiert
    return image_sources


def calculate_attenuation(distance_val, material, frequency, material_properties):
    """
    Berechnet die Abschwächung der Signalamplitude basierend auf dem inversen Quadrat der Entfernung
    und materialabhängigen Absorptionsfaktoren.

    :param distance_val: Entfernung zwischen Quelle und Mikrofon in Metern
    :param material: Material der reflektierenden Ebene (z.B. 'air', 'wood', 'metal')
    :param frequency: Frequenz des Signals in Hz
    :param material_properties: Dictionary mit materialabhängigen Dämpfungsfaktoren
    :return: Abschwächungsfaktor als float
    """
    # Geometrische Abschwächung (inverse Entfernung)
    if distance_val == 0:
        distance_val = 1e-6  # Sehr kleine Distanz zur Vermeidung von Division durch Null
    geometrical_attenuation = 1 / distance_val

    # Materialabhängige Dämpfung
    absorption_coeff = material_properties.get(f'{material}_absorption', 0.01)  # Angepasster Standardwert

    # Frequenzabhängige Dämpfung (exponentielle Beziehung)
    frequency_attenuation = np.exp(-material_properties.get(f'{material}_freq', 0.1) * frequency * distance_val)

    # Exponentielle Absorption entlang der Strecke
    absorption = np.exp(-absorption_coeff * distance_val)

    # Gesamtabschwächung
    attenuation = geometrical_attenuation * frequency_attenuation * absorption

    return attenuation


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
    # Normalisieren des Signals
    normalized_signal = normalize_signal(signal)

    # Logarithmische Kompression
    compressed_signal = np.sign(normalized_signal) * np.log1p(np.abs(normalized_signal) / threshold + epsilon)

    # Skalierung zurück zum Originalbereich
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


def generate_pink_noise(fs, duration):
    """
    Generiert pinkes Rauschen mittels Fourier-Transformation.

    :param fs: Abtastrate in Hz
    :param duration: Dauer des Rauschens in Sekunden
    :return: Pinkes Rauschen als numpy-Array
    """
    num_samples = int(fs * duration)
    # Generate white noise
    white = np.random.randn(num_samples)
    # Perform FFT
    fft_white = np.fft.rfft(white)
    # Frequenzen
    freqs = np.fft.rfftfreq(num_samples, d=1./fs)
    # Vermeiden von Division durch Null für DC-Komponente
    scaling = np.ones_like(freqs)
    scaling[1:] = 1 / np.sqrt(freqs[1:])
    scaling[0] = 0  # Entfernt die DC-Komponente
    fft_pink = fft_white * scaling
    # Perform inverse FFT
    pink = np.fft.irfft(fft_pink, n=num_samples)
    # Normalisierung und dynamische Bereichskompression
    pink = normalize_signal(pink)
    pink = dynamic_range_compression(pink)
    return pink


def generate_signal(signal_type, fs, duration, freq):
    """
    Generiert ein Signal basierend auf dem angegebenen Typ.

    :param signal_type: Typ des Signals ('sine', 'noise', 'chirp', 'speech')
    :param fs: Abtastrate in Hz
    :param duration: Dauer des Signals in Sekunden
    :param freq: Frequenz des Signals in Hz
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


def fractional_delay(signal, delay, fs):
    """
    Verzögert das Signal um eine nicht-ganzzahlige Anzahl von Samples mittels Fourier-basierter Phase-Verschiebung.

    :param signal: Eingabesignal als numpy-Array
    :param delay: Verzögerung in Sekunden
    :param fs: Abtastrate in Hz
    :return: Verzögertes Signal
    """
    N = len(signal)
    # Zero-Padding zur Minimierung von Artefakten
    padded_length = 2 * N
    SIGNAL = np.fft.fft(signal, n=padded_length)
    freqs = np.fft.fftfreq(padded_length, d=1./fs)
    # Phase-Verschiebung unter Berücksichtigung der Phasenbeziehungen
    phase_shift = np.exp(-1j * 2 * np.pi * freqs * delay)
    delayed_signal = np.fft.ifft(SIGNAL * phase_shift)
    # Rückgabe des realen Teils und Kürzen auf ursprüngliche Länge
    delayed_signal = delayed_signal.real[:N]
    # Anwendung einer Fensterfunktion zur Glättung der Übergänge
    window = get_window('hann', N)
    delayed_signal *= window
    return delayed_signal


def phat_correlation(sig1, sig2):
    """
    Führt die PHAT-Kreuzkorrelation durch.

    :param sig1: Erstes Signal als numpy-Array
    :param sig2: Zweites Signal als numpy-Array
    :return: PHAT-kreuzkorreliertes Signal als numpy-Array
    """
    SIG1 = np.fft.fft(sig1)
    SIG2 = np.fft.fft(sig2)
    R = SIG1 * np.conj(SIG2)
    R /= np.abs(R) + 1e-10  # PHAT-Normalisierung
    corr = np.fft.ifft(R).real
    return corr


def get_time_delays_phat(sig1, sig2, fs, num_peaks=1):
    """
    Berechnet die Zeitverzögerung zwischen zwei Signalen mittels PHAT-Kreuzkorrelation mit Mehrpeaks-Analyse.

    :param sig1: Erstes Signal als numpy-Array
    :param sig2: Zweites Signal als numpy-Array
    :param fs: Abtastrate in Hz
    :param num_peaks: Anzahl der direktesten Pfade, die identifiziert werden sollen
    :return: Liste der Zeitverzögerungen, Kreuzkorrelation, Lags
    """
    # PHAT-Kreuzkorrelation
    corr = phat_correlation(sig1, sig2)
    lags = correlation_lags(len(sig1), len(sig2), mode='full')
    time_lags = lags / fs

    # Finden der Peaks mit adaptivem Schwellenwert
    median_corr = np.median(np.abs(corr))
    peaks, properties = find_peaks(corr, height=median_corr, distance=int(fs * 0.001))
    if len(peaks) == 0:
        # Fallback: Verwenden der Kreuzkorrelation ohne Peaks oder Schätzung basierend auf vorherigen Daten
        logging.warning("Keine Peaks in der Kreuzkorrelation gefunden. Verwende maximale Korrelation als Verzögerung.")
        max_peak_idx = np.argmax(corr)
        return [time_lags[max_peak_idx]], corr, time_lags

    # Auswahl der höchsten Peaks
    sorted_peaks = peaks[np.argsort(properties['peak_heights'])[::-1]]
    selected_peaks = sorted_peaks[:num_peaks]
    time_delays = time_lags[selected_peaks]

    return list(time_delays), corr, time_lags


def bootstrap_significance(sig1, sig2, fs, num_bootstrap=1000, alpha=0.05):
    """
    Bestimmt die Signifikanz eines Peaks in der Kreuzkorrelation mittels Bootstrap.

    :param sig1: Erstes Signal als numpy-Array
    :param sig2: Zweites Signal als numpy-Array
    :param fs: Abtastrate in Hz
    :param num_bootstrap: Anzahl der Bootstrap-Wiederholungen
    :param alpha: Signifikanzniveau
    :return: Signifikanzschwelle für Peaks
    """
    corr_original = phat_correlation(sig1, sig2)
    peak_original = np.max(corr_original)

    bootstrap_peaks = []
    for _ in range(num_bootstrap):
        # Zufällige Permutation von sig2
        sig2_permuted = np.random.permutation(sig2)
        corr_bootstrap = phat_correlation(sig1, sig2_permuted)
        peak_bootstrap = np.max(corr_bootstrap)
        bootstrap_peaks.append(peak_bootstrap)
    
    # Bestimmen der (1 - alpha) Quantil als Signifikanzschwelle
    threshold = np.percentile(bootstrap_peaks, 100 * (1 - alpha))
    return threshold


def perform_significance_test_bootstrap(sig1, sig2, fs, alpha=0.05):
    """
    Führt einen Signifikanztest mittels Bootstrap durch.

    :param sig1: Erstes Signal als numpy-Array
    :param sig2: Zweites Signal als numpy-Array
    :param fs: Abtastrate in Hz
    :param alpha: Signifikanzniveau
    :return: Tuple (peak, significant)
    """
    corr = phat_correlation(sig1, sig2)
    peak = np.max(corr)
    threshold = bootstrap_significance(sig1, sig2, fs, alpha=alpha)
    significant = peak > threshold
    return peak, significant


def compute_peak_to_peak_ratio(corr):
    """
    Berechnet das Peak-to-Peak-Verhältnis der Kreuzkorrelation.

    :param corr: Kreuzkorrelationsarray
    :return: Peak-to-Peak-Verhältnis
    """
    peak = np.max(corr)
    trough = np.min(corr)
    if trough == 0:
        return np.inf
    return peak / abs(trough)


def compute_snr(corr):
    """
    Berechnet das Signal-Rausch-Verhältnis (SNR) der Kreuzkorrelation, wobei der Peak ausgeschlossen wird.

    :param corr: Kreuzkorrelationsarray
    :return: SNR-Wert
    """
    peak = np.max(corr)
    peak_idx = np.argmax(corr)
    # Definieren eines Bereichs um den Peak, der als Signal betrachtet wird (z.B. 1% des gesamten Signals)
    window_size = int(0.01 * len(corr))
    start = max(0, peak_idx - window_size)
    end = min(len(corr), peak_idx + window_size)
    # Rauschen berechnen, indem der Peakbereich ausgeschlossen wird
    noise = np.std(np.concatenate((corr[:start], corr[end:])))
    if noise == 0:
        return np.inf
    return peak / noise


def perform_significance_test(corr, sig1, sig2, fs, alpha=0.05, snr_threshold=2.0):
    """
    Führt einen angepassten Signifikanztest durch, um die Signifikanz der Kreuzkorrelation zu bestimmen.

    :param corr: Kreuzkorrelationsarray
    :param sig1: Erstes Signal als numpy-Array
    :param sig2: Zweites Signal als numpy-Array
    :param fs: Abtastrate in Hz
    :param alpha: Signifikanzniveau
    :param snr_threshold: Schwellenwert für SNR zur Bestimmung der Signifikanz
    :return: Tuple (SNR, significant)
    """
    # SNR-Berechnung
    snr = compute_snr(corr)

    # Signifikanztest mittels Bootstrap
    peak, significant_peak = perform_significance_test_bootstrap(sig1, sig2, fs, alpha=alpha)
    
    # Signifikanz basierend auf Bootstrap und dynamischem SNR-Schwellenwert
    significant = significant_peak and snr > snr_threshold

    return snr, significant


def compute_cross_correlation_metrics(corr, sig1, sig2, fs, alpha=0.05):
    """
    Berechnet verschiedene mathematische Kennzahlen der Kreuzkorrelation mit Bootstrap-Signifikanztest.

    :param corr: Kreuzkorrelationsarray
    :param sig1: Erstes Signal als numpy-Array
    :param sig2: Zweites Signal als numpy-Array
    :param fs: Abtastrate in Hz
    :param alpha: Signifikanzniveau für den Bootstrap-Test
    :return: Dictionary mit Peak-to-Peak-Verhältnis, SNR und Signifikanztest-Ergebnissen
    """
    ppt_ratio = compute_peak_to_peak_ratio(corr)
    snr, significant = perform_significance_test(corr, sig1, sig2, fs, alpha=alpha)
    return {
        'peak_to_peak_ratio': ppt_ratio,
        'snr': snr,
        'significant': significant
    }


def synchronize_signals_improved(signals, fs):
    """
    Synchronisiert die Signale basierend auf maximaler Kreuzkorrelation und füllt sie auf die längste Länge auf.
    Verbesserte Version zur Vermeidung von Datenverlust und unerwünschten Verzögerungen.

    :param signals: Liste von Signalen als numpy-Arrays
    :param fs: Abtastrate in Hz
    :return: Liste von synchronisierten Signalen
    """
    reference = signals[0]
    synchronized_signals = [reference]
    shifts = []

    # Bestimmen des maximalen Shift-Werts
    for sig in signals[1:]:
        corr = correlate(sig, reference, mode='full')
        shift = np.argmax(corr) - (len(reference) - 1)
        shifts.append(shift)

    # Bestimmen der maximalen positiven und negativen Verschiebung
    max_shift_positive = max([shift for shift in shifts if shift > 0], default=0)
    max_shift_negative = min([shift for shift in shifts if shift < 0], default=0)

    # Synchronisieren und Auffüllen der Signale individuell
    for sig, shift in zip(signals[1:], shifts):
        if shift > 0:
            aligned_sig = np.pad(sig, (shift, max_shift_positive - shift), 'constant')
        elif shift < 0:
            aligned_sig = np.pad(sig, (max_shift_negative - shift, -shift), 'constant')
        else:
            aligned_sig = np.pad(sig, (0, max_shift_positive + abs(max_shift_negative)), 'constant')
        synchronized_signals.append(aligned_sig)

    # Bestimmen der maximalen Länge nach Verschiebung
    max_length = max(len(sig) for sig in synchronized_signals)

    # Auffüllen aller Signale auf die maximale Länge
    synchronized_signals = [np.pad(sig, (0, max_length - len(sig)), 'constant') for sig in synchronized_signals]

    return synchronized_signals


def plot_correlation_heatmap(corr_matrix, mic_positions, title="Heatmap der Peak-Korrelationen zwischen Mikrofonpaaren"):
    """
    Erstellt eine Heatmap der Peak-Korrelationen zwischen Mikrofonpaaren.

    :param corr_matrix: 2D-Array der Peak-Korrelationen
    :param mic_positions: Array der Mikrofonpositionen
    :param title: Titel der Heatmap
    """
    num_mics = len(mic_positions)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr_matrix, cmap='viridis')

    # Beschriftungen
    ax.set_xticks(np.arange(num_mics))
    ax.set_yticks(np.arange(num_mics))
    ax.set_xticklabels([f'Mic {i+1}' for i in range(num_mics)])
    ax.set_yticklabels([f'Mic {i+1}' for i in range(num_mics)])
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Farbskala
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Peak-Korrelation', rotation=-90, va="bottom")

    # Titel
    ax.set_title(title)
    fig.tight_layout()
    plt.show()


def plot_correlation_3d(corr_data, mic_pairs, fs, title="3D-Kreuzkorrelationsplots"):
    """
    Erstellt einen 3D-Plot der Kreuzkorrelationen für alle Mikrofonpaare.

    :param corr_data: Liste von Kreuzkorrelationsarrays
    :param mic_pairs: Liste von Mikrofonpaaren als Tuple (i, j)
    :param fs: Abtastrate in Hz
    :param title: Titel des 3D-Plots
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    for idx, (corr, pair) in enumerate(zip(corr_data, mic_pairs)):
        lags = np.linspace(-(len(corr)-1)/fs, (len(corr)-1)/fs, len(corr))
        ax.plot(lags, [idx]*len(lags), corr, label=f'Mic {pair[0]+1} - Mic {pair[1]+1}')

    ax.set_xlabel('Lags (s)')
    ax.set_ylabel('Mikrofonpaare')
    ax.set_zlabel('Korrelation')
    ax.set_title(title)
    ax.legend()
    plt.show()


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


def einlesen_audiodaten(dateipfad, expected_fs):
    """
    Liest Audiodaten von einer Datei ein, überprüft die Abtastrate und resampelt bei Bedarf.

    :param dateipfad: Pfad zur Audiodatei
    :param expected_fs: Erwartete Abtastrate in Hz
    :return: Tuple (signal, fs) mit dem eingelesenen und ggf. resampelten Signal sowie der Abtastrate
    """
    try:
        signal, fs = sf.read(dateipfad)
        if signal.ndim > 1:
            # Durchschnitt aller Kanäle bilden, um ein monokanal Signal zu erhalten
            signal = np.mean(signal, axis=1)
        if fs != expected_fs:
            logging.info(f"Resampling der Datei '{dateipfad}' von {fs} Hz auf {expected_fs} Hz.")
            signal = resample_audio(signal, fs, expected_fs)
            fs = expected_fs
        # Normalisierung und dynamische Bereichskompression
        signal = normalize_signal(signal)
        signal = dynamic_range_compression(signal)
        return signal, fs
    except Exception as e:
        logging.error(f"Fehler beim Einlesen der Audiodatei '{dateipfad}': {e}")
        raise RuntimeError(f"Fehler beim Einlesen der Audiodatei '{dateipfad}': {e}")


def determine_optimal_number_of_clusters(data, max_clusters=5):
    """
    Bestimmt die optimale Anzahl von Clustern basierend auf dem Silhouette-Score.

    :param data: Liste von geschätzten Positionen
    :param max_clusters: Maximale Anzahl von Clustern, die getestet werden sollen
    :return: Optimale Anzahl von Clustern als int
    """
    if len(data) < 2:
        return 1
    best_score = -1
    best_k = 1
    for k in range(2, min(max_clusters, len(data)) + 1):
        kmeans = KMeans(n_clusters=k, random_state=0).fit(data)
        score = silhouette_score(data, kmeans.labels_)
        if score > best_score:
            best_score = score
            best_k = k
    return best_k


def heuristic_initialization_adaptive(mic_positions, mic_pairs, tdoas, c):
    """
    Heuristische Schätzung der Schallquellenposition basierend auf TDOAs unter Einbeziehung mehrerer Mikrofonpaare.

    :param mic_positions: Array mit den Positionen der Mikrofone (nx3)
    :param mic_pairs: Liste von Mikrofonpaaren als Tuple (i, j)
    :param tdoas: Array von TDOAs für die Mikrofonpaare
    :param c: Schallgeschwindigkeit in m/s
    :return: Liste von geschätzten Startpositionen als numpy-Arrays [x, y, z]
    """
    if not tdoas:
        # Fallback auf geometrischen Mittelpunkt
        return [np.mean(mic_positions, axis=0)]
    
    # Gewichtete Mittelung basierend auf inverser TDOA
    tdoas = np.array(tdoas)
    # Vermeidung von Division durch Null
    weights = np.where(tdoas != 0, 1 / np.abs(tdoas), 1.0)
    weights /= np.sum(weights)
    
    estimated_positions = []
    for (i, j), td in zip(mic_pairs, tdoas):
        if td == 0:
            continue
        mic1, mic2 = mic_positions[i], mic_positions[j]
        # Richtung von mic1 zu mic2
        direction = mic2 - mic1
        norm_dir = np.linalg.norm(direction)
        direction_norm = direction / norm_dir if norm_dir != 0 else direction
        # Geschätzte Position entlang der Richtung basierend auf der TDOA
        estimated_distance = (td * c) / 2
        estimated_position = mic1 + direction_norm * estimated_distance
        estimated_positions.append(estimated_position)
    
    if not estimated_positions:
        # Falls alle TDOAs Null sind
        return [np.mean(mic_positions, axis=0)]
    
    # Dynamisch die optimale Anzahl der Cluster bestimmen
    num_clusters = determine_optimal_number_of_clusters(estimated_positions)
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(estimated_positions)
    initial_guesses = kmeans.cluster_centers_.tolist()
    
    # Ergänzen Sie den geometrischen Mittelpunkt als zusätzlichen Startpunkt
    initial_guesses.append(np.mean(mic_positions, axis=0))
    
    return initial_guesses


def dynamic_bounds_extended(mic_positions, tdoas, c, buffer=5.0):
    """
    Berechnet die dynamischen Optimierungsgrenzen basierend auf den Mikrofonpositionen und den TDOAs.

    :param mic_positions: Array mit den Positionen der Mikrofone (nx3)
    :param tdoas: Array von TDOAs für die Mikrofonpaare
    :param c: Schallgeschwindigkeit in m/s
    :param buffer: Zusätzlicher Puffer in Metern für die Grenzen
    :return: Liste von Tupeln (min, max) für x, y, z
    """
    # Berechnung der minimalen und maximalen möglichen Positionen basierend auf Mikrofonanordnung und TDOAs
    # Dies ist eine vereinfachte Annahme; in komplexen Szenarien könnte eine geometrische Analyse erforderlich sein
    min_coords = np.min(mic_positions, axis=0) - buffer
    max_coords = np.max(mic_positions, axis=0) + buffer
    return [
        (min_coords[0], max_coords[0]),
        (min_coords[1], max_coords[1]),
        (min_coords[2], max_coords[2])
    ]


def equations(vars, mic_positions, mic_pairs, tdoas, c, weights=None):
    """
    Gleichungen für die Least Squares Optimierung mit Gewichtung.

    :param vars: Array [x, y, z] der geschätzten Schallquellenposition
    :param mic_positions: Array der Mikrofonpositionen (nx3)
    :param mic_pairs: Liste von Mikrofonpaaren als Tuple (i, j)
    :param tdoas: Array von TDOAs für die Mikrofonpaare
    :param c: Schallgeschwindigkeit in m/s
    :param weights: Array von Gewichten für die Residuen
    :return: Residuen als numpy-Array
    """
    x, y, z = vars
    source = np.array([x, y, z])
    residuals = []
    for idx, ((i, j), td) in enumerate(zip(mic_pairs, tdoas)):
        d_i = np.linalg.norm(source - mic_positions[i])
        d_j = np.linalg.norm(source - mic_positions[j])
        residual = (d_j - d_i) - c * td
        if weights is not None and idx < len(weights):
            residual *= weights[idx]
        residuals.append(residual)
    return residuals


def compute_weights(correlation_metrics, mic_pairs):
    """
    Berechnet Gewichte basierend auf der Signifikanz der Kreuzkorrelationen.
    Höhere Gewichte für signifikante TDOAs.

    :param correlation_metrics: Dictionary mit Kreuzkorrelationsmetriken
    :param mic_pairs: Liste von Mikrofonpaaren als Tuple (i, j)
    :return: Liste von Gewichten
    """
    weights = []
    for pair in mic_pairs:
        metrics = correlation_metrics.get(pair, {})
        snr = metrics.get('snr', 1.0)
        significant = metrics.get('significant', False)
        weight = snr if significant else 0.5  # Beispielgewichtung
        weights.append(weight)
    return weights


def lokalisieren_schallquelle(
    temperatur=20,
    luftfeuchtigkeit=50,
    fs=44100,
    mic_positions=np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ]),
    source_position=None,  # Optional, nur für Simulation benötigt
    duration=1.0,
    signal_type='sine',
    freq=1000,
    reflektierende_ebenen=None,  # Neue Parameter für reflektierende Ebenen
    material_properties=None,  # Angepasst für flexiblere Materialzuweisung
    filter_method='butterworth',
    use_simulation=True,
    audiodateien=None,
    max_distance=10,  # Maximal erwarteter Abstand der Schallquelle in Metern
    analyze_correlation=False,  # Neuer Parameter zur Aktivierung der erweiterten Analyse
    visualize_correlation=False,  # Neuer Parameter zur Aktivierung der erweiterten Visualisierung
    num_paths=1,  # Anzahl der Pfade, die identifiziert werden sollen
    clustering_eps=None,  # Epsilon für DBSCAN-Clustering in Sekunden, dynamisch angepasst
    clustering_min_samples=2,  # Mindestanzahl von Peaks pro Cluster für DBSCAN
    max_reflections=2,  # Maximale Anzahl der Reflexionen
    absorption_threshold=0.01,  # Minimale Abschwächung zur Berücksichtigung einer Bildquelle
    num_sources=1  # Anzahl der Schallquellen
):
    """
    Führt die Lokalisierung der Schallquelle durch, mit optionaler erweiterter Analyse und Visualisierung der Kreuzkorrelationen.

    :param temperatur: Temperatur in °C
    :param luftfeuchtigkeit: Luftfeuchtigkeit in %
    :param fs: Abtastrate in Hz
    :param mic_positions: Array der Mikrofonpositionen (nx3)
    :param source_position: Tatsächliche Position der Schallquelle (nur für Simulation)
    :param duration: Dauer des Signals in Sekunden
    :param signal_type: Typ des Signals ('sine', 'noise', 'chirp', 'speech')
    :param freq: Frequenz des Signals in Hz
    :param reflektierende_ebenen: Liste von reflektierenden Ebenen, jede definiert durch {'plane': [a, b, c, d], 'material': 'material_name'}
    :param material_properties: Dictionary mit Materialeigenschaften für Dämpfungen
    :param filter_method: Methode zur Rauschunterdrückung ('butterworth', 'wiener')
    :param use_simulation: Boolean, ob Simulation verwendet werden soll
    :param audiodateien: Liste der Pfade zu Audiodateien (nur wenn use_simulation=False)
    :param max_distance: Maximal erwarteter Abstand der Schallquelle in Metern
    :param analyze_correlation: Boolean, ob erweiterte Analyse der Kreuzkorrelation durchgeführt werden soll
    :param visualize_correlation: Boolean, ob erweiterte Visualisierung der Kreuzkorrelation durchgeführt werden soll
    :param num_paths: Anzahl der Pfade, die identifiziert werden sollen
    :param clustering_eps: Epsilon-Wert für DBSCAN-Clustering in Sekunden (dynamisch angepasst)
    :param clustering_min_samples: Mindestanzahl von Peaks pro Cluster für DBSCAN
    :param max_reflections: Maximale Anzahl der Reflexionen
    :param absorption_threshold: Minimale Abschwächung zur Berücksichtigung einer Bildquelle
    :param num_sources: Anzahl der Schallquellen
    :return: Dictionary mit Ergebnissen der Lokalisierung
    """
    try:
        # Eingabevalidierung
        if not isinstance(mic_positions, np.ndarray):
            raise TypeError("mic_positions muss ein numpy-Array sein.")
        if mic_positions.ndim != 2 or mic_positions.shape[1] != 3:
            raise ValueError("mic_positions muss ein nx3-Array sein.")
        if len(mic_positions) < 2:
            raise ValueError("Mindestens zwei Mikrofone sind für die Lokalisierung erforderlich.")

        # 1. Berechnung der Schallgeschwindigkeit
        c = schallgeschwindigkeit(temperatur, luftfeuchtigkeit)
        logging.info(f"Berechnete Schallgeschwindigkeit: {c:.2f} m/s")

        # 2. Signal-Generierung oder Einlesen
        if use_simulation:
            # Simulation-spezifische Logik
            if source_position is None:
                raise ValueError("source_position muss bei use_simulation=True angegeben werden.")
            if reflektierende_ebenen is None:
                reflektierende_ebenen = []
            if material_properties is None:
                # Standard-Materialeigenschaften setzen
                material_properties = {
                    'air': 1.0,    # Geringe Dämpfung in Luft
                    'air_absorption': 0.01,  # Beispielwert für Luftabsorption
                    'wood': 0.5,   # Moderate Dämpfung durch Holz
                    'metal': 0.2,  # Starke Dämpfung durch Metall
                    'wood_freq': 0.8,
                    'metal_freq': 0.6,
                    'wood_absorption': 0.05,   # Beispielwerte
                    'metal_absorption': 0.1
                }
            signals = simulate_signals_with_multipath(
                source_pos=source_position,
                mic_positions=mic_positions,
                fs=fs,
                c=c,
                duration=duration,
                signal_type=signal_type,
                freq=freq,
                reflektierende_ebenen=reflektierende_ebenen,
                material_properties=material_properties,
                max_reflections=max_reflections,
                max_distance=max_distance,
                absorption_threshold=absorption_threshold
            )
            logging.info(f"Simulierte Signale generiert mit Signaltyp '{signal_type}'.")
        else:
            # Verarbeitung realer Audiodaten
            if audiodateien is None:
                raise ValueError("Audiodateien müssen angegeben werden, wenn use_simulation=False ist.")
            if len(audiodateien) != len(mic_positions):
                raise ValueError("Anzahl der Audiodateien muss mit der Anzahl der Mikrofone übereinstimmen.")
            for file in audiodateien:
                if not os.path.isfile(file):
                    raise FileNotFoundError(f"Audiodatei nicht gefunden: {file}")
            signals = [einlesen_audiodaten(file, fs)[0] for file in audiodateien]
            logging.info("Echte Audiodaten eingelesen.")

        # 3. Synchronisierung der Signale
        signals = synchronize_signals_improved(signals, fs)
        logging.info("Signale synchronisiert.")

        # 4. Rauschunterdrückung
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

        gefilterte_signale = [rauschunterdrueckung(sig, fs, method=filter_method) for sig in signals]
        for i in range(len(gefilterte_signale)):
            logging.info(f"Signal {i+1} nach Rauschunterdrückung mit Methode '{filter_method}' gefiltert.")

        # 5. Laufzeitdifferenzen berechnen
        td_diffs = []
        mic_pairs = []
        cross_correlations = []
        cross_corr_lags = []
        correlation_metrics = {}
        corr_matrix = np.zeros((len(mic_positions), len(mic_positions)))
        corr_data_for_3d = []
        pairs_for_3d = []

        for i in range(len(gefilterte_signale)):
            for j in range(i + 1, len(gefilterte_signale)):
                time_delays, corr, lags = get_time_delays_phat(
                    gefilterte_signale[i],
                    gefilterte_signale[j],
                    fs,
                    num_peaks=num_paths
                )
                if not time_delays:
                    logging.warning(f"Keine Zeitverzögerungen für Mikrofon {i+1} - Mikrofon {j+1} gefunden.")
                    continue
                for td in time_delays:
                    td_diffs.append(td)
                    mic_pairs.append((i, j))
                    logging.info(f"Laufzeitdifferenz zwischen Mikrofon {i+1} und Mikrofon {j+1}: {td:.6f} Sekunden")

                # Mathematische Analyse der Kreuzkorrelationen
                if analyze_correlation:
                    metrics = compute_cross_correlation_metrics(corr, gefilterte_signale[i], gefilterte_signale[j], fs, alpha=0.05)
                    correlation_metrics[(i, j)] = metrics
                    logging.info(f"Kreuzkorrelationsmetriken zwischen Mikrofon {i+1} und Mikrofon {j+1}: {metrics}")

                # Sammeln der Peak-Korrelationen für Heatmap
                peak_correlation = np.max(corr)
                corr_matrix[i, j] = peak_correlation
                corr_matrix[j, i] = peak_correlation  # Symmetrisch

                # Sammeln der gesamten Kreuzkorrelationsdaten für 3D-Plot
                if visualize_correlation:
                    corr_data_for_3d.append(corr)
                    pairs_for_3d.append((i, j))

        if not mic_pairs:
            raise RuntimeError("Keine gültigen Mikrofonpaare mit gefundenen Zeitverzögerungen.")

        # 6. Entfernungsdifferenzen berechnen
        dd_diffs = [c * td for td in td_diffs]
        for i, dd in enumerate(dd_diffs, start=1):
            pair = mic_pairs[i-1]
            logging.info(f"Entfernungsdifferenz Mikrofonpaar {pair[0]+1}-{pair[1]+1}: {dd:.3f} Meter")

        # 7. Positionsschätzung mittels Least Squares mit adaptiver Initialisierung und dynamischen Grenzen
        # 7.1 Heuristische Initialisierung
        initial_guesses = heuristic_initialization_adaptive(mic_positions, mic_pairs, td_diffs, c)
        logging.info(f"Heuristische Startpositionen: {initial_guesses}")

        # 7.2 Dynamische Optimierungsgrenzen
        bounds = dynamic_bounds_extended(mic_positions, td_diffs, c, buffer=5.0)
        lower_bounds = [b[0] for b in bounds]
        upper_bounds = [b[1] for b in bounds]
        bounds_tuple = (lower_bounds, upper_bounds)
        logging.info(f"Dynamische Optimierungsgrenzen gesetzt: {bounds_tuple}")

        # 7.3 Berechnung der Gewichte für die Optimierung
        if analyze_correlation and correlation_metrics:
            weights = compute_weights(correlation_metrics, mic_pairs)
        else:
            weights = np.ones(len(mic_pairs))  # Standardgewichtung

        # 7.4 Least Squares Optimierung mit gewichteten Residuen
        best_result = None
        best_cost = np.inf
        for guess in initial_guesses:
            result = least_squares(
                equations,
                guess,
                args=(mic_positions, mic_pairs, td_diffs, c, weights),
                bounds=(lower_bounds, upper_bounds),
                method='trf',
                ftol=1e-6,
                xtol=1e-6,
                gtol=1e-6
            )
            if result.success and result.cost < best_cost:
                best_cost = result.cost
                best_result = result

        if best_result is not None:
            x_source, y_source, z_source = best_result.x
            logging.info(f"Die geschätzte Position der Schallquelle ist ({x_source:.3f}, {y_source:.3f}, {z_source:.3f}) Meter")
        else:
            logging.warning("Optimierung mit least_squares fehlgeschlagen. Versuch mit Differential Evolution.")
            # Versuch mit Differential Evolution
            result_de = differential_evolution(
                lambda vars: np.sum(np.square(equations(vars, mic_positions, mic_pairs, td_diffs, c, weights))),
                bounds=bounds,
                strategy='best1bin',
                maxiter=1000,
                popsize=15,
                tol=1e-6,
                mutation=(0.5, 1),
                recombination=0.7,
                seed=None,
                callback=None,
                disp=False,
                polish=True,
                init='latinhypercube'
            )

            if result_de.success:
                x_source, y_source, z_source = result_de.x
                logging.info(f"Die geschätzte Position der Schallquelle ist ({x_source:.3f}, {y_source:.3f}, {z_source:.3f}) Meter (Differential Evolution).")
            else:
                logging.error("Differential Evolution Optimierung ebenfalls fehlgeschlagen: " + result_de.message)
                # Letzter Versuch: Verwenden der heuristischen Initialisierung
                initial_guess = initial_guesses[0]
                x_source, y_source, z_source = initial_guess
                logging.info(f"Verwende heuristische Schätzung als geschätzte Position: ({x_source:.3f}, {y_source:.3f}, {z_source:.3f}) Meter")

        # 8. Visualisierung der Ergebnisse
        if use_simulation:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(mic_positions[:, 0], mic_positions[:, 1], mic_positions[:, 2], c='r', marker='o', label='Mikrofone')
            ax.scatter(source_position[0], source_position[1], source_position[2], c='g', marker='*', s=100, label='Tatsächliche Quelle')
            if not np.isnan(x_source):
                ax.scatter(x_source, y_source, z_source, c='b', marker='x', s=100, label='Geschätzte Quelle')
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
            ax.legend()
            plt.title('Schallquellenlokalisierung')
            plt.show()

        # Erweiterte Visualisierung der Kreuzkorrelationen (einmalig)
        if visualize_correlation:
            # Heatmap der Peak-Korrelationen
            plot_correlation_heatmap(corr_matrix, mic_positions)

            # 3D-Korrelationsplots
            plot_correlation_3d(corr_data_for_3d, pairs_for_3d, fs)

        # Optional: Erweiterte mathematische Kennzahlen ausgeben
        if analyze_correlation:
            logging.info("\nErweiterte mathematische Kennzahlen der Kreuzkorrelationen:")
            for pair, metrics in correlation_metrics.items():
                mic1, mic2 = pair
                logging.info(f"Mikrofon {mic1+1} - Mikrofon {mic2+1}:")
                logging.info(f"  Peak-to-Peak-Verhältnis: {metrics['peak_to_peak_ratio']:.2f}")
                logging.info(f"  SNR: {metrics['snr']:.2f}")
                logging.info(f"  Signifikant: {'Ja' if metrics['significant'] else 'Nein'}")

        # Rückgabe der Ergebnisse
        return {
            'geschätzte_position': np.array([x_source, y_source, z_source]),
            'tatsächliche_position': source_position if use_simulation else None,
            'mikrofon_positionen': mic_positions,
            'korrelation_metrics': correlation_metrics if analyze_correlation else None,
            'korrelationsmatrix': corr_matrix if visualize_correlation else None
        }

def simulate_signals_with_multipath(
    source_pos,
    mic_positions,
    fs,
    c,
    duration=1.0,
    signal_type='sine',
    freq=1000,
    reflektierende_ebenen=None,
    material_properties=None,
    max_reflections=2
):
    """
    Simuliert Signale für alle Mikrofone unter Berücksichtigung von Mehrwegeausbreitung.

    :param source_pos: Position der Schallquelle (3D) als numpy-Array [x, y, z]
    :param mic_positions: Array der Mikrofonpositionen (nx3)
    :param fs: Abtastrate in Hz
    :param c: Schallgeschwindigkeit in m/s
    :param duration: Dauer des Signals in Sekunden
    :param signal_type: Typ des Signals ('sine', 'noise', 'chirp', 'speech')
    :param freq: Frequenz des Signals in Hz
    :param reflektierende_ebenen: Liste von Ebenen, jede definiert durch {'plane': [a, b, c, d], 'material': 'material_name'}
    :param material_properties: Dictionary mit Materialeigenschaften für Dämpfungen
    :param max_reflections: Maximale Anzahl von Reflexionen
    :return: Liste von simulierten Signalen für jedes Mikrofon
    """
    base_signal = generate_signal(signal_type, fs, duration, freq)
    all_image_sources = generate_image_sources_iterative(source_pos, reflektierende_ebenen, max_reflections, freq, material_properties)
    signals = []
    
    # Bestimmen der maximalen Verzögerung
    max_delay = 0
    for mic_pos in mic_positions:
        direct_distance = distance(source_pos, mic_pos)
        reflection_distances = [distance(img['source'], mic_pos) for img in all_image_sources]
        max_distance = direct_distance + (max(reflection_distances) if reflection_distances else 0)
        max_delay = max(max_delay, max_distance / c)
    
    total_samples = int((duration + max_delay) * fs)
    base_signal_padded = np.pad(base_signal, (0, total_samples - len(base_signal)), 'constant')
    
    for mic_pos in mic_positions:
        signal_total = np.zeros(total_samples)
        # Direkter Pfad
        distance_direct = np.linalg.norm(source_pos - mic_pos)
        time_delay_direct = distance_direct / c
        attenuation_direct = calculate_attenuation(distance_direct, 'air', freq, material_properties)
        delayed_direct = fractional_delay(base_signal_padded, time_delay_direct, fs)
        signal_total += delayed_direct * attenuation_direct
        logging.debug(f"Direkter Pfad für Mikrofon {mic_pos}: Delta d = {distance_direct:.3f} m, Delta t = {time_delay_direct:.6f} s, Attenuation = {attenuation_direct:.6f}")

        # Reflexionen
        for img in all_image_sources:
            image_source = img['source']
            material = img['material']
            distance_val = distance(image_source, mic_pos)
            time_delay = distance_val / c
            attenuation = calculate_attenuation(distance_val, material, freq, material_properties)
            delayed_signal = fractional_delay(base_signal_padded, time_delay, fs)
            signal_total += delayed_signal * attenuation
            delta_d = distance_val - distance_direct
            logging.debug(f"Reflexion über Material '{material}' für Mikrofon {mic_pos}: Delta d = {delta_d:.3f} m, Delta t = {time_delay:.6f} s, Attenuation = {attenuation:.6f}")

        # Trimmen auf ursprüngliche Länge
        signal_total = signal_total[:int(duration * fs)]

        # Hinzufügen zum Signalset
        signals.append(signal_total)

    # Normalisierung und dynamische Bereichskompression nach dem gesamten Mixing
    # Dies wird bereits im Signalfluss nach der Simulation gemacht, daher ist hier keine weitere Normalisierung notwendig
    return signals

# Beispiel zur Nutzung
if __name__ == "__main__":
    # Beispielparameter
    temperatur = 20  # in °C
    luftfeuchtigkeit = 50  # in %
    fs = 44100  # Sampling-Rate in Hz
    mic_positions = np.array([
        [0.0, 0.0, 0.0],    # Mikrofon 1
        [1.0, 0.0, 0.0],    # Mikrofon 2 entlang der x-Achse
        [0.0, 1.0, 0.0],    # Mikrofon 3 entlang der y-Achse
        [0.0, 0.0, 1.0]     # Mikrofon 4 entlang der z-Achse
    ])
    source_position = np.array([0.5, 0.5, 0.5])  # Tatsächliche Position der Schallquelle
    duration = 1.0  # Dauer des Signals in Sekunden
    freq = 1000  # Frequenz des Sinussignals in Hz
    max_distance = 10  # Maximal erwarteter Abstand der Schallquelle in Metern

    # Definieren der reflektierenden Ebenen mit Materialeigenschaften
    reflektierende_ebenen = [
        {'plane': np.array([1, 0, 0, -5]), 'material': 'wood'},    # Ebene: x - 5 = 0, Material: Holz
        {'plane': np.array([0, 1, 0, -5]), 'material': 'metal'},   # Ebene: y - 5 = 0, Material: Metall
        {'plane': np.array([0, 0, 1, -5]), 'material': 'wood'}     # Ebene: z - 5 = 0, Material: Holz
    ]

    # Materialeigenschaften festlegen, falls nicht bereits festgelegt
    material_properties = {
        'air': 1.0,    # Geringe Dämpfung in Luft
        'air_absorption': 0.01,  # Beispielwert für Luftabsorption
        'wood': 0.5,   # Moderate Dämpfung durch Holz
        'metal': 0.2,  # Starke Dämpfung durch Metall
        'wood_freq': 0.8,
        'metal_freq': 0.6,
        'wood_absorption': 0.05,   # Beispielwerte
        'metal_absorption': 0.1
    }

    # Lokalisierung mit Simulation und Butterworth-Rauschunterdrückung, erweiterte Analyse und Visualisierung aktiviert
    logging.info("\nLokalisierung mit Butterworth-Filter, erweiterten Analysen und Visualisierungen:")
    lokalisieren_schallquelle(
        temperatur=temperatur,
        luftfeuchtigkeit=luftfeuchtigkeit,
        fs=fs,
        mic_positions=mic_positions,
        source_position=source_position,
        duration=duration,
        signal_type='sine',  # 'sine', 'noise', 'chirp', 'speech'
        freq=freq,
        reflektierende_ebenen=reflektierende_ebenen,  # Angepasst für Materialeigenschaften
        material_properties=material_properties,
        filter_method='butterworth',
        use_simulation=True,
        max_distance=max_distance,
        analyze_correlation=True,
        visualize_correlation=True,
        num_paths=1,  # Kann auf höhere Werte gesetzt werden, um mehrere Pfade zu identifizieren
        clustering_eps=None,  # Dynamisch angepasst
        clustering_min_samples=2,
        max_reflections=3,  # Erhöhte Anzahl der Reflexionen
        absorption_threshold=0.01
    )

    # Weitere Tests mit verschiedenen Signaltypen
    signal_types = ['sine', 'noise', 'chirp', 'speech']
    for sig_type in signal_types:
        logging.info(f"\nLokalisierung mit Butterworth-Filter, Signaltyp '{sig_type}', erweiterten Analysen und Visualisierungen:")
        lokalisieren_schallquelle(
            temperatur=temperatur,
            luftfeuchtigkeit=luftfeuchtigkeit,
            fs=fs,
            mic_positions=mic_positions,
            source_position=source_position,
            duration=duration,
            signal_type=sig_type,
            freq=freq,
            reflektierende_ebenen=reflektierende_ebenen,  # Angepasst für Materialeigenschaften
            material_properties=material_properties,
            filter_method='butterworth',
            use_simulation=True,
            max_distance=max_distance,
            analyze_correlation=True,
            visualize_correlation=True,
            num_paths=1,  # Kann auf höhere Werte gesetzt werden
            clustering_eps=0.001,
            clustering_min_samples=2,
            max_reflections=3,
            absorption_threshold=0.01
        )

    # Lokalisierung mit realen Audiodaten und Butterworth-Filter, erweiterte Analyse und Visualisierung aktiviert
    audiodateien = ['mic1.wav', 'mic2.wav', 'mic3.wav', 'mic4.wav']
    logging.info("\nLokalisierung mit realen Audiodaten, Butterworth-Filter, erweiterten Analysen und Visualisierungen:")
    lokalisieren_schallquelle(
        temperatur=temperatur,
        luftfeuchtigkeit=luftfeuchtigkeit,
        fs=fs,
        mic_positions=mic_positions,
        use_simulation=False,
        audiodateien=audiodateien,
        filter_method='butterworth',
        max_distance=max_distance,
        analyze_correlation=True,
        visualize_correlation=True,
        num_paths=1,  # Kann auf höhere Werte gesetzt werden
        clustering_eps=0.001,
        clustering_min_samples=2,
        max_reflections=3,
        absorption_threshold=0.01
    )

    # Lokalisierung mit realen Audiodaten und Wiener-Filter, erweiterte Analyse und Visualisierung aktiviert
    logging.info("\nLokalisierung mit realen Audiodaten, Wiener-Filter, erweiterten Analysen und Visualisierungen:")
    lokalisieren_schallquelle(
        temperatur=temperatur,
        luftfeuchtigkeit=luftfeuchtigkeit,
        fs=fs,
        mic_positions=mic_positions,
        use_simulation=False,
        audiodateien=audiodateien,
        filter_method='wiener',
        max_distance=max_distance,
        analyze_correlation=True,
        visualize_correlation=True,
        num_paths=1,  # Kann auf höhere Werte gesetzt werden
        clustering_eps=0.001,
        clustering_min_samples=2,
        max_reflections=3,
        absorption_threshold=0.01
    )
