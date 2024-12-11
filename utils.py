# utils.py

import numpy as np
from scipy.signal import find_peaks, correlate, correlation_lags
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import soundfile as sf
import logging
import os
from signal_processing import resample_audio, normalize_signal, dynamic_range_compression
from materials import material_properties

def schallgeschwindigkeit(temp, feuchte, druck=101.325):
    if temp < -50 or temp > 50:
        logging.warning("Ungewöhnliche Temperatur. Verwende Standardwert 20°C.")
        temp = 20
    if feuchte < 0 or feuchte > 100:
        logging.warning("Ungewöhnliche Luftfeuchtigkeit. Verwende Standardwert 50%.")
        feuchte = 50
    return 331 + 0.6 * temp + 0.0124 * feuchte + 0.0006 * (druck / 1000)

def reflect_point_across_plane(point, plane):
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
    return np.linalg.norm(point1 - point2)

def calculate_attenuation(distance_val, material, frequency, material_properties):
    if distance_val == 0:
        distance_val = 1e-6
    geometrical_attenuation = 1 / distance_val
    if material not in material_properties:
        logging.warning(f"Material '{material}' ist nicht definiert. Verwende 'air' als Standard.")
        material = 'air'
    absorption_coeff = material_properties[material]['absorption']
    frequency_factor = material_properties[material]['freq']
    frequency_attenuation = np.exp(-frequency_factor * frequency * distance_val)
    absorption = np.exp(-absorption_coeff * distance_val)
    attenuation = geometrical_attenuation * frequency_attenuation * absorption
    return attenuation

def generate_image_sources_iterative(source, planes, max_order, frequency, material_properties, mic_positions, absorption_threshold=0.01):
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
                image_tuple = tuple(np.round(image, decimals=6))
                if image_tuple not in seen_sources:
                    material = plane.get('material', 'air')
                    if material not in material_properties:
                        raise ValueError(f"Material '{material}' ist nicht definiert. Bitte im Dictionary ergänzen.")
                    if 'absorption' not in material_properties[material] or 'freq' not in material_properties[material]:
                        raise ValueError(f"Absorptions- oder Frequenzeigenschaft für Material '{material}' fehlt.")
                    attenuations = [calculate_attenuation(distance(image, mic_pos), material, frequency, material_properties) for mic_pos in mic_positions]
                    if any(att > absorption_threshold for att in attenuations):
                        seen_sources.add(image_tuple)
                        image_sources.append({'source': image, 'material': material})
                        new_sources.append(image)
        current_sources = new_sources
        if not current_sources:
            break
    return image_sources

def phat_correlation(sig1, sig2):
    SIG1 = np.fft.fft(sig1)
    SIG2 = np.fft.fft(sig2)
    R = SIG1 * np.conj(SIG2)
    R /= np.abs(R) + 1e-10
    corr = np.fft.ifft(R).real
    return corr

def get_time_delays_phat(sig1, sig2, fs, num_peaks=1):
    corr = phat_correlation(sig1, sig2)
    lags = correlation_lags(len(sig1), len(sig2), mode='full')
    time_lags = lags / fs

    median_corr = np.median(np.abs(corr))
    peaks, properties = find_peaks(corr, height=median_corr, distance=int(fs * 0.001))
    if len(peaks) == 0:
        logging.warning("Keine Peaks in der Kreuzkorrelation gefunden. Verwende maximale Korrelation als Verzögerung.")
        max_peak_idx = np.argmax(corr)
        return [time_lags[max_peak_idx]], corr, time_lags

    sorted_peaks = peaks[np.argsort(properties['peak_heights'])[::-1]]
    selected_peaks = sorted_peaks[:num_peaks]
    time_delays = time_lags[selected_peaks]

    return list(time_delays), corr, time_lags

def bootstrap_significance(sig1, sig2, fs, num_bootstrap=1000, alpha=0.05):
    corr_original = phat_correlation(sig1, sig2)
    peak_original = np.max(corr_original)

    bootstrap_peaks = []
    for _ in range(num_bootstrap):
        sig2_permuted = np.random.permutation(sig2)
        corr_bootstrap = phat_correlation(sig1, sig2_permuted)
        peak_bootstrap = np.max(corr_bootstrap)
        bootstrap_peaks.append(peak_bootstrap)

    threshold = np.percentile(bootstrap_peaks, 100 * (1 - alpha))
    return threshold

def perform_significance_test_bootstrap(sig1, sig2, fs, alpha=0.05):
    corr = phat_correlation(sig1, sig2)
    peak = np.max(corr)
    threshold = bootstrap_significance(sig1, sig2, fs, alpha=alpha)
    significant = peak > threshold
    return peak, significant

def compute_peak_to_peak_ratio(corr):
    peak = np.max(corr)
    trough = np.min(corr)
    if trough == 0:
        return np.inf
    return peak / abs(trough)

def compute_snr(corr):
    peak = np.max(corr)
    peak_idx = np.argmax(corr)
    window_size = int(0.01 * len(corr))
    start = max(0, peak_idx - window_size)
    end = min(len(corr), peak_idx + window_size)
    noise = np.std(np.concatenate((corr[:start], corr[end:])))
    if noise == 0:
        return np.inf
    return peak / noise

def perform_significance_test(corr, sig1, sig2, fs, alpha=0.05, snr_threshold=2.0):
    snr = compute_snr(corr)
    peak, significant_peak = perform_significance_test_bootstrap(sig1, sig2, fs, alpha=alpha)
    significant = significant_peak and snr > snr_threshold
    return snr, significant

def compute_cross_correlation_metrics(corr, sig1, sig2, fs, alpha=0.05):
    ppt_ratio = compute_peak_to_peak_ratio(corr)
    snr, significant = perform_significance_test(corr, sig1, sig2, fs, alpha=alpha)
    return {
        'peak_to_peak_ratio': ppt_ratio,
        'snr': snr,
        'significant': significant
    }

def determine_optimal_number_of_clusters(data, max_clusters=5, method='kmeans', eps=0.001, min_samples=2):
    if len(data) < 2:
        return 1
    if method == 'kmeans':
        best_score = -1
        best_k = 1
        for k in range(2, min(max_clusters, len(data)) + 1):
            kmeans = KMeans(n_clusters=k, random_state=0).fit(data)
            score = silhouette_score(data, kmeans.labels_)
            if score > best_score:
                best_score = score
                best_k = k
        return best_k
    elif method == 'dbscan':
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(data)
        labels = clustering.labels_
        if len(set(labels)) <= 1:
            return 1
        score = silhouette_score(data, labels)
        return len(set(labels)) if score > 0 else 1
    else:
        raise ValueError("Unbekannte Clustering-Methode. Verfügbare Methoden: 'kmeans', 'dbscan'")

def heuristic_initialization_adaptive(mic_positions, mic_pairs, tdoas, c, clustering_method='kmeans', eps=0.001, min_samples=2):
    if not tdoas:
        return [np.mean(mic_positions, axis=0)]

    tdoas = np.array(tdoas)
    weights = np.where(tdoas != 0, 1 / np.abs(tdoas), 1.0)
    weights /= np.sum(weights)

    estimated_positions = []
    for (i, j), td in zip(mic_pairs, tdoas):
        if td == 0:
            continue
        mic1, mic2 = mic_positions[i], mic_positions[j]
        direction = mic2 - mic1
        norm_dir = np.linalg.norm(direction)
        direction_norm = direction / norm_dir if norm_dir != 0 else direction
        estimated_distance = (td * c) / 2
        estimated_position = mic1 + direction_norm * estimated_distance
        estimated_positions.append(estimated_position)

    if not estimated_positions:
        return [np.mean(mic_positions, axis=0)]

    num_clusters = determine_optimal_number_of_clusters(estimated_positions, method=clustering_method, eps=eps, min_samples=min_samples)
    if clustering_method == 'kmeans':
        kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(estimated_positions)
        initial_guesses = kmeans.cluster_centers_.tolist()
    elif clustering_method == 'dbscan':
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(estimated_positions)
        labels = clustering.labels_
        unique_labels = set(labels)
        unique_labels.discard(-1)  # Entfernt Rauschen
        initial_guesses = [estimated_positions[i] for i in range(len(estimated_positions)) if labels[i] in unique_labels]
        if not initial_guesses:
            initial_guesses = [np.mean(mic_positions, axis=0)]
    else:
        initial_guesses = [np.mean(mic_positions, axis=0)]

    initial_guesses.append(np.mean(mic_positions, axis=0))
    return initial_guesses

def dynamic_bounds_extended(mic_positions, tdoas, c, buffer=5.0):
    min_coords = np.min(mic_positions, axis=0) - buffer
    max_coords = np.max(mic_positions, axis=0) + buffer
    return [
        (min_coords[0], max_coords[0]),
        (min_coords[1], max_coords[1]),
        (min_coords[2], max_coords[2])
    ]

def equations(vars, mic_positions, mic_pairs, tdoas, c, weights=None):
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
    weights = []
    for pair in mic_pairs:
        metrics = correlation_metrics.get(pair, {})
        snr = metrics.get('snr', 1.0)
        significant = metrics.get('significant', False)
        weight = snr if significant else 0.5
        weights.append(weight)
    return weights

def synchronize_signals_improved(signals, fs):
    reference = signals[0]
    synchronized_signals = [reference]
    shifts = []

    for sig in signals[1:]:
        corr = correlate(sig, reference, mode='full')
        shift = np.argmax(corr) - (len(reference) - 1)
        shifts.append(shift)

    max_shift_positive = max([shift for shift in shifts if shift > 0], default=0)
    max_shift_negative = min([shift for shift in shifts if shift < 0], default=0)

    for sig, shift in zip(signals[1:], shifts):
        if shift > 0:
            aligned_sig = np.pad(sig, (shift, max_shift_positive - shift), 'constant')
        elif shift < 0:
            aligned_sig = np.pad(sig, (max_shift_negative - shift, -shift), 'constant')
        else:
            aligned_sig = np.pad(sig, (0, max_shift_positive + abs(max_shift_negative)), 'constant')
        synchronized_signals.append(aligned_sig)

    max_length = max(len(sig) for sig in synchronized_signals)
    synchronized_signals = [np.pad(sig, (0, max_length - len(sig)), 'constant') for sig in synchronized_signals]

    return synchronized_signals

def read_audio_files(audiodateien, expected_fs):
    signals = []
    for file in audiodateien:
        if not os.path.isfile(file):
            logging.error(f"Audiodatei nicht gefunden: {file}")
            raise FileNotFoundError(f"Audiodatei nicht gefunden: {file}")
        try:
            signal, fs = sf.read(file)
            if signal.ndim > 1:
                signal = np.mean(signal, axis=1)
            if fs != expected_fs:
                logging.info(f"Resampling der Datei '{file}' von {fs} Hz auf {expected_fs} Hz.")
                signal = resample_audio(signal, fs, expected_fs)
                fs = expected_fs
            signal = normalize_signal(signal)
            signal = dynamic_range_compression(signal)
            signals.append(signal)
        except Exception as e:
            logging.error(f"Fehler beim Einlesen der Audiodatei '{file}': {e}")
            raise RuntimeError(f"Fehler beim Einlesen der Audiodatei '{file}': {e}")
    return signals
