######### utils.py #########

import numpy as np
import logging
from scipy.signal import find_peaks, correlate, correlation_lags
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import soundfile as sf
import os
from signal_processing import resample_audio, normalize_signal, dynamic_range_compression
from materials import material_properties
from typing import List, Tuple, Dict, Any, Optional
from scipy.interpolate import CubicSpline  # Für verbesserte Interpolation in der Synchronisation

def speed_of_sound(temperature: float, humidity: float, pressure: float = 101.325) -> float:
    """
    Calculate the speed of sound based on temperature, humidity, and pressure.
    (Hinweis: Bei Bedarf kann hier ein detaillierteres Modell eingesetzt werden.)
    """
    if temperature < -50 or temperature > 50:
        logging.warning("Ungewöhnliche Temperatur. Verwende Standardwert 20°C.")
        temperature = 20
    if humidity < 0 or humidity > 100:
        logging.warning("Ungewöhnliche Luftfeuchtigkeit. Verwende Standardwert 50%.")
        humidity = 50
    pressure_correction = 0.0006 * (pressure - 101.325)
    return 331 + 0.6 * temperature + 0.0124 * humidity + pressure_correction

def reflect_point_across_plane(point: List[float], plane: List[float]) -> np.ndarray:
    """
    Reflect a point across a plane defined by coefficients [a, b, c, d].
    """
    x_s, y_s, z_s = point
    a, b, c, d = plane
    denominator = a**2 + b**2 + c**2
    if denominator == 0:
        raise ValueError("Ungültige Ebene: a^2 + b^2 + c^2 ist 0.")
    factor = 2 * (a * x_s + b * y_s + c * z_s + d) / denominator
    x_prime = x_s - a * factor
    y_prime = y_s - b * factor
    z_prime = z_s - c * factor
    return np.array([x_prime, y_prime, z_prime])

def distance(point1: List[float], point2: List[float]) -> float:
    """
    Compute the Euclidean distance between two points.
    """
    return np.linalg.norm(np.array(point1) - np.array(point2))

def calculate_attenuation(distance_val: float, material: str, frequency: float, material_properties: Dict[str, Any]) -> float:
    """
    Calculate the attenuation of a signal based on distance and material properties.
    """
    d0 = 0.1  # Minimaler Abstand, um unrealistische Werte zu vermeiden
    distance_val = max(distance_val, d0)
    geometrical_attenuation = 1 / distance_val
    if material not in material_properties:
        logging.warning(f"Material '{material}' nicht definiert. Nutze 'air' als Standard.")
        material = 'air'
    absorption_coeff = material_properties[material]['absorption']
    frequency_factor = material_properties[material]['freq']
    frequency_attenuation = np.exp(-frequency_factor * frequency * distance_val)
    absorption = np.exp(-absorption_coeff * distance_val)
    attenuation = geometrical_attenuation * frequency_attenuation * absorption
    return attenuation

def generate_image_sources_iterative(source: List[float],
                                     planes: List[Dict[str, Any]],
                                     max_order: int,
                                     frequency: float,
                                     material_properties: Dict[str, Any],
                                     mic_positions: List[List[float]],
                                     absorption_threshold: float = 0.01,
                                     round_decimals: int = 6) -> List[Dict[str, Any]]:
    """
    Generate image sources iteratively considering reflections.
    (Hinweis: Die Dämpfungs-Kriterien basieren auf dem Mittelwert und Minimum der berechneten Attenuationen.)
    """
    image_sources = []
    current_sources = [source]
    seen_sources = set()
    source_tuple = tuple(np.round(source, decimals=round_decimals))
    seen_sources.add(source_tuple)

    for order in range(1, max_order + 1):
        new_sources = []
        for src in current_sources:
            for plane in planes:
                image = reflect_point_across_plane(src, plane['plane'])
                image_tuple = tuple(np.round(image, decimals=round_decimals))
                if image_tuple not in seen_sources:
                    material = plane.get('material', 'air')
                    if material not in material_properties:
                        raise ValueError(f"Material '{material}' ist nicht definiert. Bitte zum Dictionary hinzufügen.")
                    if 'absorption' not in material_properties[material] or 'freq' not in material_properties[material]:
                        raise ValueError(f"Absorptions- oder Frequenzeigenschaft für Material '{material}' fehlt.")
                    attenuations = [calculate_attenuation(distance(image, mic_pos), material, frequency, material_properties)
                                    for mic_pos in mic_positions]
                    if np.mean(attenuations) > absorption_threshold and np.min(attenuations) > (absorption_threshold / 2):
                        seen_sources.add(image_tuple)
                        image_sources.append({'source': image, 'material': material})
                        new_sources.append(image)
        current_sources = new_sources
        if not current_sources:
            break
    return image_sources

def phat_correlation(sig1: np.ndarray, sig2: np.ndarray) -> np.ndarray:
    """
    Compute the PHAT (Phase Transform) cross-correlation between two signals.
    """
    n1, n2 = len(sig1), len(sig2)
    n = n1 + n2 - 1  # Länge für lineare Kreuzkorrelation
    SIG1 = np.fft.fft(sig1, n=n)
    SIG2 = np.fft.fft(sig2, n=n)
    R = SIG1 * np.conj(SIG2)
    R /= np.abs(R) + 1e-10
    corr = np.fft.ifft(R).real
    return corr

def get_time_delays_phat(sig1: np.ndarray,
                         sig2: np.ndarray,
                         fs: float,
                         num_peaks: int = 1,
                         threshold_method: str = 'median',
                         threshold_multiplier: float = 1.0,
                         max_expected_delay: Optional[float] = None) -> Tuple[List[float], np.ndarray, np.ndarray]:
    """
    Estimate time delays between two signals using PHAT cross-correlation.

    :param sig1: Erstes Signal.
    :param sig2: Zweites Signal.
    :param fs: Abtastrate.
    :param num_peaks: Anzahl der zu extrahierenden Peaks.
    :param threshold_method: Methode zur Bestimmung des Schwellenwerts ('median' oder 'adaptive').
    :param threshold_multiplier: Multiplikator zur Anpassung des berechneten Schwellenwerts.
    :param max_expected_delay: Maximal erwartete Verzögerung in Sekunden.
    :return: Tuple (Liste der Zeitverzögerungen, Korrelationsarray, Lags in Sekunden).
    """
    corr = phat_correlation(sig1, sig2)
    lags = correlation_lags(len(sig1), len(sig2), mode='full')
    time_lags = lags / fs

    if threshold_method == 'median':
        threshold = threshold_multiplier * np.median(np.abs(corr))
    elif threshold_method == 'adaptive':
        threshold = threshold_multiplier * (np.mean(np.abs(corr)) + np.std(np.abs(corr)))
    else:
        threshold = threshold_multiplier * np.median(np.abs(corr))

    peak_distance = int(fs * 0.001)
    peaks, properties = find_peaks(corr, height=threshold, distance=peak_distance)
    if len(peaks) == 0:
        logging.warning(f"Keine Peaks mit Schwellenwertmethode '{threshold_method}' gefunden. Versuche alternativen Schwellenwert.")
        alternative_threshold = np.mean(np.abs(corr))
        peaks, properties = find_peaks(corr, height=alternative_threshold, distance=peak_distance)
        if len(peaks) == 0:
            logging.warning("Keine Peaks auch mit alternativen Schwellenwert gefunden. Nutze Maximum der Korrelation als Verzögerung.")
            max_peak_idx = np.argmax(corr)
            return [time_lags[max_peak_idx]], corr, time_lags

    if max_expected_delay is not None:
        valid_indices = [i for i in range(len(peaks)) if abs(time_lags[peaks[i]]) <= max_expected_delay]
        if not valid_indices:
            logging.warning("Keine Peaks innerhalb des erwarteten Verzögerungsbereichs gefunden. Versuche alternativen Schwellenwert.")
            alternative_threshold = np.mean(np.abs(corr))
            peaks, properties = find_peaks(corr, height=alternative_threshold, distance=peak_distance)
            valid_indices = [i for i in range(len(peaks)) if abs(time_lags[peaks[i]]) <= max_expected_delay]
            if not valid_indices:
                logging.warning("Keine gültigen Peaks nach alternativer Filterung. Nutze Maximum der Korrelation als Verzögerung.")
                max_peak_idx = np.argmax(corr)
                return [time_lags[max_peak_idx]], corr, time_lags
        peaks = peaks[valid_indices]
        properties['peak_heights'] = properties['peak_heights'][valid_indices]

    sorted_indices = np.argsort(properties['peak_heights'])[::-1]
    sorted_peaks = peaks[sorted_indices]
    selected_peaks = sorted_peaks[:num_peaks]
    time_delays = time_lags[selected_peaks]

    return list(time_delays), corr, time_lags

def bootstrap_significance(sig1: np.ndarray,
                           sig2: np.ndarray,
                           fs: float,
                           num_bootstrap: int = 1000,
                           alpha: float = 0.05,
                           bootstrap_mode: str = "permutation",
                           block_size: int = 50) -> float:
    """
    Bootstrap significance test for cross-correlation peaks.
    Unterstützte bootstrap_mode: 'permutation', 'block' und 'circular' (beibehält die zeitliche Struktur via zyklischer Verschiebung).
    """
    corr_original = phat_correlation(sig1, sig2)
    peak_original = np.max(corr_original)

    bootstrap_peaks = []
    for _ in range(num_bootstrap):
        if bootstrap_mode == "permutation":
            sig2_shuffled = np.random.permutation(sig2)
        elif bootstrap_mode == "block":
            num_blocks = int(np.ceil(len(sig2) / block_size))
            blocks = [sig2[i*block_size:(i+1)*block_size] for i in range(num_blocks)]
            np.random.shuffle(blocks)
            sig2_shuffled = np.concatenate(blocks)[:len(sig2)]
        elif bootstrap_mode == "circular":
            shift = np.random.randint(0, len(sig2))
            sig2_shuffled = np.roll(sig2, shift)
        else:
            raise ValueError("Unbekannter bootstrap_mode. Nutze 'permutation', 'block' oder 'circular'.")
        corr_bootstrap = phat_correlation(sig1, sig2_shuffled)
        peak_bootstrap = np.max(corr_bootstrap)
        bootstrap_peaks.append(peak_bootstrap)

    threshold = np.percentile(bootstrap_peaks, 100 * (1 - alpha))
    return threshold

def perform_significance_test_bootstrap(sig1: np.ndarray, sig2: np.ndarray, fs: float, alpha: float = 0.05) -> Tuple[float, bool]:
    """
    Perform significance test using the bootstrap method.
    """
    corr = phat_correlation(sig1, sig2)
    peak = np.max(corr)
    threshold = bootstrap_significance(sig1, sig2, fs, alpha=alpha)
    significant = peak > threshold
    return peak, significant

def compute_peak_to_peak_ratio(corr: np.ndarray) -> float:
    """
    Compute the peak-to-peak ratio of the cross-correlation.
    """
    peak = np.max(corr)
    trough = np.min(corr)
    if trough == 0:
        return np.inf
    return peak / abs(trough)

def compute_snr(corr: np.ndarray) -> float:
    """
    Compute the signal-to-noise ratio (SNR) from the cross-correlation.
    """
    peak = np.max(corr)
    peak_idx = np.argmax(corr)
    window_size = max(1, int(0.01 * len(corr)))
    start = max(0, peak_idx - window_size)
    end = min(len(corr), peak_idx + window_size)
    noise = np.std(np.concatenate((corr[:start], corr[end:])))
    if noise == 0:
        return np.inf
    return peak / noise

def perform_significance_test(corr: np.ndarray, sig1: np.ndarray, sig2: np.ndarray, fs: float, alpha: float = 0.05, snr_threshold: float = 2.0) -> Tuple[float, bool]:
    """
    Combine bootstrap and SNR criteria to perform a significance test.
    """
    snr = compute_snr(corr)
    peak, significant_peak = perform_significance_test_bootstrap(sig1, sig2, fs, alpha=alpha)
    significant = significant_peak and snr > snr_threshold
    return snr, significant

def compute_cross_correlation_metrics(corr: np.ndarray, sig1: np.ndarray, sig2: np.ndarray, fs: float, alpha: float = 0.05) -> Dict[str, Any]:
    """
    Compute various metrics from the cross-correlation.
    """
    ppt_ratio = compute_peak_to_peak_ratio(corr)
    snr, significant = perform_significance_test(corr, sig1, sig2, fs, alpha=alpha)
    return {
        'peak_to_peak_ratio': ppt_ratio,
        'snr': snr,
        'significant': significant
    }

def determine_optimal_number_of_clusters(data: List[List[float]], max_clusters: int = 5, method: str = 'kmeans', eps: float = 0.001, min_samples: int = 2) -> int:
    """
    Determine the optimal number of clusters using a clustering algorithm.
    """
    data_np = np.array(data)
    if len(data_np) < 2:
        return 1
    if method == 'kmeans':
        best_score = -1
        best_k = 1
        for k in range(2, min(max_clusters, len(data_np)) + 1):
            kmeans = KMeans(n_clusters=k, random_state=0).fit(data_np)
            score = silhouette_score(data_np, kmeans.labels_)
            if score > best_score:
                best_score = score
                best_k = k
        return best_k
    elif method == 'dbscan':
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(data_np)
        labels = clustering.labels_
        valid = labels != -1
        if np.sum(valid) < 2:
            return 1
        filtered_data = data_np[valid]
        filtered_labels = labels[valid]
        score = silhouette_score(filtered_data, filtered_labels)
        num_clusters = len(set(filtered_labels))
        return num_clusters if score > 0 else 1
    else:
        raise ValueError("Unbekannte Clustering-Methode. Verfügbare Methoden: 'kmeans', 'dbscan'")

def heuristic_initialization_adaptive(mic_positions: List[List[float]],
                                      mic_pairs: List[Tuple[int, int]],
                                      tdoas: List[float],
                                      c: float,
                                      clustering_method: str = 'kmeans',
                                      eps: float = 0.001,
                                      min_samples: int = 2) -> List[List[float]]:
    """
    Generate initial source position guesses based on TDOA estimates using clustering.
    Die Schätzung basiert auf der Hyperbel-Geometrie der TDOA-Differenzen.
    """
    mic_positions_np = np.array(mic_positions)
    if np.size(tdoas) == 0:
        return [np.mean(mic_positions_np, axis=0).tolist()]
    
    tdoas_np = np.array(tdoas)
    estimated_positions = []
    for (i, j), td in zip(mic_pairs, tdoas_np):
        mic1, mic2 = np.array(mic_positions[i]), np.array(mic_positions[j])
        direction = mic2 - mic1
        norm_dir = np.linalg.norm(direction)
        if norm_dir == 0:
            continue
        unit_direction = direction / norm_dir
        midpoint = (mic1 + mic2) / 2
        offset = (c * abs(td)) / 2
        if td > 0:
            estimated_position = midpoint - offset * unit_direction
        else:
            estimated_position = midpoint + offset * unit_direction
        estimated_positions.append(estimated_position.tolist())
    
    if not estimated_positions:
        return [np.mean(mic_positions_np, axis=0).tolist()]
    
    if clustering_method == 'kmeans':
        num_clusters = determine_optimal_number_of_clusters(estimated_positions, method=clustering_method, eps=eps, min_samples=min_samples)
        kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(estimated_positions)
        initial_guesses = kmeans.cluster_centers_.tolist()
    elif clustering_method == 'dbscan':
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(estimated_positions)
        labels = clustering.labels_
        unique_labels = set(labels)
        unique_labels.discard(-1)
        initial_guesses = []
        for label in unique_labels:
            cluster_points = [estimated_positions[i] for i in range(len(estimated_positions)) if labels[i] == label]
            if cluster_points:
                initial_guesses.append(np.mean(cluster_points, axis=0).tolist())
        if not initial_guesses:
            initial_guesses = [np.mean(mic_positions_np, axis=0).tolist()]
    else:
        initial_guesses = [np.mean(mic_positions_np, axis=0).tolist()]
    
    # Sicherstellen, dass der Durchschnitt der Mikrofonpositionen als zusätzliche Startposition verwendet wird
    mean_pos = np.mean(mic_positions_np, axis=0)
    if not any(np.allclose(mean_pos, guess, atol=1e-6) for guess in initial_guesses):
        initial_guesses.append(mean_pos.tolist())
    return initial_guesses

def dynamic_bounds_extended(mic_positions: List[List[float]],
                            tdoas: List[float],
                            c: float,
                            buffer: float = 5.0) -> List[Tuple[float, float]]:
    """
    Compute extended bounds for the optimization based on microphone positions and TDOA-derived distances.
    """
    mic_positions_np = np.array(mic_positions)
    if np.size(tdoas) > 0:
        additional_margin = np.percentile(c * np.abs(np.array(tdoas)), 75)
        minimum_margin = 1.0
        additional_margin = max(additional_margin, minimum_margin)
    else:
        additional_margin = 0.0
    margin = buffer + additional_margin
    min_coords = np.min(mic_positions_np, axis=0) - margin
    max_coords = np.max(mic_positions_np, axis=0) + margin
    dims = mic_positions_np.shape[1] if mic_positions_np.ndim > 1 else 1
    return [(min_coords[i], max_coords[i]) for i in range(dims)]

def equations(vars: List[float],
              mic_positions: List[List[float]],
              mic_pairs: List[Tuple[int, int]],
              tdoas: List[float],
              c: float,
              weights: Optional[np.ndarray] = None) -> List[float]:
    """
    System of equations representing the differences in signal arrival times.
    Konvention: (d_j - d_i) = c * (td)
    """
    if weights is not None and len(weights) != len(mic_pairs):
        raise ValueError("Länge der Gewichte muss der Anzahl der Mikrofonpaare entsprechen.")
    source = np.array(vars)
    residuals = []
    for idx, ((i, j), td) in enumerate(zip(mic_pairs, tdoas)):
        d_i = np.linalg.norm(source - np.array(mic_positions[i]))
        d_j = np.linalg.norm(source - np.array(mic_positions[j]))
        residual = (d_j - d_i) - c * td
        if weights is not None:
            residual *= weights[idx]
        residuals.append(residual)
    return residuals

def synchronize_signals_improved(signals: List[np.ndarray],
                                 fs: float,
                                 use_interpolation: bool = True) -> List[np.ndarray]:
    """
    Synchronize multiple signals by aligning them based on cross-correlation with a reference signal.
    Zur Verbesserung der subpixelgenauen Schätzung wird optional ein kubischer Interpolator über ein kleines Fenster verwendet.
    """
    from scipy.signal import correlate, get_window
    energies = [np.sum(sig**2) for sig in signals]
    ref_idx = np.argmax(energies)
    reference = signals[ref_idx]
    ref_corr = correlate(reference, reference, mode='full')
    ref_peak = np.max(np.abs(ref_corr))
    shifts = []
    max_shift_samples = int(fs * 0.05)  # 50 ms Schwelle
    for idx, sig in enumerate(signals):
        if idx == ref_idx:
            shifts.append(0)
            continue
        corr = correlate(sig, reference, mode='full')
        peak_index = np.argmax(np.abs(corr))
        if np.abs(corr[peak_index]) < 0.3 * ref_peak:
            logging.warning(f"Niedriger Korrelationspeak für Signal {idx} während Synchronisation. Setze Shift=0.")
            refined_peak = peak_index
        elif use_interpolation and peak_index > 1 and peak_index < len(corr) - 2:
            # Verwende kubische Interpolation über ein Fenster von 5 Punkten
            indices = np.arange(peak_index - 2, peak_index + 3)
            window_corr = corr[peak_index - 2: peak_index + 3]
            cs = CubicSpline(indices, window_corr)
            fine_indices = np.linspace(peak_index - 2, peak_index + 2, 100)
            fine_vals = cs(fine_indices)
            refined_peak = fine_indices[np.argmax(np.abs(fine_vals))]
        else:
            refined_peak = peak_index
        base_index = len(reference) - 1
        shift = refined_peak - base_index
        if abs(shift) > max_shift_samples:
            logging.warning(f"Berechneter Shift ({shift} Samples) für Signal {idx} überschreitet plausiblen Bereich. Setze Shift=0.")
            shift = 0
        shifts.append(shift)
    
    min_shift = min(shifts)
    adjusted_signals = []
    for sig, shift in zip(signals, shifts):
        pad_left = int(round(shift - min_shift))
        pad_left = max(0, pad_left)
        adjusted_sig = np.pad(sig, (pad_left, 0), mode='constant')
        adjusted_signals.append(adjusted_sig)
    max_length = max(len(s) for s in adjusted_signals)
    synchronized_signals = [np.pad(s, (0, max_length - len(s)), mode='constant') for s in adjusted_signals]
    return synchronized_signals

def read_audio_files(audio_files: List[str], expected_fs: float) -> List[np.ndarray]:
    """
    Read audio files, resample if necessary, and apply normalization and dynamic range compression.
    """
    signals = []
    for file in audio_files:
        if not os.path.isfile(file):
            logging.error(f"Audio file nicht gefunden: {file}")
            raise FileNotFoundError(f"Audio file nicht gefunden: {file}")
        try:
            signal, fs = sf.read(file)
            if signal.ndim > 1:
                signal = np.mean(signal, axis=1)
            if fs != expected_fs:
                logging.info(f"Resampling von '{file}' von {fs} Hz auf {expected_fs} Hz.")
                signal = resample_audio(signal, fs, expected_fs)
                fs = expected_fs
            signal = normalize_signal(signal)
            signal = dynamic_range_compression(signal)
            signals.append(signal)
        except Exception as e:
            logging.error(f"Fehler beim Lesen der Audio-Datei '{file}': {e}")
            raise RuntimeError(f"Fehler beim Lesen der Audio-Datei '{file}': {e}")
    return signals

def compute_weights(correlation_metrics: Dict[Tuple[int, int], Dict[str, float]],
                    mic_pairs: List[Tuple[int, int]]) -> np.ndarray:
    """
    Compute weights for microphone pairs based on correlation metrics.
    """
    weights = []
    for pair in mic_pairs:
        metrics = correlation_metrics.get(pair, None)
        weight = metrics.get('snr', 1.0) if metrics is not None else 1.0
        weights.append(weight)
    weights = np.array(weights)
    if np.mean(weights) != 0:
        weights = weights / np.mean(weights)
    return weights
