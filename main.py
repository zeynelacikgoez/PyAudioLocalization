######### main.py #########

import numpy as np
import logging
from scipy.optimize import least_squares, differential_evolution
import matplotlib.pyplot as plt

from calibration import run_calibration
from plotting import plot_correlation_heatmap, plot_correlation_3d
from signal_processing import (
    generate_signal, generate_realistic_speech, fractional_delay, normalize_signal,
    dynamic_range_compression, dynamic_range_compression_soft_clip, resample_audio,
    noise_reduction
)
from utils import (
    speed_of_sound, reflect_point_across_plane, distance,
    calculate_attenuation, generate_image_sources_iterative, phat_correlation,
    get_time_delays_phat, compute_cross_correlation_metrics, perform_significance_test,
    perform_significance_test_bootstrap, compute_peak_to_peak_ratio, compute_snr,
    determine_optimal_number_of_clusters,
    heuristic_initialization_adaptive, dynamic_bounds_extended, equations,
    compute_weights, synchronize_signals_improved, read_audio_files
)
from materials import material_properties  # Zentrale Quelle der Materialeigenschaften

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
    # Materialeigenschaften werden nun zentral in materials.py verwaltet.
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
        "max_expected_delay": 0.05  # maximal erwartete Verzögerung in Sekunden (z.B. 50 ms)
    }
}

def simulate_signals_with_multipath(
    source_pos,
    mic_positions,
    fs,
    c,
    duration=1.0,
    signal_type='sine',
    freq=1000,
    reflective_planes=None,
    material_properties=None,
    max_reflections=2,
    absorption_threshold=0.01,
    trim_to_duration=True
):
    from signal_processing import fractional_delay, normalize_signal, dynamic_range_compression, generate_signal
    from utils import distance, calculate_attenuation
    base_signal = generate_signal(signal_type, fs, duration, freq)
    all_image_sources = generate_image_sources_iterative(
        source=source_pos,
        planes=reflective_planes,
        max_order=max_reflections,
        frequency=freq,
        material_properties=material_properties,
        mic_positions=mic_positions,
        absorption_threshold=absorption_threshold
    )
    signals = []
    max_delay = 0
    for mic_pos in mic_positions:
        direct_distance = distance(source_pos, mic_pos)
        reflection_distances = [distance(img['source'], mic_pos) for img in all_image_sources]
        if reflection_distances:
            max_distance = max(direct_distance, max(reflection_distances))
        else:
            max_distance = direct_distance
        max_delay = max(max_delay, max_distance / c)
    total_samples = int((duration + max_delay) * fs)
    base_signal_padded = np.pad(base_signal, (0, total_samples - len(base_signal)), 'constant')
    for mic_pos in mic_positions:
        signal_total = np.zeros(total_samples)
        distance_direct = distance(source_pos, mic_pos)
        time_delay_direct = distance_direct / c
        attenuation_direct = calculate_attenuation(distance_direct, 'air', freq, material_properties)
        delayed_direct = fractional_delay(base_signal_padded, time_delay_direct, fs)
        signal_total += delayed_direct * attenuation_direct
        for img in all_image_sources:
            image_source = img['source']
            material = img['material']
            distance_val = distance(image_source, mic_pos)
            time_delay = distance_val / c
            attenuation = calculate_attenuation(distance_val, material, freq, material_properties)
            delayed_signal = fractional_delay(base_signal_padded, time_delay, fs)
            signal_total += delayed_signal * attenuation
        if trim_to_duration:
            signal_total = signal_total[:int(duration * fs)]
        signal_total = normalize_signal(signal_total)
        signal_total = dynamic_range_compression(signal_total)
        signals.append(signal_total)
    return signals

def localize_sound_source(config, calibration_data=None, audio_files=None, use_simulation=True, show_plots=True):
    fs = config["fs"]
    duration = config["duration"]
    mic_positions = np.array(config["mic_positions"])
    source_position = config["source_position"]
    signal_type = config["signal_type"]
    freq = config["freq"]
    reflective_planes = config.get("reflective_planes", [])
    # Materialeigenschaften werden nun zentral aus materials.py bezogen.
    material_props = material_properties
    localization_params = config.get("localization", {})
    filter_method = localization_params.get("filter_method", "butterworth")
    max_reflections = localization_params.get("max_reflections", 2)
    absorption_threshold = localization_params.get("absorption_threshold", 0.01)
    analyze_correlation = localization_params.get("analyze_correlation", False)
    visualize_correlation = localization_params.get("visualize_correlation", False)
    clustering_method = localization_params.get("clustering_method", "kmeans")
    clustering_eps = localization_params.get("clustering_eps", 0.001)
    clustering_min_samples = localization_params.get("clustering_min_samples", 2)
    max_expected_delay = localization_params.get("max_expected_delay", None)

    calib_delays = None
    if calibration_data is not None:
        if len(calibration_data) != len(mic_positions):
            logging.warning("Anzahl der Kalibrierdaten stimmt nicht mit der Anzahl der Mikrofone überein. Ignoriere Kalibrierung für diesen Durchlauf.")
        else:
            try:
                calib_delays = np.array([d.get('delay', 0.0) for d in calibration_data], dtype=float)
                logging.info("Kalibrierungskorrektur wird angewendet.")
            except Exception as e:
                logging.warning(f"Fehler beim Verarbeiten der Kalibrierdaten: {e}. Ignoriere Kalibrierung.")
                calib_delays = None
    
    c = speed_of_sound(config["celsius"], config["humidity"])
    logging.info(f"Berechnete Schallgeschwindigkeit: {c:.2f} m/s")
    
    if use_simulation:
        if source_position is None:
            raise ValueError("source_position muss angegeben werden, wenn use_simulation=True.")
        signals = simulate_signals_with_multipath(
            source_pos=source_position,
            mic_positions=mic_positions,
            fs=fs,
            c=c,
            duration=duration,
            signal_type=signal_type,
            freq=freq,
            reflective_planes=reflective_planes,
            material_properties=material_props,
            max_reflections=max_reflections,
            absorption_threshold=absorption_threshold,
            trim_to_duration=True
        )
        logging.info("Simulierte Signale erzeugt.")
    else:
        if audio_files is None:
            raise ValueError("Audio-Dateien müssen angegeben werden, wenn use_simulation=False.")
        if len(audio_files) != len(mic_positions):
            raise ValueError("Die Anzahl der Audio-Dateien muss mit der Anzahl der Mikrofone übereinstimmen.")
        signals = read_audio_files(audio_files, fs)
        logging.info("Echte Audiodaten geladen.")
    
    signals = synchronize_signals_improved(signals, fs)
    logging.info("Signale synchronisiert.")
    
    filtered_signals = [noise_reduction(sig, fs, method=filter_method) for sig in signals]
    for i in range(len(filtered_signals)):
        logging.info(f"Signal {i+1} gefiltert mit '{filter_method}' Noise Reduction.")
    
    td_diffs = []
    mic_pairs = []
    corr_matrix = np.zeros((len(mic_positions), len(mic_positions)))
    correlation_metrics = {}
    corr_data_for_3d = []
    pairs_for_3d = []
    
    for i in range(len(filtered_signals)):
        for j in range(i+1, len(filtered_signals)):
            time_delays, corr, lags = get_time_delays_phat(filtered_signals[i], filtered_signals[j], fs, num_peaks=1, max_expected_delay=max_expected_delay)
            if not time_delays:
                logging.warning(f"Keine Zeitverzögerung für Mikrofonpaar {i+1}-{j+1} gefunden.")
                continue
            for td in time_delays:
                if calib_delays is not None:
                    correction = calib_delays[j] - calib_delays[i]
                    td_corrected = td - correction
                    td_diffs.append(td_corrected)
                    mic_pairs.append((i, j))
                    logging.info(f"Mikrofonpaar {i+1}-{j+1}: TDOA gemessen={td:.6f}s, Korrektur={correction:+.6f}s, TDOA korrigiert={td_corrected:.6f}s")
                else:
                    td_diffs.append(td)
                    mic_pairs.append((i, j))
                    logging.info(f"Zeitdifferenz für Mikrofonpaar {i+1}-{j+1}: {td:.6f} s (ohne Kalibrierung)")
            if analyze_correlation:
                metrics = compute_cross_correlation_metrics(corr, filtered_signals[i], filtered_signals[j], fs, alpha=0.05)
                correlation_metrics[(i, j)] = metrics
                logging.info(f"Cross-Correlation-Metriken für Mikrofonpaar {i+1}-{j+1}: {metrics}")
            peak_correlation = np.max(corr)
            corr_matrix[i, j] = peak_correlation
            corr_matrix[j, i] = peak_correlation
            if visualize_correlation:
                corr_data_for_3d.append(corr)
                pairs_for_3d.append((i, j))
    
    if not mic_pairs:
        raise RuntimeError("Keine gültigen Mikrofonpaare mit ermittelten Zeitverzögerungen.")
    
    dd_diffs = [c * td for td in td_diffs]
    for i, dd in enumerate(dd_diffs, start=1):
        pair = mic_pairs[i-1]
        logging.info(f"Differenz der Distanzen für Mikrofonpaar {pair[0]+1}-{pair[1]+1}: {dd:.3f} m")
    
    initial_guesses = heuristic_initialization_adaptive(
        mic_positions, mic_pairs, td_diffs, c,
        clustering_method=clustering_method,
        eps=clustering_eps,
        min_samples=clustering_min_samples
    )
    logging.info(f"Heuristisch initiale Positionen: {initial_guesses}")
    
    bounds = dynamic_bounds_extended(mic_positions, td_diffs, c, buffer=5.0)
    lower_bounds = [b[0] for b in bounds]
    upper_bounds = [b[1] for b in bounds]
    
    def clip_guess(guess, lower, upper):
        return np.array([np.clip(guess[i], lower[i], upper[i]) for i in range(len(guess))])
    initial_guesses = [clip_guess(guess, lower_bounds, upper_bounds) for guess in initial_guesses]
    
    if analyze_correlation and correlation_metrics:
        weights = compute_weights(correlation_metrics, mic_pairs)
    else:
        weights = np.ones(len(mic_pairs))
    
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
        logging.info(f"Geschätzte Quelle: ({x_source:.3f}, {y_source:.3f}, {z_source:.3f}) m")
    else:
        logging.warning("Least Squares Optimierung fehlgeschlagen, versuche Differential Evolution.")
        result_de = differential_evolution(
            lambda vars: np.sum(np.square(equations(vars, mic_positions, mic_pairs, td_diffs, c, weights))),
            bounds=list(zip(lower_bounds, upper_bounds)),
            strategy='best1bin',
            maxiter=1000,
            popsize=15,
            tol=1e-6,
            mutation=(0.5, 1),
            recombination=0.7,
            polish=True,
            init='latinhypercube'
        )
        if result_de.success:
            x_source, y_source, z_source = result_de.x
            logging.info(f"Geschätzte Quelle (Differential Evolution): ({x_source:.3f}, {y_source:.3f}, {z_source:.3f}) m")
        else:
            logging.error("Differential Evolution Optimierung fehlgeschlagen. Verwende den ersten initialen Schätzwert als Fallback.")
            x_source, y_source, z_source = initial_guesses[0]
    
    if use_simulation:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(mic_positions[:,0], mic_positions[:,1], mic_positions[:,2], c='r', marker='o', label='Mikrofone')
        ax.scatter(source_position[0], source_position[1], source_position[2], c='g', marker='*', s=100, label='Tatsächliche Quelle')
        ax.scatter(x_source, y_source, z_source, c='b', marker='x', s=100, label='Geschätzte Quelle')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.legend()
        plt.title('Sound Source Localization')
        if show_plots:
            plt.show()
        else:
            plt.savefig("localization_result.png")
        plt.close(fig)
    
    if visualize_correlation:
        plot_correlation_heatmap(corr_matrix, mic_positions, show_plot=show_plots, save_path="heatmap.png")
        plot_correlation_3d(corr_data_for_3d, pairs_for_3d, fs, show_plot=show_plots, save_path="correlation_3d.png")
    
    if analyze_correlation:
        logging.info("Erweiterte Cross-Correlation Metriken:")
        for pair, metrics in correlation_metrics.items():
            logging.info(f"Mikrofonpaar {pair[0]+1}-{pair[1]+1}: {metrics}")
    
    return {
        "estimated_position": np.array([x_source, y_source, z_source]),
        "actual_position": source_position if use_simulation else None,
        "mic_positions": mic_positions,
        "correlation_metrics": correlation_metrics if analyze_correlation else None,
        "correlation_matrix": corr_matrix if visualize_correlation else None,
        "calibration_data": calibration_data
    }

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    
    calibration_results, calib_signal, recorded_signals = run_calibration(config)
    for idx, res in enumerate(calibration_results):
        logging.info(f"Calibration - Mikrofon {idx+1}: Delay = {res['delay']:.6f} s, Amplitude = {res['amplitude']:.3f}")
    
    avg_delay = np.mean([r['delay'] for r in calibration_results])
    avg_amplitude = np.mean([r['amplitude'] for r in calibration_results])
    logging.info(f"Average calibration delay: {avg_delay:.6f} s, Average amplitude: {avg_amplitude:.3f}")
    
    localization_results = localize_sound_source(config, calibration_data=calibration_results, use_simulation=True, show_plots=True)
    logging.info(f"Localization result: {localization_results['estimated_position']}")
