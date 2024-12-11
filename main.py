# main.py

import numpy as np
import logging
from scipy.optimize import least_squares, differential_evolution
from signal_processing import (
    generate_signal, generate_realistic_speech, fractional_delay, normalize_signal,
    dynamic_range_compression, dynamic_range_compression_soft_clip, resample_audio,
    rauschunterdrueckung
)
from plotting import plot_correlation_heatmap, plot_correlation_3d
from utils import (
    schallgeschwindigkeit, reflect_point_across_plane, distance,
    calculate_attenuation, generate_image_sources_iterative, phat_correlation,
    get_time_delays_phat, compute_cross_correlation_metrics, perform_significance_test,
    perform_significance_test_bootstrap, compute_peak_to_peak_ratio, compute_snr,
    perform_significance_test, determine_optimal_number_of_clusters,
    heuristic_initialization_adaptive, dynamic_bounds_extended, equations,
    compute_weights, synchronize_signals_improved, read_audio_files
)
from materials import material_properties
import os

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
    max_reflections=2,
    absorption_threshold=0.01
):
    base_signal = generate_signal(signal_type, fs, duration, freq)
    all_image_sources = generate_image_sources_iterative(
        source=source_pos,
        planes=reflektierende_ebenen,
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
        max_distance = direct_distance + (max(reflection_distances) if reflection_distances else 0)
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
        logging.debug(f"Direkter Pfad für Mikrofon {mic_pos}: Delta d = {distance_direct:.3f} m, Delta t = {time_delay_direct:.6f} s, Attenuation = {attenuation_direct:.6f}")

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

        signal_total = signal_total[:int(duration * fs)]
        signal_total = normalize_signal(signal_total)
        signal_total = dynamic_range_compression(signal_total)
        signals.append(signal_total)

    return signals

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
    source_position=None,
    duration=1.0,
    signal_type='sine',
    freq=1000,
    reflektierende_ebenen=None,
    material_properties=None,
    filter_method='butterworth',
    use_simulation=True,
    audiodateien=None,
    absorption_threshold=0.01,
    analyze_correlation=False,
    visualize_correlation=False,
    clustering_method='kmeans',
    clustering_eps=0.001,
    clustering_min_samples=2,
    max_reflections=2,
    show_plots=True
):
    try:
        # Eingabevalidierung
        if not isinstance(mic_positions, np.ndarray):
            raise TypeError("mic_positions muss ein numpy-Array sein.")
        if mic_positions.ndim != 2 or mic_positions.shape[1] != 3:
            raise ValueError("mic_positions muss ein nx3-Array sein.")
        if len(mic_positions) < 2:
            raise ValueError("Mindestens zwei Mikrofone sind für die Lokalisierung erforderlich.")

        # Schallgeschwindigkeit berechnen
        c = schallgeschwindigkeit(temperatur, luftfeuchtigkeit)
        logging.info(f"Berechnete Schallgeschwindigkeit: {c:.2f} m/s")

        # Signal-Generierung oder Einlesen
        if use_simulation:
            if source_position is None:
                raise ValueError("source_position muss bei use_simulation=True angegeben werden.")
            if reflektierende_ebenen is None:
                reflektierende_ebenen = []
            if material_properties is None:
                material_properties = {
                    'air': {'absorption': 0.01, 'freq': 0.1},
                    'wood': {'absorption': 0.05, 'freq': 0.8},
                    'metal': {'absorption': 0.1, 'freq': 0.6}
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
                absorption_threshold=absorption_threshold
            )
            logging.info(f"Simulierte Signale generiert mit Signaltyp '{signal_type}'.")
        else:
            if audiodateien is None:
                raise ValueError("Audiodateien müssen angegeben werden, wenn use_simulation=False ist.")
            if len(audiodateien) != len(mic_positions):
                raise ValueError("Anzahl der Audiodateien muss mit der Anzahl der Mikrofone übereinstimmen.")
            if material_properties is None:
                material_properties = {
                    'air': {'absorption': 0.01, 'freq': 0.1},
                    'wood': {'absorption': 0.05, 'freq': 0.8},
                    'metal': {'absorption': 0.1, 'freq': 0.6}
                }
            signals = read_audio_files(audiodateien, fs)
            logging.info("Echte Audiodaten eingelesen.")

        # Synchronisierung der Signale
        signals = synchronize_signals_improved(signals, fs)
        logging.info("Signale synchronisiert.")

        # Rauschunterdrückung
        gefilterte_signale = [rauschunterdrueckung(sig, fs, method=filter_method) for sig in signals]
        for i in range(len(gefilterte_signale)):
            logging.info(f"Signal {i+1} nach Rauschunterdrückung mit Methode '{filter_method}' gefiltert.")

        # Laufzeitdifferenzen berechnen
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
                    num_peaks=1
                )
                if not time_delays:
                    logging.warning(f"Keine Zeitverzögerungen für Mikrofon {i+1} - Mikrofon {j+1} gefunden.")
                    continue
                for td in time_delays:
                    td_diffs.append(td)
                    mic_pairs.append((i, j))
                    logging.info(f"Laufzeitdifferenz zwischen Mikrofon {i+1} und Mikrofon {j+1}: {td:.6f} Sekunden")

                if analyze_correlation:
                    metrics = compute_cross_correlation_metrics(corr, gefilterte_signale[i], gefilterte_signale[j], fs, alpha=0.05)
                    correlation_metrics[(i, j)] = metrics
                    logging.info(f"Kreuzkorrelationsmetriken zwischen Mikrofon {i+1} und Mikrofon {j+1}: {metrics}")

                peak_correlation = np.max(corr)
                corr_matrix[i, j] = peak_correlation
                corr_matrix[j, i] = peak_correlation

                if visualize_correlation:
                    corr_data_for_3d.append(corr)
                    pairs_for_3d.append((i, j))

        if not mic_pairs:
            raise RuntimeError("Keine gültigen Mikrofonpaare mit gefundenen Zeitverzögerungen.")

        # Entfernungsdifferenzen berechnen
        dd_diffs = [c * td for td in td_diffs]
        for i, dd in enumerate(dd_diffs, start=1):
            pair = mic_pairs[i-1]
            logging.info(f"Entfernungsdifferenz Mikrofonpaar {pair[0]+1}-{pair[1]+1}: {dd:.3f} Meter")

        # Positionsschätzung mittels Least Squares mit adaptiver Initialisierung und dynamischen Grenzen
        initial_guesses = heuristic_initialization_adaptive(
            mic_positions, mic_pairs, td_diffs, c,
            clustering_method=clustering_method,
            eps=clustering_eps,
            min_samples=clustering_min_samples
        )
        logging.info(f"Heuristische Startpositionen: {initial_guesses}")

        bounds = dynamic_bounds_extended(mic_positions, td_diffs, c, buffer=5.0)
        lower_bounds = [b[0] for b in bounds]
        upper_bounds = [b[1] for b in bounds]
        bounds_tuple = (lower_bounds, upper_bounds)
        logging.info(f"Dynamische Optimierungsgrenzen gesetzt: {bounds_tuple}")

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
            logging.info(f"Die geschätzte Position der Schallquelle ist ({x_source:.3f}, {y_source:.3f}, {z_source:.3f}) Meter")
        else:
            logging.warning("Optimierung mit least_squares fehlgeschlagen. Versuch mit Differential Evolution.")
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
                initial_guess = initial_guesses[0]
                x_source, y_source, z_source = initial_guess
                logging.info(f"Verwende heuristische Schätzung als geschätzte Position: ({x_source:.3f}, {y_source:.3f}, {z_source:.3f}) Meter")

        # Visualisierung der Ergebnisse
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
            if show_plots:
                plt.show()
            else:
                plt.savefig("localization_result.png")
            plt.close(fig)

        if visualize_correlation:
            plot_correlation_heatmap(corr_matrix, mic_positions, show_plot=show_plots, save_path="heatmap.png")
            plot_correlation_3d(corr_data_for_3d, pairs_for_3d, fs, show_plot=show_plots, save_path="correlation_3d.png")

        if analyze_correlation:
            logging.info("\nErweiterte mathematische Kennzahlen der Kreuzkorrelationen:")
            for pair, metrics in correlation_metrics.items():
                mic1, mic2 = pair
                logging.info(f"Mikrofon {mic1+1} - Mikrofon {mic2+1}:")
                logging.info(f"  Peak-to-Peak-Verhältnis: {metrics['peak_to_peak_ratio']:.2f}")
                logging.info(f"  SNR: {metrics['snr']:.2f}")
                logging.info(f"  Signifikant: {'Ja' if metrics['significant'] else 'Nein'}")

        return {
            'geschätzte_position': np.array([x_source, y_source, z_source]),
            'tatsächliche_position': source_position if use_simulation else None,
            'mikrofon_positionen': mic_positions,
            'korrelation_metrics': correlation_metrics if analyze_correlation else None,
            'korrelationsmatrix': corr_matrix if visualize_correlation else None
        }

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Konfigurieren des Loggings
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # Beispielparameter
    temperatur = 20  # in °C
    luftfeuchtigkeit = 50  # in %
    fs = 44100  # Sampling-Rate in Hz
    mic_positions = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])
    source_position = np.array([0.5, 0.5, 0.5])
    duration = 1.0
    freq = 1000

    reflektierende_ebenen = [
        {'plane': np.array([1, 0, 0, -5]), 'material': 'wood'},
        {'plane': np.array([0, 1, 0, -5]), 'material': 'metal'},
        {'plane': np.array([0, 0, 1, -5]), 'material': 'wood'}
    ]

    # Lokalisierung mit Simulation und Butterworth-Rauschunterdrückung, erweiterte Analyse und Visualisierung aktiviert
    logging.info("\nLokalisierung mit Butterworth-Filter, erweiterten Analysen und Visualisierungen:")
    lokalisieren_schallquelle(
        temperatur=temperatur,
        luftfeuchtigkeit=luftfeuchtigkeit,
        fs=fs,
        mic_positions=mic_positions,
        source_position=source_position,
        duration=duration,
        signal_type='sine',
        freq=freq,
        reflektierende_ebenen=reflektierende_ebenen,
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
            reflektierende_ebenen=reflektierende_ebenen,
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

    # Lokalisierung mit realen Audiodaten und Wiener-Filter, erweiterte Analyse und Visualisierung aktiviert
    logging.info("\nLokalisierung mit realen Audiodaten, Wiener-Filter, erweiterten Analysen und Visualisierungen:")
    lokalisieren_schallquelle(
        temperatur=temperatur,
        luftfeuchtigkeit=luftfeuchtigkeit,
        fs=fs,
        mic_positions=mic_positions,
        use_simulation=False,
        audiodateien=audiodateien,
        material_properties=material_properties,
        filter_method='wiener',
        absorption_threshold=0.01,
        analyze_correlation=True,
        visualize_correlation=True,
        clustering_method='kmeans',
        clustering_eps=0.001,
        clustering_min_samples=2,
        max_reflections=3,
        show_plots=True
    )
