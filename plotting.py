# plotting.py

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_correlation_heatmap(corr_matrix, mic_positions, title="Heatmap der Peak-Korrelationen zwischen Mikrofonpaaren", show_plot=True, save_path="heatmap.png"):
    num_mics = len(mic_positions)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr_matrix, cmap='viridis')

    ax.set_xticks(np.arange(num_mics))
    ax.set_yticks(np.arange(num_mics))
    ax.set_xticklabels([f'Mic {i+1}' for i in range(num_mics)])
    ax.set_yticklabels([f'Mic {i+1}' for i in range(num_mics)])
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Peak-Korrelation', rotation=-90, va="bottom")

    ax.set_title(title)
    fig.tight_layout()
    if show_plot:
        plt.show()
    else:
        plt.savefig(save_path)
    plt.close(fig)

def plot_correlation_3d(corr_data, mic_pairs, fs, title="3D-Kreuzkorrelationsplots", show_plot=True, save_path="correlation_3d.png"):
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
    if show_plot:
        plt.show()
    else:
        plt.savefig(save_path)
    plt.close(fig)
