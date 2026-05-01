"""
visualization/plots.py
------------------------
Spectrograms, time-series, distribution, and PCA visualisations for
soundscape analysis.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)


def plot_waveform(
    y: np.ndarray,
    sr: int,
    title: str = "Waveform",
    figsize: Tuple[int, int] = (12, 3),
    output_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Plot a 1-D audio waveform."""
    fig, ax = plt.subplots(figsize=figsize)
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    plt.tight_layout()
    if output_path:
        _save(fig, output_path)
    return fig


def plot_spectrogram(
    y: np.ndarray,
    sr: int,
    n_fft: int = 2048,
    hop_length: int = 512,
    title: str = "Log-Power Spectrogram",
    figsize: Tuple[int, int] = (12, 4),
    output_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Plot a log-power spectrogram."""
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length)) ** 2
    S_db = librosa.power_to_db(S, ref=np.max)

    fig, ax = plt.subplots(figsize=figsize)
    img = librosa.display.specshow(
        S_db, sr=sr, hop_length=hop_length, x_axis="time", y_axis="hz", ax=ax
    )
    plt.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title(title)
    plt.tight_layout()
    if output_path:
        _save(fig, output_path)
    return fig


def plot_mel_spectrogram(
    y: np.ndarray,
    sr: int,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512,
    title: str = "Mel Spectrogram",
    figsize: Tuple[int, int] = (12, 4),
    output_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Plot a mel-scaled spectrogram."""
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
    )
    S_db = librosa.power_to_db(S, ref=np.max)

    fig, ax = plt.subplots(figsize=figsize)
    img = librosa.display.specshow(
        S_db, sr=sr, hop_length=hop_length, x_axis="time", y_axis="mel", ax=ax
    )
    plt.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title(title)
    plt.tight_layout()
    if output_path:
        _save(fig, output_path)
    return fig


def plot_mfcc(
    y: np.ndarray,
    sr: int,
    n_mfcc: int = 13,
    hop_length: int = 512,
    title: str = "MFCCs",
    figsize: Tuple[int, int] = (12, 4),
    output_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Plot MFCC heatmap."""
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc,
                                  hop_length=hop_length)
    fig, ax = plt.subplots(figsize=figsize)
    img = librosa.display.specshow(
        mfcc, sr=sr, hop_length=hop_length, x_axis="time", ax=ax
    )
    plt.colorbar(img, ax=ax)
    ax.set_title(title)
    ax.set_ylabel("MFCC Coefficient")
    plt.tight_layout()
    if output_path:
        _save(fig, output_path)
    return fig


def plot_indices_timeseries(
    df: pd.DataFrame,
    index_cols: List[str],
    time_col: Optional[str] = None,
    title: str = "Ecoacoustic Indices Over Time",
    figsize: Tuple[int, int] = (14, 6),
    output_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Plot ecoacoustic indices as a time-series."""
    fig, axes = plt.subplots(len(index_cols), 1, figsize=figsize, sharex=True)
    if len(index_cols) == 1:
        axes = [axes]

    x = df[time_col] if time_col and time_col in df.columns else df.index

    for ax, col in zip(axes, index_cols):
        ax.plot(x, df[col], marker="o", linewidth=1.5, markersize=3)
        ax.set_ylabel(col)
        ax.grid(True, alpha=0.3)

    axes[0].set_title(title)
    axes[-1].set_xlabel(time_col or "Sample Index")
    plt.tight_layout()
    if output_path:
        _save(fig, output_path)
    return fig


def plot_index_distributions(
    df: pd.DataFrame,
    index_cols: List[str],
    group_col: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 8),
    output_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Plot distribution (histogram + KDE) of each ecoacoustic index."""
    n = len(index_cols)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes_flat = np.array(axes).ravel()

    for ax, col in zip(axes_flat, index_cols):
        if group_col and group_col in df.columns:
            for grp, sub in df.groupby(group_col):
                sns.kdeplot(sub[col], ax=ax, label=str(grp))
            ax.legend(fontsize=7)
        else:
            sns.histplot(df[col], kde=True, ax=ax)
        ax.set_title(col)
        ax.set_xlabel("")

    # hide unused axes
    for ax in axes_flat[n:]:
        ax.set_visible(False)

    plt.suptitle("Index Distributions", y=1.01)
    plt.tight_layout()
    if output_path:
        _save(fig, output_path)
    return fig


def plot_pca(
    X: np.ndarray,
    labels: Optional[np.ndarray] = None,
    feature_names: Optional[List[str]] = None,
    title: str = "PCA – Soundscape Feature Space",
    figsize: Tuple[int, int] = (8, 6),
    output_path: Optional[str | Path] = None,
) -> plt.Figure:
    """PCA scatter plot (first two principal components)."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=figsize)
    scatter = ax.scatter(
        pcs[:, 0], pcs[:, 1],
        c=labels if labels is not None else "steelblue",
        cmap="tab10",
        alpha=0.7,
        edgecolors="k",
        linewidths=0.3,
    )
    if labels is not None:
        plt.colorbar(scatter, ax=ax, label="Cluster")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title(title)
    plt.tight_layout()
    if output_path:
        _save(fig, output_path)
    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    title: str = "Confusion Matrix",
    figsize: Tuple[int, int] = (8, 6),
    output_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Plot a confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    if output_path:
        _save(fig, output_path)
    return fig


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, output_path: str | Path, dpi: int = 300) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    logger.info("Plot saved to %s", output_path)
