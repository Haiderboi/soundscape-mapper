"""
ml_clustering/clustering.py
-----------------------------
K-means, DBSCAN, and hierarchical clustering of soundscape feature vectors.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def scale_features(X: np.ndarray) -> Tuple[np.ndarray, StandardScaler]:
    """Standardise features to zero mean and unit variance.

    Returns
    -------
    (X_scaled, scaler)
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def kmeans_clustering(
    X: np.ndarray,
    n_clusters: int = 5,
    random_state: int = 42,
    n_init: int = 10,
) -> Tuple[np.ndarray, KMeans]:
    """K-means clustering.

    Parameters
    ----------
    X:
        Feature matrix of shape ``(n_samples, n_features)``.
    n_clusters:
        Number of clusters.
    random_state:
        Random seed for reproducibility.
    n_init:
        Number of initialisations.

    Returns
    -------
    (labels, model)
    """
    X_scaled, _ = scale_features(X)
    model = KMeans(
        n_clusters=n_clusters, random_state=random_state, n_init=n_init
    )
    labels = model.fit_predict(X_scaled)
    if len(set(labels)) > 1:
        sil = silhouette_score(X_scaled, labels)
        logger.info(
            "K-means k=%d  inertia=%.2f  silhouette=%.4f",
            n_clusters, model.inertia_, sil,
        )
    return labels, model


def dbscan_clustering(
    X: np.ndarray,
    eps: float = 0.5,
    min_samples: int = 5,
) -> Tuple[np.ndarray, DBSCAN]:
    """DBSCAN density-based clustering.

    Noise points are assigned label ``-1``.

    Returns
    -------
    (labels, model)
    """
    X_scaled, _ = scale_features(X)
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X_scaled)
    unique = set(labels)
    n_clusters = len(unique - {-1})
    n_noise = int(np.sum(labels == -1))
    logger.info(
        "DBSCAN eps=%.2f min=%d  clusters=%d  noise=%d",
        eps, min_samples, n_clusters, n_noise,
    )
    return labels, model


def hierarchical_clustering(
    X: np.ndarray,
    n_clusters: int = 5,
    linkage: str = "ward",
) -> Tuple[np.ndarray, AgglomerativeClustering]:
    """Agglomerative hierarchical clustering.

    Parameters
    ----------
    X:
        Feature matrix.
    n_clusters:
        Number of clusters.
    linkage:
        Linkage criterion: ``"ward"``, ``"complete"``, ``"average"``,
        ``"single"``.

    Returns
    -------
    (labels, model)
    """
    X_scaled, _ = scale_features(X)
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = model.fit_predict(X_scaled)
    if len(set(labels)) > 1:
        sil = silhouette_score(X_scaled, labels)
        logger.info(
            "Hierarchical n_clusters=%d linkage=%s  silhouette=%.4f",
            n_clusters, linkage, sil,
        )
    return labels, model


def silhouette_analysis(
    X: np.ndarray,
    k_range: range = range(2, 11),
    random_state: int = 42,
) -> pd.DataFrame:
    """Compute silhouette scores for a range of k values.

    Helps select the optimal number of clusters for K-means.

    Returns
    -------
    pd.DataFrame
        Columns: ``k``, ``inertia``, ``silhouette``.
    """
    X_scaled, _ = scale_features(X)
    records = []
    for k in k_range:
        model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = model.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels) if len(set(labels)) > 1 else 0.0
        records.append(
            {"k": k, "inertia": model.inertia_, "silhouette": sil}
        )
    df = pd.DataFrame(records)
    logger.info("Silhouette analysis: best k=%d", int(df.loc[df["silhouette"].idxmax(), "k"]))
    return df
