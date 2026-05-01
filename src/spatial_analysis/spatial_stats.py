"""
spatial_analysis/spatial_stats.py
-----------------------------------
Spatial autocorrelation: global and local Moran's I via PySAL / ESDA.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _build_weights(gdf, k: int = 8):
    """Build a KNN spatial weights matrix from a GeoDataFrame.

    Parameters
    ----------
    gdf:
        GeoDataFrame with geometry column.
    k:
        Number of nearest neighbours.

    Returns
    -------
    libpysal.weights.KNN
        Spatial weights object.
    """
    from libpysal.weights import KNN

    w = KNN.from_dataframe(gdf, k=k)
    w.transform = "r"        # row-standardise
    return w


def global_morans_i(
    gdf,
    column: str,
    k: int = 8,
) -> Dict[str, float]:
    """Compute Global Moran's I for spatial autocorrelation.

    Parameters
    ----------
    gdf:
        GeoDataFrame containing the variable of interest.
    column:
        Column name to test for spatial autocorrelation.
    k:
        Number of nearest neighbours for weight matrix.

    Returns
    -------
    dict with keys ``I``, ``E_I``, ``z_score``, ``p_value``.
    """
    from esda.moran import Moran

    w = _build_weights(gdf, k=k)
    mi = Moran(gdf[column], w)
    result = {
        "I": float(mi.I),
        "E_I": float(mi.EI),
        "z_score": float(mi.z_norm),
        "p_value": float(mi.p_norm),
    }
    logger.info(
        "Global Moran's I for '%s': I=%.4f  z=%.4f  p=%.4f",
        column, result["I"], result["z_score"], result["p_value"],
    )
    return result


def local_morans_i(
    gdf,
    column: str,
    k: int = 8,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Compute Local Moran's I (LISA) for each location.

    Parameters
    ----------
    gdf:
        GeoDataFrame with geometry and variable column.
    column:
        Column name to analyse.
    k:
        Number of nearest neighbours.
    alpha:
        Significance level for cluster type assignment.

    Returns
    -------
    pd.DataFrame
        Columns: ``Is``, ``p_sim``, ``q``, ``cluster``.
        ``cluster`` is one of ``HH``, ``LL``, ``HL``, ``LH``, ``NS``.
    """
    from esda.moran import Moran_Local

    w = _build_weights(gdf, k=k)
    lm = Moran_Local(gdf[column], w)

    quadrant_labels = {1: "HH", 2: "LH", 3: "LL", 4: "HL"}
    clusters = [
        quadrant_labels.get(int(q), "NS") if p < alpha else "NS"
        for q, p in zip(lm.q, lm.p_sim)
    ]

    result_df = pd.DataFrame(
        {
            "Is": lm.Is,
            "p_sim": lm.p_sim,
            "q": lm.q,
            "cluster": clusters,
        },
        index=gdf.index,
    )
    logger.info(
        "Local Moran's I for '%s': %d significant locations",
        column, int((result_df["cluster"] != "NS").sum()),
    )
    return result_df


def spatial_lag(gdf, column: str, k: int = 8) -> pd.Series:
    """Compute the spatial lag (weighted average of neighbours) for a column.

    Returns
    -------
    pd.Series
        Spatial lag values indexed like *gdf*.
    """
    from libpysal.weights import lag_spatial

    w = _build_weights(gdf, k=k)
    lag = lag_spatial(w, gdf[column].values)
    return pd.Series(lag, index=gdf.index, name=f"{column}_lag")
