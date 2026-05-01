"""
spatial_analysis/hotspot_analysis.py
---------------------------------------
Getis-Ord Gi* hotspot and cold-spot detection using PySAL / ESDA.
"""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def getis_ord_gi_star(
    gdf,
    column: str,
    k: int = 8,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Compute Getis-Ord Gi* statistic for hotspot / cold-spot detection.

    Parameters
    ----------
    gdf:
        GeoDataFrame with point geometry and the variable of interest.
    column:
        Column name to analyse.
    k:
        Number of nearest neighbours for the spatial weight matrix.
    alpha:
        Significance level for classification.

    Returns
    -------
    pd.DataFrame
        Columns: ``Gs``, ``p_sim``, ``hotspot_type``.
        ``hotspot_type`` ∈ {``"hotspot"``, ``"coldspot"``, ``"not_significant"``}.
    """
    from esda.getisord import G_Local
    from libpysal.weights import KNN

    w = KNN.from_dataframe(gdf, k=k)
    w.transform = "r"

    gs = G_Local(gdf[column], w, star=True, transform="r")

    types = []
    for z_score, p_val in zip(gs.Zs, gs.p_sim):
        if p_val < alpha:
            types.append("hotspot" if z_score > 0 else "coldspot")
        else:
            types.append("not_significant")

    result_df = pd.DataFrame(
        {
            "Gs": gs.Gs,
            "p_sim": gs.p_sim,
            "z_score": gs.Zs,
            "hotspot_type": types,
        },
        index=gdf.index,
    )

    n_hot = int((result_df["hotspot_type"] == "hotspot").sum())
    n_cold = int((result_df["hotspot_type"] == "coldspot").sum())
    logger.info(
        "Gi* for '%s': %d hotspots, %d coldspots (alpha=%.2f)",
        column, n_hot, n_cold, alpha,
    )
    return result_df


def classify_hotspots(
    hotspot_df: pd.DataFrame,
    z_col: str = "z_score",
    p_col: str = "p_sim",
    alpha: float = 0.05,
    confidence_levels: bool = True,
) -> pd.DataFrame:
    """Classify Gi* results into confidence-level categories.

    Assigns each observation to one of:
    ``"Hotspot 99%"``, ``"Hotspot 95%"``, ``"Hotspot 90%"``,
    ``"Not Significant"``,
    ``"Coldspot 90%"``, ``"Coldspot 95%"``, ``"Coldspot 99%"``.

    Parameters
    ----------
    hotspot_df:
        DataFrame returned by :func:`getis_ord_gi_star`.
    z_col:
        Column containing z-scores.
    p_col:
        Column containing p-values.
    alpha:
        Not used directly; confidence levels use fixed thresholds.
    confidence_levels:
        If ``True``, assign 99 / 95 / 90 % confidence tiers; otherwise use
        binary hotspot/coldspot/NS labels.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with an additional ``"confidence_class"`` column.
    """
    result = hotspot_df.copy()

    def _classify(z: float, p: float) -> str:
        if p > 0.10:
            return "Not Significant"
        if z > 0:
            if p <= 0.01:
                return "Hotspot 99%"
            elif p <= 0.05:
                return "Hotspot 95%"
            else:
                return "Hotspot 90%"
        else:
            if p <= 0.01:
                return "Coldspot 99%"
            elif p <= 0.05:
                return "Coldspot 95%"
            else:
                return "Coldspot 90%"

    result["confidence_class"] = [
        _classify(z, p)
        for z, p in zip(result[z_col], result[p_col])
    ]
    return result
