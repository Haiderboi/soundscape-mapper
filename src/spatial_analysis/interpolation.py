"""
spatial_analysis/interpolation.py
------------------------------------
Spatial interpolation: Inverse Distance Weighting (IDW) and Kriging.
Generates continuous raster surfaces from point observations.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# IDW Interpolation
# ---------------------------------------------------------------------------

def idw_interpolate(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    power: float = 2.0,
    epsilon: float = 1e-10,
) -> np.ndarray:
    """Inverse Distance Weighting interpolation.

    Parameters
    ----------
    x, y:
        1-D arrays of observation coordinates.
    values:
        Observed values at (x, y).
    grid_x, grid_y:
        2-D meshgrid arrays of the output raster.
    power:
        Distance-decay exponent.  Larger values give more local influence.
    epsilon:
        Small constant added to distances to avoid division by zero.

    Returns
    -------
    np.ndarray
        Interpolated values on the grid, shape matches *grid_x*.
    """
    gx_flat = grid_x.ravel()
    gy_flat = grid_y.ravel()
    result = np.zeros(len(gx_flat))

    for i, (gxi, gyi) in enumerate(zip(gx_flat, gy_flat)):
        dist = np.sqrt((x - gxi) ** 2 + (y - gyi) ** 2) + epsilon
        weights = 1.0 / dist ** power
        result[i] = np.sum(weights * values) / np.sum(weights)

    return result.reshape(grid_x.shape)


def idw_surface(
    gdf,
    column: str,
    grid_resolution: float = 0.01,
    power: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate an IDW-interpolated raster surface from a GeoDataFrame.

    Parameters
    ----------
    gdf:
        GeoDataFrame with point geometry and the target *column*.
    column:
        Column to interpolate.
    grid_resolution:
        Degree-spacing of the output grid.
    power:
        IDW distance-decay exponent.

    Returns
    -------
    (grid_x, grid_y, z)
        Meshgrid arrays and the interpolated surface values.
    """
    x = gdf.geometry.x.values
    y = gdf.geometry.y.values
    values = gdf[column].values.astype(float)

    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    xi = np.arange(x_min, x_max + grid_resolution, grid_resolution)
    yi = np.arange(y_min, y_max + grid_resolution, grid_resolution)
    grid_x, grid_y = np.meshgrid(xi, yi)

    z = idw_interpolate(x, y, values, grid_x, grid_y, power=power)
    logger.info(
        "IDW surface for '%s': grid shape %s", column, z.shape
    )
    return grid_x, grid_y, z


# ---------------------------------------------------------------------------
# Kriging Interpolation
# ---------------------------------------------------------------------------

def kriging_surface(
    gdf,
    column: str,
    grid_resolution: float = 0.01,
    variogram_model: str = "spherical",
    coordinates_type: str = "geographic",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Ordinary Kriging interpolation using PyKrige.

    Parameters
    ----------
    gdf:
        GeoDataFrame with point geometry.
    column:
        Column to interpolate.
    grid_resolution:
        Degree spacing of the output grid.
    variogram_model:
        PyKrige variogram model (``"linear"``, ``"power"``, ``"gaussian"``,
        ``"spherical"``, ``"exponential"``).
    coordinates_type:
        ``"geographic"`` or ``"euclidean"``.

    Returns
    -------
    (grid_x, grid_y, z, variance)
        Meshgrid arrays, interpolated values, and kriging variance.
    """
    from pykrige.ok import OrdinaryKriging

    x = gdf.geometry.x.values
    y = gdf.geometry.y.values
    values = gdf[column].values.astype(float)

    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    xi = np.arange(x_min, x_max + grid_resolution, grid_resolution)
    yi = np.arange(y_min, y_max + grid_resolution, grid_resolution)

    ok = OrdinaryKriging(
        x, y, values,
        variogram_model=variogram_model,
        coordinates_type=coordinates_type,
        verbose=False,
        enable_plotting=False,
    )
    z, variance = ok.execute("grid", xi, yi)
    grid_x, grid_y = np.meshgrid(xi, yi)

    logger.info(
        "Kriging surface for '%s': grid shape %s  model=%s",
        column, z.shape, variogram_model,
    )
    return grid_x, grid_y, np.array(z), np.array(variance)
