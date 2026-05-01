"""
visualization/gis_export.py
-----------------------------
Export processed soundscape data to GeoJSON, Shapefile and GeoTIFF formats.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


def export_geojson(
    gdf,
    output_path: str | Path,
    metadata: Optional[Dict] = None,
) -> Path:
    """Export a GeoDataFrame to GeoJSON.

    Parameters
    ----------
    gdf:
        GeoDataFrame with point or polygon geometry.
    output_path:
        Destination file path (e.g. ``outputs/soundscape.geojson``).
    metadata:
        Optional extra key/value pairs embedded in the GeoJSON root as
        ``properties``.

    Returns
    -------
    Path
        Path of the written file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    geojson_dict = json.loads(gdf.to_json())
    if metadata:
        geojson_dict["metadata"] = metadata

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(geojson_dict, f, indent=2)

    logger.info("GeoJSON exported to %s  (%d features)", output_path, len(gdf))
    return output_path


def export_shapefile(
    gdf,
    output_path: str | Path,
) -> Path:
    """Export a GeoDataFrame to ESRI Shapefile.

    Parameters
    ----------
    gdf:
        GeoDataFrame.
    output_path:
        Destination file path.  The ``.shp`` extension is appended if absent.

    Returns
    -------
    Path
        Path of the ``.shp`` file.
    """
    output_path = Path(output_path)
    if output_path.suffix.lower() != ".shp":
        output_path = output_path.with_suffix(".shp")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    gdf.to_file(str(output_path), driver="ESRI Shapefile")
    logger.info("Shapefile exported to %s", output_path)
    return output_path


def export_geotiff(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    z: np.ndarray,
    output_path: str | Path,
    crs: str = "EPSG:4326",
    nodata: float = -9999.0,
) -> Path:
    """Write a 2-D raster array to a GeoTIFF file.

    Parameters
    ----------
    grid_x:
        2-D meshgrid of longitudes.
    grid_y:
        2-D meshgrid of latitudes.
    z:
        2-D array of raster values (same shape as *grid_x*).
    output_path:
        Destination file path.
    crs:
        Coordinate reference system string (EPSG code or WKT).
    nodata:
        No-data sentinel value.

    Returns
    -------
    Path
        Path of the written GeoTIFF.
    """
    import rasterio
    from rasterio.transform import from_bounds

    output_path = Path(output_path)
    if output_path.suffix.lower() not in {".tif", ".tiff"}:
        output_path = output_path.with_suffix(".tif")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    height, width = z.shape
    x_min = float(grid_x.min())
    x_max = float(grid_x.max())
    y_min = float(grid_y.min())
    y_max = float(grid_y.max())

    transform = from_bounds(x_min, y_min, x_max, y_max, width, height)

    z_out = np.where(np.isnan(z), nodata, z).astype(np.float32)

    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(z_out, 1)

    logger.info("GeoTIFF exported to %s  shape=%s", output_path, z.shape)
    return output_path


def export_all(
    gdf,
    grid_x: Optional[np.ndarray],
    grid_y: Optional[np.ndarray],
    z: Optional[np.ndarray],
    output_dir: str | Path,
    base_name: str = "soundscape",
    crs: str = "EPSG:4326",
    metadata: Optional[Dict] = None,
) -> Dict[str, Path]:
    """Convenience function: export GeoJSON, Shapefile and optionally GeoTIFF.

    Returns
    -------
    dict with keys ``geojson``, ``shapefile``, and optionally ``geotiff``.
    """
    output_dir = Path(output_dir)
    results: Dict[str, Path] = {}

    results["geojson"] = export_geojson(
        gdf, output_dir / f"{base_name}.geojson", metadata=metadata
    )
    results["shapefile"] = export_shapefile(
        gdf, output_dir / f"{base_name}.shp"
    )

    if grid_x is not None and grid_y is not None and z is not None:
        results["geotiff"] = export_geotiff(
            grid_x, grid_y, z, output_dir / f"{base_name}.tif", crs=crs
        )

    logger.info("All exports written to %s", output_dir)
    return results
