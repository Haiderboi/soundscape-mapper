"""
utils/data_loader.py
----------------------
Audio file loading, geospatial data loading and helper utilities.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SUPPORTED_AUDIO_FORMATS = {".wav", ".mp3", ".flac"}


def discover_audio_files(
    directory: str | Path,
    recursive: bool = True,
    formats: Optional[List[str]] = None,
) -> List[Path]:
    """Return a sorted list of audio file paths in *directory*.

    Parameters
    ----------
    directory:
        Root directory to search.
    recursive:
        If ``True``, search sub-directories as well.
    formats:
        Allowed extensions (default: ``{".wav", ".mp3", ".flac"}``).

    Returns
    -------
    list of Path
    """
    directory = Path(directory)
    allowed = {fmt.lower() for fmt in (formats or SUPPORTED_AUDIO_FORMATS)}
    pattern = "**/*" if recursive else "*"
    files = [
        p for p in sorted(directory.glob(pattern))
        if p.is_file() and p.suffix.lower() in allowed
    ]
    logger.info("Discovered %d audio files in '%s'", len(files), directory)
    return files


def load_spatial_csv(
    csv_path: str | Path,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    crs: str = "EPSG:4326",
):
    """Load a CSV file with lat/lon columns and return a GeoDataFrame.

    Parameters
    ----------
    csv_path:
        Path to the CSV file.
    lat_col, lon_col:
        Column names for latitude and longitude.
    crs:
        Coordinate reference system.

    Returns
    -------
    geopandas.GeoDataFrame
    """
    import geopandas as gpd
    from shapely.geometry import Point

    df = pd.read_csv(csv_path)
    geometry = [Point(lon, lat) for lon, lat in zip(df[lon_col], df[lat_col])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=crs)
    logger.info("Loaded spatial CSV: %s  (%d rows)", csv_path, len(gdf))
    return gdf


def load_geojson(path: str | Path, crs: str = "EPSG:4326"):
    """Load a GeoJSON file as a GeoDataFrame.

    Returns
    -------
    geopandas.GeoDataFrame
    """
    import geopandas as gpd

    gdf = gpd.read_file(str(path))
    if gdf.crs is None:
        gdf = gdf.set_crs(crs)
    elif gdf.crs.to_string() != crs:
        gdf = gdf.to_crs(crs)
    logger.info("Loaded GeoJSON: %s  (%d features)", path, len(gdf))
    return gdf


def load_shapefile(path: str | Path, crs: str = "EPSG:4326"):
    """Load a Shapefile as a GeoDataFrame.

    Returns
    -------
    geopandas.GeoDataFrame
    """
    import geopandas as gpd

    gdf = gpd.read_file(str(path))
    if gdf.crs is None:
        gdf = gdf.set_crs(crs)
    elif gdf.crs.to_string() != crs:
        gdf = gdf.to_crs(crs)
    logger.info("Loaded Shapefile: %s  (%d features)", path, len(gdf))
    return gdf


def build_results_dataframe(
    records: List[Dict],
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    crs: str = "EPSG:4326",
):
    """Convert a list of per-recording result dicts into a GeoDataFrame.

    Each *record* dict is expected to contain at minimum ``latitude`` and
    ``longitude`` keys.

    Returns
    -------
    geopandas.GeoDataFrame
    """
    import geopandas as gpd
    from shapely.geometry import Point

    df = pd.DataFrame(records)
    geometry = [
        Point(row[lon_col], row[lat_col]) for _, row in df.iterrows()
    ]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=crs)
    logger.info("Built results GeoDataFrame: %d rows", len(gdf))
    return gdf
