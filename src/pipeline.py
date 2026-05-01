"""
pipeline.py
-------------
End-to-end Soundscape Mapper processing pipeline.

Workflow
--------
1. Discover audio files in the configured directory.
2. Preprocess each recording (resample, normalise, trim silence).
3. Extract ecoacoustic indices (ACI, NDSI, ADI, BI, RMS dB …).
4. Assemble results into a GeoDataFrame (requires a CSV with lat/lon).
5. Run spatial statistics (Global Moran's I) and hotspot detection (Gi*).
6. Cluster recording locations with K-means.
7. Export outputs (GeoJSON, Shapefile, GeoTIFF) and generate an interactive
   Folium map.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.audio_processing.indices import compute_all_indices
from src.audio_processing.preprocessing import preprocess_audio
from src.ml_clustering.clustering import kmeans_clustering
from src.spatial_analysis.hotspot_analysis import getis_ord_gi_star
from src.spatial_analysis.interpolation import idw_surface
from src.spatial_analysis.spatial_stats import global_morans_i
from src.utils.data_loader import (
    build_results_dataframe,
    discover_audio_files,
    load_spatial_csv,
)
from src.utils.logger import setup_logger
from src.utils.validators import validate_audio_file, validate_coordinates
from src.visualization.gis_export import export_all
from src.visualization.maps import create_soundscape_map

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Single-file processing
# ---------------------------------------------------------------------------

def process_single_file(
    audio_path: str | Path,
    latitude: float,
    longitude: float,
    sample_rate: int = 22050,
    duration: Optional[float] = None,
) -> Dict[str, Any]:
    """Process one audio file and return a flat result dict.

    Parameters
    ----------
    audio_path:
        Path to the audio recording.
    latitude, longitude:
        GPS coordinates of the recording location.
    sample_rate:
        Target sample rate for loading.
    duration:
        Maximum duration to load in seconds.

    Returns
    -------
    dict
        Contains ``file``, ``latitude``, ``longitude`` and all ecoacoustic
        index values.
    """
    validate_coordinates(latitude, longitude)
    audio_path = validate_audio_file(audio_path)

    y, sr = preprocess_audio(
        audio_path, sample_rate=sample_rate, duration=duration
    )
    indices = compute_all_indices(y, sr)

    result: Dict[str, Any] = {
        "file": str(audio_path),
        "latitude": latitude,
        "longitude": longitude,
        **indices,
    }
    logger.info(
        "Processed '%s'  ACI=%.2f  NDSI=%.4f  RMS_dB=%.2f",
        audio_path.name, indices["aci"], indices["ndsi"], indices["rms_db"],
    )
    return result


# ---------------------------------------------------------------------------
# Batch pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    audio_dir: str | Path,
    location_csv: str | Path,
    output_dir: str | Path,
    config: Optional[Dict[str, Any]] = None,
    sample_rate: int = 22050,
    kmeans_k: int = 5,
    grid_resolution: float = 0.01,
    map_center: Optional[List[float]] = None,
    export_formats: bool = True,
) -> Dict[str, Any]:
    """Run the full end-to-end soundscape mapping pipeline.

    Parameters
    ----------
    audio_dir:
        Directory containing raw audio recordings.
    location_csv:
        CSV file with columns: ``file``, ``latitude``, ``longitude``.
        The ``file`` column should contain just the filename (not the full
        path), or a path relative to *audio_dir*.
    output_dir:
        Directory where outputs will be written.
    config:
        Optional configuration overrides.
    sample_rate:
        Target sample rate.
    kmeans_k:
        Number of K-means clusters.
    grid_resolution:
        Resolution of the IDW interpolation grid in degrees.
    map_center:
        ``[lat, lon]`` for the Folium map centre.  Auto-detected if ``None``.
    export_formats:
        If ``True``, export GeoJSON, Shapefile and GeoTIFF.

    Returns
    -------
    dict
        Keys: ``gdf``, ``hotspots``, ``clusters``, ``morans``,
        ``output_files``.
    """
    audio_dir = Path(audio_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ----- load location lookup -----
    loc_df = pd.read_csv(location_csv)
    required = {"file", "latitude", "longitude"}
    missing = required - set(loc_df.columns)
    if missing:
        raise ValueError(
            f"location_csv is missing columns: {missing}"
        )

    # ----- process recordings -----
    records: List[Dict[str, Any]] = []
    for _, loc_row in loc_df.iterrows():
        file_name = loc_row["file"]
        audio_path = audio_dir / file_name
        if not audio_path.exists():
            logger.warning("File not found, skipping: %s", audio_path)
            continue
        try:
            result = process_single_file(
                audio_path,
                latitude=float(loc_row["latitude"]),
                longitude=float(loc_row["longitude"]),
                sample_rate=sample_rate,
            )
            records.append(result)
        except Exception as exc:
            logger.error("Error processing '%s': %s", audio_path, exc)

    if not records:
        raise RuntimeError("No audio files were successfully processed.")

    # ----- build GeoDataFrame -----
    gdf = build_results_dataframe(records)
    logger.info("Results GeoDataFrame: %d rows  %d columns", *gdf.shape)

    # ----- spatial statistics -----
    index_col = "aci"
    morans_result = {}
    hotspot_df = pd.DataFrame()
    if len(gdf) >= 4:
        try:
            morans_result = global_morans_i(gdf, column=index_col)
            hotspot_df = getis_ord_gi_star(gdf, column=index_col)
            gdf["hotspot_type"] = hotspot_df["hotspot_type"].values
        except Exception as exc:
            logger.warning("Spatial statistics failed: %s", exc)

    # ----- clustering -----
    feature_cols = [
        c for c in ["aci", "ndsi", "adi", "bi", "rms_db"]
        if c in gdf.columns
    ]
    cluster_labels = np.zeros(len(gdf), dtype=int)
    if len(gdf) >= kmeans_k and feature_cols:
        X = gdf[feature_cols].fillna(0).values
        cluster_labels, _ = kmeans_clustering(X, n_clusters=kmeans_k)
    gdf["cluster"] = cluster_labels

    # ----- IDW interpolation -----
    grid_x = grid_y = z = None
    if len(gdf) >= 3:
        try:
            grid_x, grid_y, z = idw_surface(
                gdf, column=index_col, grid_resolution=grid_resolution
            )
        except Exception as exc:
            logger.warning("IDW interpolation failed: %s", exc)

    # ----- exports -----
    output_files: Dict[str, Any] = {}
    if export_formats:
        output_files.update(
            export_all(
                gdf,
                grid_x=grid_x,
                grid_y=grid_y,
                z=z,
                output_dir=output_dir,
                base_name="soundscape",
            )
        )

    # ----- interactive map -----
    if map_center is None:
        map_center = [
            float(gdf.geometry.y.mean()),
            float(gdf.geometry.x.mean()),
        ]

    m = create_soundscape_map(
        gdf,
        center=map_center,
        index_col=index_col,
        cluster_col="cluster",
        output_path=output_dir / "soundscape_map.html",
    )
    output_files["map"] = output_dir / "soundscape_map.html"

    logger.info(
        "Pipeline complete. Outputs: %s",
        [str(v) for v in output_files.values()],
    )

    return {
        "gdf": gdf,
        "hotspots": hotspot_df,
        "clusters": cluster_labels,
        "morans": morans_result,
        "output_files": output_files,
    }
