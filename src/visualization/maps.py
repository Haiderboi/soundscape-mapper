"""
visualization/maps.py
-----------------------
Interactive Folium maps: point markers, heatmaps, choropleth layers and
marker clustering for soundscape data.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import folium
import folium.plugins as plugins
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Colour palette for soundscape cluster types
CLUSTER_COLOURS = [
    "red", "blue", "green", "purple", "orange",
    "darkred", "lightred", "beige", "darkblue", "darkgreen",
]


def create_base_map(
    center: List[float],
    zoom_start: int = 10,
    tiles: str = "OpenStreetMap",
) -> folium.Map:
    """Create a base Folium map centred at *center*.

    Parameters
    ----------
    center:
        ``[lat, lon]`` of the map centre.
    zoom_start:
        Initial zoom level.
    tiles:
        Tile provider name or URL.

    Returns
    -------
    folium.Map
    """
    m = folium.Map(location=center, zoom_start=zoom_start, tiles=tiles)
    logger.debug("Base map created at %s zoom=%d", center, zoom_start)
    return m


def add_soundscape_markers(
    m: folium.Map,
    gdf,
    label_col: Optional[str] = None,
    color_col: Optional[str] = None,
    popup_cols: Optional[List[str]] = None,
    radius: int = 8,
) -> folium.Map:
    """Add circle markers for each soundscape recording location.

    Parameters
    ----------
    m:
        Folium map to add markers to.
    gdf:
        GeoDataFrame with point geometry.
    label_col:
        Column to use as tooltip label.
    color_col:
        Column whose unique values determine marker colour.
    popup_cols:
        Columns to include in the click popup.
    radius:
        Circle radius in pixels.

    Returns
    -------
    folium.Map
    """
    if color_col and color_col in gdf.columns:
        unique_vals = gdf[color_col].unique()
        colour_map = {
            v: CLUSTER_COLOURS[i % len(CLUSTER_COLOURS)]
            for i, v in enumerate(unique_vals)
        }
    else:
        colour_map = {}

    for _, row in gdf.iterrows():
        lat = row.geometry.y
        lon = row.geometry.x

        colour = colour_map.get(row.get(color_col, None), "blue") \
            if color_col else "blue"

        popup_text = ""
        if popup_cols:
            lines = [
                f"<b>{c}:</b> {row[c]}"
                for c in popup_cols
                if c in row.index
            ]
            popup_text = "<br>".join(lines)

        tooltip_text = str(row[label_col]) if label_col and label_col in row.index else ""

        folium.CircleMarker(
            location=[lat, lon],
            radius=radius,
            color=colour,
            fill=True,
            fill_color=colour,
            fill_opacity=0.7,
            tooltip=tooltip_text,
            popup=folium.Popup(popup_text, max_width=300) if popup_text else None,
        ).add_to(m)

    logger.info("Added %d soundscape markers", len(gdf))
    return m


def add_heatmap(
    m: folium.Map,
    gdf,
    value_col: str,
    radius: int = 20,
    blur: int = 15,
    min_opacity: float = 0.3,
) -> folium.Map:
    """Add a heatmap layer to a Folium map.

    Parameters
    ----------
    m:
        Base Folium map.
    gdf:
        GeoDataFrame with point geometry.
    value_col:
        Column providing the intensity values.
    radius:
        Heatmap point radius.
    blur:
        Blur radius.
    min_opacity:
        Minimum opacity of the heatmap.

    Returns
    -------
    folium.Map
    """
    values = gdf[value_col].fillna(0).values
    # normalise to [0, 1] for the heatmap intensity
    v_min, v_max = values.min(), values.max()
    if v_max > v_min:
        norm = (values - v_min) / (v_max - v_min)
    else:
        norm = np.zeros_like(values)

    heat_data = [
        [row.geometry.y, row.geometry.x, float(w)]
        for row, w in zip(gdf.itertuples(), norm)
    ]
    plugins.HeatMap(
        heat_data,
        radius=radius,
        blur=blur,
        min_opacity=min_opacity,
    ).add_to(m)

    logger.info("Heatmap layer added for '%s'", value_col)
    return m


def add_marker_cluster(
    m: folium.Map,
    gdf,
    label_col: Optional[str] = None,
) -> folium.Map:
    """Add a marker cluster layer.

    Returns
    -------
    folium.Map
    """
    mc = plugins.MarkerCluster()
    for _, row in gdf.iterrows():
        lat, lon = row.geometry.y, row.geometry.x
        label = str(row[label_col]) if label_col and label_col in row.index else ""
        folium.Marker(location=[lat, lon], tooltip=label).add_to(mc)
    mc.add_to(m)
    logger.info("Marker cluster with %d points added", len(gdf))
    return m


def add_choropleth(
    m: folium.Map,
    gdf,
    value_col: str,
    key_on: str = "feature.id",
    fill_color: str = "YlOrRd",
    legend_name: Optional[str] = None,
) -> folium.Map:
    """Add a choropleth layer from a polygon GeoDataFrame.

    Returns
    -------
    folium.Map
    """
    legend_name = legend_name or value_col
    folium.Choropleth(
        geo_data=gdf.__geo_interface__,
        data=gdf[[value_col]].reset_index(),
        columns=["index", value_col],
        key_on=key_on,
        fill_color=fill_color,
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name=legend_name,
    ).add_to(m)
    return m


def save_map(m: folium.Map, output_path: str | Path) -> Path:
    """Save a Folium map to an HTML file.

    Returns
    -------
    Path
        Path of the saved file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(output_path))
    logger.info("Map saved to %s", output_path)
    return output_path


def create_soundscape_map(
    gdf,
    center: List[float],
    index_col: str = "aci",
    cluster_col: Optional[str] = None,
    output_path: Optional[str | Path] = None,
    zoom_start: int = 10,
) -> folium.Map:
    """End-to-end helper: create a map with markers + heatmap layer.

    Returns
    -------
    folium.Map
    """
    m = create_base_map(center, zoom_start=zoom_start)

    popup_cols = [c for c in gdf.columns if c != "geometry"]
    m = add_soundscape_markers(
        m, gdf,
        label_col=index_col,
        color_col=cluster_col,
        popup_cols=popup_cols[:6],
    )

    if index_col in gdf.columns:
        m = add_heatmap(m, gdf, value_col=index_col)

    folium.LayerControl().add_to(m)

    if output_path:
        save_map(m, output_path)

    return m
