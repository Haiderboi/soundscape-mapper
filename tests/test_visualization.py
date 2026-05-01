"""
tests/test_visualization.py
-----------------------------
Unit tests for visualization modules: maps, plots, and gis_export.

Map / plot rendering tests use mocking or only check that the functions
return the correct object types without raising exceptions.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_gdf(n: int = 10):
    """Create a synthetic GeoDataFrame."""
    import geopandas as gpd
    from shapely.geometry import Point

    rng = np.random.default_rng(7)
    lons = rng.uniform(72.0, 73.0, n)
    lats = rng.uniform(34.5, 35.5, n)
    aci = rng.uniform(100, 1000, n)

    return gpd.GeoDataFrame(
        {"aci": aci, "cluster": rng.integers(0, 3, n)},
        geometry=[Point(lon, lat) for lon, lat in zip(lons, lats)],
        crs="EPSG:4326",
    )


# ---------------------------------------------------------------------------
# Maps tests
# ---------------------------------------------------------------------------

class TestCreateBaseMap:
    def test_returns_folium_map(self):
        import folium
        from src.visualization.maps import create_base_map

        m = create_base_map([35.0, 72.5], zoom_start=10)
        assert isinstance(m, folium.Map)

    def test_location_set(self):
        from src.visualization.maps import create_base_map

        center = [35.2, 72.4]
        m = create_base_map(center, zoom_start=8)
        assert m.location == center


class TestAddSoundscapeMarkers:
    def test_returns_map(self):
        import folium
        from src.visualization.maps import add_soundscape_markers, create_base_map

        gdf = _make_gdf()
        m = create_base_map([35.0, 72.5])
        m2 = add_soundscape_markers(m, gdf, label_col="aci", color_col="cluster")
        assert isinstance(m2, folium.Map)


class TestAddHeatmap:
    def test_returns_map(self):
        import folium
        from src.visualization.maps import add_heatmap, create_base_map

        gdf = _make_gdf()
        m = create_base_map([35.0, 72.5])
        m2 = add_heatmap(m, gdf, value_col="aci")
        assert isinstance(m2, folium.Map)


class TestSaveMap:
    def test_creates_html_file(self):
        from src.visualization.maps import create_base_map, save_map

        m = create_base_map([35.0, 72.5])
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "test_map.html"
            saved = save_map(m, output)
            assert saved.exists()
            assert saved.suffix == ".html"
            assert saved.stat().st_size > 0


class TestCreateSoundscapeMap:
    def test_returns_folium_map(self):
        import folium
        from src.visualization.maps import create_soundscape_map

        gdf = _make_gdf()
        m = create_soundscape_map(gdf, center=[35.0, 72.5])
        assert isinstance(m, folium.Map)

    def test_saves_html(self):
        from src.visualization.maps import create_soundscape_map

        gdf = _make_gdf()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "map.html"
            create_soundscape_map(gdf, center=[35.0, 72.5], output_path=out)
            assert out.exists()


# ---------------------------------------------------------------------------
# GIS export tests
# ---------------------------------------------------------------------------

class TestExportGeoJSON:
    def test_creates_valid_geojson(self):
        from src.visualization.gis_export import export_geojson

        gdf = _make_gdf()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "test.geojson"
            export_geojson(gdf, out)
            assert out.exists()
            with open(out) as f:
                data = json.load(f)
            assert data["type"] == "FeatureCollection"
            assert len(data["features"]) == len(gdf)

    def test_metadata_embedded(self):
        from src.visualization.gis_export import export_geojson

        gdf = _make_gdf()
        meta = {"project": "soundscape-mapper", "version": "1.0"}
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "test_meta.geojson"
            export_geojson(gdf, out, metadata=meta)
            with open(out) as f:
                data = json.load(f)
            assert data.get("metadata") == meta


class TestExportGeoTIFF:
    def test_creates_tif(self):
        pytest.importorskip("rasterio")
        from src.visualization.gis_export import export_geotiff

        grid_x, grid_y = np.meshgrid(
            np.linspace(72, 73, 10), np.linspace(34.5, 35.5, 10)
        )
        z = np.random.rand(10, 10).astype(np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "test.tif"
            export_geotiff(grid_x, grid_y, z, out)
            assert out.exists()
            assert out.stat().st_size > 0


class TestExportShapefile:
    def test_creates_shp(self):
        from src.visualization.gis_export import export_shapefile

        gdf = _make_gdf()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "test.shp"
            result = export_shapefile(gdf, out)
            assert result.exists()
            assert result.suffix == ".shp"


# ---------------------------------------------------------------------------
# Plot tests (non-display: check return type only)
# ---------------------------------------------------------------------------

class TestPlots:
    """Check that plot functions return a Figure without raising."""

    @pytest.fixture(autouse=True)
    def no_display(self, monkeypatch):
        """Use non-interactive matplotlib backend."""
        import matplotlib
        matplotlib.use("Agg")

    def _sine(self):
        sr = 22050
        t = np.linspace(0, 1, sr, endpoint=False)
        return (np.sin(2 * np.pi * 440 * t)).astype(np.float32), sr

    def test_plot_waveform(self):
        import matplotlib.pyplot as plt
        from src.visualization.plots import plot_waveform

        y, sr = self._sine()
        fig = plot_waveform(y, sr)
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_plot_spectrogram(self):
        import matplotlib.pyplot as plt
        from src.visualization.plots import plot_spectrogram

        y, sr = self._sine()
        fig = plot_spectrogram(y, sr)
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_plot_mfcc(self):
        import matplotlib.pyplot as plt
        from src.visualization.plots import plot_mfcc

        y, sr = self._sine()
        fig = plot_mfcc(y, sr)
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_plot_pca(self):
        import matplotlib.pyplot as plt
        from src.visualization.plots import plot_pca

        rng = np.random.default_rng(42)
        X = rng.standard_normal((30, 5))
        labels = rng.integers(0, 3, 30)
        fig = plot_pca(X, labels=labels)
        assert isinstance(fig, plt.Figure)
        plt.close("all")
