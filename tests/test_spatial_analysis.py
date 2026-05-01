"""
tests/test_spatial_analysis.py
--------------------------------
Unit tests for spatial_analysis modules: spatial_stats, interpolation,
and hotspot_analysis.

All tests use synthetic GeoDataFrames so that no external data files are
required.
"""

import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.spatial_analysis.interpolation import idw_interpolate, idw_surface
from src.utils.validators import validate_coordinates, ValidationError

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(0)


def _make_gdf(n: int = 20, seed: int = 0):
    """Create a synthetic GeoDataFrame with random lat/lon and ACI values."""
    import geopandas as gpd
    from shapely.geometry import Point

    rng = np.random.default_rng(seed)
    lons = rng.uniform(72.0, 73.0, n)
    lats = rng.uniform(34.5, 35.5, n)
    aci = rng.uniform(100, 1000, n)

    gdf = gpd.GeoDataFrame(
        {"aci": aci, "rms_db": rng.uniform(-40, -10, n)},
        geometry=[Point(lon, lat) for lon, lat in zip(lons, lats)],
        crs="EPSG:4326",
    )
    return gdf


# ---------------------------------------------------------------------------
# IDW Interpolation tests
# ---------------------------------------------------------------------------

class TestIDWInterpolate:
    def test_exact_at_observation_points(self):
        """IDW should return the observed value at each observation point."""
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([0.0, 1.0, 2.0])
        values = np.array([10.0, 20.0, 30.0])
        grid_x, grid_y = np.meshgrid(x, y)
        z = idw_interpolate(x, y, values, grid_x, grid_y, power=2)
        # At observation points the interpolated value should be close to
        # the observed value
        for i in range(len(x)):
            row, col = i, i
            assert abs(z[row, col] - values[i]) < 1.0

    def test_output_shape(self):
        """Output shape should match the meshgrid shape."""
        x = np.array([0.0, 1.0])
        y = np.array([0.0, 1.0])
        values = np.array([5.0, 10.0])
        xi = np.linspace(0, 1, 5)
        yi = np.linspace(0, 1, 5)
        gx, gy = np.meshgrid(xi, yi)
        z = idw_interpolate(x, y, values, gx, gy)
        assert z.shape == gx.shape

    def test_values_in_range(self):
        """Interpolated values should be bounded by observed min/max."""
        rng = np.random.default_rng(1)
        x = rng.uniform(0, 10, 10)
        y = rng.uniform(0, 10, 10)
        values = rng.uniform(0, 100, 10)
        xi = np.linspace(0, 10, 20)
        yi = np.linspace(0, 10, 20)
        gx, gy = np.meshgrid(xi, yi)
        z = idw_interpolate(x, y, values, gx, gy)
        assert z.min() >= values.min() - 1e-6
        assert z.max() <= values.max() + 1e-6


class TestIDWSurface:
    def test_returns_three_arrays(self):
        gdf = _make_gdf()
        result = idw_surface(gdf, column="aci")
        assert len(result) == 3

    def test_surface_shape_consistent(self):
        gdf = _make_gdf()
        gx, gy, z = idw_surface(gdf, column="aci", grid_resolution=0.1)
        assert gx.shape == gy.shape == z.shape

    def test_surface_values_in_range(self):
        gdf = _make_gdf()
        _, _, z = idw_surface(gdf, column="aci")
        assert z.min() >= gdf["aci"].min() - 1e-3
        assert z.max() <= gdf["aci"].max() + 1e-3


# ---------------------------------------------------------------------------
# Validator tests (spatial / coordinate)
# ---------------------------------------------------------------------------

class TestValidateCoordinates:
    def test_valid(self):
        validate_coordinates(35.0, 72.5)  # should not raise

    def test_invalid_latitude_high(self):
        with pytest.raises(ValidationError):
            validate_coordinates(91.0, 0.0)

    def test_invalid_latitude_low(self):
        with pytest.raises(ValidationError):
            validate_coordinates(-91.0, 0.0)

    def test_invalid_longitude_high(self):
        with pytest.raises(ValidationError):
            validate_coordinates(0.0, 181.0)

    def test_invalid_longitude_low(self):
        with pytest.raises(ValidationError):
            validate_coordinates(0.0, -181.0)

    def test_boundary_values(self):
        validate_coordinates(90.0, 180.0)
        validate_coordinates(-90.0, -180.0)


# ---------------------------------------------------------------------------
# Spatial statistics smoke test
# ---------------------------------------------------------------------------

class TestSpatialStats:
    """Smoke-test global Moran's I.  Requires libpysal + esda."""

    def test_global_morans_keys(self):
        pytest.importorskip("esda")
        from src.spatial_analysis.spatial_stats import global_morans_i

        gdf = _make_gdf(n=20)
        result = global_morans_i(gdf, column="aci", k=4)
        assert set(result.keys()) == {"I", "E_I", "z_score", "p_value"}

    def test_global_morans_i_range(self):
        pytest.importorskip("esda")
        from src.spatial_analysis.spatial_stats import global_morans_i

        gdf = _make_gdf(n=20)
        result = global_morans_i(gdf, column="aci", k=4)
        # Moran's I is typically in [-1, 1]
        assert -1.5 <= result["I"] <= 1.5
        assert 0.0 <= result["p_value"] <= 1.0


class TestHotspotAnalysis:
    """Smoke-test Getis-Ord Gi*."""

    def test_hotspot_types(self):
        pytest.importorskip("esda")
        from src.spatial_analysis.hotspot_analysis import getis_ord_gi_star

        gdf = _make_gdf(n=20)
        result = getis_ord_gi_star(gdf, column="aci", k=4)
        assert "hotspot_type" in result.columns
        valid = {"hotspot", "coldspot", "not_significant"}
        assert set(result["hotspot_type"].unique()).issubset(valid)

    def test_output_length(self):
        pytest.importorskip("esda")
        from src.spatial_analysis.hotspot_analysis import getis_ord_gi_star

        gdf = _make_gdf(n=20)
        result = getis_ord_gi_star(gdf, column="aci", k=4)
        assert len(result) == len(gdf)
