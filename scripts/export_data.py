"""
scripts/export_data.py
------------------------
Export soundscape results to GeoJSON, Shapefile and/or GeoTIFF.

Usage
-----
::

    python scripts/export_data.py \\
        --geojson outputs/soundscape.geojson \\
        --output-dir exports \\
        --formats geojson shapefile

"""

from __future__ import annotations

import sys
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.data_loader import load_geojson
from src.utils.logger import setup_logger
from src.visualization.gis_export import export_geojson, export_shapefile


@click.command()
@click.option(
    "--geojson", "-g",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to the source GeoJSON file.",
)
@click.option(
    "--output-dir", "-o",
    default="exports",
    show_default=True,
    type=click.Path(file_okay=False),
    help="Directory to write exported files.",
)
@click.option(
    "--name", "-n",
    default="soundscape",
    show_default=True,
    help="Base filename (without extension) for exported files.",
)
@click.option(
    "--formats",
    "-f",
    multiple=True,
    default=["geojson", "shapefile"],
    show_default=True,
    type=click.Choice(["geojson", "shapefile"], case_sensitive=False),
    help="Export formats.  Repeat to select multiple.",
)
@click.option(
    "--log-level",
    default="INFO",
    show_default=True,
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
)
def main(
    geojson: str,
    output_dir: str,
    name: str,
    formats: tuple,
    log_level: str,
) -> None:
    """Export soundscape GIS data to multiple formats."""
    setup_logger(level=log_level)
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    click.echo(f"Loading GeoJSON from: {geojson}")
    gdf = load_geojson(geojson)

    for fmt in formats:
        if fmt == "geojson":
            path = export_geojson(gdf, output_dir_path / f"{name}.geojson")
            click.echo(f"✅ GeoJSON written: {path}")
        elif fmt == "shapefile":
            path = export_shapefile(gdf, output_dir_path / f"{name}.shp")
            click.echo(f"✅ Shapefile written: {path}")

    click.echo(f"\nAll exports written to: {output_dir}")


if __name__ == "__main__":
    main()
