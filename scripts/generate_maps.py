"""
scripts/generate_maps.py
--------------------------
Generate interactive Folium HTML maps from previously processed results.

Usage
-----
::

    python scripts/generate_maps.py \\
        --geojson outputs/soundscape.geojson \\
        --output outputs/map.html \\
        --index aci \\
        --center 35.2 72.4

"""

from __future__ import annotations

import sys
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.data_loader import load_geojson
from src.utils.logger import setup_logger
from src.visualization.maps import create_soundscape_map, save_map


@click.command()
@click.option(
    "--geojson", "-g",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to the GeoJSON file produced by process_batch.py.",
)
@click.option(
    "--output", "-o",
    default="outputs/soundscape_map.html",
    show_default=True,
    type=click.Path(dir_okay=False),
    help="Destination HTML file for the Folium map.",
)
@click.option(
    "--index", "-i",
    default="aci",
    show_default=True,
    help="Ecoacoustic index to use for the heatmap layer.",
)
@click.option(
    "--cluster-col", "-c",
    default="cluster",
    show_default=True,
    help="Column used for marker colouring.",
)
@click.option(
    "--center",
    nargs=2,
    type=float,
    default=(35.2, 72.4),
    show_default=True,
    metavar="LAT LON",
    help="Map centre coordinates.",
)
@click.option(
    "--zoom",
    default=10,
    show_default=True,
    type=int,
    help="Initial zoom level.",
)
@click.option(
    "--log-level",
    default="INFO",
    show_default=True,
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
)
def main(
    geojson: str,
    output: str,
    index: str,
    cluster_col: str,
    center: tuple,
    zoom: int,
    log_level: str,
) -> None:
    """Generate an interactive Folium map from a soundscape GeoJSON file."""
    setup_logger(level=log_level)

    click.echo(f"Loading GeoJSON from: {geojson}")
    gdf = load_geojson(geojson)

    click.echo(f"Generating map centred at {center} (zoom={zoom}) ...")
    m = create_soundscape_map(
        gdf,
        center=list(center),
        index_col=index,
        cluster_col=cluster_col if cluster_col in gdf.columns else None,
        output_path=output,
        zoom_start=zoom,
    )

    click.echo(f"✅ Map saved to: {output}")


if __name__ == "__main__":
    main()
