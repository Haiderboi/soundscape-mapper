"""
scripts/process_batch.py
--------------------------
Command-line tool for batch audio processing.

Usage
-----
::

    python scripts/process_batch.py \\
        --audio-dir data/raw_audio \\
        --locations data/sample_recordings/locations.csv \\
        --output data/processed \\
        --sample-rate 22050 \\
        --clusters 5

"""

from __future__ import annotations

import sys
from pathlib import Path

import click

# Make sure `src` is importable when the script is run from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import run_pipeline
from src.utils.logger import setup_logger


@click.command()
@click.option(
    "--audio-dir", "-a",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="Directory containing raw audio recordings.",
)
@click.option(
    "--locations", "-l",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="CSV file with columns: file, latitude, longitude.",
)
@click.option(
    "--output", "-o",
    default="outputs",
    show_default=True,
    type=click.Path(file_okay=False),
    help="Output directory for processed results.",
)
@click.option(
    "--sample-rate", "-sr",
    default=22050,
    show_default=True,
    type=int,
    help="Target sample rate in Hz.",
)
@click.option(
    "--clusters", "-k",
    default=5,
    show_default=True,
    type=int,
    help="Number of K-means clusters.",
)
@click.option(
    "--grid-res", "-g",
    default=0.01,
    show_default=True,
    type=float,
    help="Grid resolution (degrees) for IDW interpolation.",
)
@click.option(
    "--log-level",
    default="INFO",
    show_default=True,
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    help="Logging verbosity.",
)
def main(
    audio_dir: str,
    locations: str,
    output: str,
    sample_rate: int,
    clusters: int,
    grid_res: float,
    log_level: str,
) -> None:
    """Batch-process audio recordings and generate soundscape outputs."""
    setup_logger(level=log_level, log_file=Path(output) / "process_batch.log")

    click.echo(f"Processing audio from: {audio_dir}")
    click.echo(f"Location CSV: {locations}")
    click.echo(f"Output directory: {output}")

    results = run_pipeline(
        audio_dir=audio_dir,
        location_csv=locations,
        output_dir=output,
        sample_rate=sample_rate,
        kmeans_k=clusters,
        grid_resolution=grid_res,
    )

    click.echo(
        f"\n✅ Done! Processed {len(results['gdf'])} recordings."
    )
    click.echo(f"   Outputs written to: {output}")
    for name, path in results["output_files"].items():
        click.echo(f"   • {name}: {path}")


if __name__ == "__main__":
    main()
