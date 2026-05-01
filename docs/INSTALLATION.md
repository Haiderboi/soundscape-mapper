# Installation Guide

## Prerequisites

| Requirement | Version |
|---|---|
| Python | ≥ 3.9 |
| pip | ≥ 22 |
| GDAL / libgeos | System-level (needed by GeoPandas / Fiona) |

## System Dependencies

### Ubuntu / Debian

```bash
sudo apt-get update
sudo apt-get install -y libgdal-dev libgeos-dev libproj-dev python3-dev
```

### macOS (Homebrew)

```bash
brew install gdal geos proj
```

### Windows

Use [OSGeo4W](https://trac.osgeo.org/osgeo4w/) or install via conda:

```bash
conda install -c conda-forge gdal geopandas
```

## Python Environment

```bash
# Clone the repository
git clone https://github.com/Haiderboi/soundscape-mapper.git
cd soundscape-mapper

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install all dependencies
pip install -r requirements.txt

# (Optional) Install in development mode
pip install -e .
```

## Verify Installation

```python
import librosa
import geopandas
import folium
print("Installation successful!")
```

## Jupyter Kernel

```bash
python -m ipykernel install --user --name soundscape-mapper --display-name "Soundscape Mapper"
jupyter notebook
```

## Troubleshooting

### `ModuleNotFoundError: No module named 'gdal'`

Ensure GDAL is installed at the system level before installing `geopandas`.

### `libsndfile` errors (Windows)

Install via conda: `conda install -c conda-forge libsndfile`

### PyKrige installation

```bash
pip install pykrige
```
