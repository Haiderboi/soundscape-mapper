# Usage Guide

## Quick Start

See [README.md](../README.md) for a five-minute quick-start.

---

## Configuration

Edit `config/config.yaml` to customise paths, audio settings, and analysis
parameters.  Key sections:

```yaml
audio:
  sample_rate: 22050    # Hz
  n_fft: 2048
  hop_length: 512

spatial:
  crs: "EPSG:4326"
  interpolation_method: "idw"
  grid_resolution: 0.01   # degrees

ml:
  clustering:
    kmeans_n_clusters: 5
```

---

## Batch Processing Pipeline

```bash
python scripts/process_batch.py \
  --audio-dir data/raw_audio \
  --locations data/sample_recordings/locations.csv \
  --output outputs \
  --sample-rate 22050 \
  --clusters 5
```

The `locations.csv` must contain columns **file**, **latitude**, **longitude**:

```csv
file,latitude,longitude
recording_01.wav,35.22,72.43
recording_02.wav,35.30,72.60
```

---

## Interactive Map Generation

```bash
python scripts/generate_maps.py \
  --geojson outputs/soundscape.geojson \
  --output outputs/map.html \
  --index aci \
  --center 35.2 72.4 \
  --zoom 10
```

Open `outputs/map.html` in any browser.

---

## Data Export

```bash
python scripts/export_data.py \
  --geojson outputs/soundscape.geojson \
  --output-dir exports \
  --formats geojson shapefile
```

---

## Python API

### Process a single recording

```python
from src.audio_processing.preprocessing import preprocess_audio
from src.audio_processing.indices import compute_all_indices

y, sr = preprocess_audio("data/raw_audio/site_01.wav", sample_rate=22050)
indices = compute_all_indices(y, sr)
print(indices)
# {'aci': 342.5, 'ndsi': 0.31, 'biophony': ..., ...}
```

### Run the full pipeline

```python
from src.pipeline import run_pipeline

results = run_pipeline(
    audio_dir="data/raw_audio",
    location_csv="data/sample_recordings/locations.csv",
    output_dir="outputs",
    kmeans_k=5,
)
print(results["gdf"].head())
```

### Spatial statistics

```python
from src.spatial_analysis.spatial_stats import global_morans_i

morans = global_morans_i(results["gdf"], column="aci")
print(f"Moran's I = {morans['I']:.4f}  p = {morans['p_value']:.4f}")
```

### Clustering

```python
from src.ml_clustering.clustering import kmeans_clustering
import numpy as np

X = results["gdf"][["aci", "ndsi", "adi", "bi", "rms_db"]].values
labels, model = kmeans_clustering(X, n_clusters=5)
```

---

## Jupyter Notebooks

| Notebook | Contents |
|---|---|
| `01_audio_exploration.ipynb` | Load audio, spectrograms, feature extraction |
| `02_spatial_analysis.ipynb` | Moran's I, IDW surface, hotspot detection |
| `03_visualization.ipynb` | Interactive Folium maps and static plots |
| `04_ml_clustering.ipynb` | K-means, DBSCAN, PCA, classification |

Start Jupyter with:

```bash
jupyter notebook notebooks/
```
