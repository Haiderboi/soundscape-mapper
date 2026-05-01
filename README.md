# 🌍 Soundscape Mapper

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Soundscape Mapper** is an integrated data-science and Geographic Information
System (GIS) platform for mapping and analysing environmental soundscapes using
passive acoustic recordings.  It extracts ecoacoustic indices from audio data
collected across diverse habitats — urban markets, riverbanks, forests and
mountainous areas of the Swat Valley, Khyber Pakhtunkhwa, Pakistan — and
generates interactive GIS visualisations.

---

## ✨ Features

| Domain | Capabilities |
|---|---|
| **Audio Processing** | Load WAV/MP3/FLAC, resample, normalise, silence removal, MFCC, spectral features |
| **Ecoacoustic Indices** | ACI, NDSI, ADI, BI, RMS dB, biophony, geophony, anthrophony |
| **Spatial Analysis** | Moran's I, Getis-Ord Gi* hotspots, IDW & Kriging interpolation |
| **Machine Learning** | K-means, DBSCAN, hierarchical clustering; Random Forest classification; Isolation Forest anomaly detection |
| **GIS Visualisation** | Interactive Folium maps, heatmaps, marker clusters, choropleth layers |
| **Data Export** | GeoJSON, ESRI Shapefile, GeoTIFF rasters |

---

## 📦 Project Structure

```
soundscape-mapper/
├── config/
│   └── config.yaml            ← Project-wide settings
├── data/
│   ├── raw_audio/             ← WAV / MP3 / FLAC recordings
│   ├── processed/             ← Computed indices and arrays
│   ├── spatial_data/          ← Shapefiles and GIS layers
│   └── sample_recordings/     ← Example audio files
├── docs/                      ← Detailed documentation
├── notebooks/                 ← Jupyter workflows
├── outputs/                   ← Generated maps and exports
├── scripts/                   ← CLI tools
├── src/
│   ├── audio_processing/      ← preprocessing · feature_extraction · indices
│   ├── spatial_analysis/      ← spatial_stats · interpolation · hotspot_analysis
│   ├── ml_clustering/         ← clustering · classification
│   ├── visualization/         ← maps · plots · gis_export
│   ├── utils/                 ← data_loader · validators · logger
│   └── pipeline.py            ← End-to-end orchestration
└── tests/                     ← pytest test suite
```

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/Haiderboi/soundscape-mapper.git
cd soundscape-mapper
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run the batch processing pipeline

```bash
python scripts/process_batch.py \
  --audio-dir data/raw_audio \
  --locations data/sample_recordings/locations.csv \
  --output outputs
```

### 4. Generate an interactive map

```bash
python scripts/generate_maps.py \
  --geojson outputs/soundscape.geojson \
  --output outputs/map.html
```

### 5. Explore the notebooks

```bash
jupyter notebook notebooks/01_audio_exploration.ipynb
```

---

## 🛠 Command-Line Tools

| Script | Purpose |
|---|---|
| `scripts/process_batch.py` | Batch audio processing → indices → GeoDataFrame |
| `scripts/generate_maps.py` | Folium interactive map from a GeoJSON file |
| `scripts/export_data.py` | Re-export data to GeoJSON / Shapefile |

---

## 🧪 Tests

```bash
pytest tests/ -v
```

---

## 📚 Documentation

| Document | Description |
|---|---|
| [docs/INSTALLATION.md](docs/INSTALLATION.md) | Detailed installation guide |
| [docs/USAGE.md](docs/USAGE.md) | Usage examples and workflows |
| [docs/API.md](docs/API.md) | API reference |
| [docs/METHODOLOGY.md](docs/METHODOLOGY.md) | Scientific methodology |

---

## 📄 License

MIT License – see [LICENSE](LICENSE).
