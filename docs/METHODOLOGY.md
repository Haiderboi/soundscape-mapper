# Scientific Methodology

## Overview

Soundscape Mapper applies *soundscape ecology* principles to quantify and
spatially map acoustic complexity, biodiversity proxies, and noise pollution
across heterogeneous landscapes.

---

## 1. Passive Acoustic Monitoring

Recordings are collected with calibrated recorders (e.g., AudioMoth, Song
Meter SM4) at geo-tagged sites.  All recordings are converted to mono WAV at
22 050 Hz before analysis.

---

## 2. Ecoacoustic Indices

### 2.1 Acoustic Complexity Index (ACI)

Pieretti et al. (2011) proposed ACI to quantify the variability of intensities
within a set of time intervals.  For each frequency band *k* and temporal
sub-interval *j*:

```
ACI_j_k = Σ|I_t+1 − I_t| / Σ I_t    (summed over t in j)
ACI     = ΣΣ ACI_j_k
```

High ACI values indicate biologically rich soundscapes with rapid intensity
fluctuations (birds, insects).

### 2.2 Normalised Difference Soundscape Index (NDSI)

NDSI (Joo et al. 2011) contrasts the biophony band (2–11 kHz) with the
anthrophony band (1–2 kHz):

```
NDSI = (Biophony − Anthrophony) / (Biophony + Anthrophony)
```

Values approaching +1 indicate natural soundscapes; values approaching −1
indicate human-dominated soundscapes.

### 2.3 Acoustic Diversity Index (ADI)

ADI (Villanueva-Rivera et al. 2011) applies Shannon entropy to the
presence/absence of sound in equally-spaced frequency bands:

```
ADI = −Σ p_i · ln(p_i)
```

where *p_i* is the fraction of time frames in band *i* with energy above a dB
threshold.

### 2.4 Bioacoustic Index (BI)

BI (Boelman et al. 2007) measures the area under the mean dB spectrum curve
within the biophony frequency band (2–8 kHz), providing a proxy for the
total biological acoustic activity.

### 2.5 RMS dB

Overall RMS level in decibels, serving as a simple noise-level metric:

```
RMS_dB = 10 · log₁₀(mean(y²))
```

---

## 3. Spatial Analysis

### 3.1 Spatial Autocorrelation – Global Moran's I

Moran's I detects whether similar index values cluster spatially:

```
I = (n / S₀) · (Σᵢ Σⱼ wᵢⱼ zᵢ zⱼ) / (Σᵢ zᵢ²)
```

where *z* are mean-centred values and *w* are row-standardised KNN weights.

Positive I → clustering; negative I → dispersion; I ≈ 0 → spatial randomness.

### 3.2 Hotspot Detection – Getis-Ord Gi*

Gi* identifies statistically significant spatial clusters (hotspots / cold
spots) at the local level using a z-score-based test:

```
Gᵢ* = (Σⱼ wᵢⱼ xⱼ − X̄ Σⱼ wᵢⱼ) / S · √[(n Σⱼ wᵢⱼ² − (Σⱼ wᵢⱼ)²) / (n − 1)]
```

Positive z-scores indicate hotspots (noise pollution); negative z-scores
indicate cold spots (quiet, biodiversity-rich areas).

### 3.3 Spatial Interpolation

**Inverse Distance Weighting (IDW)**:

```
ẑ(x₀) = (Σᵢ wᵢ · zᵢ) / Σᵢ wᵢ    with wᵢ = 1 / d(x₀, xᵢ)^p
```

**Ordinary Kriging**: minimum-variance linear interpolation using a fitted
variogram model (spherical / exponential / Gaussian), providing both the
prediction surface and the associated variance.

---

## 4. Machine Learning

### 4.1 Unsupervised Clustering

- **K-means**: partitions the acoustic feature space into *k* groups by
  minimising within-cluster variance.
- **DBSCAN**: density-based clustering that automatically identifies the number
  of clusters and flags noise points.
- **Hierarchical clustering**: agglomerative Ward linkage for dendrogram-based
  exploration.

Features used: ACI, NDSI, ADI, BI, RMS dB (standardised with z-score scaling).

### 4.2 Soundscape Classification

Random Forest classifier trained on labelled sites to predict soundscape type
(urban, forest, riverbank, agricultural, etc.).  Cross-validation with 5 folds.

### 4.3 Anomaly Detection

Isolation Forest identifies recordings with unusual acoustic signatures
(equipment malfunction, rare ecological events, sudden anthropogenic noise).

---

## 5. References

* Boelman, N.T. et al. (2007) *Multi-trophic invasion resistance in Hawaii: bioacoustics, field surveys, and airborne remote sensing.* Ecological Applications.
* Joo, W. et al. (2011) *Acoustic index performance reflects the general habitat quality and biodiversity of a subtropical forest.* PLOS ONE.
* Pieretti, N. et al. (2011) *A new methodology to infer the singing activity of an avian community: the Acoustic Complexity Index (ACI).* Ecological Indicators.
* Villanueva-Rivera, L.J. et al. (2011) *A primer of acoustic analysis for landscape ecologists.* Landscape Ecology.
