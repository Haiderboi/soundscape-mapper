# API Reference

## `src.audio_processing.preprocessing`

### `load_audio(file_path, sample_rate, mono, duration, offset)`
Load an audio file and return `(y, sr)`.

### `resample(y, orig_sr, target_sr)`
Resample a waveform.

### `normalize(y)`
Peak-normalise to `[-1, 1]`.

### `remove_silence(y, sr, top_db, frame_length, hop_length)`
Remove silent intervals.

### `apply_window(y, window_type)`
Multiply by a window function.

### `preprocess_audio(file_path, sample_rate, mono, duration, normalise, silence_removal, top_db)`
Full preprocessing pipeline.

---

## `src.audio_processing.feature_extraction`

### `extract_mfcc(y, sr, n_mfcc, n_fft, hop_length)` → `ndarray (n_mfcc, T)`

### `extract_spectral_centroid(y, sr, ...)` → `ndarray (1, T)`

### `extract_spectral_bandwidth(y, sr, ...)` → `ndarray (1, T)`

### `extract_spectral_rolloff(y, sr, roll_percent, ...)` → `ndarray (1, T)`

### `extract_zero_crossing_rate(y, hop_length)` → `ndarray (1, T)`

### `extract_rms_energy(y, hop_length, frame_length)` → `ndarray (1, T)`

### `extract_chroma(y, sr, ...)` → `ndarray (12, T)`

### `extract_mel_spectrogram(y, sr, n_mels, ...)` → `ndarray (n_mels, T)`

### `extract_all_features(y, sr, ...)` → `dict`
Aggregate dictionary of all feature arrays and scalars.

---

## `src.audio_processing.indices`

### `compute_aci(y, sr, n_fft, hop_length, j_step)` → `float`
Acoustic Complexity Index.

### `compute_ndsi(y, sr, ...)` → `dict {ndsi, biophony, anthrophony}`
Normalised Difference Soundscape Index.

### `compute_adi(y, sr, max_freq, db_threshold, freq_step, ...)` → `float`
Acoustic Diversity Index.

### `compute_bi(y, sr, min_freq, max_freq, ...)` → `float`
Bioacoustic Index.

### `compute_rms_db(y, ...)` → `float`
RMS level in dB.

### `compute_geophony(y, sr, geo_min, geo_max, ...)` → `float`
Proxy for geophony energy.

### `compute_all_indices(y, sr, n_fft, hop_length)` → `dict`
All indices in a single call.

---

## `src.spatial_analysis.spatial_stats`

### `global_morans_i(gdf, column, k)` → `dict {I, E_I, z_score, p_value}`

### `local_morans_i(gdf, column, k, alpha)` → `DataFrame`

### `spatial_lag(gdf, column, k)` → `Series`

---

## `src.spatial_analysis.interpolation`

### `idw_interpolate(x, y, values, grid_x, grid_y, power, epsilon)` → `ndarray`

### `idw_surface(gdf, column, grid_resolution, power)` → `(grid_x, grid_y, z)`

### `kriging_surface(gdf, column, grid_resolution, variogram_model, ...)` → `(grid_x, grid_y, z, variance)`

---

## `src.spatial_analysis.hotspot_analysis`

### `getis_ord_gi_star(gdf, column, k, alpha)` → `DataFrame`

### `classify_hotspots(hotspot_df, z_col, p_col, alpha, confidence_levels)` → `DataFrame`

---

## `src.ml_clustering.clustering`

### `kmeans_clustering(X, n_clusters, random_state, n_init)` → `(labels, model)`

### `dbscan_clustering(X, eps, min_samples)` → `(labels, model)`

### `hierarchical_clustering(X, n_clusters, linkage)` → `(labels, model)`

### `silhouette_analysis(X, k_range, random_state)` → `DataFrame`

---

## `src.ml_clustering.classification`

### `train_soundscape_classifier(X, y, test_size, random_state, cv_folds, n_estimators)` → `dict`

### `predict_soundscape_type(model, scaler, encoder, X)` → `ndarray of str`

### `detect_anomalies(X, contamination, random_state)` → `ndarray (1 or -1)`

### `feature_importance_analysis(model, feature_names, top_n)` → `DataFrame`

---

## `src.visualization.maps`

### `create_base_map(center, zoom_start, tiles)` → `folium.Map`

### `add_soundscape_markers(m, gdf, label_col, color_col, popup_cols, radius)` → `folium.Map`

### `add_heatmap(m, gdf, value_col, radius, blur, min_opacity)` → `folium.Map`

### `add_marker_cluster(m, gdf, label_col)` → `folium.Map`

### `save_map(m, output_path)` → `Path`

### `create_soundscape_map(gdf, center, index_col, cluster_col, output_path, zoom_start)` → `folium.Map`

---

## `src.visualization.gis_export`

### `export_geojson(gdf, output_path, metadata)` → `Path`

### `export_shapefile(gdf, output_path)` → `Path`

### `export_geotiff(grid_x, grid_y, z, output_path, crs, nodata)` → `Path`

### `export_all(gdf, grid_x, grid_y, z, output_dir, base_name, crs, metadata)` → `dict`

---

## `src.pipeline`

### `process_single_file(audio_path, latitude, longitude, sample_rate, duration)` → `dict`

### `run_pipeline(audio_dir, location_csv, output_dir, ...)` → `dict`
