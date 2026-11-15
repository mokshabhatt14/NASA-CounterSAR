# CounterSAR

**Is it unusual, or just the season?** CounterSAR is a Streamlit explorer for comparing observed Sentinel-1 radar backscatter with an expected baseline over Kuttanad, Kerala. It presents statistical departures as candidate anomalies; it does not confirm floods or other disasters.

## Run the app

From the repository root:

```bash
python -m pip install -r app/requirements.txt
python -m streamlit run app/main.py
```

The app requires `results/countersar_outputs.nc`. It also reads `results/anomaly_cards.json` when available for per-date verdict and environmental context. The app does not run the anomaly-detection pipeline itself.

## App pages

- **Home:** Explains the comparison and the three verdict classes: Expected, Unusual but Explained, and Unexplained Candidate Event.
- **Time Machine:** Selects an acquisition date and plots domain-average observed and predicted backscatter, the expected uncertainty band, and rainfall when available.
- **Daily Anomaly Assessment:** Shows the selected date's verdict, candidate area, confidence, sensor, environmental context, and negative/positive Z-score hypotheses.
- **Spatial Visualization:** Shows a zoomable Kuttanad map with verdict, Z-score, observed-backscatter, or expected-backscatter overlays. Expected and observed backscatter are displayed side by side using the same dB scale.

The map uses OpenStreetMap tiles and does not require an API key. The study-area bounds are 9.30–9.50°N, 76.30–76.50°E.

## Data used by the current app

### Sentinel-1 anomaly output

The app loads `results/countersar_outputs.nc`, currently based on Sentinel-1 C-band observations. It accepts these NetCDF variable names (the first name in each list is preferred):

- Observed backscatter: `observed`, `observed_vv`
- Expected backscatter: `predicted`, `expected_mean`, `expected_mean_vv`
- Z-score: `z_scores`, `z_score_clim`, `z_score`
- Optional: `expected_sigma` (or `expected_sigma_vv`, `sigma`) and `verdicts`

If `verdicts` is absent, the app derives display classes from Z-scores: absolute Z above 2.5 is a candidate event; above 2.0 is unusual but explained; otherwise expected. Each grid cell is treated as 100 m × 100 m when calculating candidate area.

The backend source cube is `data/processed/s1_cube.nc` (VV/VH backscatter, rainfall, and elevation). The app reads the pipeline output rather than opening this source cube directly. The current result covers five Sentinel-1 acquisitions from July–August 2018.

### Environmental context

- **Rainfall:** Read from the result cards or NetCDF context when present. The project rainfall source is CHIRPS. The UI shows the accumulation period only when that metadata is supplied; otherwise it labels the period unspecified.
- **Soil moisture:** The app reads `data/processed/era5_land_soil_moisture_kuttanad.json` when present. It contains daily area means of ERA5-Land `swvl1` (volumetric soil water in the top 0–7 cm), calculated from hourly values and six valid grid cells per date. This is model reanalysis context on an approximately 9 km grid, not a direct satellite measurement. [ERA5-Land dataset](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land)
- **Crop season:** Estimated from the date using typical Kuttanad calendar windows in `config/env_context.json`: Additional Crop from May–September and Puncha from October–March. This is a seasonal estimate, not a field-specific crop stage. [Kerala Agricultural University rice calendar](https://www.celkau.in/crops/cereals/rice/climate.php)
- **SMAP quality check:** The downloaded SMAP SPL3SMP_E files had no recommended-quality retrievals in the study-area bounds, so their soil-moisture values are withheld. `src/process_smap_soil_moisture.py` can regenerate the QC summary if those HDF5 granules are available.

To regenerate the ERA5-Land summary, place the July and August NetCDF downloads in `data/raw/era5_land/` and run:

```bash
python src/process_era5_land_soil_moisture.py
```

The app still runs without this summary, but displays soil moisture as unavailable. New files under `data/raw/` and `data/processed/` are ignored by Git by default. Share the small processed summary with teammates or regenerate it from the source files.

## Verdict interpretation

| Code | Verdict | Meaning |
|---:|---|---|
| 0 | Expected | Radar behavior is consistent with the expected baseline. |
| 1 | Unusual but Explained | A departure is present but has a plausible environmental or seasonal explanation. |
| 2 | Unexplained Candidate Event | A statistical departure worth investigation; it is not a confirmed disaster. |

Negative Z-scores can be consistent with increased specular reflection from open water; positive Z-scores can be consistent with double-bounce scattering from flooded vegetation. These are physical hypotheses, not confirmed causes.

## NISAR status

NISAR is **not currently loaded by the app or the main CounterSAR pipeline**. `src/nisar/nisar_cube.py` is a separate utility for building a 100 m HH/HV cube from NISAR L2 GCOV HDF5 scenes. Sentinel-1 remains the source of the current anomaly assessments; C-band and L-band backscatter are not interchangeable.

## Repository data notes

- `data/processed/s1_cube.nc`: Sentinel-1 source cube, EPSG:32643, 100 m grid, dimensions 223 × 222.
- `data/processed/s1_table.parquet`: tabular form of the Sentinel-1 source cube.
- `results/countersar_outputs.nc`: required app input.
- `results/anomaly_cards.json`: per-date card details used when available.
- Raw and derived data may be shared separately from Git because data files are ignored by `.gitignore`.

The scientific interpretation is limited by the spatial resolution and quality of the input data. Rainfall accumulation metadata may be absent, soil moisture is coarse model context, and crop season is calendar-estimated. Findings should be treated as observed radar deviations with contextual evidence and hypotheses, not disaster declarations.
