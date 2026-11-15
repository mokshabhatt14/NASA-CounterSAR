# CounterSAR

AI-powered environmental anomaly detection using NISAR and historical Sentinel-1 SAR observations.

## Project Structure

- data/raw: Original satellite and environmental datasets.
- data/processed: Preprocessed datasets.
- results: Model outputs and analysis.
- app: Interactive application.

## Data Sources

- Sentinel-1 SAR
- NISAR
- CHIRPS Rainfall
- NASA SMAP Soil Moisture
- Copernicus DEM
- ESA WorldCover

Large datasets will be shared through Google Drive rather than GitHub.
## NISAR availability
- NISAR L-band GCOV exists over Kuttanad from Nov 2025 to Oct 2026, with a gap from Feb to May 2026.
- 22 PROVISIONAL scenes (Jun to Oct 2026) and 5 BETA pre-calibration scenes (Nov 2025 to Jan 2026).
- Chosen orbit: ascending track 113, frame 006 (mode 4005, HH+HV).
- Role: 2026 demonstration layer, not a seasonal baseline. Sentinel-1 is the baseline.
## Sentinel-1 data (Moksha)
- `data/processed/s1_cube.nc`: Sentinel-1 GRD (descending), 12-day composites, Jul-Aug 2018
- `data/processed/s1_table.parquet`: long table with columns time, y, x, vv_db, vh_db, rainfall, elevation
- Grid: EPSG:32643, 100 m, pixel centres, dims (time, y, x), y ascending, 223 x 222 (snap to NISAR grid pending)
- Variables: vv_db, vh_db (dB), rainfall (mm per 12-day window, CHIRPS), elevation (m, Copernicus GLO30)

## Rebuild the table
```bash
pip install -r app/requirements.txt
python src/clean_table.py
```

## Data notes and limitations
- Dual-pol only (VV+VH): use VH/VV and RVI, no full decompositions.
- Clip VV/VH to about -30 to +5 dB before modelling (outliers from water and buildings).
- Rainfall NaNs over water were filled with the nearest valid CHIRPS value.
- NISAR is a 2026 demonstration layer only. C-band and L-band values are not interchangeable.
## Anomaly card fields
- Rainfall: mm from CHIRPS. The accumulation window is `null` until verified against the pipeline.
- Soil moisture uses the ERA5-Land `swvl1` estimate (top 0-7 cm), daily averaged in m3/m3 from hourly values. This is model reanalysis context on a roughly 9 km grid, not a direct satellite measurement. The SMAP SPL3SMP_E Version 6 values remain withheld because their Kuttanad retrievals did not pass recommended-quality flags.
- Crop season is estimated from Kerala Agricultural University's typical Kuttanad calendar (Additional Crop: May-September; Puncha: October-March). This is contextual calendar information, not field-specific crop-stage observation: https://www.celkau.in/crops/cereals/rice/climate.php
- Sensor: Sentinel-1 C-band, descending orbit, for all 2018 dates.
- Confidence: 0-1 score of how strongly the evidence supports the verdict. It is not a probability.
- `class_pixel_counts`: `expected` is `null` (total valid pixels not stored). Each pixel is 100 m x 100 m (10,000 m2).
