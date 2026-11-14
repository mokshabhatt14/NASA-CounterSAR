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