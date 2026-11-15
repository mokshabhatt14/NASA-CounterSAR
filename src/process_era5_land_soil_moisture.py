"""Prepare ERA5-Land layer-1 soil-moisture summaries for CounterSAR dates."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "era5_land"
OUT_PATH = ROOT / "data" / "processed" / "era5_land_soil_moisture_kuttanad.json"
DATES = ("2018-07-01", "2018-07-13", "2018-07-25", "2018-08-06", "2018-08-18")
BBOX = {"west": 76.30, "east": 76.50, "south": 9.30, "north": 9.50}


def main() -> None:
    files = sorted(RAW_DIR.glob("*.nc"))
    if not files:
        raise FileNotFoundError(f"No ERA5-Land NetCDF files found in {RAW_DIR}")

    datasets = [xr.open_dataset(path) for path in files]
    try:
        cube = xr.concat(datasets, dim="valid_time").sortby("valid_time")
        if "swvl1" not in cube:
            raise ValueError("ERA5-Land files must contain the swvl1 variable.")
        if not {"valid_time", "latitude", "longitude"}.issubset(cube["swvl1"].dims):
            raise ValueError(f"Unexpected swvl1 dimensions: {cube['swvl1'].dims}")

        layer = cube["swvl1"].sel(
            latitude=slice(BBOX["north"], BBOX["south"]),
            longitude=slice(BBOX["west"], BBOX["east"]),
        )
        if layer.sizes.get("latitude", 0) == 0 or layer.sizes.get("longitude", 0) == 0:
            raise ValueError("The ERA5-Land files do not cover the Kuttanad bounding box.")

        by_date = {}
        for day in DATES:
            selected = layer.sel(valid_time=slice(f"{day}T00:00:00", f"{day}T23:59:59"))
            hourly_count = int(selected.sizes.get("valid_time", 0))
            cell_daily_means = selected.mean("valid_time", skipna=True)
            valid_cells = cell_daily_means.where(np.isfinite(cell_daily_means))
            cell_count = int(valid_cells.count())
            value = float(valid_cells.mean(skipna=True)) if cell_count else None
            by_date[day] = {
                "soil_moisture_m3_m3": round(value, 5) if value is not None else None,
                "hours_in_daily_mean": hourly_count,
                "valid_grid_cells": cell_count,
                "grid_cell_count": int(layer.sizes["latitude"] * layer.sizes["longitude"]),
                "status": (
                    f"ERA5-Land model estimate; daily mean of {hourly_count} hourly values "
                    f"across {cell_count} valid grid cells"
                    if value is not None
                    else "No valid ERA5-Land values for this date"
                ),
            }

        result = {
            "source": "ERA5-Land hourly reanalysis",
            "variable": "swvl1 (volumetric soil water layer 1, 0-7 cm)",
            "units": "m3/m3",
            "temporal_aggregation": "Mean of available hourly values for each UTC date, then mean across valid grid cells",
            "native_grid_spacing_degrees": 0.1,
            "nominal_native_resolution_km": 9,
            "spatial_bounds": BBOX,
            "dates": by_date,
        }
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Wrote {OUT_PATH}")
        for day, item in by_date.items():
            print(f"{day}: {item['soil_moisture_m3_m3']} m3/m3; {item['status']}")
    finally:
        for dataset in datasets:
            dataset.close()


if __name__ == "__main__":
    main()
