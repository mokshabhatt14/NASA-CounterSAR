import os
import numpy as np
import pandas as pd
import xarray as xr
from climatology import ClimatologyModel


def generate_dummy_epsg32643_netcdf(out_path: str = "dummy_s1_cube.nc") -> xr.Dataset:
    """
    Creates a synthetic Sentinel-1 NetCDF cube matching the locked contract:
    EPSG:32643 (100m grid over Kuttanad), dB units, ISO 8601 dates, (time, y, x).
    """
    dates = pd.date_range("2017-01-01", "2020-12-31", freq="12D")
    # EPSG:32643 meter coordinates for Kuttanad (lon 76.30-76.50, lat 9.30-9.50) at 100m step
    x = np.arange(642700, 664700, 100, dtype=np.float64)    # 220 pixels
    y = np.arange(1028400, 1050500, 100, dtype=np.float64)  # 221 pixels

    rng = np.random.default_rng(42)
    doys = dates.dayofyear.values[:, None, None]

    # Simulate normal seasonal radar cycle in dB (-11.5 dB mean +/- 1.5 dB monsoon cycle)
    seasonal_vv = -11.5 + 1.5 * np.sin(2 * np.pi * doys / 365.0)
    vv_data = (seasonal_vv + rng.normal(0.0, 0.7, size=(len(dates), len(y), len(x)))).astype(np.float32)

    ds = xr.Dataset(
        data_vars={
            "VV": (("time", "y", "x"), vv_data, {"units": "dB", "crs": "EPSG:32643"}),
        },
        coords={
            "time": dates,
            "y": ("y", y, {"units": "m", "standard_name": "projection_y_coordinate"}),
            "x": ("x", x, {"units": "m", "standard_name": "projection_x_coordinate"}),
        },
        attrs={"crs": "EPSG:32643", "resolution_m": 100, "study_area": "Kuttanad, Kerala"},
    )

    # Inject synthetic August 2018 Kerala flood (-6.5 dB backscatter drop in central Kuttanad)
    flood_dates = (ds["time"] >= np.datetime64("2018-08-10")) & (ds["time"] <= np.datetime64("2018-08-26"))
    ds["VV"].loc[dict(time=flood_dates, y=slice(1035000, 1045000), x=slice(648000, 658000))] -= 6.5

    ds.to_netcdf(out_path)
    return ds


if __name__ == "__main__":
    print("1. Generating contract-compliant EPSG:32643 NetCDF dummy cube...")
    ds = generate_dummy_epsg32643_netcdf("dummy_s1_cube.nc")

    print("2. Fitting ClimatologyModel (Step 2)...")
    model = ClimatologyModel(window_days=15).fit(ds["VV"])

    # Predict on August 2018 flood window
    eval_dates = ds["time"].sel(time=slice("2018-08-01", "2018-08-31")).values
    expected_mean, expected_sigma = model.predict(eval_dates)
    observed = ds["VV"].sel(time=eval_dates)
    z_clim = (observed - expected_mean) / expected_sigma

    # Save sample predictions + z-scores to NetCDF so Tarun (Person D) can test the swipe viewer
    out_ds = xr.Dataset(
        {
            "observed_vv": observed,
            "expected_mean_vv": expected_mean,
            "expected_sigma_vv": expected_sigma,
            "z_score_clim": z_clim,
        },
        attrs={"crs": "EPSG:32643", "units": "dB"},
    )
    out_ds.to_netcdf("dummy_step2_outputs.nc")

    print(f"SUCCESS! Cube shape (time, y, x): {observed.shape}")
    print(f"2018 Flood peak negative z-score: {float(z_clim.min()):.2f} (strong anomaly detected!)")
    print("Saved 'dummy_step2_outputs.nc' — share this file with Tarun so he can test his map swipe viewer!")
