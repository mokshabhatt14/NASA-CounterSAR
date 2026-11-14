# climatology.py
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr

from skeleton import ExpectationModel


class ClimatologyModel(ExpectationModel):
    """
    Step 2 Working Baseline Class:
    Calculates per-pixel, per day-of-year (DOY) expected mean and spread (sigma) in dB.
    """

    def __init__(
        self, window_days: int = 15, min_samples: int = 3, eps: float = 0.25
    ):
        self.window_days = window_days
        self.min_samples = min_samples
        self.eps = eps  # floor on sigma (dB) to prevent divide-by-zero
        self.train_cube: Optional[xr.DataArray] = None
        self.train_doys: Optional[np.ndarray] = None
        self.fallback_mean: Optional[xr.DataArray] = None
        self.fallback_std: Optional[xr.DataArray] = None

    def fit(
        self, sar_cube: xr.DataArray, features: Optional[xr.Dataset] = None
    ) -> "ClimatologyModel":
        if sar_cube.dims != ("time", "y", "x"):
            sar_cube = sar_cube.transpose("time", "y", "x")

        self.train_cube = sar_cube.astype(np.float32)
        self.train_doys = pd.to_datetime(sar_cube["time"].values).dayofyear.values

        # Global per-pixel fallback if a specific DOY window has too few clear passes
        self.fallback_mean = self.train_cube.mean(dim="time", skipna=True)
        self.fallback_std = self.train_cube.std(dim="time", skipna=True).clip(
            min=self.eps
        )
        return self

    def _doy_window_mask(self, target_doy: int) -> np.ndarray:
        diff = np.abs(self.train_doys - target_doy)
        circular_diff = np.minimum(diff, 365 - diff)
        return circular_diff <= self.window_days

    def predict(
        self, target_dates: np.ndarray, features: Optional[xr.Dataset] = None
    ) -> Tuple[xr.DataArray, xr.DataArray]:
        if self.train_cube is None:
            raise RuntimeError("Call fit() before predict().")

        target_index = pd.to_datetime(target_dates)
        means, sigmas = [], []

        for doy in target_index.dayofyear.values:
            mask = self._doy_window_mask(int(doy))
            subset = self.train_cube.isel(time=mask)

            if subset.sizes["time"] >= self.min_samples:
                valid_count = subset.count(dim="time")
                med = subset.median(dim="time", skipna=True)
                # Robust sigma using MAD * 1.4826 (immune to extreme flood years in training data)
                mad_sigma = (
                    np.abs(subset - med).median(dim="time", skipna=True)
                ) * 1.4826
                mu = xr.where(
                    valid_count >= self.min_samples, med, self.fallback_mean
                )
                sigma = xr.where(
                    valid_count >= self.min_samples, mad_sigma, self.fallback_std
                )
            else:
                mu = self.fallback_mean
                sigma = self.fallback_std

            means.append(mu)
            sigmas.append(sigma.clip(min=self.eps))

        mean_cube = xr.concat(means, dim="time").assign_coords(time=target_index)
        sigma_cube = xr.concat(sigmas, dim="time").assign_coords(time=target_index)
        return mean_cube.astype(np.float32), sigma_cube.astype(np.float32)