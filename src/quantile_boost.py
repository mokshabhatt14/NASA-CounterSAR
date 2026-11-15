# quantile_boost.py
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
import xarray as xr
from lightgbm import LGBMRegressor
from skeleton import ExpectationModel


class QuantileBoostModel(ExpectationModel):
    """
    Step 3: Trains quantile gradient boosting on context features to output mean and sigma (dB).
    Automatically adapts to all feature bands inside Moksha's features.nc.
    """
    def __init__(
        self,
        feature_names: Optional[List[str]] = None,
        n_estimators: int = 100,
        learning_rate: float = 0.08,
        max_depth: int = 6,
        max_train_samples: int = 120_000,
        eps: float = 0.25,
        random_state: int = 42,
    ):
        self.feature_names = feature_names
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.max_train_samples = max_train_samples
        self.eps = eps
        self.random_state = random_state

        self.model_q16: Optional[LGBMRegressor] = None
        self.model_q50: Optional[LGBMRegressor] = None
        self.model_q84: Optional[LGBMRegressor] = None
        self.used_features_: List[str] = []

    def _build_feature_matrix(
        self, target_dates: pd.DatetimeIndex, features: xr.Dataset, n_y: int, n_x: int
    ) -> Tuple[np.ndarray, List[str]]:
        n_t = len(target_dates)
        columns = []
        col_names = []

        # Cyclical day-of-year features
        doys = target_dates.dayofyear.values.astype(np.float32)
        sin_doy = np.broadcast_to(np.sin(2 * np.pi * doys / 365.0)[:, None, None], (n_t, n_y, n_x))
        cos_doy = np.broadcast_to(np.cos(2 * np.pi * doys / 365.0)[:, None, None], (n_t, n_y, n_x))
        columns.extend([sin_doy.reshape(-1), cos_doy.reshape(-1)])
        col_names.extend(["sin_doy", "cos_doy"])

        ds_aligned = (
            features.sel(time=target_dates, method="nearest")
            if "time" in features.dims
            else features
        )
        candidate_vars = self.feature_names or list(ds_aligned.data_vars)

        for feat in candidate_vars:
            if feat not in ds_aligned.data_vars:
                continue
            da = ds_aligned[feat]
            if "time" in da.dims:
                da = da.transpose("time", "y", "x")
                arr = da.values.astype(np.float32)
            else:
                da = da.transpose("y", "x")
                arr = np.broadcast_to(da.values.astype(np.float32)[None, :, :], (n_t, n_y, n_x))
            columns.append(arr.reshape(-1))
            col_names.append(feat)

        return np.column_stack(columns), col_names

    def fit(self, sar_cube: xr.DataArray, features: Optional[xr.Dataset] = None) -> "QuantileBoostModel":
        if features is None:
            raise ValueError("QuantileBoostModel requires features Dataset (features.nc).")

        if sar_cube.dims != ("time", "y", "x"):
            sar_cube = sar_cube.transpose("time", "y", "x")

        dates = pd.to_datetime(sar_cube["time"].values)
        n_t, n_y, n_x = sar_cube.shape

        X, self.used_features_ = self._build_feature_matrix(dates, features, n_y, n_x)
        y = sar_cube.values.astype(np.float32).reshape(-1)

        valid_mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
        X_valid, y_valid = X[valid_mask], y[valid_mask]

        if len(y_valid) > self.max_train_samples:
            rng = np.random.default_rng(self.random_state)
            idx = rng.choice(len(y_valid), size=self.max_train_samples, replace=False)
            X_valid, y_valid = X_valid[idx], y_valid[idx]

        params = dict(
            objective="quantile",
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=-1,
        )
        self.model_q16 = LGBMRegressor(alpha=0.16, **params).fit(X_valid, y_valid)
        self.model_q50 = LGBMRegressor(alpha=0.50, **params).fit(X_valid, y_valid)
        self.model_q84 = LGBMRegressor(alpha=0.84, **params).fit(X_valid, y_valid)
        return self

    def predict(
        self, target_dates: np.ndarray, features: Optional[xr.Dataset] = None
    ) -> Tuple[xr.DataArray, xr.DataArray]:
        if self.model_q50 is None or features is None:
            raise RuntimeError("Call fit() first and provide 'features' Dataset.")

        target_index = pd.to_datetime(target_dates)
        y_coord, x_coord = features["y"], features["x"]
        n_t, n_y, n_x = len(target_index), len(y_coord), len(x_coord)

        X, _ = self._build_feature_matrix(target_index, features, n_y, n_x)
        X_clean = np.nan_to_num(X, nan=0.0)

        q16 = self.model_q16.predict(X_clean).reshape(n_t, n_y, n_x).astype(np.float32)
        q50 = self.model_q50.predict(X_clean).reshape(n_t, n_y, n_x).astype(np.float32)
        q84 = self.model_q84.predict(X_clean).reshape(n_t, n_y, n_x).astype(np.float32)

        sigma = np.maximum((q84 - q16) / 2.0, self.eps).astype(np.float32)
        coords = {"time": target_index, "y": y_coord, "x": x_coord}

        mean_da = xr.DataArray(q50, coords=coords, dims=("time", "y", "x"), name="expected_mean")
        sigma_da = xr.DataArray(sigma, coords=coords, dims=("time", "y", "x"), name="expected_sigma")
        return mean_da, sigma_da


if __name__ == "__main__":
    print("Testing Step 3 QuantileBoostModel on dummy context features...")
    s1_ds = xr.open_dataset("dummy_s1_cube.nc")
    vv = s1_ds["VV"]
    n_t, n_y, n_x = vv.shape
    rng = np.random.default_rng(42)

    # Create dummy features.nc matching Step 3 specification
    features_ds = xr.Dataset(
        data_vars={
            "rainfall": (("time", "y", "x"), rng.gamma(2.0, 5.0, (n_t, n_y, n_x)).astype(np.float32)),
            "smap_moisture": (("time", "y", "x"), rng.uniform(0.1, 0.45, (n_t, n_y, n_x)).astype(np.float32)),
            "crop_phase": (("time", "y", "x"), rng.integers(0, 4, (n_t, n_y, n_x)).astype(np.float32)),
            "dem": (("y", "x"), rng.normal(1.5, 0.8, (n_y, n_x)).astype(np.float32)),
            "worldcover": (("y", "x"), rng.choice([10, 40, 50, 80], (n_y, n_x)).astype(np.float32)),
        },
        coords=vv.coords,
    )
    features_ds.to_netcdf("dummy_features.nc")

    model = QuantileBoostModel().fit(vv, features_ds)
    eval_dates = vv["time"].sel(time=slice("2018-08-01", "2018-08-31")).values
    mean_pred, sigma_pred = model.predict(eval_dates, features_ds)

    print(f"SUCCESS! Features used: {model.used_features_}")
    print(f"Predicted mean shape: {mean_pred.shape}, sigma shape: {sigma_pred.shape}")