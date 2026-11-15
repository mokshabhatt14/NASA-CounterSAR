import json
import os
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
import xarray as xr

from skeleton import DataSource, Pipeline as BasePipeline
from climatology import ClimatologyModel
from quantile_boost import QuantileBoostModel
from anomaly_scorer import AnomalyScorer, Interpreter


class NetCDFDataSource(DataSource):
    def __init__(self, s1_path: str, features_path: Optional[str] = None):
        self.s1_path = s1_path
        self.features_path = features_path

    def load_sar_cube(self) -> xr.Dataset:
        ds = xr.open_dataset(self.s1_path)
        for v in ds.data_vars:
            if set(("time", "y", "x")).issubset(ds[v].dims):
                ds[v] = ds[v].transpose("time", "y", "x")
        return ds

    def load_context_features(self) -> Optional[xr.Dataset]:
        if not self.features_path or not os.path.exists(self.features_path):
            return None
        return xr.open_dataset(self.features_path)


class CounterSARPipeline(BasePipeline):
    def __init__(
        self,
        data_source: NetCDFDataSource,
        clim_model: ClimatologyModel,
        weather_model: Optional[QuantileBoostModel],
        scorer: AnomalyScorer,
        interpreter: Interpreter,
    ):
        self.data_source = data_source
        self.clim_model = clim_model
        self.weather_model = weather_model
        self.scorer = scorer
        self.interpreter = interpreter
        self.clim_scale_: float = 1.0
        self.weather_scale_: float = 1.0

    @staticmethod
    def calibrate_sigma(observed: xr.DataArray, mean: xr.DataArray, sigma: xr.DataArray) -> float:
        raw_z = ((observed - mean) / sigma.clip(min=0.25)).values
        valid_z = raw_z[np.isfinite(raw_z)]
        if len(valid_z) == 0:
            return 1.0
        med = np.median(valid_z)
        robust_std = float(np.median(np.abs(valid_z - med)) * 1.4826)
        return max(robust_std, 0.5)

    def run(self, flood_cutoff: str = "2018-08-10", out_dir: str = "results") -> Dict[str, Any]:
        os.makedirs(out_dir, exist_ok=True)
        ds = self.data_source.load_sar_cube()
        sar_band = "vv_db" if "vv_db" in ds.data_vars else ("VV" if "VV" in ds.data_vars else list(ds.data_vars)[0])
        sar = ds[sar_band]

        ext_features = self.data_source.load_context_features()
        if ext_features is not None:
            features_ds = ext_features
        else:
            feat_vars = [v for v in ds.data_vars if v not in ("vv_db", "vh_db", "VV", "VH")]
            features_ds = ds[feat_vars] if feat_vars else None

        train_mask = sar["time"] < np.datetime64(flood_cutoff)
        if int(train_mask.sum()) < 2:
            train_mask = xr.ones_like(sar["time"], dtype=bool)

        train_sar = sar.sel(time=train_mask)
        train_dates = train_sar["time"].values
        all_dates = sar["time"].values

        self.clim_model.fit(train_sar)
        clim_train_mu, clim_train_sig = self.clim_model.predict(train_dates)
        self.clim_scale_ = self.calibrate_sigma(train_sar, clim_train_mu, clim_train_sig)

        clim_mu, clim_sig = self.clim_model.predict(all_dates)
        clim_sig_cal = (clim_sig * self.clim_scale_).clip(min=0.25)
        z_clim = self.scorer.apply_persistence_filter(
            self.scorer.compute_z_scores(sar, clim_mu, clim_sig_cal)
        )

        z_weather = None
        weath_mu, weath_sig_cal = clim_mu, clim_sig_cal
        if self.weather_model is not None and features_ds is not None:
            train_feat = features_ds.sel(time=train_mask) if "time" in features_ds.dims else features_ds
            self.weather_model.fit(train_sar, train_feat)
            w_train_mu, w_train_sig = self.weather_model.predict(train_dates, features_ds)
            self.weather_scale_ = self.calibrate_sigma(train_sar, w_train_mu, w_train_sig)

            weath_mu, weath_sig = self.weather_model.predict(all_dates, features_ds)
            weath_sig_cal = (weath_sig * self.weather_scale_).clip(min=0.25)
            z_weather = self.scorer.apply_persistence_filter(
                self.scorer.compute_z_scores(sar, weath_mu, weath_sig_cal)
            )

        verdicts = self.interpreter.classify(z_clim, z_weather, threshold=self.scorer.z_threshold)
        active_z = z_weather if z_weather is not None else z_clim

        cards = []
        for t_val in all_dates:
            d_str = str(pd.to_datetime(t_val).date())
            env_info = {}
            if features_ds is not None:
                for var in features_ds.data_vars:
                    da = features_ds[var]
                    val = da.sel(time=t_val, method="nearest") if "time" in da.dims else da
                    env_info[var] = round(float(val.mean(skipna=True)), 3)
            cards.append(
                self.interpreter.build_anomaly_card(
                    d_str, verdicts.sel(time=t_val), active_z.sel(time=t_val), env_info
                )
            )

        out_vars = {
            "observed_vv": sar,
            "predicted": weath_mu,
            "predicted_clim": clim_mu,
            "expected_sigma": weath_sig_cal,
            "z_clim": z_clim,
            "z_scores": active_z,
            "verdicts": verdicts,
        }
        if "vh_db" in ds.data_vars:
            out_vars["observed_vh"] = ds["vh_db"]

        out_ds = xr.Dataset(
            out_vars,
            attrs={
                "crs": "EPSG:32643",
                "units": "dB",
                "clim_uncertainty_scale": float(self.clim_scale_),
                "weather_uncertainty_scale": float(self.weather_scale_),
            },
        )
        nc_out_path = os.path.join(out_dir, "countersar_outputs.nc")
        cards_out_path = os.path.join(out_dir, "anomaly_cards.json")
        out_ds.to_netcdf(nc_out_path)
        with open(cards_out_path, "w") as f:
            json.dump(cards, f, indent=2)

        return {"dataset": out_ds, "cards": cards, "nc_path": nc_out_path, "cards_path": cards_out_path}


if __name__ == "__main__":
    s1_path = "data/processed/s1_cube.nc"
    feat_path = "data/processed/features.nc" if os.path.exists("data/processed/features.nc") else None
    print(f"Running CounterSARPipeline on real cube: {s1_path}")
    pipeline = CounterSARPipeline(
        data_source=NetCDFDataSource(s1_path=s1_path, features_path=feat_path),
        clim_model=ClimatologyModel(window_days=45, min_samples=2),
        weather_model=QuantileBoostModel(),
        scorer=AnomalyScorer(z_threshold=2.0, persistence_steps=1),
        interpreter=Interpreter(),
    )
    res = pipeline.run(out_dir="results")
    print("\nSUCCESS! Real 2018 Kerala Flood Anomaly Cards per date:")
    for card in res["cards"]:
        print(
            f"  {card['date']} -> Verdict: {card['verdict']} | "
            f"Affected Area: {card['estimated_affected_area_km2']} km2 | "
            f"Confidence: {card['confidence']} | Env: {card['environmental_drivers']}"
        )
