# anomaly_scorer.py
import json
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
import xarray as xr
from skeleton import AnomalyScorer as BaseAnomalyScorer, Interpreter as BaseInterpreter
from climatology import ClimatologyModel
from quantile_boost import QuantileBoostModel


class AnomalyScorer(BaseAnomalyScorer):
    """
    Step 4: Computes per-pixel z-scores and applies a temporal persistence filter
    to remove single-date radar speckle noise.
    """
    def __init__(self, z_threshold: float = 2.0, persistence_steps: int = 2):
        super().__init__(z_threshold=z_threshold, persistence_steps=persistence_steps)

    def compute_z_scores(
        self, observed: xr.DataArray, expected_mean: xr.DataArray, expected_sigma: xr.DataArray
    ) -> xr.DataArray:
        z = (observed - expected_mean) / expected_sigma.clip(min=0.25)
        return z.astype(np.float32).rename("z_score")

    def apply_persistence_filter(self, z_scores: xr.DataArray) -> xr.DataArray:
        """
        Requires an anomaly to persist across `persistence_steps` consecutive passes,
        suppressing isolated 1-pass radar speckle spikes.
        """
        if self.persistence_steps <= 1 or z_scores.sizes.get("time", 1) < 2:
            return z_scores

        smoothed = z_scores.rolling(time=self.persistence_steps, min_periods=1).mean()
        same_sign = (
            np.abs(np.sign(z_scores).rolling(time=self.persistence_steps, min_periods=1).mean())
            == 1.0
        )
        filtered = xr.where(same_sign, smoothed, z_scores * 0.5)
        return filtered.astype(np.float32).rename("z_score_filtered")


class Interpreter(BaseInterpreter):
    """
    Step 4: Classifies pixels into the 3 CounterSAR verdicts:
      0 = Expected
      1 = Unusual but Explained
      2 = Unexplained Candidate Event
    and builds Anomaly Card summaries for Tarun's frontend.
    """
    def classify(
        self,
        z_clim: xr.DataArray,
        z_weather: Optional[xr.DataArray] = None,
        threshold: float = 2.0,
    ) -> xr.DataArray:
        unusual_clim = np.abs(z_clim) >= threshold

        if z_weather is None:
            verdicts = xr.where(unusual_clim, 2, 0)
        else:
            unusual_weather = np.abs(z_weather) >= threshold
            verdicts = xr.where(
                ~unusual_clim,
                0,
                xr.where(~unusual_weather, 1, 2),
            )

        return verdicts.astype(np.int8).rename("verdict")

    def build_anomaly_card(
        self,
        date_str: str,
        verdict_slice: xr.DataArray,
        z_slice: xr.DataArray,
        env_summary: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        v_vals = verdict_slice.values
        unexplained_pixels = int(np.sum(v_vals == 2))
        explained_pixels = int(np.sum(v_vals == 1))
        affected_area_km2 = round(unexplained_pixels * self.PIXEL_AREA_KM2, 2)

        if unexplained_pixels > 0:
            mean_abs_z = float(np.nanmean(np.abs(z_slice.values[v_vals == 2])))
            confidence = round(float(min(0.99, 1.0 - np.exp(-mean_abs_z / 2.5))), 2)
            headline_verdict = self.VERDICT_LABELS[2]
        elif explained_pixels > 0:
            confidence = 0.75
            headline_verdict = self.VERDICT_LABELS[1]
        else:
            confidence = 0.90
            headline_verdict = self.VERDICT_LABELS[0]

        return {
            "date": date_str,
            "verdict": headline_verdict,
            "confidence": confidence,
            "estimated_affected_area_km2": affected_area_km2,
            "unexplained_pixels": unexplained_pixels,
            "explained_pixels": explained_pixels,
            "environmental_drivers": env_summary or {},
        }


if __name__ == "__main__":
    print("Testing Step 4 AnomalyScorer & Dual-Baseline Verdicts on 2018 Flood...")
    s1_ds = xr.open_dataset("dummy_s1_cube.nc")
    feat_ds = xr.open_dataset("dummy_features.nc")
    vv = s1_ds["VV"]

    train_mask = (vv["time"] < np.datetime64("2018-08-01")) | (vv["time"] > np.datetime64("2018-08-31"))
    clim_model = ClimatologyModel(window_days=15).fit(vv.sel(time=train_mask))
    weather_model = QuantileBoostModel().fit(vv.sel(time=train_mask), feat_ds.sel(time=train_mask))

    eval_dates = vv["time"].sel(time=slice("2018-08-01", "2018-08-31")).values
    obs = vv.sel(time=eval_dates)

    clim_mean, clim_sigma = clim_model.predict(eval_dates)
    weath_mean, weath_sigma = weather_model.predict(eval_dates, feat_ds)

    scorer = AnomalyScorer(z_threshold=2.0, persistence_steps=2)
    z_clim = scorer.apply_persistence_filter(scorer.compute_z_scores(obs, clim_mean, clim_sigma))
    z_weather = scorer.apply_persistence_filter(scorer.compute_z_scores(obs, weath_mean, weath_sigma))

    interpreter = Interpreter()
    verdicts = interpreter.classify(z_clim, z_weather, threshold=2.0)

    out_ds = xr.Dataset(
        {
            "observed": obs,
            "predicted_clim": clim_mean,
            "predicted_weather": weath_mean,
            "z_clim": z_clim,
            "z_weather": z_weather,
            "verdicts": verdicts,
        },
        attrs={"crs": "EPSG:32643", "units": "dB"},
    )
    out_ds.to_netcdf("dummy_step4_full_outputs.nc")

    cards = []
    for t_val in eval_dates:
        d_str = str(pd.to_datetime(t_val).date())
        env_info = {
            "rainfall_mm": round(float(feat_ds["rainfall"].sel(time=t_val).mean()), 2),
            "soil_moisture": round(float(feat_ds["smap_moisture"].sel(time=t_val).mean()), 3),
        }
        cards.append(interpreter.build_anomaly_card(d_str, verdicts.sel(time=t_val), z_weather.sel(time=t_val), env_info))

    with open("dummy_anomaly_cards.json", "w") as f:
        json.dump(cards, f, indent=2)

    print("SUCCESS! Step 4 AnomalyScorer & Verdicts complete.")
    print("Sample 2018 Flood Anomaly Card for Tarun:", json.dumps(cards[0], indent=2))