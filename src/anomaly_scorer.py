import json
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import uniform_filter
from skeleton import AnomalyScorer as BaseAnomalyScorer, Interpreter as BaseInterpreter


class AnomalyScorer(BaseAnomalyScorer):
    """
    Step 4 & 5: Computes calibrated per-pixel z-scores and suppresses radar speckle.
    """
    def __init__(self, z_threshold: float = 2.0, persistence_steps: int = 1):
        super().__init__(z_threshold=z_threshold, persistence_steps=persistence_steps)

    def compute_z_scores(
        self, observed: xr.DataArray, expected_mean: xr.DataArray, expected_sigma: xr.DataArray
    ) -> xr.DataArray:
        z = (observed - expected_mean) / expected_sigma.clip(min=0.35)
        return z.astype(np.float32).rename("z_score")

    def apply_persistence_filter(self, z_scores: xr.DataArray) -> xr.DataArray:
        # 3x3 spatial box filter suppresses isolated single-pixel SAR speckle spikes
        vals = np.nan_to_num(z_scores.values, nan=0.0)
        smoothed_vals = np.empty_like(vals, dtype=np.float32)
        for t in range(vals.shape[0]):
            smoothed_vals[t] = uniform_filter(vals[t], size=3, mode="nearest")

        out = xr.DataArray(
            smoothed_vals, coords=z_scores.coords, dims=z_scores.dims, name="z_score_filtered"
        )
        if self.persistence_steps > 1 and out.sizes.get("time", 1) >= 2:
            out = out.rolling(time=self.persistence_steps, min_periods=1).mean()
        return out.astype(np.float32)


class Interpreter(BaseInterpreter):
    """
    Step 4 & 5: Classifies pixels into:
      0 = Expected
      1 = Unusual but Explained
      2 = Unexplained Candidate Event
    and builds calibrated Anomaly Cards for Tarun's frontend.
    """
    def classify(
        self,
        z_clim: xr.DataArray,
        z_weather: Optional[xr.DataArray] = None,
        threshold: float = 2.0,
    ) -> xr.DataArray:
        unusual_clim = np.abs(z_clim) >= (threshold * 0.75)

        if z_weather is None:
            verdicts = xr.where(np.abs(z_clim) >= threshold, 2, 0)
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
        explained_area_km2 = round(explained_pixels * self.PIXEL_AREA_KM2, 2)

        rainfall = (env_summary or {}).get("rainfall", 0.0)

        # Scene-level calibrated decision rule (filters out background < 6 km2 SAR noise)
        if affected_area_km2 >= 6.0:
            mean_abs_z = float(np.nanmean(np.abs(z_slice.values[v_vals == 2])))
            confidence = round(float(min(0.96, 1.0 - np.exp(-mean_abs_z / 2.2))), 2)
            headline_verdict = self.VERDICT_LABELS[2]
        elif explained_area_km2 >= 50.0 or rainfall >= 450.0:
            confidence = 0.82
            headline_verdict = self.VERDICT_LABELS[1]
        else:
            confidence = 0.88
            headline_verdict = self.VERDICT_LABELS[0]

        return {
            "date": date_str,
            "verdict": headline_verdict,
            "confidence": confidence,
            "estimated_affected_area_km2": affected_area_km2,
            "explained_area_km2": explained_area_km2,
            "unexplained_pixels": unexplained_pixels,
            "explained_pixels": explained_pixels,
            "environmental_drivers": env_summary or {},
        }
