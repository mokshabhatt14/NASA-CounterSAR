# skeleton.py
"""
CounterSAR - AI/ML & Anomaly Detection Skeleton
Author: Nithish (Person C)
Study Area: Kuttanad, Kerala (lon 76.30 to 76.50, lat 9.30 to 9.50, 100m grid, dB units)
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
import numpy as np
import xarray as xr


class DataSource:
    """Loads processed Sentinel-1, NISAR, and environmental cubes from Moksha & Shreya."""
    def __init__(
        self,
        s1_path: str,
        features_path: Optional[str] = None,
        nisar_path: Optional[str] = None,
    ):
        self.s1_path = s1_path
        self.features_path = features_path
        self.nisar_path = nisar_path

    def load_sar_cube(self) -> xr.Dataset:
        """Loads SAR backscatter cube in dB with shape (time, y, x)."""
        raise NotImplementedError

    def load_context_features(self) -> Optional[xr.Dataset]:
        """Loads rainfall, soil moisture, DEM elevation, land-cover, and crop-phase cube."""
        raise NotImplementedError


class ExpectationModel(ABC):
    """
    Base interface for:
      1. ClimatologyModel (Step 2 seasonal baseline)
      2. QuantileBoostModel (Step 3 weather-aware LightGBM / quantile boosting)
    """
    @abstractmethod
    def fit(self, sar_cube: xr.DataArray, features: Optional[xr.Dataset] = None) -> "ExpectationModel":
        pass

    @abstractmethod
    def predict(
        self, target_dates: np.ndarray, features: Optional[xr.Dataset] = None
    ) -> Tuple[xr.DataArray, xr.DataArray]:
        """Returns (expected_mean_dB, expected_sigma_dB) with shape (time, y, x)."""
        pass


class AnomalyScorer:
    """Computes z-scores and applies temporal persistence filtering."""
    def __init__(self, z_threshold: float = 2.0, persistence_steps: int = 2):
        self.z_threshold = z_threshold
        self.persistence_steps = persistence_steps

    def compute_z_scores(
        self, observed: xr.DataArray, expected_mean: xr.DataArray, expected_sigma: xr.DataArray
    ) -> xr.DataArray:
        """Calculates per-pixel z-score: (observed - expected_mean) / expected_sigma."""
        raise NotImplementedError

    def apply_persistence_filter(self, z_scores: xr.DataArray) -> xr.DataArray:
        """Suppresses single-date speckle noise spikes."""
        raise NotImplementedError


class Interpreter:
    """
    Generates the three-category CounterSAR verdict system and Anomaly Card summaries for Tarun.
    """
    VERDICT_LABELS = {
        0: "Expected",
        1: "Unusual but Explained",
        2: "Unexplained Candidate Event",
    }
    PIXEL_AREA_KM2 = 0.01  # 100m x 100m grid pixel = 0.01 km^2

    def classify(
        self,
        z_clim: xr.DataArray,
        z_weather: Optional[xr.DataArray] = None,
        threshold: float = 2.0,
    ) -> xr.DataArray:
        """Returns integer verdict map (0, 1, 2) with shape (time, y, x)."""
        raise NotImplementedError

    def build_anomaly_card(
        self,
        date_str: str,
        verdict_slice: xr.DataArray,
        z_slice: xr.DataArray,
        env_summary: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Formats Anomaly Card payload for Tarun's frontend:
        verdict, confidence, estimated_affected_area_km2, and environmental drivers.
        """
        raise NotImplementedError


class Pipeline:
    """End-to-end CounterSAR pipeline connecting data, models, scoring, and verdicts."""
    def __init__(
        self,
        data_source: DataSource,
        clim_model: ExpectationModel,
        weather_model: Optional[ExpectationModel],
        scorer: AnomalyScorer,
        interpreter: Interpreter,
    ):
        self.data_source = data_source
        self.clim_model = clim_model
        self.weather_model = weather_model
        self.scorer = scorer
        self.interpreter = interpreter

    def run(self) -> Dict[str, Any]:
        """
        Returns dictionary containing:
          - 'observed': xr.DataArray (for Tarun's swipe viewer)
          - 'predicted': xr.DataArray (for Tarun's swipe viewer)
          - 'z_scores': xr.DataArray (anomaly layer)
          - 'verdicts': xr.DataArray (0=Expected, 1=Unusual but Explained, 2=Unexplained Candidate Event)
          - 'cards': Dict[str, Any] (Anomaly Card metadata per date)
        """
        raise NotImplementedError
