"""
tabpfn_dayahead_tuned.py — TabPFN-TS with extended features and target transform for day-ahead.

Model name: "tabpfn_dayahead_tuned"

Enhancements vs tabpfn_ts_sota:
1. Uses extended day-ahead features (feature_builder_dayahead)
2. Target transform support (asinh / clip_low50 / none)
3. Day-ahead only
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

from ..models.tabpfn_ts_adapter import TabPFNTSAdapter
from ..common.feature_builder_dayahead import build_features_dayahead
from ..common.target_transform import apply_transform, invert_transform
from ..common.output_schema import make_long_table


class TabPFNDayAheadTuned:
    """
    TabPFN-TS adapter with extended features and target transform.
    """

    def __init__(
        self,
        model_name: str = "tabpfn_dayahead_tuned",
        max_train_rows: int = 50000,
        device: str = "cpu",
        target_transform: str = "none",
        transform_scale: float = 100.0,
    ):
        self.model_name = model_name
        self.max_train_rows = max_train_rows
        self.device = device
        self.target_transform = target_transform
        self.transform_scale = transform_scale
        self._adapter: Optional[TabPFNTSAdapter] = None
        self._trained: bool = False

    @property
    def is_trained(self) -> bool:
        return self._trained

    def train(self, train_df: pd.DataFrame) -> dict:
        """Train TabPFN on transformed target."""
        # Apply transform to y column
        train_df = train_df.copy()
        train_df["y_transformed"] = apply_transform(
            train_df["y"].values.copy(),
            method=self.target_transform,
            scale=self.transform_scale,
        )
        # Temporarily replace y with y_transformed for training
        train_df["y_original"] = train_df["y"].copy()
        train_df["y"] = train_df["y_transformed"]

        self._adapter = TabPFNTSAdapter(
            max_train_rows=self.max_train_rows,
            device=self.device,
        )
        self._adapter.train(train_df)

        self._trained = True
        return {"model_name": self.model_name, "target_transform": self.target_transform}

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict, invert transform."""
        if not self._trained:
            raise RuntimeError("Not trained")
        y_pred_t = self._adapter.predict(df)
        y_pred = invert_transform(y_pred_t, self.target_transform, self.transform_scale)
        return y_pred

    def predict_day(
        self,
        full_feature_df: pd.DataFrame,
        target_date: str,
        task: str = "dayahead",
    ) -> pd.DataFrame:
        """Predict all 24 hours."""
        day_dt = pd.Timestamp(target_date)
        target_ds_start = day_dt + pd.Timedelta(hours=1)
        target_ds_end = day_dt + pd.Timedelta(hours=24)

        day_df = full_feature_df[
            (full_feature_df["ds"] >= target_ds_start)
            & (full_feature_df["ds"] <= target_ds_end)
        ].copy()

        if len(day_df) == 0:
            raise ValueError(f"No rows for {target_date}")

        pred = self.predict(day_df)
        day_df["y_pred"] = pred
        if "y" in day_df.columns:
            day_df["y_true"] = day_df["y"]

        result = make_long_table(day_df, model_name=self.model_name, task=task)
        return result

    def save_model(self, path: str | Path):
        if self._adapter:
            self._adapter.save_model(path)

    def load_model(self, path: str | Path):
        self._adapter = TabPFNTSAdapter()
        self._adapter.load_model(path)
        self._trained = True
