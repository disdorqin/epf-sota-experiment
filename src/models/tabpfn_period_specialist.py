"""
tabpfn_period_specialist.py — 3 period-specific TabPFN-TS models for day-ahead.

Model name: "tabpfn_period_specialist"

Trains 3 separate TabPFN-TS models, one for each period:
  - period "1_8"  (hours 1-8)
  - period "9_16" (hours 9-16)
  - period "17_24"(hours 17-24)
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
from ..common.output_schema import make_long_table


PERIODS = ["1_8", "9_16", "17_24"]
PERIOD_HOURS = {"1_8": list(range(1, 9)), "9_16": list(range(9, 17)), "17_24": list(range(17, 25))}


class TabPFNPeriodSpecialist:
    """
    3 period-specific TabPFN-TS models.
    """

    def __init__(
        self,
        model_name: str = "tabpfn_period_specialist",
        max_train_rows: int = 50000,
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.max_train_rows = max_train_rows
        self.device = device
        self._adapters: dict[str, TabPFNTSAdapter] = {}
        self._trained: bool = False

    @property
    def is_trained(self) -> bool:
        return self._trained

    def train(
        self,
        train_df: pd.DataFrame,
    ) -> dict:
        """Train 3 period-specific TabPFN models."""
        manifests = []

        for period in PERIODS:
            h_list = PERIOD_HOURS[period]
            p_train = train_df[train_df["hour"].isin(h_list)].copy()
            if len(p_train) < 500:
                logger.warning(f"  Period {period}: only {len(p_train)} rows, skipping")
                continue

            adapter = TabPFNTSAdapter(max_train_rows=self.max_train_rows, device=self.device)
            adapter.train(p_train)
            self._adapters[period] = adapter

            manifests.append({
                "period": period,
                "train_rows": len(p_train),
            })
            logger.info(f"  Period {period}: trained ({len(p_train)} rows)")

        self._trained = True
        return {"model_name": self.model_name, "period_models": manifests}

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict using period-specific models."""
        if not self._trained:
            raise RuntimeError("Not trained")

        result = np.full(len(df), np.nan)
        for period, adapter in self._adapters.items():
            h_list = PERIOD_HOURS[period]
            p_mask = df["hour"].isin(h_list)
            if p_mask.sum() == 0:
                continue
            y_pred = adapter.predict(df[p_mask])
            result[p_mask.values] = y_pred

        return result

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

    def save_models(self, dir_path: str | Path):
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        for period, adapter in self._adapters.items():
            path = dir_path / f"tabpfn_period_{period}.pkl"
            adapter.save_model(str(path))
        logger.info(f"Saved {len(self._adapters)} period models to {dir_path}")

    def load_models(self, dir_path: str | Path):
        dir_path = Path(dir_path)
        self._adapters = {}
        for period in PERIODS:
            path = dir_path / f"tabpfn_period_{period}.pkl"
            if path.exists():
                adapter = TabPFNTSAdapter()
                adapter.load_model(str(path))
                self._adapters[period] = adapter
        self._trained = len(self._adapters) > 0
        logger.info(f"Loaded {len(self._adapters)} period models from {dir_path}")
