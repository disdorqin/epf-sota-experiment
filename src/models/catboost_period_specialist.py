"""
catboost_period_specialist.py — 3 period-specific CatBoost models for day-ahead.

Model name: "catboost_period_specialist"

Trains 3 separate CatBoost models, one for each period:
  - period "1_8"  (hours 1-8)
  - period "9_16" (hours 9-16)
  - period "17_24"(hours 17-24)

Each model only sees data from its assigned period.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from catboost import CatBoostRegressor
    _CATBOOST_AVAILABLE = True
except ImportError:
    _CATBOOST_AVAILABLE = False

from ..common.feature_builder_dayahead import (
    build_features_dayahead,
    get_dayahead_feature_columns,
)
from ..common.target_transform import apply_transform, invert_transform
from ..common.output_schema import make_long_table


PERIODS = ["1_8", "9_16", "17_24"]
PERIOD_HOURS = {"1_8": list(range(1, 9)), "9_16": list(range(9, 17)), "17_24": list(range(17, 25))}


class CatBoostPeriodSpecialist:
    """
    3 period-specific CatBoost models.

    Each model is trained only on data from its assigned period.
    """

    def __init__(
        self,
        model_name: str = "catboost_period_specialist",
        task_type: str = "CPU",
        target_transform: str = "none",
        transform_scale: float = 100.0,
        use_extended_features: bool = True,
    ):
        if not _CATBOOST_AVAILABLE:
            raise ImportError("catboost not installed")

        self.model_name = model_name
        self.task_type = task_type
        self.target_transform = target_transform
        self.transform_scale = transform_scale
        self.use_extended_features = use_extended_features

        self._models: dict[str, CatBoostRegressor] = {}
        self._feature_names = get_dayahead_feature_columns(use_extended_features)
        self._trained: bool = False

    @property
    def is_trained(self) -> bool:
        return self._trained

    def _prepare_X(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features."""
        missing = [c for c in self._feature_names if c not in df.columns]
        if missing:
            raise ValueError(f"Missing features: {missing}")

        X = df[self._feature_names].copy()
        for col in ["hour", "month", "day_of_week", "is_weekend"]:
            if col in X.columns:
                X[col] = X[col].astype(str)
        for col in X.columns:
            if col not in ["hour", "month", "day_of_week", "is_weekend"]:
                X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)

        cat_indices = [i for i, c in enumerate(self._feature_names)
                       if c in ["hour", "month", "day_of_week", "is_weekend"]]
        return X, cat_indices

    def train(
        self,
        train_df: pd.DataFrame,
        eval_df: Optional[pd.DataFrame] = None,
    ) -> dict:
        """
        Train 3 period-specific models.
        """
        manifests = []

        for period in PERIODS:
            h_list = PERIOD_HOURS[period]
            p_train = train_df[train_df["hour"].isin(h_list)].copy()
            if len(p_train) < 500:
                logger.warning(f"  Period {period}: only {len(p_train)} rows, skipping")
                continue

            y_train = apply_transform(
                p_train["y"].values.copy(),
                method=self.target_transform,
                scale=self.transform_scale,
            )
            X_train, cat_indices = self._prepare_X(p_train)

            eval_set = None
            if eval_df is not None:
                p_eval = eval_df[eval_df["hour"].isin(h_list)].copy()
                if len(p_eval) > 50:
                    y_eval = apply_transform(
                        p_eval["y"].values.copy(),
                        method=self.target_transform,
                        scale=self.transform_scale,
                    )
                    X_eval, _ = self._prepare_X(p_eval)
                    eval_set = [(X_eval, y_eval)]

            model = CatBoostRegressor(
                loss_function="RMSE",
                eval_metric="RMSE",
                iterations=1500,
                learning_rate=0.03,
                depth=8,
                l2_leaf_reg=5.0,
                random_seed=42,
                od_type="Iter",
                od_wait=100,
                verbose=False,
                allow_writing_files=False,
                task_type=self.task_type,
                cat_features=cat_indices,
            )
            model.fit(X_train, y_train, eval_set=eval_set, plot=False)

            self._models[period] = model
            logger.info(f"  Period {period}: trained ({len(p_train)} rows)")

            manifests.append({
                "period": period,
                "train_rows": len(p_train),
                "iterations": model.get_best_iteration() or 1500,
            })

        self._trained = True
        return {"model_name": self.model_name, "period_models": manifests}

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict using period-specific models."""
        if not self._trained:
            raise RuntimeError("Not trained")

        result = np.full(len(df), np.nan)
        for period, model in self._models.items():
            h_list = PERIOD_HOURS[period]
            p_mask = df["hour"].isin(h_list)
            if p_mask.sum() == 0:
                continue
            X_p, _ = self._prepare_X(df[p_mask])
            y_pred_t = model.predict(X_p)
            y_pred = invert_transform(y_pred_t, self.target_transform, self.transform_scale)
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
        for period, model in self._models.items():
            path = dir_path / f"period_specialist_{period}.cbm"
            model.save_model(str(path))
        logger.info(f"Saved {len(self._models)} period models to {dir_path}")

    def load_models(self, dir_path: str | Path):
        dir_path = Path(dir_path)
        self._models = {}
        for period in PERIODS:
            path = dir_path / f"period_specialist_{period}.cbm"
            if path.exists():
                model = CatBoostRegressor()
                model.load_model(str(path))
                self._models[period] = model
        self._trained = len(self._models) > 0
        logger.info(f"Loaded {len(self._models)} period models from {dir_path}")
