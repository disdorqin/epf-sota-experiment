"""
catboost_dayahead_tuned.py — CatBoost with Optuna tuning for day-ahead.

Model name: "catboost_dayahead_tuned"

Enhancements vs catboost_sota:
1. Optuna hyperparameter search (optimizes sMAPE_floor50 on validation set)
2. Target transform support (asinh / clip_low50 / none)
3. Uses extended day-ahead features (feature_builder_dayahead)
4. Day-ahead only (no realtime)
"""

from __future__ import annotations

import json
import logging
import os
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
    CatBoostRegressor = None

try:
    import optuna
    _OPTUNA_AVAILABLE = True
except ImportError:
    _OPTUNA_AVAILABLE = False
    optuna = None

from ..common.feature_builder_dayahead import (
    build_features_dayahead,
    get_dayahead_feature_columns,
)
from ..common.target_transform import apply_transform, invert_transform
from ..common.output_schema import make_long_table


# ── Default CatBoost params (starting point for tuning) ──
DEFAULT_TUNED_PARAMS = {
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "iterations": 2000,
    "learning_rate": 0.03,
    "depth": 8,
    "l2_leaf_reg": 5.0,
    "random_seed": 42,
    "od_type": "Iter",
    "od_wait": 100,
    "verbose": False,
    "allow_writing_files": False,
    "task_type": "CPU",
    "bagging_temperature": 0.5,
    "random_strength": 0.5,
}

CATEGORICAL_FEATURES = ["hour", "month", "day_of_week", "is_weekend"]


class CatBoostDayAheadTuned:
    """
    CatBoost adapter with Optuna tuning for day-ahead prediction.

    Usage:
        adapter = CatBoostDayAheadTuned(target_transform="asinh")
        adapter.train(train_df, eval_df, tune=True, n_trials=30)
        pred = adapter.predict(day_df)
    """

    def __init__(
        self,
        model_name: str = "catboost_dayahead_tuned",
        task_type: str = "CPU",
        target_transform: str = "none",
        transform_scale: float = 100.0,
        use_extended_features: bool = True,
    ):
        if not _CATBOOST_AVAILABLE:
            raise ImportError("catboost not installed. Run: pip install catboost")

        self.model_name = model_name
        self.params = dict(DEFAULT_TUNED_PARAMS)
        self.params["task_type"] = task_type
        self.target_transform = target_transform
        self.transform_scale = transform_scale
        self.use_extended_features = use_extended_features

        self._model: Optional[CatBoostRegressor] = None
        self._feature_names: list[str] = get_dayahead_feature_columns(use_extended_features)
        self._cat_feature_indices: list[int] = []
        self._trained: bool = False
        self._best_trial_params: Optional[dict] = None

    @property
    def is_trained(self) -> bool:
        return self._trained

    def _prepare_X(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare feature DataFrame."""
        missing = [c for c in self._feature_names if c not in df.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")

        X = df[self._feature_names].copy()
        for col in CATEGORICAL_FEATURES:
            if col in X.columns:
                X[col] = X[col].astype(str)

        for col in X.columns:
            if col not in CATEGORICAL_FEATURES:
                X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)

        self._cat_feature_indices = [
            i for i, c in enumerate(self._feature_names)
            if c in CATEGORICAL_FEATURES
        ]
        return X

    def _prepare_train_data(self, df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        """Prepare X, y (with transform applied)."""
        X = self._prepare_X(df)
        y = df["y"].values.copy()
        y = apply_transform(y, method=self.target_transform, scale=self.transform_scale)
        valid = ~np.isnan(y)
        if valid.sum() < len(valid):
            logger.warning(f"Dropping {(~valid).sum()} rows with NaN target")
        return X[valid], y[valid]

    def train(
        self,
        train_df: pd.DataFrame,
        eval_df: Optional[pd.DataFrame] = None,
        tune: bool = False,
        n_trials: int = 30,
    ) -> dict:
        """
        Train with optional Optuna tuning.

        If tune=True, runs Optuna search optimizing sMAPE_floor50 on eval set.
        """
        if tune and _OPTUNA_AVAILABLE:
            logger.info(f"Starting Optuna tuning ({n_trials} trials)...")
            study = optuna.create_study(direction="minimize")
            study.optimize(
                lambda trial: self._optuna_objective(trial, train_df, eval_df),
                n_trials=n_trials,
            )
            self._best_trial_params = study.best_params
            self.params.update(study.best_params)
            logger.info(f"Best trial params: {study.best_params}")

        X_train, y_train = self._prepare_train_data(train_df)
        eval_set = None
        if eval_df is not None and len(eval_df) > 0:
            X_eval, y_eval = self._prepare_train_data(eval_df)
            eval_set = [(X_eval, y_eval)]

        model = CatBoostRegressor(
            **self.params,
            cat_features=self._cat_feature_indices,
        )
        model.fit(X_train, y_train, eval_set=eval_set, plot=False)

        self._model = model
        self._trained = True

        manifest = {
            "model_name": self.model_name,
            "params": self.params,
            "target_transform": self.target_transform,
            "transform_scale": self.transform_scale,
            "feature_columns": self._feature_names,
            "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "best_trial_params": self._best_trial_params,
        }
        return manifest

    def _optuna_objective(
        self,
        trial: "optuna.Trial",
        train_df: pd.DataFrame,
        eval_df: pd.DataFrame,
    ) -> float:
        """Optuna objective: minimize sMAPE_floor50 on eval set."""
        params = dict(self.params)
        params["depth"] = trial.suggest_int("depth", 4, 10)
        params["learning_rate"] = trial.suggest_float("learning_rate", 0.005, 0.1, log=True)
        params["l2_leaf_reg"] = trial.suggest_float("l2_leaf_reg", 1.0, 20.0, log=True)
        params["iterations"] = trial.suggest_int("iterations", 500, 3000)
        params["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0.0, 1.0)
        params["random_strength"] = trial.suggest_float("random_strength", 0.0, 1.0)

        X_train, y_train = self._prepare_train_data(train_df)
        X_eval, y_eval = self._prepare_train_data(eval_df)

        model = CatBoostRegressor(
            **params,
            cat_features=self._cat_feature_indices,
            verbose=False,
        )
        model.fit(X_train, y_train, eval_set=[(X_eval, y_eval)], plot=False)

        # Predict on eval set (inverted to original scale)
        y_pred_t = model.predict(X_eval)
        y_pred = invert_transform(y_pred_t, self.target_transform, self.transform_scale)
        y_true = eval_df["y"].values

        valid = ~(np.isnan(y_true) | np.isnan(y_pred))
        if valid.sum() < 2:
            return 100.0

        # Compute sMAPE_floor50
        y_true_c = np.where(y_true[valid] < 50, 50, y_true[valid])
        y_pred_c = np.where(y_pred[valid] < 50, 50, y_pred[valid])
        denom = (np.abs(y_true_c) + np.abs(y_pred_c)) / 2.0
        denom = np.where(denom < 1e-6, 1e-6, denom)
        smape = float(np.mean(np.abs(y_pred_c - y_true_c) / denom) * 100.0)
        return smape

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Run inference, invert transform."""
        if not self._trained:
            raise RuntimeError("Model not trained.")
        X = self._prepare_X(df)
        y_pred_t = self._model.predict(X)
        y_pred = invert_transform(y_pred_t, self.target_transform, self.transform_scale)
        return y_pred

    def predict_day(
        self,
        full_feature_df: pd.DataFrame,
        target_date: str,
        task: str = "dayahead",
    ) -> pd.DataFrame:
        """Predict all 24 hours for target_date."""
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
        if not self._trained:
            raise RuntimeError("No model to save.")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._model.save_model(str(path))
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str | Path):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        self._model = CatBoostRegressor()
        self._model.load_model(str(path))
        self._trained = True
        logger.info(f"Model loaded from {path}")
