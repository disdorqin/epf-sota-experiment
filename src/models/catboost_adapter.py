"""
catboost_adapter.py — CatBoostRegressor adapter for dayahead & realtime tasks.

Model name: "catboost_sota"

Replicates the LightGBM training approach but uses a single CatBoostRegressor
(trained on all hours) instead of the original 3-stage LGBM.

Key features:
- Uses same 21+ features as original LightGBM
- Treats hour, month, day_of_week, is_weekend as categorical (via astype(str))
- Default CPU, optional GPU
- Saves .cbm model, feature importance CSV, training manifest
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

# ── Lazy import CatBoost ──
try:
    from catboost import CatBoostRegressor
    _CATBOOST_AVAILABLE = True
except ImportError:
    _CATBOOST_AVAILABLE = False
    CatBoostRegressor = None


# ── Default parameters (stable, not over-tuned) ──
DEFAULT_CATBOOST_PARAMS = {
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "iterations": 1500,
    "learning_rate": 0.03,
    "depth": 8,
    "l2_leaf_reg": 5.0,
    "random_seed": 42,
    "od_type": "Iter",
    "od_wait": 100,
    "verbose": False,
    "allow_writing_files": False,
    "task_type": "CPU",
}

# Features to treat as categorical (passed as cat_features param to CatBoost)
CATEGORICAL_FEATURE_NAMES = [
    "hour", "month", "day_of_week", "is_weekend",
]

FEATURE_COLUMNS = [
    "hour", "month", "day_of_week", "is_weekend",
    "lag_price_target", "lag_price_week",
    "load", "wind", "solar", "interconnect",
    "bidding_space", "space_ratio",
    "net_load", "solar_ratio", "net_load_sq",
    "wind_ratio", "renew_penetration", "ramp_load", "ramp_solar",
    "morning_mean", "noon_min", "morning_std", "morning_trend", "is_info_fresh",
]


class CatBoostAdapter:
    """
    CatBoost adapter for electricity price prediction.
    Supports both dayahead and realtime tasks.
    """

    def __init__(
        self,
        model_name: str = "catboost_sota",
        task_type: str = "CPU",
        **catboost_kwargs,
    ):
        if not _CATBOOST_AVAILABLE:
            raise ImportError(
                "catboost is not installed. Run: pip install catboost"
            )

        self.model_name = model_name
        self.params = dict(DEFAULT_CATBOOST_PARAMS)
        self.params["task_type"] = task_type
        self.params.update(catboost_kwargs)

        self._model: Optional[CatBoostRegressor] = None
        self._feature_names: list[str] = list(FEATURE_COLUMNS)
        self._cat_feature_indices: list[int] = []
        self._trained: bool = False

    @property
    def is_trained(self) -> bool:
        return self._trained

    # ── Feature preparation ──

    def _prepare_X(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare feature DataFrame: select columns, convert categoricals to str.
        Returns a DataFrame ready for CatBoost's direct fit/predict.
        """
        missing = [c for c in self._feature_names if c not in df.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")

        X = df[self._feature_names].copy()

        # Convert categorical features to string (CatBoost's expected format
        # when using cat_features list)
        for col in CATEGORICAL_FEATURE_NAMES:
            X[col] = X[col].astype(str)

        # Fill numeric NaN
        for col in X.columns:
            if col not in CATEGORICAL_FEATURE_NAMES:
                X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)

        self._cat_feature_indices = [
            i for i, c in enumerate(self._feature_names)
            if c in CATEGORICAL_FEATURE_NAMES
        ]
        return X

    def _prepare_train_data(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, np.ndarray]:
        """Prepare X, y, dropping rows with NaN target."""
        X = self._prepare_X(df)
        y = df["y"].values
        valid = ~np.isnan(y)
        if valid.sum() < len(valid):
            dropped = (~valid).sum()
            logger.warning(f"Dropping {dropped} rows with NaN target")
        return X[valid], y[valid]

    # ── Training ──

    def train(
        self,
        train_df: pd.DataFrame,
        eval_df: Optional[pd.DataFrame] = None,
    ) -> dict:
        """
        Train CatBoostRegressor.

        Parameters
        ----------
        train_df : DataFrame with feature columns + 'y' target column
        eval_df : optional validation DataFrame

        Returns
        -------
        training_manifest dict
        """
        X_train, y_train = self._prepare_train_data(train_df)

        eval_set = None
        if eval_df is not None and len(eval_df) > 0:
            X_eval, y_eval = self._prepare_train_data(eval_df)
            eval_set = [(X_eval, y_eval)]

        model = CatBoostRegressor(
            **self.params,
            cat_features=self._cat_feature_indices,
        )
        model.fit(
            X_train, y_train,
            eval_set=eval_set,
            plot=False,
        )

        self._model = model
        self._trained = True

        manifest = {
            "model_name": self.model_name,
            "params": self.params,
            "feature_columns": self._feature_names,
            "categorical_features": CATEGORICAL_FEATURE_NAMES,
            "cat_feature_indices": self._cat_feature_indices,
            "train_rows": len(X_train),
            "eval_rows": len(eval_set[0][0]) if eval_set else 0,
            "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "best_iteration": model.get_best_iteration() if model.get_best_iteration() else None,
        }
        return manifest

    # ── Prediction ──

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Run inference. df must contain all feature columns."""
        if not self._trained:
            raise RuntimeError("Model not trained. Call train() first.")
        X = self._prepare_X(df)
        return self._model.predict(X)

    # ── Save/Load ──

    def save_model(self, path: str | Path):
        """Save .cbm model file."""
        if not self._trained:
            raise RuntimeError("No trained model to save.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._model.save_model(str(path))
        logger.info(f"CatBoost model saved to {path}")

    def load_model(self, path: str | Path):
        """Load .cbm model file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        if CatBoostRegressor is None:
            raise ImportError("catboost not installed")
        self._model = CatBoostRegressor()
        self._model.load_model(str(path))
        self._trained = True
        self._feature_names = list(self._model.feature_names_)
        logger.info(f"CatBoost model loaded from {path}")

    def get_feature_importance(self) -> pd.DataFrame:
        """Return feature importance DataFrame."""
        if not self._trained:
            raise RuntimeError("Model not trained.")
        importance = self._model.get_feature_importance()
        return pd.DataFrame({
            "feature": self._feature_names,
            "importance": importance,
        }).sort_values("importance", ascending=False).reset_index(drop=True)

    # ── Convenience: full predict-for-day flow ──

    def predict_day(
        self,
        full_feature_df: pd.DataFrame,
        target_date: str,
        task: str = "dayahead",
    ) -> pd.DataFrame:
        """
        Predict all 24 business hours for a given target day.

        Parameters
        ----------
        full_feature_df : full DataFrame with features
        target_date : str, YYYY-MM-DD

        Returns
        -------
        DataFrame with 24 rows, standard long-table format.
        """
        from ..common.output_schema import make_long_table

        # Target day business hours: D 01:00 → D+1 00:00
        day_dt = pd.Timestamp(target_date)
        target_ds_start = day_dt + pd.Timedelta(hours=1)
        target_ds_end = day_dt + pd.Timedelta(hours=24)

        # Get the feature rows for this day
        day_df = full_feature_df[
            (full_feature_df["ds"] >= target_ds_start)
            & (full_feature_df["ds"] <= target_ds_end)
        ].copy()

        if len(day_df) == 0:
            raise ValueError(
                f"No data rows found for target_date={target_date}."
            )

        pred = self.predict(day_df)
        day_df["y_pred"] = pred

        result = make_long_table(
            day_df,
            model_name=self.model_name,
            task=task,
        )
        if len(result) != 24:
            logger.warning(
                f"catboost_sota: {target_date} {task} — got {len(result)} rows, expected 24"
            )
        return result
