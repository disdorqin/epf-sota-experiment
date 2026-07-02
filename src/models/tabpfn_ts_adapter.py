"""
tabpfn_ts_adapter.py — TabPFN-TS (tabular foundation model) for electricity price forecasting.

Model name: "tabpfn_ts_sota"

Key design:
    - Uses TabPFNRegressor as a supervised tabular model
    - Reuses feature_builder features (21+ features)
    - Supports both dayahead and realtime tasks
    - Default max_train_rows = 50000
    - CPU/GPU compatible
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

# ── Lazy import ──
_TABPFN_AVAILABLE = False
try:
    from tabpfn import TabPFNRegressor
    _TABPFN_AVAILABLE = True
except ImportError:
    TabPFNRegressor = None

FEATURE_COLUMNS = [
    "hour", "month", "day_of_week", "is_weekend",
    "lag_price_target", "lag_price_week",
    "load", "wind", "solar", "interconnect",
    "bidding_space", "space_ratio",
    "net_load", "solar_ratio", "net_load_sq",
    "wind_ratio", "renew_penetration", "ramp_load", "ramp_solar",
    "morning_mean", "noon_min", "morning_std", "morning_trend", "is_info_fresh",
]


class TabPFNTSAdapter:
    """
    TabPFN-TS adapter for electricity price prediction.

    Uses TabPFNRegressor as a supervised tabular model.
    Trains on historical data and predicts each day's 24 hours.
    """

    def __init__(
        self,
        model_name: str = "tabpfn_ts_sota",
        max_train_rows: int = 50000,
        device: str = "cpu",
        random_seed: int = 42,
    ):
        if not _TABPFN_AVAILABLE:
            raise ImportError(
                "tabpfn is not installed. Run: pip install tabpfn"
            )

        self.model_name = model_name
        self.max_train_rows = max_train_rows
        self.device = device
        self.random_seed = random_seed

        self._model: Optional[TabPFNRegressor] = None
        self._feature_names: list[str] = list(FEATURE_COLUMNS)
        self._trained: bool = False

    @property
    def is_trained(self) -> bool:
        return self._trained

    # ── Feature preparation ──

    def _prepare_X(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select feature columns and fill NaN."""
        missing = [c for c in self._feature_names if c not in df.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")

        X = df[self._feature_names].copy()
        # Fill numeric NaN
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)
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
        Train TabPFNRegressor.

        Parameters
        ----------
        train_df : DataFrame with feature columns + 'y' target column
        eval_df : optional validation DataFrame (not used by TabPFN directly)

        Returns
        -------
        training_manifest dict
        """
        X_train, y_train = self._prepare_train_data(train_df)

        # Apply max_train_rows limit
        if len(X_train) > self.max_train_rows:
            logger.info(f"Truncating training from {len(X_train)} to {self.max_train_rows} rows")
            X_train = X_train.tail(self.max_train_rows)
            y_train = y_train[-self.max_train_rows:]

        logger.info(f"TabPFN training on {len(X_train)} rows with {len(self._feature_names)} features")

        model = TabPFNRegressor(
            device=self.device,
            random_state=self.random_seed,
            ignore_pretraining_limits=True,
        )
        model.fit(X_train, y_train)

        self._model = model
        self._trained = True

        manifest = {
            "model_name": self.model_name,
            "feature_columns": self._feature_names,
            "train_rows": len(X_train),
            "max_train_rows": self.max_train_rows,
            "device": self.device,
            "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
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
        """Save model to pickle."""
        if not self._trained:
            raise RuntimeError("No trained model to save.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        import joblib
        joblib.dump(self._model, str(path))
        logger.info(f"TabPFN model saved to {path}")

    def load_model(self, path: str | Path):
        """Load model from pickle."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        import joblib
        self._model = joblib.load(str(path))
        self._trained = True
        logger.info(f"TabPFN model loaded from {path}")

    def get_manifest(self) -> dict:
        return {
            "model_name": self.model_name,
            "max_train_rows": self.max_train_rows,
            "device": self.device,
            "feature_columns": self._feature_names,
            "trained": self._trained,
        }

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
        task : "dayahead" or "realtime"

        Returns
        -------
        DataFrame with 24 rows, standard long-table format.
        """
        from ..common.output_schema import make_long_table

        day_dt = pd.Timestamp(target_date)
        target_ds_start = day_dt + pd.Timedelta(hours=1)
        target_ds_end = day_dt + pd.Timedelta(hours=24)

        day_df = full_feature_df[
            (full_feature_df["ds"] >= target_ds_start)
            & (full_feature_df["ds"] <= target_ds_end)
        ].copy()

        if len(day_df) == 0:
            raise ValueError(f"No data rows found for target_date={target_date}.")

        pred = self.predict(day_df)
        day_df["y_pred"] = pred

        if "y" in day_df.columns:
            day_df["y_true"] = day_df["y"]

        result = make_long_table(
            day_df,
            model_name=self.model_name,
            task=task,
        )
        if len(result) != 24:
            logger.warning(
                f"tabpfn_ts_sota: {target_date} {task} — got {len(result)} rows, expected 24"
            )
        return result
