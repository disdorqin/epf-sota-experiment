"""
xgboost_dayahead_adapter.py — XGBoost day-ahead model with config search.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import xgboost as xgb
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False
    logger.warning("xgboost not installed")


def assign_period(hour: int) -> str:
    if 1 <= hour <= 8:
        return "1_8"
    elif 9 <= hour <= 16:
        return "9_16"
    else:
        return "17_24"


XGB_CONFIGS = {
    "squared_error_default": {
        "objective": "reg:squarederror", "max_depth": 8, "eta": 0.05,
        "subsample": 1.0, "colsample_bytree": 1.0,
        "lambda": 0.0, "alpha": 0.0, "min_child_weight": 5,
        "eval_metric": "rmse", "verbosity": 0,
    },
    "absolute_error": {
        "objective": "reg:absoluteerror", "max_depth": 6, "eta": 0.05,
        "subsample": 0.8, "colsample_bytree": 0.8,
        "lambda": 1.0, "alpha": 1.0, "min_child_weight": 10,
        "eval_metric": "mae", "verbosity": 0,
    },
    "huber": {
        "objective": "reg:squarederror", "max_depth": 6, "eta": 0.04,
        "subsample": 0.8, "colsample_bytree": 0.8,
        "lambda": 0.5, "alpha": 0.5, "min_child_weight": 8,
        "eval_metric": "rmse", "verbosity": 0,
    },
    "regularized_deep": {
        "objective": "reg:squarederror", "max_depth": 12, "eta": 0.02,
        "subsample": 0.7, "colsample_bytree": 0.7,
        "lambda": 2.0, "alpha": 2.0, "min_child_weight": 20,
        "eval_metric": "rmse", "verbosity": 0,
    },
}

FEATURE_COLS = [
    "hour", "month", "day_of_week", "is_weekend",
    "lag_price_target", "lag_price_week",
    "load", "wind", "solar", "interconnect", "bidding_space", "space_ratio",
    "net_load", "solar_ratio", "net_load_sq", "wind_ratio", "renew_penetration",
    "ramp_load", "ramp_solar", "morning_mean", "noon_min", "morning_std",
    "morning_trend", "is_info_fresh",
    "lag_24h", "lag_48h", "lag_72h", "lag_168h", "lag_336h",
    "same_hour_mean_7d", "same_hour_mean_14d", "same_hour_std_7d",
    "same_hour_max_7d", "same_hour_min_7d",
    "price_momentum_24_168", "net_load_rank_30d", "bidding_space_rank_30d",
    "is_spring_festival_window", "days_to_spring_festival",
    "days_after_spring_festival", "is_month_start", "is_month_end",
]


class XGBoostDayaheadAdapter:
    """XGBoost day-ahead model with multiple configs and window selection."""

    def __init__(self, config_name: str = "squared_error_default", model_params: Optional[dict] = None):
        if not _HAS_XGB:
            raise ImportError("xgboost not installed")
        self.config_name = config_name
        self.params = model_params or XGB_CONFIGS.get(config_name, XGB_CONFIGS["squared_error_default"]).copy()
        self.model = None
        self.feature_cols = [c for c in FEATURE_COLS]

    def _prepare_X(self, df: pd.DataFrame) -> np.ndarray:
        """Extract feature matrix from dataframe."""
        available = [c for c in self.feature_cols if c in df.columns]
        X = df[available].fillna(0)
        for col in ["hour", "month", "day_of_week", "is_weekend"]:
            if col in X.columns:
                X[col] = X[col].astype(int)
        return X.values

    def train(self, train_df: pd.DataFrame, eval_df: Optional[pd.DataFrame] = None) -> dict:
        """Train XGBoost model."""
        t0 = time.time()
        X_train = self._prepare_X(train_df)
        y_train = train_df["y"].values

        dtrain = xgb.DMatrix(X_train, label=y_train)
        if eval_df is not None and len(eval_df) > 0:
            X_eval = self._prepare_X(eval_df)
            y_eval = eval_df["y"].values
            deval = xgb.DMatrix(X_eval, label=y_eval)
            self.model = xgb.train(
                self.params, dtrain, num_boost_round=2000,
                evals=[(deval, "eval")],
                early_stopping_rounds=50, verbose_eval=False,
            )
            best_iter = self.model.best_iteration
        else:
            self.model = xgb.train(
                self.params, dtrain, num_boost_round=1000, verbose_eval=False,
            )
            best_iter = 1000

        elapsed = time.time() - t0
        return {
            "config_name": self.config_name,
            "n_train": len(train_df),
            "n_features": X_train.shape[1],
            "best_iteration": best_iter,
            "train_time": elapsed,
        }

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict on feature dataframe."""
        X = self._prepare_X(df)
        dmatrix = xgb.DMatrix(X)
        return self.model.predict(dmatrix)

    def predict_day(self, full_df: pd.DataFrame, target_date: str, task: str = "dayahead") -> pd.DataFrame:
        """Predict one day and return standard long table."""
        target_dt = pd.Timestamp(target_date)
        start = target_dt + pd.Timedelta(hours=1)
        end = target_dt + pd.Timedelta(days=1)

        mask = (full_df["ds"] >= start) & (full_df["ds"] < end)
        day_df = full_df[mask].copy()
        if len(day_df) == 0:
            mask = full_df["target_day"].astype(str) == target_date
            day_df = full_df[mask].copy()

        if len(day_df) == 0:
            return pd.DataFrame()

        y_pred = self.predict(day_df)
        day_df["y_pred"] = y_pred
        day_df["y_true"] = day_df.get("y", np.nan)
        day_df["task"] = task
        day_df["model_name"] = f"xgboost_dayahead_sota_{self.config_name}"
        day_df["hour_business"] = day_df.get("hour_business",
                                               ((day_df["ds"].dt.hour + 23) % 24 + 1))
        day_df["period"] = day_df.get("period", day_df["hour_business"].apply(assign_period))
        day_df["business_day"] = day_df.get("business_day", day_df["target_day"])

        out = day_df[["ds", "y_true", "y_pred", "hour_business", "period",
                       "business_day", "target_day", "task", "model_name"]].copy()
        return out
