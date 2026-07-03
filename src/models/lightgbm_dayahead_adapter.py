"""
lightgbm_dayahead_adapter.py — LightGBM day-ahead model with config search.
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
    import lightgbm as lgb
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False
    logger.warning("lightgbm not installed")


def assign_period(hour: int) -> str:
    if 1 <= hour <= 8:
        return "1_8"
    elif 9 <= hour <= 16:
        return "9_16"
    else:
        return "17_24"


LGB_CONFIGS = {
    "gbdt_default": {
        "boosting_type": "gbdt", "num_leaves": 31, "learning_rate": 0.05,
        "feature_fraction": 1.0, "bagging_fraction": 1.0, "bagging_freq": 0,
        "lambda_l1": 0.0, "lambda_l2": 0.0, "min_data_in_leaf": 20,
        "objective": "rmse", "metric": "rmse", "verbosity": -1,
    },
    "dart_regularized": {
        "boosting_type": "dart", "num_leaves": 63, "learning_rate": 0.03,
        "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 5,
        "lambda_l1": 1.0, "lambda_l2": 1.0, "min_data_in_leaf": 30,
        "objective": "rmse", "metric": "rmse", "verbosity": -1,
    },
    "huber_loss": {
        "boosting_type": "gbdt", "num_leaves": 31, "learning_rate": 0.05,
        "feature_fraction": 0.9, "bagging_fraction": 0.8, "bagging_freq": 4,
        "lambda_l1": 0.5, "lambda_l2": 0.5, "min_data_in_leaf": 20,
        "objective": "huber", "metric": "huber", "verbosity": -1,
    },
    "mae_loss": {
        "boosting_type": "gbdt", "num_leaves": 31, "learning_rate": 0.05,
        "feature_fraction": 0.9, "bagging_fraction": 0.8, "bagging_freq": 4,
        "lambda_l1": 0.0, "lambda_l2": 0.0, "min_data_in_leaf": 20,
        "objective": "mae", "metric": "mae", "verbosity": -1,
    },
    "high_leaf_regularized": {
        "boosting_type": "gbdt", "num_leaves": 127, "learning_rate": 0.02,
        "feature_fraction": 0.7, "bagging_fraction": 0.7, "bagging_freq": 3,
        "lambda_l1": 2.0, "lambda_l2": 2.0, "min_data_in_leaf": 50,
        "objective": "rmse", "metric": "rmse", "verbosity": -1,
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


class LightGBMDayaheadAdapter:
    """LightGBM day-ahead model with multiple configs and window selection."""

    def __init__(self, config_name: str = "gbdt_default", model_params: Optional[dict] = None):
        if not _HAS_LGB:
            raise ImportError("lightgbm not installed")
        self.config_name = config_name
        self.params = model_params or LGB_CONFIGS.get(config_name, LGB_CONFIGS["gbdt_default"]).copy()
        self.model = None
        self.feature_cols = [c for c in FEATURE_COLS]

    def _prepare_X(self, df: pd.DataFrame) -> np.ndarray:
        """Extract feature matrix from dataframe."""
        available = [c for c in self.feature_cols if c in df.columns]
        X = df[available].fillna(0)
        # Convert categorical columns
        for col in ["hour", "month", "day_of_week", "is_weekend"]:
            if col in X.columns:
                X[col] = X[col].astype(int)
        return X.values

    def train(self, train_df: pd.DataFrame, eval_df: Optional[pd.DataFrame] = None) -> dict:
        """Train LightGBM model."""
        t0 = time.time()
        X_train = self._prepare_X(train_df)
        y_train = train_df["y"].values

        n_rounds = self.params.pop("num_boost_round", 2000) if "num_boost_round" in self.params else 2000
        es_rounds = self.params.pop("early_stopping_rounds", 50) if "early_stopping_rounds" in self.params else 50

        if eval_df is not None and len(eval_df) > 0:
            X_eval = self._prepare_X(eval_df)
            y_eval = eval_df["y"].values
            self.model = lgb.train(
                self.params,
                lgb.Dataset(X_train, y_train),
                num_boost_round=n_rounds,
                valid_sets=[lgb.Dataset(X_eval, y_eval)],
                valid_names=["eval"],
                callbacks=[lgb.early_stopping(es_rounds), lgb.log_evaluation(0)],
            )
            best_iter = self.model.best_iteration
        else:
            self.model = lgb.train(
                self.params,
                lgb.Dataset(X_train, y_train),
                num_boost_round=n_rounds,
                callbacks=[lgb.log_evaluation(0)],
            )
            best_iter = n_rounds

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
        return self.model.predict(X)

    def predict_day(self, full_df: pd.DataFrame, target_date: str, task: str = "dayahead") -> pd.DataFrame:
        """Predict one day and return standard long table."""
        target_dt = pd.Timestamp(target_date)
        start = target_dt + pd.Timedelta(hours=1)
        end = target_dt + pd.Timedelta(days=1)

        mask = (full_df["ds"] >= start) & (full_df["ds"] < end)
        day_df = full_df[mask].copy()
        if len(day_df) == 0:
            logger.warning(f"  No data for {target_date}, skipping")
            return pd.DataFrame()

        y_pred = self.predict(day_df)
        day_df["y_pred"] = y_pred
        day_df["y_true"] = day_df.get("y", np.nan)
        day_df["task"] = task
        day_df["model_name"] = f"lightgbm_dayahead_sota_{self.config_name}"
        day_df["hour_business"] = day_df.get("hour_business",
                                               ((day_df["ds"].dt.hour + 23) % 24 + 1))
        day_df["period"] = day_df.get("period", day_df["hour_business"].apply(assign_period))
        day_df["target_day"] = target_date
        day_df["business_day"] = target_date

        out = day_df[["ds", "y_true", "y_pred", "hour_business", "period",
                       "business_day", "target_day", "task", "model_name"]].copy()
        return out
