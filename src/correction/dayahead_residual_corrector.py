"""
dayahead_residual_corrector.py — Lightweight residual correction for CatBoost day-ahead.

Two methods:
1. SelectedHourResidualCorrector — trains residual model for specific hours (11,12,13,17)
2. SpikeResidualCorrector — detects spike hours and corrects residual

Both apply rolling walk-forward (no future leakage) with guardrails.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Helpers ──

def _get_catboost():
    from catboost import CatBoostRegressor
    return CatBoostRegressor


def _get_catboost_classifier():
    from catboost import CatBoostClassifier
    return CatBoostClassifier


def _smape(y_true, y_pred):
    denom = np.abs(y_true) + np.abs(y_pred)
    denom = np.maximum(denom, 1e-8)
    return float(np.nanmean(200 * np.abs(y_true - y_pred) / denom))


# ── Selected Hour Residual Corrector ──

class SelectedHourResidualCorrector:
    """Train residual correction models for specific hours only.

    For each target_day and each selected hour:
        residual = y_true - catboost_pred
        train residual_model on all data < target_day for that hour
        final_pred = catboost_pred + clip(residual_pred, max_delta)
    """

    def __init__(self, target_hours: list[int] | None = None,
                 max_delta: float = 150.0,
                 model_params: dict | None = None):
        self.target_hours = target_hours or [11, 12, 13, 17]
        self.max_delta = max_delta
        self.model_params = model_params or {
            "depth": 6, "learning_rate": 0.05, "iterations": 500,
            "l2_leaf_reg": 3.0, "random_seed": 42,
            "verbose": False, "thread_count": -1,
        }

    def correct(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply rolling residual correction.

        df must have columns: ds, target_day, hour_business, y_pred, y_true, y,
                              plus all feature columns from feature builder.
        Returns df with corrected y_pred column.
        """
        result = df.copy()
        result["y_pred_original"] = result["y_pred"].values
        result["residual"] = result["y_true"] - result["y_pred"]

        # Sort by time
        result = result.sort_values("ds").reset_index(drop=True)

        # Get feature columns (everything except metadata)
        exclude = {"ds", "y", "y_pred", "y_pred_original", "y_true", "residual",
                   "hour_business", "period", "business_day", "target_day",
                   "task", "model_name", "source", "run_mode", "created_at",
                   "date_only"}
        feat_cols = [c for c in result.columns if c not in exclude
                     and result[c].dtype in (np.float64, np.int64, np.float32, np.int32)]

        # Group by hour and correct each hour separately
        days = sorted(result["target_day"].unique())

        for hour in self.target_hours:
            hour_mask = result["hour_business"] == hour
            if hour_mask.sum() < 5:
                logger.info(f"  Skipping hour {hour}: too few samples")
                continue

            # Walk forward: for each day, train on past data, predict current day
            for day_idx, day in enumerate(days):
                day_mask = hour_mask & (result["target_day"] == day)
                if day_mask.sum() == 0:
                    continue

                # Training data: same hour, all days before this day
                train_mask = hour_mask & (result["target_day"] < day)
                if train_mask.sum() < 20:
                    logger.debug(f"  Hour {hour} day {day}: insufficient train ({train_mask.sum()})")
                    continue

                X_train = result.loc[train_mask, feat_cols].values
                y_train = result.loc[train_mask, "residual"].values
                X_pred = result.loc[day_mask, feat_cols].values

                if len(X_train) < 10 or len(X_pred) == 0:
                    continue

                try:
                    CB = _get_catboost()
                    model = CB(**self.model_params)
                    model.fit(X_train, y_train, verbose=False)

                    resid_pred = model.predict(X_pred).flatten()
                    # Apply guardrail
                    resid_pred = np.clip(resid_pred, -self.max_delta, self.max_delta)

                    result.loc[day_mask, "y_pred"] = (
                        result.loc[day_mask, "y_pred_original"].values + resid_pred
                    )
                except Exception as e:
                    logger.warning(f"  Hour {hour} day {day}: fit failed: {e}")

        # Clip to non-negative (prices can't be negative for dayahead... well, they can)
        # Don't clip — let the guardrail handle extremes

        return result


# ── Spike Residual Corrector ──

class SpikeResidualCorrector:
    """Detect spike hours and apply residual correction.

    Spike definition: y_true in top 10% of rolling window.
    Corrects only when spike_prob > threshold.
    """

    def __init__(self, alpha: float = 0.5, threshold: float = 0.55,
                 max_delta: float = 150.0,
                 spike_percentile: float = 90.0,
                 lookback_days: int = 30,
                 classifier_params: dict | None = None,
                 regressor_params: dict | None = None):
        self.alpha = alpha
        self.threshold = threshold
        self.max_delta = max_delta
        self.spike_percentile = spike_percentile
        self.lookback_days = lookback_days
        self.clf_params = classifier_params or {
            "depth": 4, "learning_rate": 0.05, "iterations": 300,
            "l2_leaf_reg": 5.0, "random_seed": 42,
            "verbose": False, "thread_count": -1,
        }
        self.reg_params = regressor_params or {
            "depth": 6, "learning_rate": 0.05, "iterations": 500,
            "l2_leaf_reg": 3.0, "random_seed": 42,
            "verbose": False, "thread_count": -1,
        }

    def _compute_spike_labels(self, df: pd.DataFrame) -> pd.Series:
        """Compute spike labels: y_true in top percentile of past N days."""
        is_spike = pd.Series(False, index=df.index, dtype=bool)
        days = sorted(df["target_day"].unique())

        for day in days:
            day_mask = df["target_day"] == day
            past_mask = df["target_day"] < day
            if past_mask.sum() < 24:
                continue

            past_prices = df.loc[past_mask, "y_true"].values
            threshold_val = np.percentile(past_prices, self.spike_percentile)
            day_values = df.loc[day_mask, "y_true"].values >= threshold_val
            is_spike.iloc[day_mask.values] = day_values

        return is_spike

    def correct(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply spike residual correction with rolling walk-forward."""
        CB = _get_catboost()
        result = df.copy()
        result["y_pred_original"] = result["y_pred"].values
        result["residual"] = result["y_true"] - result["y_pred"]

        # Feature columns
        exclude = {"ds", "y", "y_pred", "y_pred_original", "y_true", "residual",
                   "hour_business", "period", "business_day", "target_day",
                   "task", "model_name", "source", "run_mode", "created_at",
                   "date_only", "is_spike"}
        feat_cols = [c for c in result.columns if c not in exclude
                     and result[c].dtype in (np.float64, np.int64, np.float32, np.int32)]

        result = result.sort_values("ds").reset_index(drop=True)
        days = sorted(result["target_day"].unique())
        n_days = len(days)

        for day_idx, day in enumerate(days):
            day_mask = result["target_day"] == day
            if day_mask.sum() == 0:
                continue

            # Past data for spike labeling
            past_mask = result["target_day"] < day
            if past_mask.sum() < 48:
                continue

            # Compute spike labels on past data
            past_df = result[past_mask].copy()
            past_df["is_spike"] = self._compute_spike_labels(past_df)

            if past_df["is_spike"].sum() < 3:
                logger.debug(f"  Day {day}: too few spike samples ({past_df['is_spike'].sum()})")
                continue

            X_train = past_df[feat_cols].values
            y_spike = past_df["is_spike"].values.astype(int)
            y_resid = past_df["residual"].values

            X_pred = result.loc[day_mask, feat_cols].values

            try:
                # Train classifier
                CB_CLF = _get_catboost_classifier()
                clf = CB_CLF(**self.clf_params)
                clf.fit(X_train, y_spike, verbose=False)
                spike_prob = clf.predict_proba(X_pred)[:, 1]

                # Train residual regressor (only on spike hours)
                spike_mask = past_df["is_spike"]
                if spike_mask.sum() >= 5:
                    CB_REG = _get_catboost()
                    reg = CB_REG(**self.reg_params)
                    reg.fit(past_df.loc[spike_mask, feat_cols].values,
                            past_df.loc[spike_mask, "residual"].values,
                            verbose=False)
                    resid_pred = reg.predict(X_pred).flatten()
                else:
                    resid_pred = np.zeros(len(X_pred))

                # Apply correction where spike_prob > threshold
                correction = np.where(
                    spike_prob >= self.threshold,
                    self.alpha * resid_pred,
                    0.0,
                )
                # Guardrail
                correction = np.clip(correction, -self.max_delta, self.max_delta)

                result.loc[day_mask, "y_pred"] = (
                    result.loc[day_mask, "y_pred_original"].values + correction
                )

            except Exception as e:
                logger.warning(f"  Day {day}: spike correction failed: {e}")

        return result
