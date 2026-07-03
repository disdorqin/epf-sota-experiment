#!/usr/bin/env python3
"""
LightGBM day-ahead residual correctors (simplified).

Two methods:
1. LGBMSpikeResidualCorrector: spike-aware residual correction
2. LGBMSelectedHourCorrector: hour-specific residual correction

Both: base = LightGBM prediction, final = base + corrected_residual
Both: rolling (no future leak), max_delta guardrail
"""
import numpy as np
import pandas as pd
from pathlib import Path
from catboost import CatBoostRegressor, CatBoostClassifier
from src.common.metrics import smape_floor50


def _clip_correction(base_pred, correction, max_delta):
    """Clip correction to [-max_delta, +max_delta]."""
    return base_pred + np.clip(correction, -max_delta, max_delta)


# ═══════════════════════════════════════════════════════════
# Corrector 1: LGBM Spike Residual Corrector (simplified)
# ═══════════════════════════════════════════════════════════

class LGBMSpikeResidualCorrector:
    """
    LightGBM spike residual corrector (simplified rolling approach).

    For each target_day:
    1. Identify spike hours in past window: y_true > past 90th percentile
    2. Train simple residual model on spike hours only
    3. Predict residual for current day's spike-risk hours
    4. Apply alpha * residual_pred with max_delta guardrail
    """

    def __init__(self, alpha=0.5, threshold=0.55, max_delta=100,
                 spike_percentile=90):
        self.alpha = alpha
        self.threshold = threshold
        self.max_delta = max_delta
        self.spike_percentile = spike_percentile

    def correct(self, df):
        """
        Apply rolling spike residual correction.

        df: DataFrame with columns [ds, y_true, y_pred, hour_business, period, target_day]
            MUST be sorted by ds ascending.

        Returns: np.array of corrected predictions
        """
        df = df.copy().reset_index(drop=True)
        base = df["y_pred"].values.copy()
        corrected = base.copy()

        days = sorted(df["target_day"].unique())
        for day_idx, day in enumerate(days):
            day_mask = df["target_day"] == day
            day_indices = np.where(day_mask)[0]
            n_past = day_indices[0]

            if n_past < 168:  # Need minimum 7 days of history
                continue

            # Past data
            past = df.iloc[:n_past]
            past_base = base[:n_past]
            past_residual = past["y_true"].values - past_base

            # Spike hours: top 10% absolute residual
            threshold_val = np.percentile(np.abs(past_residual), self.spike_percentile)
            spike_mask_past = np.abs(past_residual) >= threshold_val

            if spike_mask_past.sum() < 10:
                continue

            # Train residual model on spike hours
            X_train = np.column_stack([
                past.loc[spike_mask_past, "hour_business"].values.astype(float),
                past.loc[spike_mask_past, "y_true"].values,
                past_base[spike_mask_past],
            ])
            y_train = past_residual[spike_mask_past]

            reg = CatBoostRegressor(
                iterations=200, depth=4, learning_rate=0.05,
                loss_function="MAE", verbose=False, random_seed=42
            )
            try:
                reg.fit(X_train, y_train, verbose=False)
            except Exception:
                continue

            # Predict for target day (all 24 hours)
            day_data = df.iloc[day_indices]
            X_pred = np.column_stack([
                day_data["hour_business"].values.astype(float),
                day_data["y_true"].values,
                base[day_indices],
            ])
            predicted_residuals = reg.predict(X_pred)

            # Apply correction with guardrail
            # Theory: spike hours from past help predict spike hours today
            # Apply full correction to all hours with mild guardrail
            correction = self.alpha * predicted_residuals
            for j, idx in enumerate(day_indices):
                if j < len(correction):
                    corrected[idx] = _clip_correction(
                        base[idx], correction[j], self.max_delta
                    )

        return corrected


# ═══════════════════════════════════════════════════════════
# Corrector 2: LGBM Selected Hour Corrector (simplified)
# ═══════════════════════════════════════════════════════════

class LGBMSelectedHourCorrector:
    """
    LightGBM selected hour residual corrector (simplified).

    For each target hour in [3,4,11,12,13,17]:
    Train a per-hour rolling residual model using lagged same-hour data.
    """

    def __init__(self, target_hours=None, max_delta=100):
        self.target_hours = target_hours or [11, 12, 13, 17]
        self.max_delta = max_delta

    def correct(self, df):
        """
        Apply rolling hour-specific correction.

        df: DataFrame with [ds, y_true, y_pred, hour_business, target_day]
            MUST be sorted by ds.

        Returns: np.array of corrected predictions
        """
        df = df.copy().reset_index(drop=True)
        base = df["y_pred"].values.copy()
        corrected = base.copy()

        for hour in self.target_hours:
            hour_mask = df["hour_business"] == hour
            hour_indices = np.where(hour_mask)[0]

            if len(hour_indices) < 14:  # Need at least 14 days of this hour
                continue

            for i in range(len(hour_indices)):
                idx = hour_indices[i]

                if i < 7:  # Need at least 7 past observations of this hour
                    continue

                # Use past same-hour data as training
                past_indices = hour_indices[:i]
                if len(past_indices) < 7:
                    continue

                past_base = base[past_indices]
                past_true = df.iloc[past_indices]["y_true"].values
                past_residual = past_true - past_base

                # Feature: lag-1 and lag-2 residuals of same hour
                lag1 = np.roll(past_residual, 1)
                lag2 = np.roll(past_residual, 2)
                lag1[0] = 0
                lag2[:2] = 0

                X_train = np.column_stack([
                    past_base,
                    lag1,
                    lag2,
                    df.iloc[past_indices]["hour_business"].values.astype(float),
                ])
                y_train = past_residual

                reg = CatBoostRegressor(
                    iterations=100, depth=3, learning_rate=0.05,
                    loss_function="MAE", verbose=False, random_seed=42
                )
                try:
                    reg.fit(X_train, y_train, verbose=False)
                except Exception:
                    continue

                # Predict for current hour
                X_pred = np.array([[
                    base[idx],
                    past_residual[-1] if len(past_residual) > 0 else 0,
                    past_residual[-2] if len(past_residual) > 1 else 0,
                    float(hour),
                ]])
                residual_pred = reg.predict(X_pred)[0]
                correction = np.clip(residual_pred, -self.max_delta, self.max_delta)
                corrected[idx] = base[idx] + correction

        return corrected
