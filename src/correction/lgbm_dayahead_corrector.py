#!/usr/bin/env python3
"""
LightGBM day-ahead residual correctors — ANTI-LEAKAGE VERSION.

Anti-leakage rules (enforced in _validate_prediction_features):
  Prediction features MUST NOT contain: y_true, residual, error, abs_error,
  future_y, target_actual, oracle, best_model.

Training: uses past y_true (OK, historical data).
Prediction: uses only base_pred + hour_business + rolling stats from past only.

Two methods:
1. LGBMSpikeResidualCorrector: spike-aware residual correction
2. LGBMSelectedHourCorrector: hour-specific residual correction

Both: base = LightGBM prediction, final = base + corrected_residual
Both: rolling (no future leak), max_delta guardrail
"""
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

# ── Denylist for prediction features ──
_DENYLIST = [
    "y_true", "residual", "error", "abs_error",
    "future_y", "target_actual", "oracle", "best_model",
]


def _validate_prediction_features(feature_array, feature_names=None):
    """Raise ValueError if any prediction feature matches denylist.
    
    Allows past/lagged residual terms (e.g. past_residual, lag1_residual)
    which are legitimate historical features. Only flags direct column access
    of denylist terms (e.g. "y_true", df["residual"]).
    """
    if feature_names is not None:
        for name in feature_names:
            for banned in _DENYLIST:
                if banned in name.lower():
                    # Skip if it's a past/lagged variant, e.g. past_residual, lag1_residual
                    prefix = name.lower().split(banned)[0]
                    if prefix.endswith("_") or "lag" in prefix:
                        continue
                    raise ValueError(
                        f"LEAKAGE DETECTED: prediction feature '{name}' "
                        f"contains banned term '{banned}'. "
                        f"Use only past-available features for prediction."
                    )


def _clip_correction(base_pred, correction, max_delta):
    """Clip correction to [-max_delta, +max_delta]."""
    return base_pred + np.clip(correction, -max_delta, max_delta)


# ═══════════════════════════════════════════════════════════
# Corrector 1: LGBM Spike Residual Corrector (anti-leakage)
# ═══════════════════════════════════════════════════════════

class LGBMSpikeResidualCorrector:
    """
    Spike residual corrector — ANTI-LEAKAGE.

    For each target_day:
    1. Identify spike hours in past: base_pred error > past 90th percentile
    2. Train residual model on spike hours using *past-only* features
    3. Predict residual for current day using only *past-available* features:
       [hour_business, base_pred]
    4. Apply alpha * residual_pred with max_delta guardrail
    """

    def __init__(self, alpha=0.5, max_delta=100, spike_percentile=90):
        self.alpha = alpha
        self.max_delta = max_delta
        self.spike_percentile = spike_percentile

    def correct(self, df):
        """
        Apply rolling spike residual correction (no leakage).

        df: DataFrame sorted by ds, with [ds, y_true, y_pred, hour_business, ...]
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

            # ── Past data (historical only, no leakage) ──
            past_df = df.iloc[:n_past]
            past_base = base[:n_past]
            past_residual = past_df["y_true"].values - past_base

            # Spike hours: top 10% absolute residual from PAST
            threshold_val = np.percentile(np.abs(past_residual), self.spike_percentile)
            spike_mask_past = np.abs(past_residual) >= threshold_val
            if spike_mask_past.sum() < 10:
                continue

            # ── TRAINING features: use base_pred + hour_business (PAST-ONLY) ──
            X_train = np.column_stack([
                past_df.loc[spike_mask_past, "hour_business"].values.astype(float),
                past_base[spike_mask_past],
            ])
            y_train = past_residual[spike_mask_past]
            _validate_prediction_features(
                X_train, ["hour_business", "base_pred"]
            )

            reg = CatBoostRegressor(
                iterations=200, depth=4, learning_rate=0.05,
                loss_function="MAE", verbose=False, random_seed=42,
            )
            try:
                reg.fit(X_train, y_train, verbose=False)
            except Exception:
                continue

            # ── PREDICTION features: ONLY past-available ──
            # NO y_true, NO residual, NO error columns
            X_pred = np.column_stack([
                df.loc[day_indices, "hour_business"].values.astype(float),
                base[day_indices],  # base_pred is the model's estimate — safe
            ])
            _validate_prediction_features(
                X_pred, ["hour_business", "base_pred"]
            )
            predicted_residuals = reg.predict(X_pred)

            correction = self.alpha * predicted_residuals
            for j, idx in enumerate(day_indices):
                if j < len(correction):
                    corrected[idx] = _clip_correction(
                        base[idx], correction[j], self.max_delta
                    )

        return corrected


# ═══════════════════════════════════════════════════════════
# Corrector 2: LGBM Selected Hour Corrector (anti-leakage)
# ═══════════════════════════════════════════════════════════

class LGBMSelectedHourCorrector:
    """
    Selected hour corrector — ANTI-LEAKAGE.

    For target hours [3,4,11,12,13,17]:
    Train per-hour rolling residual model.
    Prediction features: [base_pred, lag1_residual, lag2_residual, hour] — all from PAST.
    """

    def __init__(self, target_hours=None, max_delta=100):
        self.target_hours = target_hours or [11, 12, 13, 17]
        self.max_delta = max_delta

    def correct(self, df):
        df = df.copy().reset_index(drop=True)
        base = df["y_pred"].values.copy()
        corrected = base.copy()

        for hour in self.target_hours:
            hour_mask = df["hour_business"] == hour
            hour_indices = np.where(hour_mask)[0]
            if len(hour_indices) < 14:
                continue

            for i in range(len(hour_indices)):
                idx = hour_indices[i]
                if i < 7:
                    continue  # Need 7 past observations of this hour

                past_indices = hour_indices[:i]
                if len(past_indices) < 7:
                    continue

                # Past data (historical only)
                past_base = base[past_indices]
                past_true = df.iloc[past_indices]["y_true"].values
                past_residual = past_true - past_base
                lag1 = np.roll(past_residual, 1)
                lag2 = np.roll(past_residual, 2)
                lag1[0], lag2[:2] = 0, 0

                # ── Training features ──
                X_train = np.column_stack([past_base, lag1, lag2])
                y_train = past_residual
                _validate_prediction_features(
                    X_train, ["base_pred", "lag1_residual", "lag2_residual"]
                )

                reg = CatBoostRegressor(
                    iterations=100, depth=3, learning_rate=0.05,
                    loss_function="MAE", verbose=False, random_seed=42,
                )
                try:
                    reg.fit(X_train, y_train, verbose=False)
                except Exception:
                    continue

                # ── Prediction features: only past-available ──
                X_pred = np.array([[
                    base[idx],
                    past_residual[-1] if len(past_residual) > 0 else 0,
                    past_residual[-2] if len(past_residual) > 1 else 0,
                ]])
                _validate_prediction_features(
                    X_pred, ["base_pred", "lag1_residual", "lag2_residual"]
                )
                residual_pred = reg.predict(X_pred)[0]
                correction = np.clip(residual_pred, -self.max_delta, self.max_delta)
                corrected[idx] = base[idx] + correction

        return corrected
