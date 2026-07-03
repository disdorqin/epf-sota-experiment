#!/usr/bin/env python3
"""
LightGBM day-ahead residual correctors.

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

# ── Helpers ──

def _smape(y_true, y_pred):
    return smape_floor50(y_true, y_pred)


def _clip_correction(base_pred, correction, max_delta):
    """Clip correction to [-max_delta, +max_delta]."""
    return base_pred + np.clip(correction, -max_delta, max_delta)


# ═══════════════════════════════════════════════════════════
# Corrector 1: LGBM Spike Residual Corrector
# ═══════════════════════════════════════════════════════════

class LGBMSpikeResidualCorrector:
    """
    LightGBM-dayahead spike residual corrector.

    Logic:
    1. Detect spike hours: y_true in top 10% of recent rolling window
    2. Train spike classifier + residual regressor on past data
    3. Predict spike probability for current day
    4. If spike_prob > threshold: apply alpha * residual_pred (clipped by max_delta)

    Rolling: each target_day uses data before that day only.
    """

    def __init__(self, alpha=0.5, threshold=0.55, max_delta=100,
                 spike_percentile=90, reg_params=None, clf_params=None):
        self.alpha = alpha
        self.threshold = threshold
        self.max_delta = max_delta
        self.spike_percentile = spike_percentile
        self.reg_params = reg_params or {
            "iterations": 300, "depth": 6, "learning_rate": 0.05,
            "loss_function": "MAE", "verbose": False, "random_seed": 42,
        }
        self.clf_params = clf_params or {
            "iterations": 300, "depth": 6, "learning_rate": 0.05,
            "loss_function": "Logloss", "verbose": False, "random_seed": 42,
        }

    def _build_features(self, df, day_idx):
        """Build features up to (not including) target_day at day_idx."""
        past = df.iloc[:day_idx].copy()
        if len(past) < 48:
            return None, None
        # Features: hour, period, price moments, volatility
        feats = pd.DataFrame(index=past.index)
        feats["hour"] = past["hour_business"]
        feats["period"] = past["period"].map({"1_8": 0, "9_16": 1, "17_24": 2}).fillna(1)
        feats["price"] = past["y_true"]
        feats["price_lag24"] = past["y_true"].shift(24).fillna(past["y_true"].median())
        feats["price_lag48"] = past["y_true"].shift(48).fillna(past["y_true"].median())
        feats["price_lag168"] = past["y_true"].shift(168).fillna(past["y_true"].median())
        feats["hour_mean_7d"] = past.groupby("hour_business")["y_true"].transform(
            lambda x: x.rolling(7, min_periods=2).mean().shift(1)
        ).fillna(past["y_true"].median())
        feats["hour_std_7d"] = past.groupby("hour_business")["y_true"].transform(
            lambda x: x.rolling(7, min_periods=2).std().shift(1)
        ).fillna(1)
        # Target: residual
        feats["residual"] = past["y_true"] - past["base_pred"].values[:len(past)]
        # Spike label: top 10% of rolling residuals (absolute)
        rolling_spike = feats["residual"].abs().rolling(168, min_periods=24).quantile(0.9).shift(1)
        feats["is_spike"] = (feats["residual"].abs() > rolling_spike.fillna(feats["residual"].abs().quantile(0.9))).astype(int)

        # Feature columns (no target, no is_spike)
        feature_cols = ["hour", "period", "price", "price_lag24", "price_lag48",
                        "price_lag168", "hour_mean_7d", "hour_std_7d"]
        X = feats[feature_cols].fillna(0).values
        y_spike = feats["is_spike"].values
        y_residual = feats["residual"].values
        return X, (y_spike, y_residual, feature_cols)

    def _build_features_pred(self, day_df):
        """Build features for a target day (day_df has 24 rows)."""
        feats = pd.DataFrame(index=day_df.index)
        feats["hour"] = day_df["hour_business"]
        feats["period"] = day_df["period"].map({"1_8": 0, "9_16": 1, "17_24": 2}).fillna(1)
        feats["price"] = day_df["y_true"].values  # Known from history, but careful
        # Use base_pred as price proxy for lags
        feats["price_lag24"] = day_df["base_pred"].shift(24).fillna(day_df["y_true"].median())
        feats["price_lag48"] = day_df["base_pred"].shift(48).fillna(day_df["y_true"].median())
        feats["price_lag168"] = day_df["base_pred"].shift(168).fillna(day_df["y_true"].median())
        feats["hour_mean_7d"] = day_df["base_pred"].median()
        feats["hour_std_7d"] = day_df["y_true"].std() if len(day_df) > 1 else 1
        feature_cols = ["hour", "period", "price", "price_lag24", "price_lag48",
                        "price_lag168", "hour_mean_7d", "hour_std_7d"]
        return feats[feature_cols].fillna(0).values

    def correct(self, df):
        """
        Apply rolling spike residual correction.

        df: DataFrame with columns [ds, y_true, base_pred, hour_business, period, target_day]
            MUST be sorted by ds ascending.

        Returns: np.array of corrected predictions
        """
        df = df.copy().reset_index(drop=True)
        df["base_pred"] = df["base_pred"].values.copy()
        corrected = df["base_pred"].values.copy().astype(float)

        days = sorted(df["target_day"].unique())
        for i, day in enumerate(days):
            day_mask = df["target_day"] == day
            day_idx = np.where(day_mask)[0][0]

            if day_idx < 168:  # Need minimum history
                continue

            # Build features from past data
            X_train, targets = self._build_features(df, day_idx)
            if X_train is None:
                continue

            y_spike_train, y_residual_train, _ = targets

            # Train spike classifier
            spike_mask = y_spike_train == 1
            n_spike = spike_mask.sum()

            # Train residual regressor on spike hours
            if n_spike >= 10:
                reg = CatBoostRegressor(**self.reg_params)
                reg.fit(X_train[spike_mask], y_residual_train[spike_mask], verbose=False)

                # Predict for target day
                day_data = df[day_mask].copy()
                day_data["base_pred"] = corrected.copy()  # Use cumulative corrected
                X_pred = self._build_features_pred(day_data)
                residual_pred = reg.predict(X_pred)

                # Calculate spike probability
                clf = CatBoostClassifier(**self.clf_params)
                if n_spike >= 20:
                    clf.fit(X_train, y_spike_train, verbose=False)
                    spike_prob = clf.predict_proba(X_pred)[:, 1]
                else:
                    spike_prob = np.ones(len(day_data)) * 0.5

                # Apply correction where spike_prob > threshold
                apply_mask = spike_prob > self.threshold
                correction = np.where(apply_mask, self.alpha * residual_pred, 0)
                day_idx_range = np.where(day_mask)[0]
                for j, idx in enumerate(day_idx_range):
                    if j < len(correction):
                        corrected[idx] = _clip_correction(
                            corrected[idx], correction[j], self.max_delta
                        )

        return corrected


# ═══════════════════════════════════════════════════════════
# Corrector 2: LGBM Selected Hour Corrector
# ═══════════════════════════════════════════════════════════

class LGBMSelectedHourCorrector:
    """
    LightGBM selected hour residual corrector.

    Hours: [3, 4, 11, 12, 13, 17]
    Each hour: train a rolling residual model (CatBoost) using past N days of that hour
    Only correct those hours; leave others unchanged.
    """

    def __init__(self, target_hours=None, past_days=60,
                 reg_params=None, max_delta=100):
        self.target_hours = target_hours or [3, 4, 11, 12, 13, 17]
        self.past_days = past_days
        self.max_delta = max_delta
        self.reg_params = reg_params or {
            "iterations": 200, "depth": 4, "learning_rate": 0.05,
            "loss_function": "MAE", "verbose": False, "random_seed": 42,
        }

    def correct(self, df):
        """
        Apply rolling hour-specific correction.

        df: DataFrame with [ds, y_true, base_pred, hour_business, target_day]
            MUST be sorted by ds.

        Returns: np.array of corrected predictions
        """
        df = df.copy().reset_index(drop=True)
        corrected = df["base_pred"].values.copy().astype(float)

        for hour in self.target_hours:
            hour_mask = df["hour_business"] == hour
            if hour_mask.sum() < 5:
                continue

            hour_df = df[hour_mask].copy()
            hour_indices = np.where(hour_mask)[0]

            for i in range(len(hour_df)):
                day = hour_df.iloc[i]["target_day"]
                # Training data: past N days of same hour
                cutoff = pd.Timestamp(day) - pd.Timedelta(days=self.past_days)
                train_mask = (
                    (df["hour_business"] == hour) &
                    (pd.to_datetime(df["ds"]) < pd.Timestamp(day)) &
                    (pd.to_datetime(df["ds"]) >= cutoff)
                )
                if train_mask.sum() < 10:
                    continue

                # Simple model: use lags and rolling stats of same hour
                train_df = df[train_mask].copy()
                X_train = np.column_stack([
                    train_df["y_true"].values,
                    np.roll(train_df["y_true"].values, 1),
                    np.roll(train_df["y_true"].values, 2),
                    train_df["base_pred"].values,
                ])
                # Shift to avoid future leak
                X_train[0] = [train_df["y_true"].iloc[0], 0, 0, train_df["base_pred"].iloc[0]]
                y_train = train_df["y_true"].values - train_df["base_pred"].values

                reg = CatBoostRegressor(**self.reg_params)
                reg.fit(X_train, y_train, verbose=False)

                # Predict
                idx = hour_indices[i]
                row = df.iloc[idx]
                X_pred = np.array([[row["y_true"], 0, 0, row["base_pred"]]])
                residual_pred = reg.predict(X_pred)[0]
                correction = np.clip(residual_pred, -self.max_delta, self.max_delta)
                corrected[idx] = row["base_pred"] + correction

        return corrected
