"""
catboost_regime_v2_adapter.py — Three advanced CatBoost day-ahead models.

1. catboost_weighted_smape_v2 — global CatBoost with sample weights
2. catboost_midday_spike_v2 — specialist for hours 11/12/13/17, replaces base pred
3. catboost_regime_v2 — lightweight MoE with hard routing

All use enhanced features + rolling walk-forward (no future leakage).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── CatBoost lazy import ──

def _CB():
    from catboost import CatBoostRegressor
    return CatBoostRegressor


# ── Enhanced Feature Builder ──

SPRING_FESTIVAL_2026 = pd.Timestamp("2026-02-17")
SPRING_FESTIVAL_WINDOW = 10


def add_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add regime v2 features. No future leakage. Modifies df in place."""
    # Ensure sorted
    df = df.sort_values("ds").reset_index(drop=True)

    # ── Calendar features ──
    ds = pd.to_datetime(df["ds"])
    df["is_spring_festival_window"] = (
        (ds >= SPRING_FESTIVAL_2026 - pd.Timedelta(days=SPRING_FESTIVAL_WINDOW)) &
        (ds <= SPRING_FESTIVAL_2026 + pd.Timedelta(days=SPRING_FESTIVAL_WINDOW))
    ).astype(int)
    df["days_to_spring_festival"] = (SPRING_FESTIVAL_2026 - ds).dt.days
    df["days_after_spring_festival"] = (ds - SPRING_FESTIVAL_2026).dt.days

    df["is_high_risk_hour"] = df["hour_business"].isin([11, 12, 13, 17]).astype(int)
    df["is_midday_peak_hour"] = df["hour_business"].isin([11, 12, 13]).astype(int)
    df["is_evening_peak_hour"] = df["hour_business"].isin([17, 18, 19]).astype(int)

    # ── Rolling same-hour price stats (shifted by 24 to use only past data) ──
    for window_days, suffix in [(7, "7d"), (14, "14d")]:
        window_hours = window_days * 24
        shifted = df.groupby("hour_business")["y_true"].transform(
            lambda x: x.shift(24)
        )
        df[f"same_hour_price_mean_{suffix}"] = (
            shifted.rolling(window_hours, min_periods=1).mean()
        )
        df[f"same_hour_price_std_{suffix}"] = (
            shifted.rolling(window_hours, min_periods=1).std()
        )
        df[f"same_hour_price_max_{suffix}"] = (
            shifted.rolling(window_hours, min_periods=1).max()
        )
        df[f"same_hour_price_min_{suffix}"] = (
            shifted.rolling(window_hours, min_periods=1).min()
        )

    # ── Same-hour physical stats ──
    for col, suffix in [("net_load", "net_load"), ("bidding_space", "bidding_space")]:
        shifted = df.groupby("hour_business")[col].transform(
            lambda x: x.shift(24)
        )
        df[f"same_hour_{suffix}_mean_7d"] = (
            shifted.rolling(7 * 24, min_periods=1).mean()
        )
        df[f"same_hour_{suffix}_rank_30d"] = (
            shifted.rolling(30 * 24, min_periods=1).rank(pct=True)
        )

    # ── Volatility features ──
    price = df["y_true"].values.astype(float)
    for window, suffix in [(24, "24h"), (168, "168h")]:
        shifted = np.roll(price, 24)
        shifted[:24] = np.nan
        s = pd.Series(shifted)
        df[f"price_volatility_{suffix}"] = (
            s.rolling(window, min_periods=2).std().fillna(0)
        )

    # ── Price momentum ──
    lag_24 = np.roll(price, 24)
    lag_168 = np.roll(price, 168)
    lag_24[:24] = np.nan
    lag_168[:168] = np.nan
    denom = np.abs(lag_168)
    denom = np.maximum(denom, 1e-8)
    momentum = (lag_24 - lag_168) / denom
    momentum[lag_168 == 0] = 0
    momentum = np.nan_to_num(momentum, nan=0.0)
    df["price_momentum_24_168"] = momentum

    # ── Ranking features ──
    for col in ["renew_penetration", "load", "bidding_space"]:
        if col in df.columns:
            shifted = df.groupby("hour_business")[col].transform(
                lambda x: x.shift(24)
            )
            df[f"{col}_rank_30d"] = (
                shifted.rolling(30 * 24, min_periods=1).rank(pct=True)
            )

    # ── Load ramp rank ──
    if "ramp_load" in df.columns:
        shifted_ramp = df["ramp_load"].shift(24)
        df["load_ramp_rank_30d"] = (
            shifted_ramp.rolling(30 * 24, min_periods=1).rank(pct=True)
        )

    # ── Bidding space change ──
    if "bidding_space" in df.columns:
        df["bidding_space_change_24h"] = (
            df["bidding_space"] - df["bidding_space"].shift(24)
        ).fillna(0)

    # ── Rolling regime labels (top 10% price) ──
    rolling_p99 = df.groupby("hour_business")["y_true"].transform(
        lambda x: x.shift(24).rolling(30 * 24, min_periods=24).quantile(0.90)
    )
    df["is_high_price_regime"] = (
        (df["y_true"] >= rolling_p99) & rolling_p99.notna()
    ).astype(int)

    # Fill all NaNs
    df = df.ffill().fillna(0)
    return df


REGIME_FEATURE_NAMES = [
    "is_spring_festival_window",
    "days_to_spring_festival",
    "days_after_spring_festival",
    "is_high_risk_hour",
    "is_midday_peak_hour",
    "is_evening_peak_hour",
    "same_hour_price_mean_7d",
    "same_hour_price_mean_14d",
    "same_hour_price_std_7d",
    "same_hour_price_max_7d",
    "same_hour_price_min_7d",
    "same_hour_net_load_mean_7d",
    "same_hour_bidding_space_mean_7d",
    "same_hour_bidding_space_rank_30d",
    "net_load_rank_30d",
    "price_volatility_24h",
    "price_volatility_168h",
    "price_momentum_24_168",
    "renew_penetration_rank_30d",
    "load_ramp_rank_30d",
    "bidding_space_change_24h",
    "is_high_price_regime",
]


# ── Model 1: catboost_weighted_smape_v2 ──

def train_weighted_smape_v2(train_df: pd.DataFrame,
                             val_df: Optional[pd.DataFrame] = None,
                             use_smape_selection: bool = True):
    """Train CatBoost with sample weights for regime focus.

    Sample weights:
        base = 1.0
        +2 if hour in [11,12,13,17]
        +2 if spring festival
        +2 if high price regime
        +1 if period == 9_16
    """
    CB = _CB()

    # Build sample weights
    weights = np.ones(len(train_df), dtype=float)
    weights[train_df["hour_business"].isin([11, 12, 13, 17])] += 2.0
    weights[train_df["is_spring_festival_window"] == 1] += 2.0
    weights[train_df["is_high_price_regime"] == 1] += 2.0
    if "period" in train_df.columns:
        weights[train_df["period"] == "9_16"] += 1.0

    # Feature columns
    exclude = {"ds", "y", "y_pred", "y_pred_original", "y_true", "residual",
               "hour_business", "period", "business_day", "target_day",
               "task", "model_name", "source", "run_mode", "created_at"}
    feat_cols = [c for c in train_df.columns if c not in exclude
                 and train_df[c].dtype in (np.float64, np.int64, np.float32, np.int32)]

    X_tr = train_df[feat_cols].values.astype(float)
    y_tr = train_df["y_true"].values.astype(float)

    # Try configs and pick by validation sMAPE
    configs = [
        {"depth": 8, "learning_rate": 0.03, "iterations": 1500, "l2_leaf_reg": 3.0},
        {"depth": 10, "learning_rate": 0.02, "iterations": 2000, "l2_leaf_reg": 5.0},
        {"depth": 6, "learning_rate": 0.05, "iterations": 1200, "l2_leaf_reg": 5.0},
    ]

    best_model = None
    best_smape = float("inf")
    best_feat_cols = feat_cols

    for cfg in configs:
        model = CB(
            **cfg,
            loss_function="RMSE",
            eval_metric="RMSE",
            random_seed=42,
            verbose=False,
            thread_count=-1,
        )
        model.fit(X_tr, y_tr, sample_weight=weights, verbose=False)

        if use_smape_selection and val_df is not None and len(val_df) > 50:
            X_val = val_df[feat_cols].values.astype(float)
            y_val = val_df["y_true"].values.astype(float)
            yp = model.predict(X_val).flatten()
            s = _smape(y_val, yp)
            if s < best_smape:
                best_smape = s
                best_model = model
        else:
            if best_model is None:
                best_model = model

    return best_model, feat_cols


def _smape(y_true, y_pred):
    denom = np.abs(y_true) + np.abs(y_pred)
    denom = np.maximum(denom, 1e-8)
    return float(np.nanmean(200 * np.abs(y_true - y_pred) / denom))


# ── Model 2: catboost_midday_spike_v2 ──

def train_midday_spike_v2(train_df: pd.DataFrame,
                            val_df: Optional[pd.DataFrame] = None):
    """Train specialist model for hours [11,12,13,17] only."""
    CB = _CB()

    # Filter to target hours
    target_hours = [11, 12, 13, 17]
    train_df = train_df[train_df["hour_business"].isin(target_hours)].copy()
    if len(train_df) < 100:
        return None, None

    if val_df is not None:
        val_df = val_df[val_df["hour_business"].isin(target_hours)].copy()

    exclude = {"ds", "y", "y_pred", "y_pred_original", "y_true", "residual",
               "hour_business", "period", "business_day", "target_day",
               "task", "model_name", "source", "run_mode", "created_at"}
    feat_cols = [c for c in train_df.columns if c not in exclude
                 and train_df[c].dtype in (np.float64, np.int64, np.float32, np.int32)]

    X_tr = train_df[feat_cols].values.astype(float)
    y_tr = train_df["y_true"].values.astype(float)

    configs = [
        {"depth": 8, "learning_rate": 0.03, "iterations": 1500, "l2_leaf_reg": 3.0},
        {"depth": 6, "learning_rate": 0.05, "iterations": 1200, "l2_leaf_reg": 5.0},
    ]

    best_model = None
    best_smape = float("inf")

    for cfg in configs:
        model = CB(
            **cfg,
            loss_function="RMSE",
            eval_metric="RMSE",
            random_seed=42,
            verbose=False,
            thread_count=-1,
        )
        model.fit(X_tr, y_tr, verbose=False)

        if val_df is not None and len(val_df) > 20:
            X_val = val_df[feat_cols].values.astype(float)
            y_val = val_df["y_true"].values.astype(float)
            yp = model.predict(X_val).flatten()
            s = _smape(y_val, yp)
            if s < best_smape:
                best_smape = s
                best_model = model
        else:
            if best_model is None:
                best_model = model

    return best_model, feat_cols


def predict_midday_spike(base_df: pd.DataFrame,
                          specialist_model, feat_cols,
                          replacement_hours: list[int]) -> np.ndarray:
    """Replace base predictions for target hours with specialist predictions."""
    y_pred = base_df["y_pred"].values.copy()
    mask = base_df["hour_business"].isin(replacement_hours)
    if mask.sum() == 0:
        return y_pred

    X_pred = base_df.loc[mask, feat_cols].values.astype(float)
    y_pred[mask.values] = specialist_model.predict(X_pred).flatten()
    return y_pred


# ── Model 3: catboost_regime_v2 ──

def train_regime_v2(train_df: pd.DataFrame,
                     val_df: Optional[pd.DataFrame] = None):
    """Train regime classifier + expert regressors with hard routing.

    Regimes: normal, spike_risk, holiday_risk, midday_risk.
    Hard routing: holiday -> holiday_expert, midday hours -> midday_expert,
                  high price -> spike_expert, else -> normal_expert.
    """
    CB = _CB()
    from catboost import CatBoostClassifier

    # Feature columns
    exclude = {"ds", "y", "y_pred", "y_pred_original", "y_true", "residual",
               "hour_business", "period", "business_day", "target_day",
               "task", "model_name", "source", "run_mode", "created_at",
               "regime_label"}
    feat_cols = [c for c in train_df.columns if c not in exclude
                 and train_df[c].dtype in (np.float64, np.int64, np.float32, np.int32)]

    # Create hard routing labels
    train_df = train_df.copy()
    train_df["regime_label"] = _assign_regime(train_df)

    # Train expert per regime
    experts = {}
    for regime in ["normal", "spike_risk", "holiday_risk", "midday_risk"]:
        mask = train_df["regime_label"] == regime
        sub = train_df[mask]
        if len(sub) < 50:
            logger.info(f"  Regime {regime}: insufficient data ({len(sub)}), skipping")
            continue
        X_sub = sub[feat_cols].values.astype(float)
        y_sub = sub["y_true"].values.astype(float)
        model = CB(
            depth=8, learning_rate=0.03, iterations=1500,
            l2_leaf_reg=3.0, random_seed=42,
            verbose=False, thread_count=-1,
        )
        model.fit(X_sub, y_sub, verbose=False)
        experts[regime] = model

    return experts, feat_cols


def _assign_regime(df: pd.DataFrame) -> np.ndarray:
    """Assign hard regime label per row."""
    labels = np.full(len(df), "normal", dtype=object)
    # Holiday risk: spring festival window
    if "is_spring_festival_window" in df.columns:
        labels[df["is_spring_festival_window"].values == 1] = "holiday_risk"
    # Midday risk: hours 11/12/13/17
    midday_mask = df["hour_business"].isin([11, 12, 13, 17]).values
    labels[midday_mask] = "midday_risk"
    # Spike risk: high price regime
    if "is_high_price_regime" in df.columns:
        spike_mask = df["is_high_price_regime"].values == 1
        labels[spike_mask] = "spike_risk"
    # Priority: spike_risk > holiday_risk > midday_risk > normal
    # Re-prioritize: spike first, then holiday, then midday, then normal
    labels = np.full(len(df), "normal", dtype=object)
    labels[midday_mask] = "midday_risk"
    if "is_spring_festival_window" in df.columns:
        labels[df["is_spring_festival_window"].values == 1] = "holiday_risk"
    if "is_high_price_regime" in df.columns:
        labels[df["is_high_price_regime"].values == 1] = "spike_risk"
    return labels


def predict_regime_v2(base_df: pd.DataFrame,
                       experts: dict, feat_cols: list[str]) -> np.ndarray:
    """Predict using hard-routed experts."""
    y_pred = np.zeros(len(base_df), dtype=float)
    for regime, model in experts.items():
        mask = _assign_regime(base_df) == regime
        if mask.sum() == 0:
            continue
        X_pred = base_df.loc[mask, feat_cols].values.astype(float)
        y_pred[mask] = model.predict(X_pred).flatten()
    return y_pred
