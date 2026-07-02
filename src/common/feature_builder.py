"""
feature_builder.py — Feature engineering replicating lightGBM/train_fix.py logic.

Generates 21+ features matching the original LightGBM pipeline:
    1.  Time features: hour, month, day_of_week, is_weekend
    2.  Lag features: lag_price_target (48h/168h smart select), lag_price_week
    3.  Physical features: net_load, solar_ratio, bidding_space, etc.
    4.  D-day statistics: morning_mean, noon_min, morning_std, morning_trend, is_info_fresh
"""

from __future__ import annotations

import pandas as pd
import numpy as np

FEATURE_COLUMNS = [
    "hour", "month", "day_of_week", "is_weekend",
    "lag_price_target", "lag_price_week",
    "load", "wind", "solar", "interconnect",
    "bidding_space", "space_ratio",
    "net_load", "solar_ratio", "net_load_sq",
    "wind_ratio", "renew_penetration", "ramp_load", "ramp_solar",
    "morning_mean", "noon_min", "morning_std", "morning_trend", "is_info_fresh",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering pipeline.

    Input df must have columns:
        ds (datetime), y (float), load, wind, solar, interconnect

    Returns a new DataFrame with all original columns + feature columns.
    No in-place modification — returns a copy.
    """
    df = df.copy()

    # ── 1-second offset for business time (matching original logic) ──
    adjusted_time = df["ds"] - pd.Timedelta(seconds=1)

    # ── 1. Basic time features (1-24 hour) ──
    df["hour"] = adjusted_time.dt.hour + 1          # 0→1, 23→24
    df["month"] = adjusted_time.dt.month
    df["day_of_week"] = adjusted_time.dt.dayofweek   # 0=Mon, 6=Sun
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    # ── 2. Lag features ──
    lag_step_2day = 48
    lag_step_7day = 168

    df["lag_48h"] = df["y"].shift(lag_step_2day)
    df["lag_168h"] = df["y"].shift(lag_step_7day)

    # Smart lag: workday → week-ago; non-workday → 2-days-ago
    df["lag_price_target"] = np.where(
        df["day_of_week"] < 5,
        df["lag_168h"],
        df["lag_48h"],
    )
    df["lag_price_week"] = df["lag_168h"]

    df["lag_price_target"] = df["lag_price_target"].ffill().fillna(0)
    df["lag_price_week"] = df["lag_price_week"].ffill().fillna(0)

    # ── 3. Physical features ──
    safe_load = df["load"].replace(0, 1)
    df["net_load"] = df["load"] - df["wind"] - df["solar"]
    df["solar_ratio"] = df["solar"] / safe_load
    df["net_load_sq"] = (df["net_load"] / 1000) ** 2

    # billing_space: prefer raw column from data; fallback to net_load - interconnect
    if "bidding_space_raw" in df.columns and df["bidding_space_raw"].notna().sum() > 0:
        df["bidding_space"] = df["bidding_space_raw"]
    else:
        df["bidding_space"] = df["net_load"] - df["interconnect"]

    df["space_ratio"] = df["bidding_space"] / safe_load
    df["wind_ratio"] = df["wind"] / safe_load
    df["renew_penetration"] = (df["wind"] + df["solar"]) / safe_load
    df["ramp_load"] = df["load"].diff().fillna(0)
    df["ramp_solar"] = df["solar"].diff().fillna(0)

    # ── 4. D-day latest-info features ──
    df["date_only"] = adjusted_time.dt.date

    # Morning window: business hours 1-15
    mask_morning = (df["hour"] >= 1) & (df["hour"] <= 15)
    df_morning = df[mask_morning].copy()

    def calc_trend(x):
        return x.iloc[-1] - x.iloc[0] if len(x) >= 2 else 0

    stats_basic = df_morning.groupby("date_only")["y"].agg(
        morning_mean="mean",
        morning_std="std",
    )

    # Noon window: business hours 11-15
    mask_noon = (df_morning["hour"] >= 11) & (df_morning["hour"] <= 15)
    stats_noon = df_morning[mask_noon].groupby("date_only")["y"].agg(
        noon_min="min",
        morning_trend=calc_trend,
    )

    daily_feats = pd.concat([stats_basic, stats_noon], axis=1).reset_index()

    # Shift by 1 day: predicting D+1 can only use D-day info
    cols_to_shift = ["morning_mean", "noon_min", "morning_std", "morning_trend"]
    daily_feats[cols_to_shift] = daily_feats[cols_to_shift].shift(1)
    daily_feats["is_info_fresh"] = daily_feats["morning_mean"].notna().astype(int)
    daily_feats[cols_to_shift] = daily_feats[cols_to_shift].ffill().fillna(0)

    df = df.merge(daily_feats, on="date_only", how="left")

    # Ensure all feature cols exist and are filled
    for col in cols_to_shift + ["is_info_fresh"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Clean up
    df = df.drop(columns=["date_only", "lag_48h", "lag_168h"], errors="ignore")
    return df


def get_target_columns_for_task(task: str) -> list[str]:
    """Get the true price column name from original data for a task."""
    from .data_loader import TASK_TO_Y_COL
    return [TASK_TO_Y_COL.get(task, "日前电价")]
