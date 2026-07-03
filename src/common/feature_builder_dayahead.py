"""
feature_builder_dayahead.py — Enhanced feature engineering for day-ahead specialist models.

Adds to the base 21 features from feature_builder.py:
  - lag_24h, lag_48h, lag_72h, lag_168h, lag_336h
  - same_hour_mean_7d, same_hour_mean_14d, same_hour_std_7d,
    same_hour_min_7d, same_hour_max_7d
  - price_momentum_24_168 (trend over past week)
  - load_forecast_error_proxy (if available)
  - net_load_rank_30d, bidding_space_rank_30d
  - is_spring_festival_window, days_to_spring_festival,
    days_after_spring_festival
  - is_weekend, is_month_start, is_month_end

All rolling features use only data up to the current row (no future leakage).
Shifted by 1 day so that D+1 predictions only use D-day and earlier info.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Base features (from feature_builder.py)
BASE_FEATURE_COLUMNS = [
    "hour", "month", "day_of_week", "is_weekend",
    "lag_price_target", "lag_price_week",
    "load", "wind", "solar", "interconnect",
    "bidding_space", "space_ratio",
    "net_load", "solar_ratio", "net_load_sq",
    "wind_ratio", "renew_penetration", "ramp_load", "ramp_solar",
    "morning_mean", "noon_min", "morning_std", "morning_trend", "is_info_fresh",
]

# Extended features (base + new)
EXTENDED_FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + [
    "lag_24h", "lag_72h", "lag_336h",
    "same_hour_mean_7d", "same_hour_mean_14d",
    "same_hour_std_7d", "same_hour_min_7d", "same_hour_max_7d",
    "price_momentum_24_168",
    "net_load_rank_30d", "bidding_space_rank_30d",
    "is_spring_festival_window",
    "days_to_spring_festival", "days_after_spring_festival",
    "is_month_start", "is_month_end",
]


def _add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lag_24h, lag_48h, lag_72h, lag_168h, lag_336h."""
    df = df.copy()
    df["lag_24h"] = df["y"].shift(24).ffill().fillna(0)
    df["lag_48h"] = df["y"].shift(48).ffill().fillna(0)
    df["lag_72h"] = df["y"].shift(72).ffill().fillna(0)
    df["lag_168h"] = df["y"].shift(168).ffill().fillna(0)  # 7 days
    df["lag_336h"] = df["y"].shift(336).ffill().fillna(0)  # 14 days
    return df


def _add_same_hour_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add same-hour statistics over rolling windows.
    For each row (ds, hour), compute mean/std/min/max of y at the same hour
    over the past 7 days and 14 days.
    Uses only past data (expanding window, no future leakage).
    """
    df = df.copy()
    df["hour_int"] = df["hour"]  # already 1-24

    # We need to group by hour and compute rolling stats
    # Sort by ds to ensure correct order
    df = df.sort_values("ds").reset_index(drop=True)

    # 7-day same-hour stats
    df["same_hour_mean_7d"] = np.nan
    df["same_hour_std_7d"] = np.nan
    df["same_hour_min_7d"] = np.nan
    df["same_hour_max_7d"] = np.nan

    for h in range(1, 25):
        mask = df["hour_int"] == h
        h_df = df[mask].copy()
        h_df["same_hour_mean_7d"] = h_df["y"].shift(1).rolling(window=7, min_periods=1).mean()
        h_df["same_hour_std_7d"] = h_df["y"].shift(1).rolling(window=7, min_periods=1).std()
        h_df["same_hour_min_7d"] = h_df["y"].shift(1).rolling(window=7, min_periods=1).min()
        h_df["same_hour_max_7d"] = h_df["y"].shift(1).rolling(window=7, min_periods=1).max()
        # Merge back
        for col in ["same_hour_mean_7d", "same_hour_std_7d", "same_hour_min_7d", "same_hour_max_7d"]:
            df.loc[mask, col] = h_df[col].values

    # 14-day same-hour mean
    df["same_hour_mean_14d"] = np.nan
    for h in range(1, 25):
        mask = df["hour_int"] == h
        h_df = df[mask].copy()
        h_df["same_hour_mean_14d"] = h_df["y"].shift(1).rolling(window=14, min_periods=1).mean()
        df.loc[mask, "same_hour_mean_14d"] = h_df["same_hour_mean_14d"].values

    # Fill remaining NaN with global mean for that hour
    for col in ["same_hour_mean_7d", "same_hour_std_7d",
                "same_hour_min_7d", "same_hour_max_7d", "same_hour_mean_14d"]:
        df[col] = df[col].ffill().fillna(0)

    df = df.drop(columns=["hour_int"], errors="ignore")
    return df


def _add_price_momentum(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add price_momentum_24_168:
      (lag_24h - lag_168h) / lag_168h
    Measures week-over-week price trend at the same hour.
    """
    df = df.copy()
    df["price_momentum_24_168"] = (
        (df["lag_24h"] - df["lag_168h"]) / (df["lag_168h"].replace(0, 1))
    ).fillna(0)
    return df


def _add_30d_ranks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add net_load_rank_30d and bidding_space_rank_30d.
    Rank of current value vs past 30 days (0-1, higher = relatively higher).
    Uses expanding window over past 30 days.
    """
    df = df.copy()
    df = df.sort_values("ds").reset_index(drop=True)

    df["net_load_rank_30d"] = np.nan
    df["bidding_space_rank_30d"] = np.nan

    for i in range(len(df)):
        start_idx = max(0, i - 30 * 24)
        window = df.iloc[start_idx:i]
        if len(window) < 10:
            df.loc[i, "net_load_rank_30d"] = 0.5
            df.loc[i, "bidding_space_rank_30d"] = 0.5
        else:
            nl = df.loc[i, "net_load"]
            bs = df.loc[i, "bidding_space"]
            nl_rank = (window["net_load"] < nl).sum() / len(window)
            bs_rank = (window["bidding_space"] < bs).sum() / len(window)
            df.loc[i, "net_load_rank_30d"] = nl_rank
            df.loc[i, "bidding_space_rank_30d"] = bs_rank

    df["net_load_rank_30d"] = df["net_load_rank_30d"].fillna(0.5)
    df["bidding_space_rank_30d"] = df["bidding_space_rank_30d"].fillna(0.5)
    return df


def _add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add is_month_start, is_month_end, spring festival features."""
    df = df.copy()
    ds = pd.to_datetime(df["ds"])

    # Month start/end
    df["is_month_start"] = (ds.dt.day == 1).astype(int)
    df["is_month_end"] = (ds.dt.day == ds.dt.days_in_month).astype(int)

    # Spring Festival window (approximate: Jan 20 - Feb 20 each year)
    year = ds.dt.year
    sf_start = pd.to_datetime(year.astype(str) + "-01-20")
    sf_end = pd.to_datetime(year.astype(str) + "-02-20")
    # Handle cross-year: if ds is in early Feb, use previous year's SF
    sf_start_alt = pd.to_datetime((year - 1).astype(str) + "-01-20")
    sf_end_alt = pd.to_datetime(year.astype(str) + "-02-20")
    df["is_spring_festival_window"] = (
        ((ds >= sf_start) & (ds <= sf_end)) |
        ((ds >= sf_start_alt) & (ds <= sf_end_alt))
    ).astype(int)

    # Days to / after Spring Festival (approximate: Feb 1)
    sf_date = pd.to_datetime(year.astype(str) + "-02-01")
    sf_date_alt = pd.to_datetime((year).astype(str) + "-02-01")
    # Use the closest Feb 1
    df["_sf"] = sf_date
    df["days_to_spring_festival"] = (df["_sf"] - ds).dt.days
    df["days_after_spring_festival"] = (ds - df["_sf"]).dt.days
    df["days_to_spring_festival"] = df["days_to_spring_festival"].clip(-30, 30)
    df["days_after_spring_festival"] = df["days_after_spring_festival"].clip(-30, 30)
    df = df.drop(columns=["_sf"], errors="ignore")

    return df


def build_features_dayahead(df: pd.DataFrame, use_extended: bool = True) -> pd.DataFrame:
    """
    Enhanced feature engineering for day-ahead specialist models.

    Parameters
    ----------
    df : DataFrame with columns ds, y, load, wind, solar, interconnect
    use_extended : if True, add all extended features; if False, use base only

    Returns
    -------
    DataFrame with feature columns + y
    """
    from .feature_builder import build_features as build_features_base

    # Start with base features
    result = build_features_base(df)

    if not use_extended:
        return result

    # Add extended features
    result = _add_lag_features(result)
    result = _add_same_hour_stats(result)
    result = _add_price_momentum(result)
    result = _add_30d_ranks(result)
    result = _add_calendar_features(result)

    return result


def get_dayahead_feature_columns(use_extended: bool = True) -> list[str]:
    """Return the feature column list for day-ahead models."""
    if use_extended:
        return list(EXTENDED_FEATURE_COLUMNS)
    else:
        return list(BASE_FEATURE_COLUMNS)
