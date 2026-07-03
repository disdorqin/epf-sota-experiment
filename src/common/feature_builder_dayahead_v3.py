"""
feature_builder_dayahead_v3.py — Day-ahead feature engineering v3.

Extends feature_builder_dayahead.py with:
  - price_volatility_24h, price_volatility_168h
  - renewable_penetration_rank_30d, load_ramp_rank_30d
  - bidding_space_change_24h, net_load_change_24h, renewable_change_24h
  - is_spring_festival_window (exact 2026 date), days_to/after (exact)
  - Interaction features: hour_x_bidding_space, hour_x_net_load,
    period_x_bidding_space, period_x_renewable_penetration
  - lag_48h (explicit in v3)

All rolling/rank/change features use ONLY data visible before prediction day.
No y_true, residual, error, or future information leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# v3 = base + extended (v2) + new v3 features
V3_NEW_COLUMNS = [
    "price_volatility_24h",
    "price_volatility_168h",
    "renewable_penetration_rank_30d",
    "load_ramp_rank_30d",
    "bidding_space_change_24h",
    "net_load_change_24h",
    "renewable_change_24h",
    "is_spring_festival_exact",
    "days_to_spring_festival_exact",
    "days_after_spring_festival_exact",
    "hour_x_bidding_space",
    "hour_x_net_load",
    "period_x_bidding_space",
    "period_x_renewable_penetration",
]


def _add_volatility(df: pd.DataFrame) -> pd.DataFrame:
    """Price volatility: rolling std of lagged prices."""
    df = df.copy()
    # Use shift(24) to ensure no same-day leakage
    lagged = df["y"].shift(24)
    df["price_volatility_24h"] = lagged.rolling(24, min_periods=2).std().fillna(0)
    df["price_volatility_168h"] = lagged.rolling(168, min_periods=2).std().fillna(0)
    return df


def _add_additional_ranks(df: pd.DataFrame) -> pd.DataFrame:
    """Add renewable_penetration_rank_30d and load_ramp_rank_30d."""
    df = df.copy()
    df = df.sort_values("ds").reset_index(drop=True)

    n = len(df)
    window = 30 * 24  # 30 days

    rp_rank = np.full(n, 0.5)
    lr_rank = np.full(n, 0.5)

    rp_vals = df["renew_penetration"].values if "renew_penetration" in df.columns else np.zeros(n)
    lr_vals = df["ramp_load"].values if "ramp_load" in df.columns else np.zeros(n)

    for i in range(max(1, min(10, n)), n):
        start = max(0, i - window)
        w_rp = rp_vals[start:i]
        w_lr = lr_vals[start:i]
        if len(w_rp) >= 10:
            rp_rank[i] = (w_rp < rp_vals[i]).sum() / len(w_rp)
            lr_rank[i] = (w_lr < lr_vals[i]).sum() / len(w_lr)

    df["renewable_penetration_rank_30d"] = rp_rank
    df["load_ramp_rank_30d"] = lr_rank
    return df


def _add_change_features(df: pd.DataFrame) -> pd.DataFrame:
    """24-hour change in key physical features."""
    df = df.copy()
    if "bidding_space" in df.columns:
        df["bidding_space_change_24h"] = df["bidding_space"].diff(24).fillna(0)
    else:
        df["bidding_space_change_24h"] = 0.0

    if "net_load" in df.columns:
        df["net_load_change_24h"] = df["net_load"].diff(24).fillna(0)
    else:
        df["net_load_change_24h"] = 0.0

    # renewable_change = wind_change + solar_change
    wind_change = df["wind"].diff(24).fillna(0) if "wind" in df.columns else 0.0
    solar_change = df["solar"].diff(24).fillna(0) if "solar" in df.columns else 0.0
    df["renewable_change_24h"] = wind_change + solar_change
    return df


def _add_exact_spring_festival(df: pd.DataFrame) -> pd.DataFrame:
    """Spring Festival features with exact 2026 date (Feb 17, 2026)."""
    df = df.copy()
    ds = pd.to_datetime(df["ds"])

    # Exact 2026 Spring Festival: Jan 29 is New Year's Eve, Feb 17 is the actual date
    # For 2026: Chinese New Year falls on February 17
    sf_date = pd.Timestamp("2026-02-17")

    # Window: 7 days before to 7 days after
    df["is_spring_festival_exact"] = (
        (ds >= sf_date - pd.Timedelta(days=7)) & (ds <= sf_date + pd.Timedelta(days=7))
    ).astype(int)

    df["days_to_spring_festival_exact"] = (sf_date - ds).dt.days.clip(-30, 30)
    df["days_after_spring_festival_exact"] = (ds - sf_date).dt.days.clip(-30, 30)

    return df


def _add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Hour/period x physical feature interactions."""
    df = df.copy()

    hour = df["hour"].values if "hour" in df.columns else np.ones(len(df))
    period_num = np.where(
        df["hour_business"].between(1, 8) if "hour_business" in df.columns else hour <= 8,
        1, np.where(
            df["hour_business"].between(9, 16) if "hour_business" in df.columns else hour <= 16,
            2, 3
        )
    ).astype(float)

    if "bidding_space" in df.columns:
        df["hour_x_bidding_space"] = hour * df["bidding_space"].values
        df["period_x_bidding_space"] = period_num * df["bidding_space"].values
    else:
        df["hour_x_bidding_space"] = 0.0
        df["period_x_bidding_space"] = 0.0

    if "net_load" in df.columns:
        df["hour_x_net_load"] = hour * df["net_load"].values
    else:
        df["hour_x_net_load"] = 0.0

    if "renew_penetration" in df.columns:
        df["period_x_renewable_penetration"] = period_num * df["renew_penetration"].values
    else:
        df["period_x_renewable_penetration"] = 0.0

    return df


def build_features_dayahead_v3(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build v3 features on top of the base + extended (v2) features.

    Parameters
    ----------
    df : DataFrame with columns ds, y, load, wind, solar, interconnect, etc.
         Must already have base + extended features from feature_builder_dayahead.

    Returns
    -------
    DataFrame with all v2 features + new v3 features.
    """
    from .feature_builder_dayahead import build_features_dayahead as build_v2

    # Start with v2 features
    result = build_v2(df, use_extended=True)

    # Add v3 features
    result = _add_volatility(result)
    result = _add_additional_ranks(result)
    result = _add_change_features(result)
    result = _add_exact_spring_festival(result)
    result = _add_interaction_features(result)

    # Fill NaN
    result = result.ffill().fillna(0)

    return result


def get_v3_feature_columns() -> list[str]:
    """Return the full v3 feature column list."""
    from .feature_builder_dayahead import get_dayahead_feature_columns
    v2_cols = get_dayahead_feature_columns(use_extended=True)
    return v2_cols + V3_NEW_COLUMNS
