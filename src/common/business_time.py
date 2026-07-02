"""
business_time.py — Business-day / business-hour mapping.

CRITICAL RULE (matching original repo):
    Physical time 00:00 → business hour 24 of the PREVIOUS business day.
    Achieved via: (ds - 1s).hour + 1 = 1..24
    So business_day = (ds - 1s).date
"""

from __future__ import annotations

import pandas as pd
import numpy as np

# ── Period definitions ─────────────────────────────────────────────
VALID_PERIODS = {"1_8", "9_16", "17_24"}


def infer_period(hour_business: int) -> str:
    """Map business hour (1-24) to period label."""
    h = int(hour_business)
    if 1 <= h <= 8:
        return "1_8"
    if 9 <= h <= 16:
        return "9_16"
    if 17 <= h <= 24:
        return "17_24"
    raise ValueError(f"hour_business out of range: {h}")


def business_time_mapping(ds: pd.Series) -> pd.DataFrame:
    """
    Convert physical timestamp Series to business-time DataFrame.

    Returns DataFrame with columns:
        ds              original timestamp
        business_day    date of the business day (YYYY-MM-DD)
        hour_business   business hour (1-24)
        period          period label (1_8 / 9_16 / 17_24)
    """
    adjusted = ds - pd.Timedelta(seconds=1)
    hour_business = (adjusted.dt.hour + 1).astype(int)
    business_day = adjusted.dt.date

    out = pd.DataFrame({
        "ds": ds,
        "business_day": business_day,
        "hour_business": hour_business,
    })
    out["period"] = out["hour_business"].map(infer_period)
    return out


def build_business_hour_grid(target_day: str, target: str = "dayahead") -> pd.DataFrame:
    """
    Build the standard 24-row grid for a given target business day.

    For a business day D, business hours 1-24 correspond to:
        hour 1..24  →  ds = D 01:00 .. D 24:00  (physical time)
    Note: physical D+1 00:00 → business hour 24, D 24:00 does NOT exist,
    so we generate from D 01:00 to D+1 00:00.
    """
    day_dt = pd.Timestamp(target_day)
    # Generate 24 timestamps: business hour 1..24
    ds_list = [day_dt + pd.Timedelta(hours=h) for h in range(1, 25)]
    df = pd.DataFrame({"ds": ds_list})
    biz = business_time_mapping(df["ds"])
    df["business_day"] = biz["business_day"]
    df["hour_business"] = biz["hour_business"]
    df["period"] = biz["period"]
    df["task"] = target
    return df
