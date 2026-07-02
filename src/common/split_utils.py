"""
split_utils.py — Time-based train/test splitting for walk-forward evaluation.

Always splits chronologically (no random shuffle).
"""

from __future__ import annotations

import pandas as pd
from typing import Optional


def time_split(
    df: pd.DataFrame,
    train_end: str,
    test_start: str,
    test_end: Optional[str] = None,
    min_train_rows: int = 2000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame chronologically.

    Parameters
    ----------
    df : DataFrame with 'ds' (datetime) column
    train_end : str, last timestamp for training (exclusive)
    test_start : str, first timestamp for testing (inclusive)
    test_end : str, optional, last timestamp for testing (inclusive)
    min_train_rows : int, minimum training rows required

    Returns
    -------
    train_df, test_df
    """
    train_end_dt = pd.to_datetime(train_end)
    test_start_dt = pd.to_datetime(test_start)

    train_df = df[df["ds"] < train_end_dt].copy()
    if test_end:
        test_end_dt = pd.to_datetime(test_end)
        test_df = df[(df["ds"] >= test_start_dt) & (df["ds"] <= test_end_dt)].copy()
    else:
        test_df = df[df["ds"] >= test_start_dt].copy()

    if len(train_df) < min_train_rows:
        raise ValueError(
            f"Training set too small: {len(train_df)} rows (min {min_train_rows})"
        )
    if len(test_df) == 0:
        raise ValueError("Test set is empty")

    return train_df, test_df


def walk_forward_windows(
    df: pd.DataFrame,
    start_date: str,
    end_date: str,
    train_months: int = 12,
) -> list[tuple[str, str, str, str]]:
    """
    Generate walk-forward window definitions.

    For each day D in [start_date, end_date]:
        train: D - train_months → D (exclusive D)
        test:  D 01:00 → D+1 00:00  (24 business hours)

    Returns list of (train_start, train_end, test_start, test_end) date strings.
    """
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)

    windows = []
    current = start_dt
    while current <= end_dt:
        train_start = current - pd.DateOffset(months=train_months)
        train_end = current  # exclusive
        test_start = current + pd.Timedelta(hours=1)   # business hour 1 = 01:00
        test_end = current + pd.Timedelta(hours=24)    # business hour 24 = next day 00:00

        windows.append((
            train_start.strftime("%Y-%m-%d %H:%M:%S"),
            train_end.strftime("%Y-%m-%d %H:%M:%S"),
            test_start.strftime("%Y-%m-%d %H:%M:%S"),
            test_end.strftime("%Y-%m-%d %H:%M:%S"),
        ))
        current += pd.Timedelta(days=1)

    return windows
