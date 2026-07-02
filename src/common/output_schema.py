"""
output_schema.py — Long-table output format matching fusion/contracts.py.

Final output (long table) has columns:
    task, model_name, target_day, ds, hour_business, period, y_pred, y_true,
    source, run_mode, created_at
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from datetime import datetime

from .business_time import business_time_mapping, infer_period

# ── Schema ─────────────────────────────────────────────────────────
REQUIRED_COLUMNS = [
    "task",
    "model_name",
    "target_day",
    "ds",
    "hour_business",
    "period",
    "y_true",
    "y_pred",
]

VALID_TASKS = {"dayahead", "realtime"}
VALID_PERIODS = {"1_8", "9_16", "17_24"}


def make_long_table(
    df: pd.DataFrame,
    model_name: str,
    task: str,
    source: str = "sota_exp",
    run_mode: str = "eval",
) -> pd.DataFrame:
    """
    Build a standardized long-table prediction output.

    Input df must have at minimum: ds, y_pred
    Optional: y_true, target_day

    The function:
    1. Derives business_day and hour_business from ds
    2. Infers period from hour_business
    3. Sets default / fills missing columns
    """
    out = df.copy()
    n = len(out)

    # ── Derive business time ──
    biz = business_time_mapping(out["ds"])
    out["hour_business"] = biz["hour_business"]
    out["period"] = biz["period"]

    # ── target_day: use provided or derive from business_day ──
    if "target_day" not in out.columns:
        out["target_day"] = biz["business_day"].astype(str)
    else:
        out["target_day"] = pd.to_datetime(out["target_day"]).dt.strftime("%Y-%m-%d")

    # ── y_true ──
    if "y_true" not in out.columns:
        out["y_true"] = np.nan

    # ── Metadata columns ──
    out["task"] = task
    out["model_name"] = model_name
    out["source"] = source
    out["run_mode"] = run_mode
    out["created_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── Ensure required columns exist ──
    for col in REQUIRED_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan

    return out
