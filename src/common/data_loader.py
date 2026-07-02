"""
data_loader.py — Load raw CSV/Excel with Chinese column names.
Replicates the loading logic from lightGBM/train_fix.py but using pathlib.

The CSV has these columns (all Chinese):
    时刻, 日前电价, 实时电价, ... (see source_repo_scan.json)
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

# ── Column name constants (Chinese) ────────────────────────────────
COL_TIME = "时刻"
COL_DAYAHEAD = "日前电价"
COL_REALTIME = "实时电价"
COL_LOAD = "直调负荷预测值"
COL_WIND = "风电总加预测值"
COL_SOLAR = "光伏总加预测值"
COL_INTERCONNECT = "联络线受电负荷预测值"
COL_BIDDING_SPACE = "竞价空间预测值"
COL_NUCLEAR = "核电总加预测值"
COL_SELF_USE = "自备机组总加预测值"
COL_TEST = "试验机组总加预测值"
COL_LOCAL_PLANT = "地方电厂总加预测值"
COL_RENEWABLE = "新能源总加预测值"

TASK_TO_Y_COL = {
    "dayahead": COL_DAYAHEAD,
    "realtime": COL_REALTIME,
}


def load_data(
    data_path: str | Path,
    target: str = "realtime",
) -> pd.DataFrame:
    """
    Load and return a DataFrame with ds, y, and raw physical columns.
    Target can be 'dayahead' (日前电价) or 'realtime' (实时电价).
    """
    data_path = Path(data_path)
    # Read CSV (try GBK first, then UTF-8)
    if data_path.suffix in (".xlsx", ".xls"):
        raw = pd.read_excel(str(data_path))
    else:
        try:
            raw = pd.read_csv(str(data_path), encoding="gbk")
        except UnicodeDecodeError:
            raw = pd.read_csv(str(data_path), encoding="utf-8")

    y_col = TASK_TO_Y_COL.get(target, COL_REALTIME)

    df = pd.DataFrame()
    df["ds"] = pd.to_datetime(raw[COL_TIME], errors="coerce")
    df["y"] = pd.to_numeric(raw[y_col], errors="coerce")

    # Physical features
    df["load"] = pd.to_numeric(raw.get(COL_LOAD, np.nan), errors="coerce").ffill()
    df["wind"] = pd.to_numeric(raw.get(COL_WIND, np.nan), errors="coerce").ffill()
    df["solar"] = pd.to_numeric(raw.get(COL_SOLAR, np.nan), errors="coerce").ffill()
    df["interconnect"] = pd.to_numeric(raw.get(COL_INTERCONNECT, np.nan), errors="coerce").ffill()

    df = df.sort_values("ds").reset_index(drop=True)
    return df


def load_raw_full(data_path: str | Path) -> pd.DataFrame:
    """Load full DataFrame with ALL original columns, parsed types."""
    data_path = Path(data_path)
    if data_path.suffix in (".xlsx", ".xls"):
        raw = pd.read_excel(str(data_path))
    else:
        try:
            raw = pd.read_csv(str(data_path), encoding="gbk")
        except UnicodeDecodeError:
            raw = pd.read_csv(str(data_path), encoding="utf-8")
    raw[COL_TIME] = pd.to_datetime(raw[COL_TIME], errors="coerce")
    return raw
