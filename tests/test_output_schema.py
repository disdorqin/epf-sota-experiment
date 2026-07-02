"""
test_output_schema.py — Tests for output_schema module.
"""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SRC))

import pandas as pd
import numpy as np

from src.common.output_schema import make_long_table, REQUIRED_COLUMNS


def test_make_long_table_required_columns():
    """All required columns must be present in output."""
    ds = pd.date_range("2026-02-16 01:00", "2026-02-17 00:00", freq="h")
    df = pd.DataFrame({"ds": ds, "y_pred": np.random.randn(24) * 50 + 200})
    result = make_long_table(df, model_name="catboost_sota", task="dayahead")

    for col in REQUIRED_COLUMNS:
        assert col in result.columns, f"Missing required column: {col}"
    print("✓ test_make_long_table_required_columns")


def test_make_long_table_hour_range():
    """Output must have hour_business 1-24."""
    ds = pd.date_range("2026-02-16 01:00", "2026-02-17 00:00", freq="h")
    df = pd.DataFrame({"ds": ds, "y_pred": np.random.randn(24) * 50 + 200})
    result = make_long_table(df, model_name="catboost_sota", task="dayahead")
    hours = sorted(result["hour_business"].unique())
    assert hours == list(range(1, 25)), f"Hours: {hours}"
    print("✓ test_make_long_table_hour_range")


def test_make_long_table_24_rows():
    """Exactly 24 rows per day."""
    ds = pd.date_range("2026-02-16 01:00", "2026-02-17 00:00", freq="h")
    df = pd.DataFrame({"ds": ds, "y_pred": np.random.randn(24) * 50 + 200})
    result = make_long_table(df, model_name="catboost_sota", task="dayahead")
    assert len(result) == 24, f"Expected 24 rows, got {len(result)}"
    print("✓ test_make_long_table_24_rows")


def test_make_long_table_y_true():
    """y_true should carry through if provided."""
    ds = pd.date_range("2026-02-16 01:00", "2026-02-17 00:00", freq="h")
    y_true_vals = np.random.randn(24) * 50 + 200
    df = pd.DataFrame({"ds": ds, "y_pred": np.random.randn(24) * 50 + 200, "y_true": y_true_vals})
    result = make_long_table(df, model_name="catboost_sota", task="dayahead")
    assert np.allclose(result["y_true"].values, y_true_vals), "y_true mismatch"
    print("✓ test_make_long_table_y_true")


def test_make_long_table_task_model_name():
    """Task and model_name must be correctly set."""
    ds = pd.date_range("2026-02-16 01:00", "2026-02-17 00:00", freq="h")
    df = pd.DataFrame({"ds": ds, "y_pred": np.random.randn(24) * 50 + 200})
    result = make_long_table(df, model_name="catboost_sota", task="realtime")
    assert result["task"].iloc[0] == "realtime"
    assert result["model_name"].iloc[0] == "catboost_sota"
    print("✓ test_make_long_table_task_model_name")


if __name__ == "__main__":
    test_make_long_table_required_columns()
    test_make_long_table_hour_range()
    test_make_long_table_24_rows()
    test_make_long_table_y_true()
    test_make_long_table_task_model_name()
    print("\n✅ All output_schema tests passed!")
