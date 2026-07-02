"""
test_business_time.py — Tests for business_time module.
"""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SRC))

import pandas as pd
import numpy as np

from src.common.business_time import (
    business_time_mapping,
    infer_period,
    build_business_hour_grid,
)
from src.common.output_schema import make_long_table


def test_infer_period():
    assert infer_period(1) == "1_8"
    assert infer_period(8) == "1_8"
    assert infer_period(9) == "9_16"
    assert infer_period(16) == "9_16"
    assert infer_period(17) == "17_24"
    assert infer_period(24) == "17_24"

    try:
        infer_period(0)
        assert False, "Should raise"
    except ValueError:
        pass

    try:
        infer_period(25)
        assert False, "Should raise"
    except ValueError:
        pass

    print("✓ test_infer_period")


def test_business_time_mapping_00_to_24():
    """
    CRITICAL: physical 00:00 → business hour 24 of previous day.
    """
    ds = pd.Series([pd.Timestamp("2026-02-16 00:00:00")])
    result = business_time_mapping(ds)
    assert result["hour_business"].iloc[0] == 24, f"Got {result['hour_business'].iloc[0]}"
    assert str(result["business_day"].iloc[0]) == "2026-02-15", f"Got {result['business_day'].iloc[0]}"
    print("✓ test_business_time_mapping_00_to_24")


def test_business_time_mapping_01_to_23():
    """Physical hours 01:00-23:00 → hour_business 1-23 of same day."""
    for h in range(1, 24):
        ds = pd.Series([pd.Timestamp(f"2026-02-16 {h:02d}:00:00")])
        result = business_time_mapping(ds)
        assert result["hour_business"].iloc[0] == h, f"Hour {h}: got {result['hour_business'].iloc[0]}"
        assert str(result["business_day"].iloc[0]) == "2026-02-16", (
            f"Hour {h}: got {result['business_day'].iloc[0]}"
        )
    print("✓ test_business_time_mapping_01_to_23")


def test_build_business_hour_grid():
    """24-row grid: hours 1-24, business_day = target_day, hour 24 maps to next day."""
    grid = build_business_hour_grid("2026-02-16", "dayahead")
    assert len(grid) == 24, f"Expected 24, got {len(grid)}"
    assert sorted(grid["hour_business"].unique()) == list(range(1, 25))
    # hour 24's ds should be 2026-02-17 00:00
    hour24 = grid[grid["hour_business"] == 24]
    assert hour24["ds"].iloc[0] == pd.Timestamp("2026-02-17 00:00")
    print("✓ test_build_business_hour_grid")


def test_make_long_table():
    """make_long_table produces correct output columns and hour_business."""
    ds = pd.date_range("2026-02-16 01:00", "2026-02-17 00:00", freq="h")
    df = pd.DataFrame({"ds": ds, "y_pred": np.random.randn(24) * 50 + 200, "y_true": np.random.randn(24) * 50 + 200})

    result = make_long_table(df, model_name="catboost_sota", task="dayahead")
    assert len(result) == 24
    for col in ["task", "model_name", "target_day", "ds", "hour_business", "period", "y_pred", "y_true"]:
        assert col in result.columns, f"Missing column: {col}"
    assert sorted(result["hour_business"].unique()) == list(range(1, 25))
    # Hour 24 should map to next day's target_day conceptually
    print("✓ test_make_long_table")


if __name__ == "__main__":
    test_infer_period()
    test_business_time_mapping_00_to_24()
    test_business_time_mapping_01_to_23()
    test_build_business_hour_grid()
    test_make_long_table()
    print("\n✅ All business_time tests passed!")
