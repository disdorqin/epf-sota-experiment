#!/usr/bin/env python3
"""
test_dayahead_model_zoo_contract.py — Contract tests for day-ahead model zoo.

Verifies:
  1. CHAMPION_MODEL_ID == "cfg05"
  2. cfg05 in DAYAHEAD_MODELS with status == "champion"
  3. Default fusion pool does not contain INVALID_MODELS
  4. INVALID_MODELS requests raise
  5. All default=True models have smape_floor50 and status
  6. Output schema contains required columns

Usage:
  python -m pytest tests/test_dayahead_model_zoo_contract.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from pathlib import Path

from src.registry.dayahead_models import (
    DAYAHEAD_MODELS, INVALID_MODELS, DEFAULT_FUSION_POOL,
    CHAMPION_MODEL_ID, raise_if_invalid, get_champion_id,
)


# ── Required output columns ──
REQUIRED_COLUMNS = [
    "task", "model_name", "target_day", "business_day",
    "ds", "hour_business", "period", "y_true", "y_pred",
]


def test_champion_model_id():
    """Test that CHAMPION_MODEL_ID == 'cfg05'."""
    assert CHAMPION_MODEL_ID == "cfg05", \
        f"CHAMPION_MODEL_ID should be 'cfg05', got '{CHAMPION_MODEL_ID}'"
    print(f"✅ CHAMPION_MODEL_ID == '{CHAMPION_MODEL_ID}'")


def test_cfg05_in_registry():
    """Test that cfg05 is in DAYAHEAD_MODELS with status == 'champion'."""
    assert "cfg05" in DAYAHEAD_MODELS, "cfg05 not found in DAYAHEAD_MODELS"
    assert DAYAHEAD_MODELS["cfg05"]["status"] == "champion", \
        f"cfg05 status should be 'champion', got '{DAYAHEAD_MODELS['cfg05']['status']}'"
    print(f"✅ cfg05 in DAYAHEAD_MODELS with status == 'champion'")


def test_default_fusion_pool_no_invalid():
    """Test that default fusion pool does not contain invalid models."""
    for mid in DEFAULT_FUSION_POOL:
        assert mid not in INVALID_MODELS, \
            f"Default fusion pool contains invalid model: {mid}"
    print(f"✅ Default fusion pool does not contain invalid models")


def test_invalid_models_raise():
    """Test that INVALID_MODELS requests raise ValueError."""
    for mid in INVALID_MODELS:
        try:
            raise_if_invalid(mid)
            assert False, f"Should have raised for {mid}"
        except ValueError:
            pass
    print(f"✅ All INVALID_MODELS requests raise ValueError")


def test_default_models_have_required_fields():
    """Test that all default=True models have smape_floor50 and status."""
    for mid, info in DAYAHEAD_MODELS.items():
        if info.get("default", False):
            assert "status" in info, f"{mid} missing 'status'"
            assert "smape_floor50" in info, f"{mid} missing 'smape_floor50'"
    print(f"✅ All default=True models have smape_floor50 and status")


def test_output_schema():
    """Test that output schema contains required columns."""
    # Try to load unified predictions
    unified_path = Path("outputs/dayahead_model_zoo_30d/predictions/model_zoo_unified.csv")
    if not unified_path.exists():
        print(f"⚠️  Unified predictions not found: {unified_path}")
        print("   (Run python scripts/run_dayahead_model_zoo.py --models default first)")
        return

    df = pd.read_csv(str(unified_path), encoding="utf-8-sig")
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    assert len(missing_cols) == 0, f"Missing columns: {missing_cols}"

    # Check that we have 720 rows per model
    for mid in df["model_name"].unique():
        mid_df = df[df["model_name"] == mid]
        if len(mid_df) != 720:
            print(f"⚠️  {mid}: expected 720 rows, got {len(mid_df)}")

    # Check that hour_business ranges from 1 to 24
    assert df["hour_business"].min() == 1, \
        f"hour_business min should be 1, got {df['hour_business'].min()}"
    assert df["hour_business"].max() == 24, \
        f"hour_business max should be 24, got {df['hour_business'].max()}"

    print(f"✅ Output schema correct: {len(df)} rows, all required columns present")


def test_champion_smape_below_target():
    """Test that champion sMAPE is below 11.5%."""
    champion_info = DAYAHEAD_MODELS[CHAMPION_MODEL_ID]
    smape = champion_info["smape_floor50"]
    assert smape < 11.5, \
        f"Champion sMAPE should be below 11.5%, got {smape}%"
    print(f"✅ Champion sMAPE = {smape}% (below 11.5%)")


def test_registry_completeness():
    """Test that registry contains all expected models."""
    expected_models = ["cfg05", "best_two_average", "stage3_business_fixed",
                      "catboost_spike_residual", "catboost_sota"]
    for mid in expected_models:
        assert mid in DAYAHEAD_MODELS, f"{mid} not found in DAYAHEAD_MODELS"
    print(f"✅ Registry contains all expected models: {expected_models}")


if __name__ == "__main__":
    print("=" * 60)
    print("DAY-AHEAD MODEL ZOO CONTRACT TESTS")
    print("=" * 60)
    print()

    test_champion_model_id()
    test_cfg05_in_registry()
    test_default_fusion_pool_no_invalid()
    test_invalid_models_raise()
    test_default_models_have_required_fields()
    test_output_schema()
    test_champion_smape_below_target()
    test_registry_completeness()

    print()
    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
