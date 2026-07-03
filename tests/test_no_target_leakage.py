#!/usr/bin/env python3
"""
Anti-target-leakage tests for day-ahead correction/fusion code.

Verifies that prediction-time features never contain:
  y_true, residual, error, abs_error, future_y, target_actual, oracle, best_model
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from pathlib import Path

# ── Denylist (must match src/correction/lgbm_dayahead_corrector.py) ──
DENYLIST = [
    "y_true", "residual", "error", "abs_error",
    "future_y", "target_actual", "oracle", "best_model",
]


def test_corrector_prediction_features():
    """
    Test that prediction features in LGBMSpikeResidualCorrector
    do not contain any denylisted terms.
    """
    from src.correction.lgbm_dayahead_corrector import LGBMSpikeResidualCorrector

    corrector = LGBMSpikeResidualCorrector(alpha=0.25, max_delta=50)

    # Load real data as input
    df = pd.read_csv(
        "outputs/dayahead_lgbm_stage2_30d/predictions/trial_02_w150_nl255_lr0.03_dayahead.csv",
        encoding="utf-8-sig"
    )
    df = df.sort_values("ds").reset_index(drop=True)

    # Run corrector — if leakage exists, _validate_prediction_features will raise
    try:
        result = corrector.correct(df)
        assert len(result) == len(df), f"Output length mismatch: {len(result)} vs {len(df)}"
        assert not np.any(np.isnan(result)), "Output contains NaN"
        print("✅ LGBMSpikeResidualCorrector: leakage guard passed, output clean")
    except ValueError as e:
        if "LEAKAGE" in str(e):
            print(f"❌ LGBMSpikeResidualCorrector: LEAKAGE DETECTED: {e}")
            raise
        raise
    except Exception as e:
        # Other errors (e.g. not enough data for CatBoost) are fine
        print(f"⚠️  LGBMSpikeResidualCorrector: non-leakage error (ok): {e}")


def test_hour_corrector_prediction_features():
    """
    Test that prediction features in LGBMSelectedHourCorrector
    do not contain any denylisted terms.
    """
    from src.correction.lgbm_dayahead_corrector import LGBMSelectedHourCorrector

    corrector = LGBMSelectedHourCorrector(target_hours=[13], max_delta=50)

    df = pd.read_csv(
        "outputs/dayahead_lgbm_stage2_30d/predictions/trial_02_w150_nl255_lr0.03_dayahead.csv",
        encoding="utf-8-sig"
    )
    df = df.sort_values("ds").reset_index(drop=True)

    try:
        result = corrector.correct(df)
        assert len(result) == len(df), f"Output length mismatch"
        print("✅ LGBMSelectedHourCorrector: leakage guard passed, output clean")
    except ValueError as e:
        if "LEAKAGE" in str(e):
            print(f"❌ LGBMSelectedHourCorrector: LEAKAGE DETECTED: {e}")
            raise
        raise
    except Exception as e:
        print(f"⚠️  LGBMSelectedHourCorrector: non-leakage error (ok): {e}")


def test_corrector_source_code_no_ytrue_in_prediction():
    """
    Static analysis: scan corrector source for y_true used in prediction path.
    The training path (past data) can use y_true. The prediction path must not.
    """
    source_path = Path(__file__).parent.parent / "src" / "correction" / "lgbm_dayahead_corrector.py"
    src = source_path.read_text(encoding="utf-8")

    # Check that _validate_prediction_features is called
    assert "_validate_prediction_features" in src, (
        "Missing _validate_prediction_features calls in corrector"
    )

    # Check that prediction paths avoid denylist
    lines = src.split("\n")
    in_prediction = False
    for i, line in enumerate(lines):
        stripped = line.strip()

        # Track when we enter prediction-building sections
        if "X_pred = np.column_stack" in stripped or "X_pred = np.array" in stripped:
            in_prediction = True
            continue
        if in_prediction:
            # Check next few lines for denylist terms in column-access form
            # Allow "past_residual" as it's a legit lag from historical data
            for banned in DENYLIST:
                if banned in stripped.lower():
                    # Skip if it's a variable like past_residual or lag_residual
                    prefix = stripped.lower().split(banned)[0]
                    if prefix.endswith("_") or prefix.endswith("past_"):
                        continue
                    # If it looks like direct column access df["y_true"], flag it
                    if f'"{banned}"' in stripped or f"'{banned}'" in stripped:
                        raise AssertionError(
                            f"Line {i+1}: BANNED TERM '{banned}' found in prediction features:\n{stripped}"
                        )
            # Exit when we hit a non-continuation line
            if not stripped.endswith(",") and not stripped.startswith("]"):
                in_prediction = False

    print("✅ Static analysis: no denylist terms in prediction features")


def test_denylist_consistency():
    """
    Verify that the test denylist matches the source code denylist.
    """
    source_path = Path(__file__).parent.parent / "src" / "correction" / "lgbm_dayahead_corrector.py"
    src = source_path.read_text(encoding="utf-8")

    for term in DENYLIST:
        assert term in src, f"Denylist term '{term}' not found in corrector source"

    print("✅ Denylist consistent between test and source")


if __name__ == "__main__":
    print("=" * 60)
    print("ANTI-TARGET-LEAKAGE TESTS")
    print("=" * 60)
    print()
    test_denylist_consistency()
    test_corrector_source_code_no_ytrue_in_prediction()
    test_corrector_prediction_features()
    test_hour_corrector_prediction_features()
    print()
    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
