"""
smoke_check.py — Quick smoke test for the SOTA experiment zone.

Checks:
1. All imports work
2. business_day / hour_business mapping correctness
3. sMAPE_floor50 runs without error
4. Output schema includes business_day
5. Prints SMOKE PASS at the end

Usage:
    python scripts/smoke_check.py
"""

from __future__ import annotations

import os
import sys

# ── Ensure src on path ──
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_DIR)

_PASSED = 0
_FAILED = 0


def check(description: str, condition: bool, detail: str = ""):
    global _PASSED, _FAILED
    if condition:
        _PASSED += 1
        print(f"  ✅ {description}")
    else:
        _FAILED += 1
        print(f"  ❌ {description}  {detail}")


def main():
    global _PASSED, _FAILED

    # Parse --data-path if provided
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default=None)
    cli_args, _ = parser.parse_known_args()

    print("=" * 60)
    print("SOTA Experiment — Smoke Check")
    print("=" * 60)

    # ── 1. Import check ──
    print("\n[1] Import check:")
    try:
        import numpy as np
        import pandas as pd
        import yaml
        check("numpy, pandas, pyyaml imported", True)
    except ImportError as e:
        check(f"Core imports failed: {e}", False)

    try:
        from src.common.data_loader import load_data
        from src.common.business_time import business_time_mapping, infer_period, build_business_hour_grid
        from src.common.metrics import smape_floor50, compute_all_metrics
        from src.common.output_schema import make_long_table
        from src.common.feature_builder import build_features
        check("All src.common modules imported", True)
    except ImportError as e:
        check(f"src.common imports failed: {e}", False)

    try:
        from src.models.catboost_adapter import CatBoostAdapter
        check("CatBoostAdapter imported", True)
    except ImportError as e:
        check(f"CatBoostAdapter import failed: {e}", False)

    try:
        from src.models.chronos_adapter import ChronosAdapter
        check("ChronosAdapter imported", True)
    except ImportError as e:
        check(f"ChronosAdapter import failed: {e}", False)

    # ── 2. Business time mapping ──
    print("\n[2] Business time mapping:")
    import pandas as pd
    import numpy as np
    from src.common.business_time import business_time_mapping

    # Case 1: 2026-02-02 00:00:00 → business_day=2026-02-01, hour_business=24
    ts_00 = pd.Series([pd.Timestamp("2026-02-02 00:00:00")])
    biz_00 = business_time_mapping(ts_00)
    check("00:00 → hour_business=24",
          biz_00["hour_business"].iloc[0] == 24,
          f"Got {biz_00['hour_business'].iloc[0]}")
    check("00:00 → business_day=prev day",
          str(biz_00["business_day"].iloc[0]) == "2026-02-01",
          f"Got {biz_00['business_day'].iloc[0]}")

    # Case 2: 2026-02-01 01:00:00 → business_day=2026-02-01, hour_business=1
    ts_01 = pd.Series([pd.Timestamp("2026-02-01 01:00:00")])
    biz_01 = business_time_mapping(ts_01)
    check("01:00 → hour_business=1",
          biz_01["hour_business"].iloc[0] == 1,
          f"Got {biz_01['hour_business'].iloc[0]}")
    check("01:00 → business_day=same day",
          str(biz_01["business_day"].iloc[0]) == "2026-02-01",
          f"Got {biz_01['business_day'].iloc[0]}")

    # Case 3: All hours 1-24 from build_business_hour_grid
    from src.common.business_time import build_business_hour_grid
    grid = build_business_hour_grid("2026-02-01", "dayahead")
    check("build_business_hour_grid → 24 rows", len(grid) == 24, f"Got {len(grid)}")
    check("build_business_hour_grid → hours 1-24",
          sorted(grid["hour_business"].unique()) == list(range(1, 25)))

    # ── 3. sMAPE_floor50 ──
    print("\n[3] Metrics:")
    from src.common.metrics import smape_floor50, compute_all_metrics
    y_true = np.array([100.0, 200.0, 300.0, -10.0, 400.0])
    y_pred = np.array([110.0, 190.0, 290.0, -5.0, 420.0])
    smape_val = smape_floor50(y_true, y_pred)
    check("sMAPE_floor50 runs without error", not np.isnan(smape_val), f"Got {smape_val:.4f}")
    metrics = compute_all_metrics(y_true, y_pred)
    for k in ["MAE", "RMSE", "sMAPE_floor50", "peak_MAE_q90"]:
        check(f"  {k} present in compute_all_metrics", k in metrics)

    # ── 4. Output schema with business_day ──
    print("\n[4] Output schema:")
    from src.common.output_schema import make_long_table, REQUIRED_COLUMNS
    ds = pd.date_range("2026-02-01 01:00", "2026-02-02 00:00", freq="h")
    test_df = pd.DataFrame({
        "ds": ds,
        "y_pred": np.random.randn(24) * 50 + 200,
        "y_true": np.random.randn(24) * 50 + 200,
    })
    result = make_long_table(test_df, model_name="catboost_sota", task="dayahead")
    check("business_day column present", "business_day" in result.columns)
    check("24 rows output", len(result) == 24, f"Got {len(result)}")

    # Verify the business_day for hour=24
    hour24 = result[result["hour_business"] == 24]
    if len(hour24) == 1:
        check("hour 24 → ds = D+1 00:00",
              str(hour24["ds"].iloc[0]) == "2026-02-02 00:00:00",
              f"Got {hour24['ds'].iloc[0]}")
        check("hour 24 → business_day = D",
              str(hour24["business_day"].iloc[0]) == "2026-02-01",
              f"Got {hour24['business_day'].iloc[0]}")
        check("hour 24 → target_day = D",
              str(hour24["target_day"].iloc[0]) == "2026-02-01",
              f"Got {hour24['target_day'].iloc[0]}")

    # All required columns
    missing = [c for c in REQUIRED_COLUMNS if c not in result.columns]
    check("All required columns present", len(missing) == 0, f"Missing: {missing}")

    # ── 5. Feature builder with bidding_space_raw ──
    print("\n[5] Feature builder:")
    from src.common.data_loader import load_data
    # Try to find data
    data_path = cli_args.data_path if cli_args.data_path else None
    if data_path is None:
        data_path = os.environ.get("SOTA_DATA_PATH")
    if data_path is None:
        try:
            from src.common.repo_paths import get_data_path
            data_path = str(get_data_path())
        except FileNotFoundError:
            data_path = None

    if data_path and os.path.exists(data_path):
        df = load_data(data_path, target="dayahead")
        check("bidding_space_raw loaded", "bidding_space_raw" in df.columns)

        full = build_features(df)
        check("Feature engineering runs", len(full) > 0)
        check("bidding_space in features", "bidding_space" in full.columns)
        if "bidding_space" in full.columns:
            check("bidding_space has valid values",
                  full["bidding_space"].notna().sum() > 0)
    else:
        print("  ⚠️  Data file not found in any default location — skipping feature builder check.")
        print(f"     Provide data path via: python scripts/smoke_check.py --data-path=...")

    # ── Summary ──
    print(f"\n{'=' * 60}")
    total = _PASSED + _FAILED
    print(f"Results: {_PASSED}/{total} passed, {_FAILED} failed")
    if _FAILED == 0:
        print("\n  🎉 SMOKE PASS")
    else:
        print(f"\n  ❌ SMOKE FAIL — {_FAILED} check(s) failed")
    print(f"{'=' * 60}")

    return 0 if _FAILED == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
