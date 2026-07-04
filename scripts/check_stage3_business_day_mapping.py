#!/usr/bin/env python3
"""
check_stage3_business_day_mapping.py

Verifies that Stage3's business-day / business-hour mapping is correct.

Rule:
  business_day D has hour_business 1-24 mapping to ds = D 01:00 ... D+1 00:00
  Hour 24 → business_day = D (NOT D+1)

Checks:
  1. target_day=2026-02-01 has exactly 24 rows
  2. Those 24 rows have ds: 2026-02-01 01:00:00 ... 2026-02-02 00:00:00
  3. hour_business = 1..24
  4. hour_business=24 → target_day = 2026-02-01
  5. No natural-day (00:00-23:00) grouping
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
from src.common.business_time import business_time_mapping
from src.common.data_loader import load_data
from src.common.repo_paths import get_data_path


def check_business_day_mapping(df, label="Stage3"):
    """Run all checks on a DataFrame. Returns (passed, list of messages)."""
    messages = []
    passed = True

    def _check(cond, msg):
        nonlocal passed
        if cond:
            messages.append(f"  ✅ {msg}")
        else:
            messages.append(f"  ❌ {msg}")
            passed = False

    # Ensure sorted
    df = df.sort_values("ds").reset_index(drop=True)

    # Check columns exist
    for col in ["target_day", "business_day", "hour_business", "period", "ds"]:
        assert col in df.columns, f"Missing column: {col}"

    # Pick a representative day
    test_day = "2026-02-01"
    day_df = df[df["target_day"] == test_day]

    _check(len(day_df) == 24, f"1. target_day={test_day} has {len(day_df)} rows (expect 24)")

    if len(day_df) == 24:
        day_df = day_df.sort_values("hour_business")

        # Check ds sequence: hours 1-23 = D HH:00:00, hour 24 = D+1 00:00:00
        actual_ds = day_df["ds"].astype(str).values[:24]
        # Build expected: hours 1-23 = f"{test_day} {h:02d}:00:00", hour 24 = f"D+1 00:00:00"
        expected_ds = [f"{test_day} {h:02d}:00:00" for h in range(1, 24)]
        expected_ds.append(f"2026-02-02 00:00:00")
        ds_match = all(a == e for a, e in zip(actual_ds, expected_ds))
        _check(ds_match, f"2. ds sequence matches business hours (1-24)")
        if not ds_match:
            for i in range(24):
                if actual_ds[i] != expected_ds[i]:
                    messages.append(f"     Mismatch at hour {i+1}: got '{actual_ds[i]}', expected '{expected_ds[i]}'")

        # Show the mapping
        messages.append(f"     Hour 1: ds={actual_ds[0]}, target_day={day_df.iloc[0]['target_day']}")
        messages.append(f"     Hour 24: ds={actual_ds[23]}, target_day={day_df.iloc[23]['target_day']}")

        # Check hour_business range
        hours = sorted(day_df["hour_business"].unique())
        _check(
            list(hours) == list(range(1, 25)),
            f"3. hour_business = {min(hours)}..{max(hours)} (expect 1..24)"
        )

        # Check hour 24 → target_day = D (business day, not D+1)
        h24 = day_df[day_df["hour_business"] == 24]
        _check(
            len(h24) == 1 and h24.iloc[0]["target_day"] == test_day,
            f"4. hour 24 target_day = {h24.iloc[0]['target_day'] if len(h24) > 0 else 'N/A'} (expect {test_day})"
        )

        # Verify ds for hour 24 is D+1 00:00:00
        h24_ds = str(h24.iloc[0]["ds"]) if len(h24) > 0 else "N/A"
        expected_h24_ds = f"2026-02-02 00:00:00"
        _check(
            h24_ds == expected_h24_ds,
            f"     hour 24 ds = {h24_ds} (expect {expected_h24_ds})"
        )

    # Check that ALL 30 evaluation business days have exactly 24 rows
    eval_days = df[(df["target_day"] >= "2026-02-01") & (df["target_day"] <= "2026-03-02")]
    biz_day_counts = eval_days.groupby("target_day").size()
    non_24 = [d for d, c in biz_day_counts.items() if c != 24]
    _check(
        len(non_24) == 0,
        f"5. All 30 evaluation business days have exactly 24 rows " +
        f"(found {len(non_24)} exceptions: {non_24[:3]})"
    )

    return passed, messages


def _build_v3_features_direct(raw):
    """Build v3 features (same logic as run_lightgbm_stage3.build_v3_features, without optuna)."""
    from src.common.feature_builder import build_features as build_base
    from src.common.feature_builder_dayahead import (
        _add_lag_features, _add_same_hour_stats, _add_price_momentum,
        _add_calendar_features,
    )
    from src.common.feature_builder_dayahead_v3 import (
        _add_volatility, _add_change_features,
        _add_exact_spring_festival, _add_interaction_features,
    )
    df = build_base(raw)
    biz = business_time_mapping(df["ds"])
    df["business_day"] = biz["business_day"].astype(str)
    df["hour_business"] = biz["hour_business"]
    df["period"] = biz["period"]
    df["target_day"] = df["business_day"]
    df = _add_lag_features(df)
    df = _add_same_hour_stats(df)
    df = _add_price_momentum(df)
    df = _add_calendar_features(df)
    df = df.sort_values("ds").reset_index(drop=True)
    df["net_load_rank_30d"] = _fast_rank_rolling(df["net_load"], 720)
    df["bidding_space_rank_30d"] = _fast_rank_rolling(df["bidding_space"], 720)
    df = _add_volatility(df)
    df = _add_change_features(df)
    df = _add_exact_spring_festival(df)
    df = _add_interaction_features(df)
    df = df.ffill().fillna(0).reset_index(drop=True)
    return df


def _fast_rank_rolling(series, window=720):
    return series.rolling(window, min_periods=max(10, window//4)).apply(
        lambda x: (x < x[-1]).sum() / len(x) if len(x) >= 10 else 0.5,
        raw=True,
    ).fillna(0.5)


def main():
    print("=" * 65)
    print("STAGE 3 BUSINESS DAY MAPPING CHECK")
    print("=" * 65)
    print()

    # Build features directly (same code path as Stage3, without Optuna import)
    print("Loading data and building v3 features...")
    data_path = str(get_data_path())
    raw = load_data(data_path, target="dayahead")
    df = _build_v3_features_direct(raw)
    df = df[df["ds"] >= "2025-08-01"].reset_index(drop=True)

    print(f"Feature DF: {len(df)} rows ({df['ds'].min()} → {df['ds'].max()})")
    print()

    passed, messages = check_business_day_mapping(df, "Stage3")

    print()
    for m in messages:
        print(m)

    print()
    print("=" * 65)
    if passed:
        print("ALL CHECKS PASSED ✅")
    else:
        print("SOME CHECKS FAILED ❌")
    print("=" * 65)

    # Also check that 2026-02-01 has hour 24 correctly
    print()
    print("Full 30-day view:")
    for d in sorted(df["target_day"].unique()):
        if d < "2026-02-01" or d > "2026-03-02":
            continue
        dd = df[df["target_day"] == d]
        h24 = dd[dd["hour_business"] == 24]
        h24_ds = str(h24.iloc[0]["ds"]) if len(h24) > 0 else "MISSING"
        print(f"  Business day {d}: {len(dd)} rows, hour24 ds={h24_ds}")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
