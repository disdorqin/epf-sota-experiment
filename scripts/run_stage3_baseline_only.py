#!/usr/bin/env python3
"""
run_stage3_baseline_only.py — Run Stage3 baseline configuration ONLY (no Optuna)
with correct business-day mapping from business_time_mapping.

Usage:
    python scripts/run_stage3_baseline_only.py
"""
import sys, os, json, logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from src.common.metrics import smape_floor50, compute_all_metrics
from src.common.data_loader import load_data
from src.common.repo_paths import get_data_path
from src.common.business_time import business_time_mapping
from src.common.feature_builder import build_features as build_base
from src.common.feature_builder_dayahead import (
    _add_lag_features, _add_same_hour_stats, _add_price_momentum, _add_calendar_features,
)
from src.common.feature_builder_dayahead_v3 import (
    _add_volatility, _add_change_features, _add_exact_spring_festival, _add_interaction_features,
)
import lightgbm as lgb

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

OUT_ROOT = "outputs/dayahead_lgbm_stage3_business_fixed_30d"

EVAL_START = "2026-02-01"
EVAL_END = "2026-03-02"


def _fast_rank_rolling(series, window=720):
    return series.rolling(window, min_periods=max(10, window // 4)).apply(
        lambda x: (x < x[-1]).sum() / len(x) if len(x) >= 10 else 0.5,
        raw=True,
    ).fillna(0.5)


def build_v3_features(raw):
    df = build_base(raw)
    # Use business_time_mapping for correct day/hour mapping
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


def get_feature_cols(df):
    exclude = {
        "ds", "y", "target_day", "business_day", "hour_business", "period",
        "date_only", "y_pred", "y_true", "model_name", "task",
        "lag_48h_raw", "lag_168h_raw",
    }
    numeric = df.select_dtypes(include=[np.float64, np.int64, np.float32, np.int32, np.int8, bool]).columns
    return [c for c in numeric if c not in exclude and c != "y"]


def main():
    output_root = Path(OUT_ROOT)
    for sub in ["predictions", "metrics", "reports"]:
        (output_root / sub).mkdir(parents=True, exist_ok=True)

    logger.info("Loading data and building v3 features...")
    raw = load_data(str(get_data_path()), target="dayahead")
    df = build_v3_features(raw)
    df = df[df["ds"] >= "2025-08-01"].reset_index(drop=True)
    logger.info(f"Feature DF: {len(df)} rows ({df['ds'].min()} -> {df['ds'].max()})")

    feat_cols = get_feature_cols(df)
    logger.info(f"Feature columns: {len(feat_cols)}")

    # Evaluation days
    all_days = sorted(df[
        (df["ds"] >= EVAL_START) & (df["ds"] <= f"{EVAL_END} 23:00:00")
    ]["target_day"].unique())
    logger.info(f"Evaluation: {len(all_days)} days ({all_days[0]} -> {all_days[-1]})")

    # Baseline config (from Stage3)
    params = {
        "num_leaves": 127, "min_data_in_leaf": 50,
        "lambda_l1": 0.1, "lambda_l2": 2.0,
        "learning_rate": 0.02, "feature_fraction": 0.85,
        "bagging_fraction": 0.85, "bagging_freq": 5,
        "objective": "mae", "n_estimators": 2000,
    }
    MAX_TRAIN_ROWS = 5000
    window = 90
    preds = []

    for day in all_days:
        target_dt = pd.Timestamp(day)
        train_all = df[df["target_day"] < day]
        if len(train_all) < 200:
            continue

        train_start = target_dt - timedelta(days=window)
        train_df = train_all[train_all["ds"] >= train_start].copy()
        if len(train_df) > MAX_TRAIN_ROWS:
            train_df = train_df.tail(MAX_TRAIN_ROWS)
        if len(train_df) < 100:
            continue

        val_start = target_dt - timedelta(days=30)
        val_df = train_all[train_all["ds"].between(val_start, target_dt - timedelta(hours=1))]
        if len(val_df) > 2000:
            val_df = val_df.tail(2000)

        X_tr = train_df[feat_cols].values.astype(float)
        y_tr = train_df["y"].values.astype(float)

        try:
            lgb_params = {
                "boosting_type": "gbdt",
                "num_leaves": params["num_leaves"],
                "min_data_in_leaf": params["min_data_in_leaf"],
                "lambda_l1": params["lambda_l1"],
                "lambda_l2": params["lambda_l2"],
                "learning_rate": params["learning_rate"],
                "feature_fraction": params["feature_fraction"],
                "bagging_fraction": params["bagging_fraction"],
                "bagging_freq": params["bagging_freq"],
                "objective": params["objective"],
                "metric": params["objective"],
                "verbosity": -1,
            }
            if len(val_df) >= 50:
                X_val = val_df[feat_cols].values.astype(float)
                y_val = val_df["y"].values.astype(float)
                model = lgb.train(
                    lgb_params, lgb.Dataset(X_tr, y_tr),
                    num_boost_round=params["n_estimators"],
                    valid_sets=[lgb.Dataset(X_val, y_val)],
                    valid_names=["eval"],
                    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
                )
            else:
                model = lgb.train(
                    lgb_params, lgb.Dataset(X_tr, y_tr),
                    num_boost_round=params["n_estimators"],
                    callbacks=[lgb.log_evaluation(0)],
                )

            day_df = df[df["target_day"] == day].copy()
            X_pred = day_df[feat_cols].values.astype(float)
            day_df["y_pred"] = model.predict(X_pred)
            day_df["y_true"] = day_df["y"].values
            day_df["model_name"] = "stage3_baseline_90d_mae"
            preds.append(day_df)
            del model
        except Exception as e:
            logger.debug(f"  Day {day}: {e}")
            continue

    if not preds:
        logger.error("No predictions generated!")
        return

    full = pd.concat(preds, ignore_index=True)
    logger.info(f"Predictions: {len(full)} rows")

    # ── Save ──
    full.to_csv(str(output_root / "predictions" / "stage3_baseline_90d_mae_dayahead.csv"),
                index=False, encoding="utf-8-sig")

    # Metrics
    s_smape = smape_floor50(full["y_true"].values, full["y_pred"].values)
    m = compute_all_metrics(full["y_true"].values, full["y_pred"].values)
    m["model_name"] = "stage3_baseline_90d_mae"
    m["task"] = "dayahead"
    m["n"] = len(full)
    summary = pd.DataFrame([m])
    summary.to_csv(str(output_root / "metrics" / "summary.csv"), index=False, encoding="utf-8-sig")

    # Hour metrics
    hour_rows = []
    for h, grp in full.groupby("hour_business"):
        hm = compute_all_metrics(grp["y_true"].values, grp["y_pred"].values)
        hm["hour_business"] = int(h)
        hour_rows.append(hm)
    pd.DataFrame(hour_rows).to_csv(str(output_root / "metrics" / "hour_metrics.csv"),
                                    index=False, encoding="utf-8-sig")

    # Period metrics
    period_rows = []
    for p, grp in full.groupby("period"):
        pm = compute_all_metrics(grp["y_true"].values, grp["y_pred"].values)
        pm["period"] = p
        period_rows.append(pm)
    pd.DataFrame(period_rows).to_csv(str(output_root / "metrics" / "period_metrics.csv"),
                                      index=False, encoding="utf-8-sig")

    # ── Report ──
    trusted_champion = 11.85  # best_two_average
    old_stage3_natural = 11.64  # old (invalid) Stage3 result

    lines = []
    lines.append("# Stage3 Business-Day Fixed Report")
    lines.append(f"> Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"> Fix: business_time_mapping replaces ds.date() for target_day")
    lines.append("")
    lines.append("## Result")
    lines.append("")
    lines.append(f"**stage3_baseline_90d_mae = {s_smape:.4f}%**")
    lines.append(f"Rows: {len(full)}")
    lines.append(f"")
    lines.append("## Comparison")
    lines.append("")
    lines.append("| Model | sMAPE | Note |")
    lines.append("|-------|:-----:|------|")
    lines.append(f"| Stage3 (business fixed) | **{s_smape:.4f}%** | This run |")
    lines.append(f"| Old Stage3 (natural day) | {old_stage3_natural:.2f}% | ⚠️ Invalid (natural-day grouping) |")
    lines.append(f"| Trusted champion (best_two_average) | {trusted_champion:.2f}% | Clean reference |")
    lines.append(f"|")
    below = s_smape < trusted_champion
    improvement = trusted_champion - s_smape
    lines.append(f"## Target Check")
    lines.append(f"")
    lines.append(f"| Target | Status |")
    lines.append(f"|:-------|:------:|")
    lines.append(f"| Business-day mapping fixed? | ✅ |")
    lines.append(f"| 720 rows? | {'✅' if len(full) == 720 else '❌'} {len(full)} rows |")
    lines.append(f"| Below old Stage3 ({old_stage3_natural:.2f}%)? | {'✅' if s_smape < old_stage3_natural else '❌'} |")
    lines.append(f"| **Below trusted champion ({trusted_champion:.2f}%)?** | **{'✅' if below else '❌'} {s_smape:.2f}%** |")
    lines.append(f"| Below 11.5%? | {'✅' if s_smape < 11.5 else '❌'} |")
    lines.append(f"| Below 11%? | {'✅' if s_smape < 11 else '❌'} |")
    lines.append(f"")
    if below:
        lines.append(f"**Stage3 is {improvement:.2f}pp better than trusted champion.**")
        lines.append(f"Stage3 CAN be considered as new champion candidate.")
    else:
        lines.append(f"**Stage3 is NOT better than trusted champion.**")
        lines.append(f"The old 11.64% was an artifact of natural-day grouping error.")
        lines.append(f"Trusted champion (11.85%) remains the best verified result.")
    lines.append("")

    report = "\n".join(lines)
    report_path = output_root / "reports" / "stage3_business_fixed_report.md"
    report_path.write_text(report, encoding="utf-8")
    logger.info(f"Report: {report_path}")

    print()
    print("=" * 65)
    print("STAGE3 BUSINESS-FIXED RESULT")
    print("=" * 65)
    print(f"  sMAPE_floor50: {s_smape:.4f}%")
    print(f"  Rows: {len(full)}")
    print(f"  vs trusted champion (11.85%): {'BEATEN' if below else 'NOT BEATEN'}")
    print(f"  vs old Stage3 natural-day (11.64%): {'BEATEN' if s_smape < 11.64 else 'WORSE'}")
    print("=" * 65)


if __name__ == "__main__":
    main()
