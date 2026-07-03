"""
run_dayahead_regime_catboost_v2.py — Run 3 regime v2 CatBoost models.

Usage:
    python scripts/run_dayahead_regime_catboost_v2.py ^
        --input-pred outputs/dayahead_30d_core/predictions/catboost_sota_dayahead.csv ^
        --output-root outputs/dayahead_regime_v2_30d
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_DIR)

from src.common.metrics import compute_all_metrics
from src.common.output_schema import make_long_table
from src.models.catboost_regime_v2_adapter import (
    add_regime_features,
    train_weighted_smape_v2,
    train_midday_spike_v2,
    predict_midday_spike,
    train_regime_v2,
    predict_regime_v2,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Day-ahead regime v2 CatBoost experiment")
    p.add_argument("--input-pred", type=str,
                   default="outputs/dayahead_30d_core/predictions/catboost_sota_dayahead.csv")
    p.add_argument("--output-root", type=str, default="outputs/dayahead_regime_v2_30d")
    p.add_argument("--baseline-smape", type=float, default=12.58)
    p.add_argument("--spike-smape", type=float, default=12.47)
    return p.parse_args()


def compute_metrics_dict(df: pd.DataFrame, model_name: str, task: str = "dayahead") -> dict:
    yt = df["y_true"].values if "y_true" in df.columns else df["y"].values
    yp = df["y_pred"].values
    v = ~(np.isnan(yt) | np.isnan(yp))
    if v.sum() < 2:
        return {"model_name": model_name, "n": 0}
    m = compute_all_metrics(yt[v], yp[v])
    m["model_name"] = model_name
    m["task"] = task
    m["n"] = int(v.sum())
    return m


def main():
    args = parse_args()
    input_path = Path(args.input_pred)
    output_root = Path(args.output_root)
    for sub in ["predictions", "metrics", "reports", "debug"]:
        (output_root / sub).mkdir(parents=True, exist_ok=True)

    baseline_smape = args.baseline_smape
    spike_smape = args.spike_smape

    # Load CatBoost predictions (has features + y_true + y_pred)
    logger.info(f"Loading baseline predictions from {input_path}")
    df = pd.read_csv(str(input_path), encoding="utf-8-sig")
    df = df.sort_values("ds").reset_index(drop=True)
    logger.info(f"Loaded {len(df)} rows")

    # Add regime features
    df["ds"] = pd.to_datetime(df["ds"])
    df = add_regime_features(df)
    logger.info(f"Added regime features. Total cols: {len(df.columns)}")

    # Get all unique target days
    days = sorted(df["target_day"].unique())
    logger.info(f"Evaluation days: {len(days)} ({days[0]} → {days[-1]})")

    all_results = {}  # model_name -> list of (df_pred)

    # Pre-compute baseline predictions per day
    baseline_by_day = {}
    for day in days:
        day_df = df[df["target_day"] == day].copy()
        day_df["y_pred"] = day_df["y_pred"].values  # keep CatBoost's original pred
        baseline_by_day[day] = day_df

    # =========================================================
    # 1. catboost_weighted_smape_v2
    # =========================================================
    logger.info("\n" + "=" * 60)
    logger.info("MODEL 1: catboost_weighted_smape_v2")
    logger.info("=" * 60)

    weighted_preds = []

    for day_idx, day in enumerate(days):
        target_dt = pd.Timestamp(day)
        train_df = df[df["target_day"] < day].copy()
        if len(train_df) < 200:
            continue

        val_start = target_dt - timedelta(days=30)
        val_df = df[df["ds"].between(val_start, target_dt - timedelta(hours=1))].copy() if len(df[df["target_day"] < day]) > 100 else None

        try:
            model, feat_cols = train_weighted_smape_v2(train_df, val_df)
            if model is None:
                continue

            # Predict current day
            day_df = df[df["target_day"] == day].copy()
            X_pred = day_df[feat_cols].values.astype(float)
            day_df["y_pred"] = model.predict(X_pred).flatten()
            day_df["y_true"] = day_df["y"].values
            weighted_preds.append(day_df)

        except Exception as e:
            logger.warning(f"  Day {day}: weighted_smape_v2 failed: {e}")

    if weighted_preds:
        all_results["catboost_weighted_smape_v2"] = pd.concat(weighted_preds, ignore_index=True)
        m = compute_metrics_dict(all_results["catboost_weighted_smape_v2"], "catboost_weighted_smape_v2")
        logger.info(f"  sMAPE = {m.get('sMAPE_floor50', 0):.4f}%")
    else:
        logger.warning("  Weighted SMAPE v2 produced no results")

    # =========================================================
    # 2. catboost_midday_spike_v2
    # =========================================================
    logger.info("\n" + "=" * 60)
    logger.info("MODEL 2: catboost_midday_spike_v2")
    logger.info("=" * 60)

    # Train specialist model (rolling)
    spike_specialist_models = []  # list of (day, model, feat_cols)
    for day in days:
        target_dt = pd.Timestamp(day)
        train_df = df[df["target_day"] < day].copy()
        if len(train_df) < 200:
            continue
        val_df = df[df["ds"].between(target_dt - timedelta(days=30), target_dt - timedelta(hours=1))].copy() if len(train_df) > 100 else None

        try:
            model, feat_cols = train_midday_spike_v2(train_df, val_df)
            if model is None:
                continue
            spike_specialist_models.append((day, model, feat_cols))
        except Exception as e:
            logger.warning(f"  Day {day}: midday_spike_v2 train failed: {e}")

    if spike_specialist_models:
        # Test different replacement strategies
        replacement_sets = {
            "replace_hour_11": [11],
            "replace_hour_12": [12],
            "replace_hour_13": [13],
            "replace_hour_17": [17],
            "replace_11_12_13_17": [11, 12, 13, 17],
        }

        for strategy_name, hours in replacement_sets.items():
            preds = []
            for day, model, feat_cols in spike_specialist_models:
                day_df = df[df["target_day"] == day].copy()
                day_df["y_pred"] = predict_midday_spike(day_df, model, feat_cols, hours)
                day_df["y_true"] = day_df["y"].values
                preds.append(day_df)

            if preds:
                combined = pd.concat(preds, ignore_index=True)
                model_name = f"catboost_midday_spike_v2_{strategy_name}"
                all_results[model_name] = combined
                m = compute_metrics_dict(combined, model_name)
                logger.info(f"  {strategy_name}: sMAPE = {m.get('sMAPE_floor50', 0):.4f}%")

    # =========================================================
    # 3. catboost_regime_v2
    # =========================================================
    logger.info("\n" + "=" * 60)
    logger.info("MODEL 3: catboost_regime_v2")
    logger.info("=" * 60)

    regime_preds = []

    for day_idx, day in enumerate(days):
        target_dt = pd.Timestamp(day)
        train_df = df[df["target_day"] < day].copy()
        if len(train_df) < 200:
            continue

        try:
            experts, feat_cols = train_regime_v2(train_df)
            if not experts:
                continue

            day_df = df[df["target_day"] == day].copy()
            day_df["y_pred"] = predict_regime_v2(day_df, experts, feat_cols)
            day_df["y_true"] = day_df["y"].values
            regime_preds.append(day_df)

        except Exception as e:
            logger.warning(f"  Day {day}: regime_v2 failed: {e}")

    if regime_preds:
        all_results["catboost_regime_v2"] = pd.concat(regime_preds, ignore_index=True)
        m = compute_metrics_dict(all_results["catboost_regime_v2"], "catboost_regime_v2")
        logger.info(f"  sMAPE = {m.get('sMAPE_floor50', 0):.4f}%")

    # =========================================================
    # Save predictions
    # =========================================================
    for name, result_df in all_results.items():
        path = output_root / "predictions" / f"{name}_dayahead.csv"
        result_df.to_csv(str(path), index=False, encoding="utf-8-sig")
        logger.info(f"Saved {path} ({len(result_df)} rows)")

    # =========================================================
    # Metrics
    # =========================================================
    all_metrics = []
    hour_rows = []
    period_rows = []

    for name, result_df in all_results.items():
        m = compute_metrics_dict(result_df, name)
        all_metrics.append(m)

        for hour, grp in result_df.groupby("hour_business"):
            hm = compute_metrics_dict(grp, name)
            hm["hour_business"] = hour
            hour_rows.append(hm)

        for period, grp in result_df.groupby("period"):
            pm = compute_metrics_dict(grp, name)
            pm["period"] = period
            period_rows.append(pm)

    summary = pd.DataFrame(all_metrics)
    summary.to_csv(str(output_root / "metrics" / "summary.csv"), index=False, encoding="utf-8-sig")

    hour_metrics = pd.DataFrame(hour_rows)
    hour_metrics.to_csv(str(output_root / "metrics" / "hour_metrics.csv"), index=False, encoding="utf-8-sig")

    period_metrics = pd.DataFrame(period_rows)
    period_metrics.to_csv(str(output_root / "metrics" / "period_metrics.csv"), index=False, encoding="utf-8-sig")

    # =========================================================
    # Report
    # =========================================================
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = []
    def w(s=""):
        lines.append(s)

    w(f"# Day-Ahead Regime v2 Report")
    w(f"**Generated**: {now}")
    w(f"**CatBoost baseline**: {baseline_smape:.2f}%")
    w(f"**Spike residual corrector**: {spike_smape:.2f}%")
    w()

    # Rankings
    sorted_metrics = sorted(all_metrics, key=lambda m: m.get("sMAPE_floor50", 1e9))

    w("## 1. Model Rankings (by sMAPE)")
    w("| Rank | Model | sMAPE | MAE | RMSE | Beats baseline? | Beats spike? |")
    w("|---|---|---|---|---|---|---|")
    for rank, m in enumerate(sorted_metrics, 1):
        name = m.get("model_name", "?")
        smape = m.get("sMAPE_floor50", 0)
        mae = m.get("MAE", 0)
        rmse = m.get("RMSE", 0)
        beats_bl = "✅" if smape < baseline_smape else "❌"
        beats_sp = "✅" if smape < spike_smape else "❌"
        w(f"| {rank} | {name} | {smape:.2f}% | {mae:.2f} | {rmse:.2f} | {beats_bl} | {beats_sp} |")
    w()

    # Target check
    best_smape = sorted_metrics[0].get("sMAPE_floor50", 1e9) if sorted_metrics else 1e9
    best_model = sorted_metrics[0].get("model_name", "?") if sorted_metrics else "?"

    w("## 2. Target Check")
    w(f"- Best model: **{best_model}** (sMAPE = {best_smape:.2f}%)")
    w(f"- Below 12.58% (CatBoost)? {'✅' if best_smape < baseline_smape else '❌'}")
    w(f"- Below 12.47% (spike)? {'✅' if best_smape < spike_smape else '❌'}")
    w(f"- Below 12%? {'✅' if best_smape < 12 else '❌'}")
    w(f"- Below 10%? {'✅' if best_smape < 10 else '❌'}")
    w(f"- Below 8%? {'✅' if best_smape < 8 else '❌'}")
    w()

    # Hour 11/12/13/17 analysis
    w("## 3. Hour 11/12/13/17 Comparison")
    w("| Model | H11 | H12 | H13 | H17 |")
    w("|---|---|---|---|---|")
    for m in sorted_metrics:
        name = m["model_name"]
        h11 = next((hm.get("sMAPE_floor50", 0) for hm in hour_rows
                    if hm.get("model_name") == name and hm.get("hour_business") == 11), 0)
        h12 = next((hm.get("sMAPE_floor50", 0) for hm in hour_rows
                    if hm.get("model_name") == name and hm.get("hour_business") == 12), 0)
        h13 = next((hm.get("sMAPE_floor50", 0) for hm in hour_rows
                    if hm.get("model_name") == name and hm.get("hour_business") == 13), 0)
        h17 = next((hm.get("sMAPE_floor50", 0) for hm in hour_rows
                    if hm.get("model_name") == name and hm.get("hour_business") == 17), 0)
        w(f"| {name} | {h11:.2f}% | {h12:.2f}% | {h13:.2f}% | {h17:.2f}% |")
    w()

    # Summary
    w("## 4. Summary")
    if best_smape < spike_smape:
        w(f"✅ **Best model ({best_model}) beats both baseline and spike residual corrector.**")
    elif best_smape < baseline_smape:
        w(f"⚠️ **Improves over CatBoost baseline but not over spike residual corrector.**")
    else:
        w(f"❌ **No model improves over CatBoost baseline.**")

    if best_smape < 12:
        w("✅ **Below 12%**")
    else:
        w(f"❌ **Still above 12% (gap: {best_smape - 12:.2f}pp)**")

    w()
    w("## 5. Recommendations")
    if best_smape < baseline_smape:
        w(f"- Best approach: **{best_model}**")
        w("- Consider combining with spike residual correction")
    else:
        w("- All regime v2 models failed to improve over baseline.")
        w("- Need different architecture (deep learning, LSTNet, TimesNet)")
    w(f"- 8% target gap: **{best_smape - 8:.2f}pp** — {'achievable' if best_smape < 12 else 'very difficult'}")

    report = "\n".join(lines)
    report_path = output_root / "reports" / "dayahead_regime_v2_report.md"
    with open(str(report_path), "w", encoding="utf-8") as f:
        f.write(report)
    logger.info(f"Report saved to {report_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("REGIME V2 SUMMARY")
    print("=" * 70)
    print(f"{'Model':50s} {'sMAPE':>8s}")
    print("-" * 60)
    for m in sorted_metrics:
        name = m.get("model_name", "?")
        smape = m.get("sMAPE_floor50", 0)
        beats = "✅" if smape < baseline_smape else "❌"
        print(f"{name:50s} {smape:7.2f}% {beats}")
    print("-" * 60)
    print(f"{'CatBoost baseline':50s} {baseline_smape:7.2f}%")
    print(f"{'Spike residual corrector':50s} {spike_smape:7.2f}%")
    print("=" * 70)
    print(f"Best: {best_model} @ {best_smape:.2f}%")
    print(f"Below 12%? {'✅' if best_smape < 12 else '❌'}")
    print(f"Below 10%? {'✅' if best_smape < 10 else '❌'}")
    print(f"Below 8%?  {'✅' if best_smape < 8 else '❌'}")

    # Save manifest
    manifest = {
        "baseline_smape": baseline_smape,
        "spike_smape": spike_smape,
        "best_model": best_model,
        "best_smape": best_smape,
        "all_metrics": {m["model_name"]: m.get("sMAPE_floor50", 0) for m in sorted_metrics},
        "completed_at": now,
    }
    with open(str(output_root / "debug" / "run_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
