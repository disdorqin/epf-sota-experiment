"""
evaluate_dayahead_corrections.py — Compare correction results vs baseline.

Usage:
    python scripts/evaluate_dayahead_corrections.py ^
        --correction-root outputs/dayahead_corrections_30d ^
        --baseline-path outputs/dayahead_30d_core/predictions/catboost_sota_dayahead.csv
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_DIR)

from src.common.metrics import compute_all_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate day-ahead corrections")
    p.add_argument("--correction-root", type=str,
                   default="outputs/dayahead_corrections_30d")
    p.add_argument("--baseline-path", type=str,
                   default="outputs/dayahead_30d_core/predictions/catboost_sota_dayahead.csv")
    return p.parse_args()


def compute_model_metrics(df, name, task="dayahead"):
    yt = df["y_true"].values
    yp = df["y_pred"].values
    v = ~(np.isnan(yt) | np.isnan(yp))
    if v.sum() < 2:
        return {"model_name": name, "n": 0}
    m = compute_all_metrics(yt[v], yp[v])
    m["model_name"] = name
    m["task"] = task
    m["n"] = int(v.sum())
    return m


def main():
    args = parse_args()
    correction_root = Path(args.correction_root)
    baseline_path = Path(args.baseline_path)

    # Load baseline
    baseline = pd.read_csv(str(baseline_path), encoding="utf-8-sig")
    bl_metrics = compute_model_metrics(baseline, "catboost_sota")
    bl_smape = bl_metrics.get("sMAPE_floor50", 0)
    logger.info(f"Baseline: sMAPE={bl_smape:.4f}%")

    # Load corrections
    pred_dir = correction_root / "predictions"
    if not pred_dir.exists():
        logger.error(f"No predictions found at {pred_dir}")
        return

    all_results = [bl_metrics]
    for csv_path in sorted(pred_dir.glob("*.csv")):
        name = csv_path.stem.replace("_dayahead", "")
        df = pd.read_csv(str(csv_path), encoding="utf-8-sig")
        m = compute_model_metrics(df, name)
        all_results.append(m)
        logger.info(f"{name}: sMAPE={m.get('sMAPE_floor50', 0):.4f}%")

    # Print comparison table
    print("\n" + "=" * 80)
    print("CORRECTION EVALUATION vs BASELINE")
    print("=" * 80)
    print(f"{'Model':45s} {'sMAPE':>8s} {'MAE':>8s} {'RMSE':>8s} {'vs Base':>10s}")
    print("-" * 80)
    for m in all_results:
        name = m.get("model_name", "?")
        smape = m.get("sMAPE_floor50", 0)
        mae = m.get("MAE", 0)
        rmse = m.get("RMSE", 0)
        delta = smape - bl_smape
        vs = f"{'+' if delta > 0 else ''}{delta:.2f}pp"
        beats = "✅" if smape < bl_smape else "❌" if smape > bl_smape else "➡️"
        print(f"{name:45s} {smape:7.2f}% {mae:7.2f} {rmse:7.2f} {beats + ' ' + vs:>10s}")
    print("=" * 80)

    # Hour 11/12/13 analysis
    print("\nHOUR 11/12/13 ANALYSIS:")
    print(f"{'Model':35s} {'H11':>8s} {'H12':>8s} {'H13':>8s} {'H17':>8s}")
    print("-" * 65)
    for m in all_results:
        name = m.get("model_name", "?")
    # Reload and compute per-hour
    for csv_path in [baseline_path] + sorted(pred_dir.glob("*.csv")):
        name = csv_path.stem.replace("_dayahead", "")
        df = pd.read_csv(str(csv_path), encoding="utf-8-sig")
        h11 = compute_model_metrics(df[df["hour_business"] == 11], "h11")
        h12 = compute_model_metrics(df[df["hour_business"] == 12], "h12")
        h13 = compute_model_metrics(df[df["hour_business"] == 13], "h13")
        h17 = compute_model_metrics(df[df["hour_business"] == 17], "h17")
        print(f"{name:35s} {h11.get('sMAPE_floor50', 0):7.2f}% "
              f"{h12.get('sMAPE_floor50', 0):7.2f}% "
              f"{h13.get('sMAPE_floor50', 0):7.2f}% "
              f"{h17.get('sMAPE_floor50', 0):7.2f}%")

    # Worst days
    print("\nWORST 5 DAYS (baseline):")
    day_smape = baseline.groupby("target_day").apply(
        lambda g: compute_model_metrics(g, "tmp").get("sMAPE_floor50", 0)
    ).sort_values(ascending=False).head(5)
    for day, sm in day_smape.items():
        print(f"  {day}: {sm:.2f}%")


if __name__ == "__main__":
    main()
