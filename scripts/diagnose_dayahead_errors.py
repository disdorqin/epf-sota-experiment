"""
diagnose_dayahead_errors.py — Error diagnosis for day-ahead predictions.

Output:
    outputs/dayahead_diagnosis/
    ├── worst_5_hours.csv
    ├── worst_5_days.csv
    ├── error_by_period.csv
    ├── error_by_hour.csv
    ├── error_distribution.png (ASCII)
    └── diagnosis_report.md
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_DIR)

from src.common.metrics import compute_all_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Day-ahead error diagnosis")
    p.add_argument("--input-root", type=str, default="outputs/dayahead_30d_core")
    p.add_argument("--output-root", type=str, default="outputs/dayahead_diagnosis")
    return p.parse_args()


def load_preds(pred_dir: Path) -> dict[str, pd.DataFrame]:
    """Load all prediction CSVs from a directory."""
    result = {}
    for f in sorted(pred_dir.glob("*.csv")):
        name = f.stem
        result[name] = pd.read_csv(str(f), encoding="utf-8-sig")
    return result


def main():
    args = parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    pred_dir = input_root / "predictions"
    if not pred_dir.exists():
        logger.error(f"Predictions not found: {pred_dir}")
        sys.exit(1)

    preds = load_preds(pred_dir)
    logger.info(f"Loaded {len(preds)} prediction sets: {list(preds.keys())}")

    # Build diagnosis for each model
    diagnosis = {}

    for name, df in preds.items():
        # Align on ds + hour_business
        model_diagnosis = {}

        # Worst 5 hours (absolute error)
        df["abs_error"] = np.abs(df["y_true"] - df["y_pred"])
        df["smape_hour"] = 200 * df["abs_error"] / (np.abs(df["y_true"]) + np.abs(df["y_pred"]) + 1e-8)
        worst_hours = df.nlargest(5, "smape_hour")[
            ["ds", "hour_business", "period", "y_true", "y_pred", "smape_hour"]
        ]
        model_diagnosis["worst_5_hours"] = worst_hours
        worst_hours.to_csv(str(output_root / f"{name}_worst_5_hours.csv"), index=False, encoding="utf-8-sig")

        # Worst 5 days
        daily_smape = df.groupby("target_day")["smape_hour"].mean().reset_index()
        worst_days = daily_smape.nlargest(5, "smape_hour")
        model_diagnosis["worst_5_days"] = worst_days
        worst_days.to_csv(str(output_root / f"{name}_worst_5_days.csv"), index=False, encoding="utf-8-sig")

        # Error by period
        period_err = df.groupby("period").agg(
            count=("smape_hour", "count"),
            mean_sMAPE=("smape_hour", "mean"),
            max_sMAPE=("smape_hour", "max"),
            mean_abs_err=("abs_error", "mean"),
        ).reset_index()
        model_diagnosis["error_by_period"] = period_err
        period_err.to_csv(str(output_root / f"{name}_error_by_period.csv"), index=False, encoding="utf-8-sig")

        # Error by hour
        hour_err = df.groupby("hour_business").agg(
            count=("smape_hour", "count"),
            mean_sMAPE=("smape_hour", "mean"),
            max_sMAPE=("smape_hour", "max"),
            mean_abs_err=("abs_error", "mean"),
        ).reset_index()
        model_diagnosis["error_by_hour"] = hour_err
        hour_err.to_csv(str(output_root / f"{name}_error_by_hour.csv"), index=False, encoding="utf-8-sig")

        diagnosis[name] = model_diagnosis
        logger.info(f"Analyzed {name}")

    # Generate report
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = []
    def w(s=""):
        lines.append(s)

    w(f"# Day-Ahead Error Diagnosis Report")
    w(f"**Generated**: {now}")
    w()

    for name, diag in diagnosis.items():
        w(f"## {name}")
        w()

        w("### Worst 5 Hours")
        w("| ds | hour | period | y_true | y_pred | sMAPE |")
        w("|---|---|---|---|---|---|")
        for _, r in diag["worst_5_hours"].iterrows():
            w(f"| {r['ds']} | {r['hour_business']} | {r['period']} | {r['y_true']:.1f} | {r['y_pred']:.1f} | {r['smape_hour']:.2f}% |")
        w()

        w("### Worst 5 Days")
        w("| target_day | mean_sMAPE |")
        w("|---|---|")
        for _, r in diag["worst_5_days"].iterrows():
            w(f"| {r['target_day']} | {r['smape_hour']:.2f}% |")
        w()

        w("### Error by Period")
        w("| period | count | mean_sMAPE | max_sMAPE | mean_abs_err |")
        w("|---|---|---|---|---|")
        for _, r in diag["error_by_period"].iterrows():
            w(f"| {r['period']} | {r['count']} | {r['mean_sMAPE']:.2f}% | {r['max_sMAPE']:.2f}% | {r['mean_abs_err']:.2f} |")
        w()

        w("### Error by Hour")
        w("| hour | count | mean_sMAPE | max_sMAPE | mean_abs_err |")
        w("|---|---|---|---|---|")
        for _, r in diag["error_by_hour"].iterrows():
            w(f"| {r['hour_business']} | {r['count']} | {r['mean_sMAPE']:.2f}% | {r['max_sMAPE']:.2f}% | {r['mean_abs_err']:.2f} |")
        w()

        # Overall
        total_smape = df["smape_hour"].mean()
        total_mae = df["abs_error"].mean()
        w(f"### Overall")
        w(f"- Mean sMAPE: **{total_smape:.2f}%**")
        w(f"- Mean MAE: {total_mae:.2f}")
        w(f"- Total hours: {len(df)}")
        w()

    report = "\n".join(lines)
    report_path = output_root / "diagnosis_report.md"
    with open(str(report_path), "w", encoding="utf-8") as f:
        f.write(report)
    logger.info(f"Diagnosis report: {report_path}")
    print(report)


if __name__ == "__main__":
    main()
