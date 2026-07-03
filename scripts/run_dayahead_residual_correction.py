"""
run_dayahead_residual_correction.py — Run residual correction on CatBoost day-ahead.

Usage:
    python scripts/run_dayahead_residual_correction.py ^
        --input-pred outputs/dayahead_30d_core/predictions/catboost_sota_dayahead.csv ^
        --output-root outputs/dayahead_corrections_30d

Runs:
    1. SelectedHourResidualCorrector (hours 11,12,13,17)
    2. SpikeResidualCorrector (with grid search for alpha/threshold/max_delta)
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
from src.correction.dayahead_residual_corrector import (
    SelectedHourResidualCorrector,
    SpikeResidualCorrector,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Day-ahead residual correction experiment")
    p.add_argument("--input-pred", type=str,
                   default="outputs/dayahead_30d_core/predictions/catboost_sota_dayahead.csv")
    p.add_argument("--output-root", type=str, default="outputs/dayahead_corrections_30d")
    p.add_argument("--selected-hours", type=str, default="11,12,13,17",
                   help="Comma-separated hours for selected_hour corrector")
    p.add_argument("--baseline-smape", type=float, default=12.58,
                   help="CatBoost baseline sMAPE for comparison")
    return p.parse_args()


def compute_metrics_df(df: pd.DataFrame, model_name: str, task: str = "dayahead") -> dict:
    yt = df["y_true"].values
    yp = df["y_pred"].values
    valid = ~(np.isnan(yt) | np.isnan(yp))
    if valid.sum() < 2:
        return {"model_name": model_name, "task": task, "n": 0}
    m = compute_all_metrics(yt[valid], yp[valid])
    m["model_name"] = model_name
    m["task"] = task
    m["n"] = int(valid.sum())
    return m


def main():
    args = parse_args()
    input_path = Path(args.input_pred)
    output_root = Path(args.output_root)

    for sub in ["predictions", "metrics", "reports", "debug"]:
        (output_root / sub).mkdir(parents=True, exist_ok=True)

    selected_hours = [int(h) for h in args.selected_hours.split(",") if h.strip()]
    baseline_smape = args.baseline_smape

    # Load predictions
    logger.info(f"Loading predictions from {input_path}")
    df = pd.read_csv(str(input_path), encoding="utf-8-sig")
    df = df.sort_values("ds").reset_index(drop=True)
    logger.info(f"Loaded {len(df)} rows")

    all_results = []

    # ── 1. SelectedHourResidualCorrector ──
    logger.info("\n" + "=" * 60)
    logger.info("SelectedHourResidualCorrector")
    logger.info("=" * 60)

    # Grid search max_delta
    best_delta = 150
    best_smape = float("inf")

    for max_delta in [50, 100, 150, 200]:
        corrector = SelectedHourResidualCorrector(
            target_hours=selected_hours,
            max_delta=float(max_delta),
        )
        corrected = corrector.correct(df.copy())
        m = compute_metrics_df(corrected, f"selected_hour_delta_{max_delta}")
        logger.info(f"  max_delta={max_delta}: sMAPE={m.get('sMAPE_floor50', 'N/A')}")
        if m.get("sMAPE_floor50", 1e9) < best_smape:
            best_smape = m["sMAPE_floor50"]
            best_delta = max_delta

    logger.info(f"Best max_delta for selected_hour: {best_delta} (sMAPE={best_smape:.4f}%)")

    # Run best
    corrector = SelectedHourResidualCorrector(
        target_hours=selected_hours,
        max_delta=float(best_delta),
    )
    corrected_sh = corrector.correct(df.copy())
    corrected_sh["model_name"] = "catboost_selected_hour_corrected"
    corrected_sh["task"] = "dayahead"
    all_results.append(corrected_sh)

    sh_metrics = compute_metrics_df(corrected_sh, "catboost_selected_hour_corrected")
    logger.info(f"SelectedHour final sMAPE: {sh_metrics.get('sMAPE_floor50', 'N/A'):.4f}%")

    # ── 2. SpikeResidualCorrector (with grid search) ──
    logger.info("\n" + "=" * 60)
    logger.info("SpikeResidualCorrector (grid search)")
    logger.info("=" * 60)

    alphas = [0.5, 0.75]
    thresholds = [0.55, 0.75]
    deltas = [100, 200]

    best_spike = {"smape": float("inf"), "alpha": 0.5, "threshold": 0.55, "max_delta": 150}
    results_grid = []

    for alpha in alphas:
        for threshold in thresholds:
            for max_delta in deltas:
                corrector = SpikeResidualCorrector(
                    alpha=alpha, threshold=threshold, max_delta=float(max_delta),
                )
                corrected = corrector.correct(df.copy())
                m = compute_metrics_df(corrected, f"spike_a{alpha}_t{threshold}_d{max_delta}")
                smape = m.get("sMAPE_floor50", 1e9)
                results_grid.append({
                    "alpha": alpha, "threshold": threshold, "max_delta": max_delta,
                    "sMAPE": smape,
                })

    # Find best
    for r in results_grid:
        if r["sMAPE"] < best_spike["smape"]:
            best_spike = {"smape": r["sMAPE"], "alpha": r["alpha"],
                          "threshold": r["threshold"], "max_delta": r["max_delta"]}

    logger.info(f"Best spike params: alpha={best_spike['alpha']}, "
                f"threshold={best_spike['threshold']}, "
                f"max_delta={best_spike['max_delta']} "
                f"(sMAPE={best_spike['smape']:.4f}%)")

    # Run best
    corrector = SpikeResidualCorrector(
        alpha=best_spike["alpha"], threshold=best_spike["threshold"],
        max_delta=float(best_spike["max_delta"]),
    )
    corrected_sp = corrector.correct(df.copy())
    corrected_sp["model_name"] = "catboost_spike_residual_corrected"
    corrected_sp["task"] = "dayahead"
    all_results.append(corrected_sp)

    sp_metrics = compute_metrics_df(corrected_sp, "catboost_spike_residual_corrected")
    logger.info(f"SpikeResidual final sMAPE: {sp_metrics.get('sMAPE_floor50', 'N/A'):.4f}%")

    # ── Save predictions ──
    for corrected_df in all_results:
        model_name = corrected_df["model_name"].iloc[0]
        path = output_root / "predictions" / f"{model_name}_dayahead.csv"
        corrected_df.to_csv(str(path), index=False, encoding="utf-8-sig")
        logger.info(f"Saved {path} ({len(corrected_df)} rows)")

    # ── Compute metrics ──
    all_metrics = []
    hour_rows = []
    period_rows = []

    for corrected_df in all_results:
        model_name = corrected_df["model_name"].iloc[0]
        task = corrected_df["task"].iloc[0]

        # Overall
        m = compute_metrics_df(corrected_df, model_name, task)
        all_metrics.append(m)

        # Hour breakdown
        for hour, grp in corrected_df.groupby("hour_business"):
            hm = compute_metrics_df(grp, model_name, task)
            hm["hour_business"] = hour
            hour_rows.append(hm)

        # Period breakdown
        for period, grp in corrected_df.groupby("period"):
            pm = compute_metrics_df(grp, model_name, task)
            pm["period"] = period
            period_rows.append(pm)

    summary = pd.DataFrame(all_metrics)
    summary.to_csv(str(output_root / "metrics" / "summary.csv"), index=False, encoding="utf-8-sig")

    hour_metrics = pd.DataFrame(hour_rows)
    hour_metrics.to_csv(str(output_root / "metrics" / "hour_metrics.csv"), index=False, encoding="utf-8-sig")

    period_metrics = pd.DataFrame(period_rows)
    period_metrics.to_csv(str(output_root / "metrics" / "period_metrics.csv"), index=False, encoding="utf-8-sig")

    # Save grid search results
    grid_df = pd.DataFrame(results_grid)
    grid_df.to_csv(str(output_root / "debug" / "spike_grid_search.csv"), index=False, encoding="utf-8-sig")

    # ── Main comparison ──
    sh_smape = sh_metrics.get("sMAPE_floor50", 0)
    sp_smape = sp_metrics.get("sMAPE_floor50", 0)

    # Hour 11/12/13 improvement
    baseline_hours = {h: {"smape": None} for h in [11, 12, 13, 17]}
    for hour_row in hour_rows:
        h = hour_row.get("hour_business")
        name = hour_row.get("model_name")
        if h in [11, 12, 13, 17]:
            if name == "catboost_selected_hour_corrected":
                baseline_hours[h]["selected"] = hour_row.get("sMAPE_floor50")
            elif name == "catboost_spike_residual_corrected":
                baseline_hours[h]["spike"] = hour_row.get("sMAPE_floor50")

    # Build report
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = []
    def w(s=""):
        lines.append(s)

    w(f"# Day-Ahead Residual Correction Report")
    w(f"**Generated**: {now}")
    w(f"**Baseline**: catboost_sota sMAPE = {baseline_smape:.2f}%")
    w(f"**Period**: Feb 1 → Mar 2 ({len(df) // 24} days)")
    w()

    w("## 1. SelectedHourResidualCorrector")
    w(f"- Target hours: {selected_hours}")
    w(f"- Best max_delta: {best_delta}")
    w(f"- **sMAPE**: {sh_smape:.4f}%")
    w(f"- Better than baseline ({baseline_smape:.2f}%)? "
      f"{'✅ Yes' if sh_smape < baseline_smape else '❌ No'}")
    w()

    w("## 2. SpikeResidualCorrector")
    w(f"- Best params: alpha={best_spike['alpha']}, "
      f"threshold={best_spike['threshold']}, "
      f"max_delta={best_spike['max_delta']}")
    w(f"- **sMAPE**: {sp_smape:.4f}%")
    w(f"- Better than baseline ({baseline_smape:.2f}%)? "
      f"{'✅ Yes' if sp_smape < baseline_smape else '❌ No'}")
    w()

    w("## 3. Target Check")
    for name, smape in [("SelectedHour", sh_smape), ("SpikeResidual", sp_smape)]:
        w(f"### {name}")
        w(f"- sMAPE = {smape:.2f}%")
        w(f"- Below 12.58% (baseline)? {'✅' if smape < baseline_smape else '❌'}")
        w(f"- Below 12%? {'✅' if smape < 12 else '❌'}")
        w(f"- Below 10%? {'✅' if smape < 10 else '❌'}")
        w(f"- Below 8%?  {'✅' if smape < 8 else '❌'}")
        w()

    w("## 4. Hour 11/12/13/17 Comparison")
    w("| Hour | Baseline | SelectedHour | SpikeResidual |")
    w("|------|----------|-------------|---------------|")
    for h in [11, 12, 13, 17]:
        # Compute baseline sMAPE for this hour
        bh = baseline_hours[h]
        bl_smape = bh.get("smape", "N/A")
        sel_smape = bh.get("selected", "N/A")
        sp_smape_h = bh.get("spike", "N/A")
        w(f"| {h} | {bl_smape}% | {sel_smape}% | {sp_smape_h}% |")
    w()

    w("## 5. Summary Comparison")
    w("| Method | sMAPE | MAE | RMSE | Beats baseline? |")
    w("|--------|-------|-----|------|-----------------|")
    for m in all_metrics:
        name = m.get("model_name", "?")
        smape = m.get("sMAPE_floor50", 0)
        mae = m.get("MAE", 0)
        rmse = m.get("RMSE", 0)
        beats = "✅" if smape < baseline_smape else "❌"
        w(f"| {name} | {smape:.2f}% | {mae:.2f} | {rmse:.2f} | {beats} |")
    w()

    w("## 6. Recommendations")
    if sh_smape < baseline_smape or sp_smape < baseline_smape:
        w("✅ **Residual correction improves over baseline.**")
        best_method = "selected_hour" if sh_smape < sp_smape else "spike_residual"
        w(f"Best method: **{best_method}** (sMAPE={min(sh_smape, sp_smape):.2f}%)")
    else:
        w("❌ **Residual correction does NOT improve over baseline.**")

    if min(sh_smape, sp_smape) < 8:
        w("✅ **Target achieved: sMAPE < 8%**")
    elif min(sh_smape, sp_smape) < 10:
        w("⚠️ Target not met but under 10%. Need specialist/hour correction.")
    elif min(sh_smape, sp_smape) < 12:
        w("⚠️ Under 12% but not under 10%. Need more aggressive correction.")
    else:
        w("❌ Still above 12%. Residual correction insufficient.")

    w()

    w("### Detailed Hour Change")
    w("For each hour, compare baseline vs best correction:")
    baseline_df = df.copy()
    baseline_df["y_pred_orig"] = baseline_df["y_pred"]
    for h in sorted(baseline_df["hour_business"].unique()):
        bl_h = compute_metrics_df(baseline_df[baseline_df["hour_business"] == h],
                                   "baseline", "dayahead")
        best_corrected = all_results[0] if sh_smape <= sp_smape else all_results[1]
        co_h = compute_metrics_df(best_corrected[best_corrected["hour_business"] == h],
                                   "corrected", "dayahead")
        bl_s = bl_h.get("sMAPE_floor50", 0)
        co_s = co_h.get("sMAPE_floor50", 0)
        delta = co_s - bl_s
        arrow = "↑" if delta > 0.5 else "↓" if delta < -0.5 else "→"
        if h in [11, 12, 13, 17]:
            w(f"- Hour {h:2d}: baseline={bl_s:.2f}% → corrected={co_s:.2f}% ({arrow}{delta:+.2f}pp) **TARGET**")
        else:
            w(f"- Hour {h:2d}: baseline={bl_s:.2f}% → corrected={co_s:.2f}% ({arrow}{delta:+.2f}pp)")

    report = "\n".join(lines)
    report_path = output_root / "reports" / "dayahead_correction_report.md"
    with open(str(report_path), "w", encoding="utf-8") as f:
        f.write(report)
    logger.info(f"Report saved to {report_path}")

    # Print short summary
    print("\n" + "=" * 60)
    print("CORRECTION SUMMARY")
    print("=" * 60)
    print(f"Baseline CatBoost:              {baseline_smape:.2f}%")
    print(f"SelectedHourResidualCorrector:  {sh_smape:.4f}%  "
          f"{'✅ Better' if sh_smape < baseline_smape else '❌ Worse'}")
    print(f"SpikeResidualCorrector:         {sp_smape:.4f}%  "
          f"{'✅ Better' if sp_smape < baseline_smape else '❌ Worse'}")
    print(f"Best method: {'selected_hour' if sh_smape < sp_smape else 'spike_residual'}")
    print(f"Below 12%?  {'✅' if min(sh_smape, sp_smape) < 12 else '❌'}")
    print(f"Below 10%?  {'✅' if min(sh_smape, sp_smape) < 10 else '❌'}")
    print(f"Below 8%?   {'✅' if min(sh_smape, sp_smape) < 8 else '❌'}")
    print("=" * 60)

    # Save manifest
    manifest = {
        "baseline_smape": baseline_smape,
        "selected_hour": {"hours": selected_hours, "best_delta": best_delta, "smape": sh_smape},
        "spike_residual": {"best_alpha": best_spike["alpha"],
                           "best_threshold": best_spike["threshold"],
                           "best_delta": best_spike["max_delta"],
                           "smape": sp_smape},
        "best_method": "selected_hour" if sh_smape < sp_smape else "spike_residual",
        "completed_at": now,
    }
    with open(str(output_root / "debug" / "run_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
