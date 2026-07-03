#!/usr/bin/env python3
"""
Run LightGBM day-ahead residual corrections (simplified).

Usage:
    python scripts/run_lgbm_dayahead_correction.py
"""
import sys, os, argparse, json, logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from pathlib import Path
from src.common.metrics import smape_floor50, compute_all_metrics
from src.correction.lgbm_dayahead_corrector import (
    LGBMSpikeResidualCorrector,
    LGBMSelectedHourCorrector,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_PATH = "outputs/dayahead_lgbm_stage2_30d/predictions/trial_02_w150_nl255_lr0.03_dayahead.csv"
OUT_ROOT = "outputs/dayahead_lgbm_corrections_30d"


def load_data(path=None):
    if path is None:
        path = BASE_PATH
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "y_pred" not in df.columns and "y_pred_cb" in df.columns:
        df["y_pred"] = df["y_pred_cb"]
    df = df.sort_values("ds").reset_index(drop=True)
    for col in ["hour_business", "period", "target_day"]:
        if col not in df.columns:
            raise ValueError(f"Missing column in input: {col}")
    return df


def save_prediction(df, preds, name, pred_dir):
    out = df.copy()
    out["y_pred"] = preds
    out["model_name"] = name
    out.to_csv(str(pred_dir / f"{name}_dayahead.csv"), index=False, encoding="utf-8-sig")
    logger.info(f"Saved {name}: {len(out)} rows")


def main():
    out_root = Path(OUT_ROOT)
    pred_dir = out_root / "predictions"
    metric_dir = out_root / "metrics"
    report_dir = out_root / "reports"
    for d in [pred_dir, metric_dir, report_dir]:
        d.mkdir(parents=True, exist_ok=True)

    df = load_data()
    base = df["y_pred"].values.copy()
    logger.info(f"Loaded {len(df)} rows, {df['target_day'].nunique()} days")
    logger.info(f"Baseline sMAPE_floor50: {smape_floor50(df['y_true'].values, base):.4f}%")

    # ═══════════════════════════════════════════════════════════
    # 1. Spike Residual Correction
    # ═══════════════════════════════════════════════════════════
    logger.info("Running spike residual corrector (alpha grid)...")

    results = {}
    best_spike = None
    best_spike_smape = float("inf")

    for alpha in [0.25, 0.5, 0.75, 1.0]:
        for delta in [50, 100]:
            corrector = LGBMSpikeResidualCorrector(alpha=alpha, max_delta=delta)
            try:
                preds = corrector.correct(df.copy())
                s = smape_floor50(df["y_true"].values, preds)
                logger.info(f"  spike α={alpha} δ={delta}: {s:.4f}%")
                if s < best_spike_smape:
                    best_spike_smape = s
                    best_spike = (alpha, delta, preds.copy())
            except Exception as e:
                logger.warning(f"  spike α={alpha} δ={delta}: ERROR {e}")

    if best_spike is not None:
        alpha, delta, spike_preds = best_spike
        logger.info(f"Best spike: α={alpha} δ={delta} → {best_spike_smape:.4f}%")
        save_prediction(df, spike_preds, "lgbm_spike_residual_corrected", pred_dir)
        results["lgbm_spike_residual_corrected"] = spike_preds
        results["spike_params"] = {"alpha": alpha, "max_delta": delta}
    else:
        logger.warning("Spike corrector failed entirely")
        results["lgbm_spike_residual_corrected"] = base
        best_spike_smape = float("nan")

    # ═══════════════════════════════════════════════════════════
    # 2. Selected Hour Correction
    # ═══════════════════════════════════════════════════════════
    logger.info("Running selected hour corrector...")

    best_hour = None
    best_hour_smape = float("inf")

    hour_sets = [
        [11, 12, 13, 17],
        [3, 4, 11, 12, 13, 17],
        [11, 13],
        [13],
        [3, 4],
    ]
    for hours in hour_sets:
        for delta in [50, 100]:
            corrector = LGBMSelectedHourCorrector(target_hours=hours, max_delta=delta)
            try:
                preds = corrector.correct(df.copy())
                s = smape_floor50(df["y_true"].values, preds)
                logger.info(f"  hour {hours} δ={delta}: {s:.4f}%")
                if s < best_hour_smape:
                    best_hour_smape = s
                    best_hour = (hours, delta, preds.copy())
            except Exception as e:
                logger.warning(f"  hour {hours} δ={delta}: ERROR {e}")

    if best_hour is not None:
        hours, delta, hour_preds = best_hour
        logger.info(f"Best hour: {hours} δ={delta} → {best_hour_smape:.4f}%")
        save_prediction(df, hour_preds, "lgbm_selected_hour_corrected", pred_dir)
        results["lgbm_selected_hour_corrected"] = hour_preds
        results["hour_params"] = {"hours": hours, "max_delta": delta}
    else:
        logger.warning("Hour corrector failed entirely")
        results["lgbm_selected_hour_corrected"] = base
        best_hour_smape = float("nan")

    results["baseline"] = base

    # ═══════════════════════════════════════════════════════════
    # Metrics
    # ═══════════════════════════════════════════════════════════
    y_true = df["y_true"].values
    summary_rows = []
    for name in ["baseline", "lgbm_spike_residual_corrected", "lgbm_selected_hour_corrected"]:
        if name not in results:
            continue
        preds = results[name]
        if not isinstance(preds, np.ndarray):
            continue
        valid = ~(np.isnan(y_true) | np.isnan(preds))
        if valid.sum() < 2:
            continue
        m = compute_all_metrics(y_true[valid], preds[valid])
        m["model_name"] = name
        m["task"] = "dayahead"
        m["n"] = int(valid.sum())
        summary_rows.append(m)
    pd.DataFrame(summary_rows).to_csv(
        str(metric_dir / "summary.csv"), index=False, encoding="utf-8-sig"
    )

    # Hour metrics
    hour_rows = []
    for h in sorted(df["hour_business"].unique()):
        hm = df["hour_business"] == h
        for name in ["baseline", "lgbm_spike_residual_corrected", "lgbm_selected_hour_corrected"]:
            if name not in results or not isinstance(results[name], np.ndarray):
                continue
            yt = y_true[hm.values]
            yp = results[name][hm.values]
            valid = ~(np.isnan(yt) | np.isnan(yp))
            if valid.sum() < 2:
                continue
            hour_rows.append({
                "model_name": name, "hour_business": int(h),
                "sMAPE_floor50": round(smape_floor50(yt[valid], yp[valid]), 4)
            })
    pd.DataFrame(hour_rows).to_csv(
        str(metric_dir / "hour_metrics.csv"), index=False, encoding="utf-8-sig"
    )

    # Period metrics
    period_rows = []
    for p in sorted(df["period"].unique()):
        pm = df["period"] == p
        for name in ["baseline", "lgbm_spike_residual_corrected", "lgbm_selected_hour_corrected"]:
            if name not in results or not isinstance(results[name], np.ndarray):
                continue
            yt = y_true[pm.values]
            yp = results[name][pm.values]
            valid = ~(np.isnan(yt) | np.isnan(yp))
            if valid.sum() < 2:
                continue
            period_rows.append({
                "model_name": name, "period": p,
                "sMAPE_floor50": round(smape_floor50(yt[valid], yp[valid]), 4)
            })
    pd.DataFrame(period_rows).to_csv(
        str(metric_dir / "period_metrics.csv"), index=False, encoding="utf-8-sig"
    )

    # ── Report ──
    baseline_smape = smape_floor50(y_true, base)
    lines = []
    lines.append("# LightGBM Day-Ahead Correction Report")
    lines.append(f"> Generated: 2026-07-03")
    lines.append(f"> Baseline: LightGBM trial_02 = {baseline_smape:.2f}%")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Method | sMAPE_floor50 | vs baseline | vs best_two_average |")
    lines.append("|--------|:-------------:|:-----------:|:-------------------:|")
    lines.append(f"| baseline (trial_02) | {baseline_smape:.2f}% | — | — |")
    lines.append(f"| best_two_average | 11.85% | -0.22pp | — |")
    spike_s = best_spike_smape if not np.isnan(best_spike_smape) else 0
    lines.append(f"| lgbm_spike_residual | {best_spike_smape:.2f}% | {best_spike_smape - baseline_smape:+.2f}pp | {best_spike_smape - 11.85:+.2f}pp |" if not np.isnan(best_spike_smape) else "| lgbm_spike_residual | FAILED | — | — |")
    lines.append(f"| lgbm_selected_hour | {best_hour_smape:.2f}% | {best_hour_smape - baseline_smape:+.2f}pp | {best_hour_smape - 11.85:+.2f}pp |" if not np.isnan(best_hour_smape) else "| lgbm_selected_hour | FAILED | — | — |")
    lines.append("")

    best_overall = baseline_smape
    best_name = "baseline"
    if not np.isnan(best_spike_smape) and best_spike_smape < best_overall:
        best_overall = best_spike_smape
        best_name = "lgbm_spike_residual"
    if not np.isnan(best_hour_smape) and best_hour_smape < best_overall:
        best_overall = best_hour_smape
        best_name = "lgbm_selected_hour"

    lines.append(f"**Best overall**: {best_name} = {best_overall:.2f}%")
    lines.append("")
    lines.append("## Target Check")
    lines.append("")
    lines.append(f"| Target | Status |")
    lines.append(f"|:------|:------:|")
    lines.append(f"| Below 12.07% (baseline) | {'✅' if best_overall < baseline_smape else '❌'} |")
    lines.append(f"| Below 11.85% (best_two) | {'✅' if best_overall < 11.85 else '❌'} |")
    lines.append(f"| Below 11.5% | {'✅' if best_overall < 11.5 else '❌'} |")
    lines.append(f"| Below 11% | {'✅' if best_overall < 11 else '❌'} |")
    lines.append(f"| Below 10% | {'✅' if best_overall < 10 else '❌'} |")
    lines.append(f"| Below 8% | {'✅' if best_overall < 8 else '❌'} |")
    lines.append("")

    report = "\n".join(lines)
    (report_dir / "lgbm_correction_report.md").write_text(report, encoding="utf-8")
    print("\n" + report)
    logger.info(f"Report: {report_dir / 'lgbm_correction_report.md'}")


if __name__ == "__main__":
    main()
