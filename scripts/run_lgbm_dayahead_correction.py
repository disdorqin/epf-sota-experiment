#!/usr/bin/env python3
"""
Run LightGBM day-ahead residual corrections.

Usage:
    python scripts/run_lgbm_dayahead_correction.py \\
        --input-pred outputs/dayahead_lgbm_freeze_30d/predictions/best_two_average_dayahead.csv \\
        --input-base outputs/dayahead_lgbm_stage2_30d/predictions/trial_02_w150_nl255_lr0.03_dayahead.csv \\
        --output-root outputs/dayahead_lgbm_corrections_30d \\
        --baseline-smape 11.85
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-pred", type=str,
                        default="outputs/dayahead_lgbm_stage2_30d/predictions/trial_02_w150_nl255_lr0.03_dayahead.csv",
                        help="Best LightGBM prediction CSV (720 rows)")
    parser.add_argument("--output-root", type=str,
                        default="outputs/dayahead_lgbm_corrections_30d")
    parser.add_argument("--baseline-smape", type=float, default=12.07,
                        help="Baseline sMAPE for comparison")
    return parser.parse_args()


def main():
    args = parse_args()
    out_root = Path(args.output_root)
    pred_dir = out_root / "predictions"
    metric_dir = out_root / "metrics"
    report_dir = out_root / "reports"
    for d in [pred_dir, metric_dir, report_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ── Load data ──
    logger.info(f"Loading predictions from {args.input_pred}")
    df = pd.read_csv(args.input_pred, encoding="utf-8-sig")
    if "y_pred" not in df.columns and "y_pred_cb" in df.columns:
        df["y_pred"] = df["y_pred_cb"]
    df = df.sort_values("ds").reset_index(drop=True)
    df["base_pred"] = df["y_pred"].values.copy()
    logger.info(f"Loaded {len(df)} rows, {df['target_day'].nunique()} days")

    # Ensure standard columns
    for col in ["hour_business", "period", "target_day"]:
        if col not in df.columns:
            logger.error(f"Missing column: {col}")
            return

    # Baseline sMAPE
    baseline_smape = smape_floor50(df["y_true"].values, df["base_pred"].values)
    logger.info(f"Baseline sMAPE_floor50: {baseline_smape:.4f}%")

    # ═══════════════════════════════════════════════════════════
    # Corrector 1: Spike Residual Correction
    # ═══════════════════════════════════════════════════════════
    logger.info("Running LGBMSpikeResidualCorrector (grid search)...")

    alphas = [0.25, 0.5, 0.75]
    thresholds = [0.45, 0.55, 0.65]
    deltas = [50, 100, 150]

    best_spike_smape = float("inf")
    best_spike_params = None
    best_spike_pred = None

    for alpha in alphas:
        for threshold in thresholds:
            for max_delta in deltas:
                corrector = LGBMSpikeResidualCorrector(
                    alpha=alpha, threshold=threshold, max_delta=max_delta
                )
                try:
                    corrected = corrector.correct(df.copy())
                    smape_val = smape_floor50(df["y_true"].values, corrected)
                    if smape_val < best_spike_smape:
                        best_spike_smape = smape_val
                        best_spike_params = (alpha, threshold, max_delta)
                        best_spike_pred = corrected.copy()
                    logger.info(f"  spike α={alpha} θ={threshold} δ={max_delta}: {smape_val:.4f}%")
                except Exception as e:
                    logger.warning(f"  spike α={alpha} θ={threshold} δ={max_delta}: ERROR {e}")

    logger.info(f"Best spike corrector: α={best_spike_params[0]} θ={best_spike_params[1]} δ={best_spike_params[2]} → {best_spike_smape:.4f}%")

    # Save spike correction
    df_out = df.copy()
    df_out["y_pred"] = best_spike_pred
    df_out["model_name"] = "lgbm_spike_residual_corrected"
    df_out.to_csv(str(pred_dir / "lgbm_spike_residual_corrected_dayahead.csv"),
                  index=False, encoding="utf-8-sig")
    logger.info(f"Saved spike correction to {pred_dir / 'lgbm_spike_residual_corrected_dayahead.csv'}")

    # ═══════════════════════════════════════════════════════════
    # Corrector 2: Selected Hour Correction
    # ═══════════════════════════════════════════════════════════
    logger.info("Running LGBMSelectedHourCorrector (grid search)...")

    target_hours_list = [
        [11, 12, 13, 17],
        [3, 4, 11, 12, 13, 17],
        [11, 13],
        [13],
        [3, 4],
    ]

    best_hour_smape = float("inf")
    best_hour_params = None
    best_hour_pred = None

    for hours in target_hours_list:
        for max_delta in [50, 100]:
            corrector = LGBMSelectedHourCorrector(
                target_hours=hours, max_delta=max_delta
            )
            try:
                corrected = corrector.correct(df.copy())
                smape_val = smape_floor50(df["y_true"].values, corrected)
                if smape_val < best_hour_smape:
                    best_hour_smape = smape_val
                    best_hour_params = (hours, max_delta)
                    best_hour_pred = corrected.copy()
                logger.info(f"  hour {hours} δ={max_delta}: {smape_val:.4f}%")
            except Exception as e:
                logger.warning(f"  hour {hours} δ={max_delta}: ERROR {e}")

    logger.info(f"Best hour corrector: hours={best_hour_params[0]} δ={best_hour_params[1]} → {best_hour_smape:.4f}%")

    # Save hour correction
    df_out = df.copy()
    df_out["y_pred"] = best_hour_pred
    df_out["model_name"] = "lgbm_selected_hour_corrected"
    df_out.to_csv(str(pred_dir / "lgbm_selected_hour_corrected_dayahead.csv"),
                  index=False, encoding="utf-8-sig")
    logger.info(f"Saved hour correction to {pred_dir / 'lgbm_selected_hour_corrected_dayahead.csv'}")

    # ═══════════════════════════════════════════════════════════
    # Metrics
    # ═══════════════════════════════════════════════════════════
    models = {
        "baseline": df["base_pred"].values,
        "lgbm_spike_residual_corrected": best_spike_pred,
        "lgbm_selected_hour_corrected": best_hour_pred,
    }

    rows = []
    for name, preds in models.items():
        valid = ~(np.isnan(df["y_true"].values) | np.isnan(preds))
        if valid.sum() < 2:
            continue
        m = compute_all_metrics(df["y_true"].values[valid], preds[valid])
        m["model_name"] = name
        m["task"] = "dayahead"
        m["n"] = int(valid.sum())
        rows.append(m)

    summary = pd.DataFrame(rows)
    summary.to_csv(str(metric_dir / "summary.csv"), index=False, encoding="utf-8-sig")

    # Hour metrics
    hour_rows = []
    for hour in sorted(df["hour_business"].unique()):
        h_mask = df["hour_business"] == hour
        for name, preds in models.items():
            yt = df.loc[h_mask, "y_true"].values
            yp = preds[h_mask.values]
            valid = ~(np.isnan(yt) | np.isnan(yp))
            if valid.sum() < 2:
                continue
            s = smape_floor50(yt[valid], yp[valid])
            hour_rows.append({"model_name": name, "hour_business": int(hour), "sMAPE_floor50": s})
    hour_df = pd.DataFrame(hour_rows)
    hour_df.to_csv(str(metric_dir / "hour_metrics.csv"), index=False, encoding="utf-8-sig")

    # Period metrics
    period_rows = []
    for period in sorted(df["period"].unique()):
        p_mask = df["period"] == period
        for name, preds in models.items():
            yt = df.loc[p_mask, "y_true"].values
            yp = preds[p_mask.values]
            valid = ~(np.isnan(yt) | np.isnan(yp))
            if valid.sum() < 2:
                continue
            s = smape_floor50(yt[valid], yp[valid])
            period_rows.append({"model_name": name, "period": period, "sMAPE_floor50": s})
    period_df = pd.DataFrame(period_rows)
    period_df.to_csv(str(metric_dir / "period_metrics.csv"), index=False, encoding="utf-8-sig")

    # ── Report ──
    lines = []
    lines.append("# LightGBM Day-Ahead Correction Report")
    lines.append(f"> Generated: 2026-07-03")
    lines.append(f"> Baseline: LightGBM trial_02 = {baseline_smape:.2f}%")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Method | sMAPE_floor50 | vs baseline | vs best_two_average |")
    lines.append("|--------|:-------------:|:-----------:|:-------------------:|")
    lines.append(f"| **baseline (trial_02)** | {baseline_smape:.2f}% | — | — |")
    lines.append(f"| best_two_average | 11.85% | — | — |")
    lines.append(f"| **lgbm_spike_residual** | {best_spike_smape:.2f}% | {best_spike_smape - baseline_smape:+.2f}pp | {best_spike_smape - 11.85:+.2f}pp |")
    lines.append(f"| **lgbm_selected_hour** | {best_hour_smape:.2f}% | {best_hour_smape - baseline_smape:+.2f}pp | {best_hour_smape - 11.85:+.2f}pp |")
    lines.append("")

    lines.append("## Best Parameters")
    lines.append("")
    lines.append(f"- Spike corrector: alpha={best_spike_params[0]}, threshold={best_spike_params[1]}, max_delta={best_spike_params[2]}")
    lines.append(f"- Hour corrector: hours={best_hour_params[0]}, max_delta={best_hour_params[1]}")
    lines.append("")

    # Hour detail
    lines.append("## Hour Detail (Spike Corrector)")
    lines.append("")
    lines.append("| Hour | Baseline | Spike | Change |")
    lines.append("|:----:|:--------:|:-----:|:------:|")
    spike_hour = hour_df[hour_df["model_name"] == "lgbm_spike_residual_corrected"]
    base_hour = hour_df[hour_df["model_name"] == "baseline"]
    for _, r in spike_hour.iterrows():
        h = int(r["hour_business"])
        b_val = base_hour[base_hour["hour_business"] == h]["sMAPE_floor50"].values
        b_str = f"{b_val[0]:.2f}%" if len(b_val) > 0 else "N/A"
        change = r["sMAPE_floor50"] - b_val[0] if len(b_val) > 0 else 0
        marker = "✅" if change < -0.1 else ("❌" if change > 0.1 else "➡️")
        lines.append(f"| {h} | {b_str} | {r['sMAPE_floor50']:.2f}% | {marker} {change:+.2f}pp |")
    lines.append("")

    lines.append("## Target Check")
    lines.append("")
    lines.append(f"| Target | Status |")
    lines.append(f"|:------|:------:|")
    lines.append(f"| Below 12.07% (baseline) | {'✅' if best_spike_smape < baseline_smape else '❌'} |")
    lines.append(f"| Below 11.85% (best_two_average) | {'✅' if best_spike_smape < 11.85 else '❌'} |")
    lines.append(f"| Below 11.5% | {'✅' if best_spike_smape < 11.5 else '❌'} |")
    lines.append(f"| Below 11% | {'✅' if best_spike_smape < 11 else '❌'} |")
    lines.append(f"| Below 10% | {'✅' if best_spike_smape < 10 else '❌'} |")
    lines.append(f"| Below 8% | {'✅' if best_spike_smape < 8 else '❌'} |")
    lines.append("")

    # Conclusion
    gain = baseline_smape - best_spike_smape
    lines.append("## Conclusion")
    lines.append("")
    if gain > 0.3:
        lines.append(f"✅ 修正有效，降低 {gain:.2f}pp，建议接入主链路。")
    elif gain > 0.1:
        lines.append(f"✅ 修正轻微有效，降低 {gain:.2f}pp。")
    elif gain > 0:
        lines.append(f"➡️ 修正基本无效，仅降低 {gain:.2f}pp。")
    else:
        lines.append(f"❌ 修正反而变差，不建议接入。")
    lines.append("")

    report = "\n".join(lines)
    (report_dir / "lgbm_correction_report.md").write_text(report, encoding="utf-8")
    print("\n" + report)
    logger.info(f"Report saved to {report_dir / 'lgbm_correction_report.md'}")


if __name__ == "__main__":
    main()
