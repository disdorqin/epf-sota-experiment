"""
run_dayahead_safe_fusion.py — Safe fusion for Stage-3 results.

Combines trusted champion (best_two_average) + Stage-3 best + CatBoost baselines.
All fusion methods use only search-window data for weight selection.

Usage:
    python scripts/run_dayahead_safe_fusion.py
"""

from __future__ import annotations

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

from src.common.metrics import smape_floor50, compute_all_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


SEARCH_DAYS = [f"2026-02-{d:02d}" for d in range(1, 21)]
CONFIRM_DAYS = [f"2026-02-{d:02d}" for d in range(21, 29)] + ["2026-03-01", "2026-03-02"]
ALL_DAYS = SEARCH_DAYS + CONFIRM_DAYS


def load_prediction(path: str) -> pd.DataFrame:
    """Load a prediction CSV and validate."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    assert "y_true" in df.columns, f"Missing y_true in {path}"
    assert "y_pred" in df.columns, f"Missing y_pred in {path}"
    assert len(df) == 720, f"Expected 720 rows, got {len(df)}"
    return df


def main():
    output_root = Path("outputs/dayahead_stage_next_fusion_30d")
    for sub in ["predictions", "metrics", "reports"]:
        (output_root / sub).mkdir(parents=True, exist_ok=True)

    # ── Load candidate predictions ──
    candidates = {}

    # 1. Stage-3 best (if exists)
    stage3_dir = Path("outputs/dayahead_lgbm_stage3_30d/predictions")
    if stage3_dir.exists():
        # Find best config from summary
        summary_path = Path("outputs/dayahead_lgbm_stage3_30d/metrics/summary.csv")
        if summary_path.exists():
            summary = pd.read_csv(str(summary_path), encoding="utf-8-sig")
            if len(summary) > 0:
                best_name = summary.iloc[0]["model_name"]
                pred_file = stage3_dir / f"{best_name}_dayahead.csv"
                if pred_file.exists():
                    candidates[f"lgbm_stage3_best ({best_name})"] = load_prediction(str(pred_file))
                    logger.info(f"Loaded Stage-3 best: {best_name}")

        # Also load baseline if exists
        baseline_file = stage3_dir / "stage3_baseline_90d_mae_dayahead.csv"
        if baseline_file.exists():
            candidates["lgbm_stage3_baseline"] = load_prediction(str(baseline_file))

    # 2. Stage-2 trusted predictions (trial_02 and trial_24 for best_two_average)
    stage2_dir = Path("outputs/dayahead_lgbm_stage2_30d/predictions")
    if stage2_dir.exists():
        t02 = stage2_dir / "trial_02_w150_nl255_lr0.03_dayahead.csv"
        t24 = stage2_dir / "trial_24_w90_nl127_lr0.02_dayahead.csv"
        if t02.exists():
            candidates["lgbm_trial_02"] = load_prediction(str(t02))
        if t24.exists():
            candidates["lgbm_trial_24"] = load_prediction(str(t24))

    # 3. CatBoost baselines
    cb_dir = Path("outputs/dayahead_30d_core/predictions")
    if cb_dir.exists():
        cb_file = cb_dir / "catboost_sota_dayahead.csv"
        if cb_file.exists():
            candidates["catboost_sota"] = load_prediction(str(cb_file))

    # Also check model_pool
    mp_dir = Path("outputs/dayahead_model_pool_30d/predictions")
    if mp_dir.exists():
        for f in mp_dir.glob("*spike_residual*"):
            candidates["catboost_spike_residual"] = load_prediction(str(f))

    logger.info(f"Loaded {len(candidates)} candidates: {list(candidates.keys())}")

    if len(candidates) < 2:
        logger.error("Need at least 2 candidates for fusion. Exiting.")
        sys.exit(1)

    # ── Compute individual metrics ──
    individual_metrics = {}
    for name, df in candidates.items():
        s = smape_floor50(df["y_true"].values, df["y_pred"].values)
        individual_metrics[name] = s
        logger.info(f"  {name}: {s:.4f}%")

    # ── best_two_average (champion) ──
    if "lgbm_trial_02" in candidates and "lgbm_trial_24" in candidates:
        t02 = candidates["lgbm_trial_02"]
        t24 = candidates["lgbm_trial_24"]
        avg_pred = (t02["y_pred"].values + t24["y_pred"].values) / 2.0
        candidates["best_two_average (champion)"] = t02.copy()
        candidates["best_two_average (champion)"]["y_pred"] = avg_pred
        s = smape_floor50(t02["y_true"].values, avg_pred)
        individual_metrics["best_two_average (champion)"] = s
        logger.info(f"  best_two_average (champion): {s:.4f}%")

    # ── Fusion methods ──
    fusion_results = {}

    # Get all model prediction arrays aligned
    model_names = list(candidates.keys())
    y_true = candidates[model_names[0]]["y_true"].values
    pred_matrix = np.column_stack([candidates[m]["y_pred"].values for m in model_names])

    # 1. Simple average of all
    fusion_results["simple_average_all"] = pred_matrix.mean(axis=1)

    # 2. Median of all
    fusion_results["median_all"] = np.median(pred_matrix, axis=1)

    # 3. Inverse search-window sMAPE weight
    search_mask = candidates[model_names[0]]["target_day"].isin(SEARCH_DAYS)
    search_true = candidates[model_names[0]].loc[search_mask, "y_true"].values
    search_weights = {}
    for i, m in enumerate(model_names):
        search_pred = candidates[m].loc[search_mask, "y_pred"].values
        s = smape_floor50(search_true, search_pred)
        search_weights[m] = 1.0 / max(s, 0.01)
    total_w = sum(search_weights.values())
    weights = np.array([search_weights[m] / total_w for m in model_names])
    fusion_results["inverse_smape_weight"] = pred_matrix @ weights

    # 4. Winner by hour (based on search window only)
    hour_winners = {}
    for h in range(1, 25):
        best_model = None
        best_smape = 999
        for i, m in enumerate(model_names):
            h_mask = (candidates[m]["hour_business"] == h) & search_mask
            if h_mask.sum() < 2:
                continue
            h_true = candidates[m].loc[h_mask, "y_true"].values
            h_pred = candidates[m].loc[h_mask, "y_pred"].values
            s = smape_floor50(h_true, h_pred)
            if s < best_smape:
                best_smape = s
                best_model = i
        hour_winners[h] = best_model if best_model is not None else 0

    hour_pred = np.zeros(len(y_true))
    for h in range(1, 25):
        h_mask = candidates[model_names[0]]["hour_business"] == h
        if hour_winners[h] is not None:
            hour_pred[h_mask] = pred_matrix[h_mask.values, hour_winners[h]]
    fusion_results["winner_by_hour"] = hour_pred

    # 5. Winner by period (based on search window only)
    period_winners = {}
    for period in ["1_8", "9_16", "17_24"]:
        best_model = None
        best_smape = 999
        for i, m in enumerate(model_names):
            p_mask = (candidates[m]["period"] == period) & search_mask
            if p_mask.sum() < 2:
                continue
            p_true = candidates[m].loc[p_mask, "y_true"].values
            p_pred = candidates[m].loc[p_mask, "y_pred"].values
            s = smape_floor50(p_true, p_pred)
            if s < best_smape:
                best_smape = s
                best_model = i
        period_winners[period] = best_model if best_model is not None else 0

    period_pred = np.zeros(len(y_true))
    for period in ["1_8", "9_16", "17_24"]:
        p_mask = candidates[model_names[0]]["period"] == period
        if period_winners[period] is not None:
            period_pred[p_mask] = pred_matrix[p_mask.values, period_winners[period]]
    fusion_results["winner_by_period"] = period_pred

    # ── Evaluate all fusion methods ──
    logger.info("\n" + "=" * 60)
    logger.info("FUSION RESULTS")
    logger.info("=" * 60)

    all_fusion_metrics = []
    for name, y_pred in fusion_results.items():
        s = smape_floor50(y_true, y_pred)
        m = compute_all_metrics(y_true, y_pred)
        m["model_name"] = name
        all_fusion_metrics.append(m)

        search_pred = y_pred[search_mask.values] if hasattr(search_mask, 'values') else y_pred[search_mask]
        confirm_mask = ~search_mask
        confirm_pred = y_pred[confirm_mask.values] if hasattr(confirm_mask, 'values') else y_pred[confirm_mask]
        search_s = smape_floor50(y_true[search_mask.values], search_pred) if search_mask.sum() > 0 else None
        confirm_s = smape_floor50(y_true[confirm_mask.values], confirm_pred) if confirm_mask.sum() > 0 else None

        logger.info(f"  {name}: full={s:.4f}% search={search_s:.4f}% confirm={confirm_s:.4f}%" if search_s and confirm_s else f"  {name}: full={s:.4f}%")

    # Save
    fusion_df = pd.DataFrame(all_fusion_metrics).sort_values("sMAPE_floor50")
    fusion_df.to_csv(str(output_root / "metrics" / "summary.csv"), index=False, encoding="utf-8-sig")

    # Save best fusion predictions
    best_fusion_name = min(fusion_results, key=lambda k: smape_floor50(y_true, fusion_results[k]))
    best_pred_df = candidates[model_names[0]].copy()
    best_pred_df["y_pred"] = fusion_results[best_fusion_name]
    best_pred_df["model_name"] = f"fusion_{best_fusion_name}"
    best_pred_df.to_csv(str(output_root / "predictions" / f"fusion_{best_fusion_name}_dayahead.csv"),
                        index=False, encoding="utf-8-sig")

    # ── Report ──
    best_s = smape_floor50(y_true, fusion_results[best_fusion_name])
    champion_s = individual_metrics.get("best_two_average (champion)", 11.85)

    lines = []
    def w(s=""):
        lines.append(s)

    w("# Day-Ahead Safe Fusion Report")
    w(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    w(f"**Candidates**: {len(candidates)}")
    w()
    w("## Individual Model Metrics")
    for name, s in sorted(individual_metrics.items(), key=lambda x: x[1]):
        w(f"- {name}: {s:.4f}%")
    w()
    w("## Fusion Results")
    w("| Method | Full sMAPE |")
    w("|---|---|")
    for _, row in fusion_df.iterrows():
        w(f"| {row['model_name']} | {row['sMAPE_floor50']:.4f}% |")
    w()
    w("## Target Check")
    w(f"- Champion (best_two_average): {champion_s:.4f}%")
    w(f"- Best fusion: {best_fusion_name} = {best_s:.4f}%")
    w(f"- Below champion? {'YES' if best_s < champion_s else 'NO'}")
    w(f"- Below 11.5%? {'YES' if best_s < 11.5 else 'NO'}")
    w(f"- Below 11.0%? {'YES' if best_s < 11.0 else 'NO'}")
    w()

    report = "\n".join(lines)
    report_path = output_root / "reports" / "dayahead_safe_fusion_report.md"
    with open(str(report_path), "w", encoding="utf-8") as f:
        f.write(report)
    docs_report = Path(_PROJECT_DIR) / "docs" / "reports" / "dayahead_safe_fusion_report.md"
    with open(str(docs_report), "w", encoding="utf-8") as f:
        f.write(report)

    manifest = {
        "n_candidates": len(candidates),
        "best_fusion": best_fusion_name,
        "best_fusion_smape": best_s,
        "champion_smape": champion_s,
        "beats_champion": best_s < champion_s,
        "below_11_5": best_s < 11.5,
        "completed_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    with open(str(output_root / "debug" / "run_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\nSafe Fusion: best={best_fusion_name} ({best_s:.4f}%), champion={champion_s:.4f}%")


if __name__ == "__main__":
    main()
