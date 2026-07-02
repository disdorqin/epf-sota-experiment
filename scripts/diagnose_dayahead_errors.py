"""
diagnose_dayahead_errors.py — Day-ahead error diagnosis script.

Reads day-ahead prediction CSVs and produces a diagnostic report:
1. Daily sMAPE (worst 10 / best 10 days)
2. Per period (1_8, 9_16, 17_24)
3. Per hour_business (worst / most stable)
4. Per price range
5. Spike hours (top 10% / top 5% y_true)
6. Spring Festival window
7. Model complementarity (CatBoost vs TabPFN)

Usage:
    python scripts/diagnose_dayahead_errors.py ^
        --input-root outputs/dayahead_30d_core ^
        --output-root outputs/dayahead_diagnosis
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Ensure project root is on sys.path so `import src` works
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Metrics ─────────────────────────────────────────────────────────────────────


def _smape_floor50(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true_clip = np.where(y_true < 50.0, 50.0, y_true)
    pred_clip = np.where(y_pred < 50.0, 50.0, y_pred)
    denom = (np.abs(true_clip) + np.abs(pred_clip)) / 2.0
    denom = np.where(denom < 1e-8, 1e-8, denom)
    return float(np.mean(np.abs(pred_clip - true_clip) / denom) * 100.0)


def _smape_raw(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.abs(y_true) + np.abs(y_pred)
    denom = np.where(denom < 1e-8, 1e-8, denom)
    return float(np.mean(2.0 * np.abs(y_true - y_pred) / denom) * 100.0)


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


# ── Load predictions ────────────────────────────────────────────────────────────


def _find_input_root(cli_root: str) -> Path:
    candidates = [
        Path(cli_root),
        Path("outputs/catboost_tabpfn_30d"),
        Path("outputs/sota_walkforward"),
        Path("outputs/sota_walkforward_7d"),
    ]
    for p in candidates:
        pred_dir = p / "predictions"
        if pred_dir.exists():
            csvs = list(pred_dir.glob("*dayahead*.csv"))
            if len(csvs) >= 1:
                logger.info(f"Found input root: {p}")
                return p
    return None


def _load_dayahead_preds(input_root: Path) -> dict[str, pd.DataFrame]:
    pred_dir = input_root / "predictions"
    results = {}
    if not pred_dir.exists():
        logger.warning(f"Predictions dir not found: {pred_dir}")
        return results
    for csv_path in sorted(pred_dir.glob("*.csv")):
        name = csv_path.stem
        if "dayahead" not in name and "day_ahead" not in name:
            continue
        try:
            df = pd.read_csv(str(csv_path), encoding="utf-8-sig")
            if "task" in df.columns:
                df = df[df["task"] == "dayahead"].copy()
            if len(df) == 0:
                continue
            results[name] = df
            logger.info(f"  Loaded {name}: {len(df)} rows")
        except Exception as e:
            logger.warning(f"  Failed to load {csv_path}: {e}")
    return results


# ── Diagnosis ───────────────────────────────────────────────────────────────────


def _diagnose(preds: dict[str, pd.DataFrame], output_root: Path) -> None:
    """
    Run all diagnostic analyses and write reports.
    """
    report_dir = output_root / "reports"
    debug_dir = output_root / "debug"
    report_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    # ── Find primary model pair (catboost + tabpfn) ───────────────────────────
    cb_name = None
    tp_name = None
    for name in preds:
        if "catboost" in name and "dayahead" in name:
            cb_name = name
        if "tabpfn" in name and "dayahead" in name:
            tp_name = name

    if cb_name is None or tp_name is None:
        logger.warning("Could not find both catboost and tabpfn dayahead predictions")
        # Still diagnose whatever is available
        all_dfs = list(preds.values())
        if len(all_dfs) > 0:
            diag_df = all_dfs[0].copy()
        else:
            logger.error("No dayahead predictions found at all")
            return
    else:
        # Use catboost as base (has all rows)
        diag_df = preds[cb_name].copy()
        if tp_name in preds:
            tp_df = preds[tp_name].copy()
            # Merge tabpfn preds
            merge_keys = ["ds", "y_true"]
            for col in ["task", "target_day", "hour_business", "period"]:
                if col in diag_df.columns and col in tp_df.columns:
                    if col not in merge_keys:
                        merge_keys.append(col)
            diag_df = diag_df.merge(
                tp_df[merge_keys + ["y_pred"]].rename(columns={"y_pred": "y_pred_tabpfn"}),
                on=merge_keys,
                how="left",
            )
            diag_df["y_pred_tabpfn"] = diag_df["y_pred_tabpfn"].fillna(diag_df["y_pred"])
        diag_df["y_pred_catboost"] = diag_df["y_pred"]

    # Ensure required columns
    for col in ["y_true", "y_pred", "target_day", "period", "hour_business"]:
        if col not in diag_df.columns:
            logger.error(f"Required column '{col}' not found in predictions")
            return

    # ── 1. Daily sMAPE ───────────────────────────────────────────────────────
    logger.info("Computing daily sMAPE...")
    daily = (
        diag_df.groupby("target_day")
        .apply(lambda g: pd.Series({
            "smape_floor50": _smape_floor50(g["y_true"].values, g["y_pred"].values),
            "mae": _mae(g["y_true"].values, g["y_pred"].values),
            "n": len(g),
        }))
        .sort_values("smape_floor50", ascending=False)
    )
    worst_10 = daily.head(10)
    best_10 = daily.sort_values("smape_floor50").head(10)

    # ── 2. Per period ──────────────────────────────────────────────────────────
    logger.info("Computing per-period sMAPE...")
    period_stats = (
        diag_df.groupby("period")
        .apply(lambda g: pd.Series({
            "smape_floor50": _smape_floor50(g["y_true"].values, g["y_pred"].values),
            "mae": _mae(g["y_true"].values, g["y_pred"].values),
            "n": len(g),
        }))
        .sort_values("smape_floor50")
    )

    # ── 3. Per hour_business ───────────────────────────────────────────────────
    logger.info("Computing per-hour sMAPE...")
    hour_stats = (
        diag_df.groupby("hour_business")
        .apply(lambda g: pd.Series({
            "smape_floor50": _smape_floor50(g["y_true"].values, g["y_pred"].values),
            "mae": _mae(g["y_true"].values, g["y_pred"].values),
            "n": len(g),
        }))
        .sort_values("smape_floor50")
    )
    worst_5_hours = hour_stats.sort_values("smape_floor50", ascending=False).head(5)
    best_5_hours = hour_stats.head(5)

    # ── 4. Per price range ─────────────────────────────────────────────────────
    logger.info("Computing per-price-range sMAPE...")

    def _price_range(y):
        if y < 0:
            return "y_true < 0"
        elif y < 50:
            return "0 ≤ y_true < 50"
        elif y < 200:
            return "50 ≤ y_true < 200"
        elif y < 500:
            return "200 ≤ y_true < 500"
        else:
            return "y_true ≥ 500"

    diag_df["price_range"] = diag_df["y_true"].apply(_price_range)
    price_stats = (
        diag_df.groupby("price_range")
        .apply(lambda g: pd.Series({
            "smape_floor50": _smape_floor50(g["y_true"].values, g["y_pred"].values),
            "mae": _mae(g["y_true"].values, g["y_pred"].values),
            "n": len(g),
        }))
    )

    # ── 5. Spike hours ─────────────────────────────────────────────────────────
    logger.info("Computing spike hour performance...")
    y_true_all = diag_df["y_true"].values
    spike_threshold_10 = np.quantile(y_true_all, 0.90)
    spike_threshold_5 = np.quantile(y_true_all, 0.95)

    spike_10_mask = diag_df["y_true"] >= spike_threshold_10
    spike_5_mask = diag_df["y_true"] >= spike_threshold_5

    spike_10_smape = _smape_floor50(diag_df.loc[spike_10_mask, "y_true"].values, diag_df.loc[spike_10_mask, "y_pred"].values) if spike_10_mask.sum() > 0 else np.nan
    spike_5_smape = _smape_floor50(diag_df.loc[spike_5_mask, "y_true"].values, diag_df.loc[spike_5_mask, "y_pred"].values) if spike_5_mask.sum() > 0 else np.nan

    # Also diagnose catboost vs tabpfn on spikes
    spike_10_catboost = np.nan
    spike_10_tabpfn = np.nan
    if "y_pred_catboost" in diag_df.columns and spike_10_mask.sum() > 0:
        spike_10_catboost = _smape_floor50(diag_df.loc[spike_10_mask, "y_true"].values, diag_df.loc[spike_10_mask, "y_pred_catboost"].values)
    if "y_pred_tabpfn" in diag_df.columns and spike_10_mask.sum() > 0:
        spike_10_tabpfn = _smape_floor50(diag_df.loc[spike_10_mask, "y_true"].values, diag_df.loc[spike_10_mask, "y_pred_tabpfn"].values)

    # ── 6. Spring Festival window ──────────────────────────────────────────────
    logger.info("Checking Spring Festival window...")
    spring_festival_2026 = pd.Timestamp("2026-02-17")
    diag_df["days_to_sf"] = (spring_festival_2026 - pd.to_datetime(diag_df["target_day"])).dt.days
    diag_df["is_sf_window"] = diag_df["days_to_sf"].between(-7, 7)

    sf_window = diag_df[diag_df["is_sf_window"]]
    non_sf = diag_df[~diag_df["is_sf_window"]]

    sf_smape = _smape_floor50(sf_window["y_true"].values, sf_window["y_pred"].values) if len(sf_window) > 0 else np.nan
    non_sf_smape = _smape_floor50(non_sf["y_true"].values, non_sf["y_pred"].values) if len(non_sf) > 0 else np.nan

    # ── 7. Model complementarity ───────────────────────────────────────────────
    logger.info("Computing model complementarity...")
    complementarity = {}
    if "y_pred_catboost" in diag_df.columns and "y_pred_tabpfn" in diag_df.columns:
        valid = ~(diag_df["y_pred_catboost"].isna() | diag_df["y_pred_tabpfn"].isna() | diag_df["y_true"].isna())
        vdf = diag_df[valid].copy()
        vdf["err_cb"] = np.abs(vdf["y_true"] - vdf["y_pred_catboost"])
        vdf["err_tp"] = np.abs(vdf["y_true"] - vdf["y_pred_tabpfn"])
        vdf["cb_correct"] = vdf["err_cb"] < vdf["err_tp"]
        vdf["tp_correct"] = vdf["err_tp"] < vdf["err_cb"]

        n_cb_only = ((vdf["cb_correct"]) & ~(vdf["tp_correct"])).sum()
        n_tp_only = ((vdf["tp_correct"]) & ~(vdf["cb_correct"])).sum()
        n_both_wrong = (~(vdf["cb_correct"]) & ~(vdf["tp_correct"])).sum()
        n_both_right = (vdf["cb_correct"] & vdf["tp_correct"]).sum()

        # Correlation of absolute errors
        corr_err = float(vdf["err_cb"].corr(vdf["err_tp"])) if len(vdf) > 2 else np.nan

        # Disagreement → high error risk
        vdf["disagreement"] = np.abs(vdf["y_pred_catboost"] - vdf["y_pred_tabpfn"])
        high_err_thresh = np.quantile(vdf["err_cb"].values, 0.8)
        high_err_mask = vdf["err_cb"] >= high_err_thresh
        low_err_mask = ~high_err_mask
        mean_disagree_high = float(vdf.loc[high_err_mask, "disagreement"].mean()) if high_err_mask.sum() > 0 else np.nan
        mean_disagree_low = float(vdf.loc[low_err_mask, "disagreement"].mean()) if low_err_mask.sum() > 0 else np.nan

        complementarity = {
            "n_cb_only": n_cb_only,
            "n_tp_only": n_tp_only,
            "n_both_wrong": n_both_wrong,
            "n_both_right": n_both_right,
            "corr_abs_error": corr_err,
            "mean_disagreement_high_err": mean_disagree_high,
            "mean_disagreement_low_err": mean_disagree_low,
            "disagreement_predicts_error": mean_disagree_high > mean_disagree_low if not np.isnan(mean_disagree_high) and not np.isnan(mean_disagree_low) else None,
        }

    # ── Save error segments CSV ─────────────────────────────────────────────────
    error_segments = daily.reset_index()[["target_day", "smape_floor50", "mae", "n"]]
    error_segments.to_csv(debug_dir / "dayahead_error_segments.csv", index=False, encoding="utf-8-sig")

    # ── Build report ────────────────────────────────────────────────────────────
    logger.info("Building diagnostic report...")
    overall_smape = _smape_floor50(diag_df["y_true"].values, diag_df["y_pred"].values)
    overall_mae = _mae(diag_df["y_true"].values, diag_df["y_pred"].values)

    lines = []
    lines.append("# Day-Ahead Error Diagnosis Report")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Summary
    lines.append("## Summary")
    lines.append(f"- **Overall day-ahead sMAPE (floor50):** `{overall_smape:.2f}%`")
    lines.append(f"- **Overall day-ahead MAE:** `{overall_mae:.2f}`")
    lines.append(f"- **Total predictions:** `{len(diag_df)}`")
    lines.append(f"- **Days evaluated:** `{daily.shape[0]}`")
    lines.append("")
    gap_to_8 = overall_smape - 8.0
    if gap_to_8 > 0:
        lines.append(f"⚠️ **Still `{gap_to_8:.2f}` pp above 8% target.**")
    else:
        lines.append(f"✅ **Below 8% target!** (`{overall_smape:.2f}%`)")
    lines.append("")

    # 1. Daily sMAPE
    lines.append("## 1. Daily sMAPE")
    lines.append("")
    lines.append("**Worst 10 days:**")
    lines.append("")
    lines.append("| target_day | sMAPE_floor50 | MAE | n |")
    lines.append("|------------|---------------|-----|---|")
    for _, row in worst_10.iterrows():
        lines.append(f"| {row.name} | {row['smape_floor50']:.2f}% | {row['mae']:.2f} | {int(row['n'])} |")
    lines.append("")
    lines.append("**Best 10 days:**")
    lines.append("")
    lines.append("| target_day | sMAPE_floor50 | MAE | n |")
    lines.append("|------------|---------------|-----|---|")
    for _, row in best_10.iterrows():
        lines.append(f"| {row.name} | {row['smape_floor50']:.2f}% | {row['mae']:.2f} | {int(row['n'])} |")
    lines.append("")

    # 2. Per period
    lines.append("## 2. Per Period")
    lines.append("")
    lines.append("| period | sMAPE_floor50 | MAE | n |")
    lines.append("|--------|---------------|-----|---|")
    for period, row in period_stats.iterrows():
        lines.append(f"| {period} | {row['smape_floor50']:.2f}% | {row['mae']:.2f} | {int(row['n'])} |")
    lines.append("")

    # 3. Per hour
    lines.append("## 3. Per Hour (hour_business)")
    lines.append("")
    lines.append("**Worst 5 hours:**")
    lines.append("")
    lines.append("| hour_business | sMAPE_floor50 | MAE | n |")
    lines.append("|---------------|---------------|-----|---|")
    for _, row in worst_5_hours.iterrows():
        lines.append(f"| {row.name} | {row['smape_floor50']:.2f}% | {row['mae']:.2f} | {int(row['n'])} |")
    lines.append("")
    lines.append("**Most stable 5 hours:**")
    lines.append("")
    lines.append("| hour_business | sMAPE_floor50 | MAE | n |")
    lines.append("|---------------|---------------|-----|---|")
    for _, row in best_5_hours.iterrows():
        lines.append(f"| {row.name} | {row['smape_floor50']:.2f}% | {row['mae']:.2f} | {int(row['n'])} |")
    lines.append("")

    # 4. Per price range
    lines.append("## 4. Per Price Range")
    lines.append("")
    lines.append("| price_range | sMAPE_floor50 | MAE | n |")
    lines.append("|-------------|---------------|-----|---|")
    for rng, row in price_stats.iterrows():
        lines.append(f"| {rng} | {row['smape_floor50']:.2f}% | {row['mae']:.2f} | {int(row['n'])} |")
    lines.append("")

    # 5. Spike hours
    lines.append("## 5. Spike Hours")
    lines.append(f"- **Top 10% threshold (y_true):** `{spike_threshold_10:.1f}`")
    lines.append(f"- **Top 5% threshold (y_true):** `{spike_threshold_5:.1f}`")
    lines.append(f"- **sMAPE on top 10% spikes:** `{spike_10_smape:.2f}%`" if not np.isnan(spike_10_smape) else "- sMAPE on top 10% spikes: N/A")
    lines.append(f"- **sMAPE on top 5% spikes:** `{spike_5_smape:.2f}%`" if not np.isnan(spike_5_smape) else "- sMAPE on top 5% spikes: N/A")
    if not np.isnan(spike_10_catboost):
        lines.append(f"- **CatBoost sMAPE on top 10% spikes:** `{spike_10_catboost:.2f}%`")
    if not np.isnan(spike_10_tabpfn):
        lines.append(f"- **TabPFN sMAPE on top 10% spikes:** `{spike_10_tabpfn:.2f}%`")
    lines.append("")

    # 6. Spring Festival window
    lines.append("## 6. Spring Festival Window (±7 days around 2026-02-17)")
    lines.append(f"- **SF window sMAPE:** `{sf_smape:.2f}%`" if not np.isnan(sf_smape) else "- SF window sMAPE: N/A (no data)")
    lines.append(f"- **Non-SF sMAPE:** `{non_sf_smape:.2f}%`" if not np.isnan(non_sf_smape) else "- Non-SF sMAPE: N/A (no data)")
    if not np.isnan(sf_smape) and not np.isnan(non_sf_smape):
        diff = sf_smape - non_sf_smape
        if diff > 2.0:
            lines.append(f"⚠️ **SF window is `{diff:.2f}` pp worse — major drag.**")
        elif diff > 0.5:
            lines.append(f"⚠️ SF window is `{diff:.2f}` pp worse (moderate drag).")
        else:
            lines.append(f"✅ SF window is within `{diff:.2f}` pp (not a major drag).")
    lines.append("")

    # 7. Model complementarity
    lines.append("## 7. Model Complementarity (CatBoost vs TabPFN)")
    if len(complementarity) > 0:
        lines.append(f"- **CatBoost-only wins:** `{complementarity['n_cb_only']}` hours")
        lines.append(f"- **TabPFN-only wins:** `{complementarity['n_tp_only']}` hours")
        lines.append(f"- **Both wrong:** `{complementarity['n_both_wrong']}` hours")
        lines.append(f"- **Both right:** `{complementarity['n_both_right']}` hours")
        lines.append(f"- **Correlation of abs errors:** `{complementarity['corr_abs_error']:.4f}`" if not np.isnan(complementarity['corr_abs_error']) else "- Correlation: N/A")
        if complementarity["disagreement_predicts_error"] is not None:
            if complementarity["disagreement_predicts_error"]:
                lines.append(f"✅ **Disagreement predicts high error** (mean disagree on high-err: `{complementarity['mean_disagreement_high_err']:.1f}`, low-err: `{complementarity['mean_disagreement_low_err']:.1f}`)")
            else:
                lines.append(f"⚠️ Disagreement does NOT strongly predict high error.")
        lines.append("")
        fusion_worth = (complementarity['n_cb_only'] > 0 and complementarity['n_tp_only'] > 0) or (complementarity['corr_abs_error'] < 0.7)
        lines.append(f"**Fusion value:** {'✅ High (models make different errors)' if fusion_worth else '⚠️ Low (models make similar errors)'}")
    else:
        lines.append("- N/A (need both CatBoost and TabPFN predictions)")
    lines.append("")

    # Final diagnosis
    lines.append("---")
    lines.append("")
    lines.append("## Diagnosis & Recommendations")
    lines.append("")
    lines.append(f"1. **Current day-ahead sMAPE:** `{overall_smape:.2f}%`")
    lines.append(f"2. **Gap to 8% target:** `{gap_to_8:.2f}` pp")
    lines.append("")

    # Find biggest error source
    worst_period = period_stats.sort_values("smape_floor50", ascending=False).iloc[0]
    worst_hour = hour_stats.sort_values("smape_floor50", ascending=False).iloc[0]
    lines.append(f"3. **Biggest error source:**")
    lines.append(f"   - Worst period: `{worst_period.name}` (sMAPE = `{worst_period['smape_floor50']:.2f}%`)")
    lines.append(f"   - Worst hour: `hour {worst_hour.name}` (sMAPE = `{worst_hour['smape_floor50']:.2f}%`)")
    lines.append("")

    # Priority
    priorities = []
    if gap_to_8 > 2.0:
        priorities.append("🚨 **Urgent:** gap to 8% is large, consider target transformation (asinh), hour specialists, or more aggressive feature engineering")
    if not np.isnan(sf_smape) and not np.isnan(non_sf_smape) and (sf_smape - non_sf_smape) > 2.0:
        priorities.append("🔴 **Spring Festival window** is a major drag — add `is_spring_festival_window` feature or exclude/expost-correct")
    if spike_10_smape > overall_smape + 3.0:
        priorities.append("🔴 **Spike hours** are under-predicted — consider spike correction or asymmetric loss")
    if worst_hour["smape_floor50"] > overall_smape + 3.0:
        priorities.append(f"🔴 **Hour {worst_hour.name}** is a major outlier — consider hour specialist model")
    if len(complementarity) > 0 and complementarity["fusion_worth"]:
        priorities.append("✅ **Fusion helps** — CatBoost and TabPFN make different errors, fusion should improve")
    else:
        priorities.append("⚠️ **Fusion may not help much** — models make similar errors")

    lines.append("4. **Priority optimization targets:**")
    for p in priorities:
        lines.append(f"   - {p}")
    lines.append("")

    report_path = report_dir / "dayahead_error_diagnosis.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info(f"Report written to {report_path}")

    # Also print summary
    print(f"\n{'='*60}")
    print(f"Day-Ahead Diagnosis Summary")
    print(f"{'='*60}")
    print(f"  Overall sMAPE (floor50): {overall_smape:.2f}%")
    print(f"  Gap to 8% target:         {gap_to_8:.2f} pp")
    print(f"  Worst period:               {worst_period.name} ({worst_period['smape_floor50']:.2f}%)")
    print(f"  Worst hour:                hour {worst_hour.name} ({worst_hour['smape_floor50']:.2f}%)")
    if len(complementarity) > 0:
        print(f"  Fusion value:               {'High' if fusion_worth else 'Low'}")
    print(f"{'='*60}\n")


# ── CLI ─────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Day-ahead error diagnosis")
    parser.add_argument("--input-root", type=str, default="outputs/dayahead_30d_core", help="Root directory containing predictions/")
    parser.add_argument("--output-root", type=str, default="outputs/dayahead_diagnosis", help="Output directory")
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)

    # Find actual input root
    found_root = _find_input_root(str(input_root))
    if found_root is None:
        print("day-ahead predictions not found yet. Run day-ahead walk-forward first.")
        sys.exit(1)

    print(f"Using input root: {found_root}")
    preds = _load_dayahead_preds(found_root)
    if len(preds) == 0:
        print("day-ahead predictions not found yet. Run day-ahead walk-forward first.")
        sys.exit(1)

    print(f"Loaded {len(preds)} day-ahead prediction files")
    _diagnose(preds, output_root)


if __name__ == "__main__":
    main()
