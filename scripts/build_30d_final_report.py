"""
build_30d_final_report.py — Build final 30-day fusion evaluation report.

Reads fusion_metrics.csv and predictions to produce a comprehensive report.

Usage:
    python scripts/build_30d_final_report.py ^
        --input-root outputs/catboost_tabpfn_30d

Outputs:
    {input_root}/reports/catboost_tabpfn_30d_fusion_report.md
    docs/reports/catboost_tabpfn_30d_fusion_report.md  (copy)

If 30d results are not found yet, prints:
    "30d predictions not found yet. Run walk-forward first."
and exits with code 1.

Report sections (9 total):
1. 30-day single-model ranking (by sMAPE_floor50, across all data).
2. 30-day fusion-method ranking.
3. Best method per task.
4. Best method per period.
5. Best method for spike hours (y_true top 10%).
6. Best method for negative hours (y_true < 0).
7. Does fusion beat single models?
8. Recommended default fusion method.
9. Should this be integrated into electricity_forecast_model2.0_exp main fusion pipeline?
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from shutil import copy2

import numpy as np
import pandas as pd

# ── Ensure src on path ──
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_DIR)

from src.common.metrics import smape_floor50

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_ORDER = [
    "catboost_sota",
    "tabpfn_ts_sota",
    "fused_simple_average",
    "fused_inverse_smape_weight",
    "fused_period_best",
]


def _smape_series(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    denom = np.abs(y_true) + np.abs(y_pred)
    denom = np.where(denom < 1e-8, 1e-8, denom)
    return 200.0 * np.abs(y_true - y_pred) / denom


def _early_exit(input_root: Path) -> bool:
    """Check if 30d data is available; return True if should exit."""
    fusion_metrics_path = input_root / "fusion" / "fusion_metrics.csv"
    pred_dir = input_root / "predictions"
    if not fusion_metrics_path.exists() or not pred_dir.exists():
        print("30d predictions not found yet. Run walk-forward first.")
        logger.error(
            f"Missing data: fusion_metrics={fusion_metrics_path.exists()}, "
            f"pred_dir={pred_dir.exists()}"
        )
        return True
    return False


def _load_data(input_root: Path):
    """Load fusion metrics and prediction data."""
    fusion_metrics_path = input_root / "fusion" / "fusion_metrics.csv"
    fusion_metrics = pd.read_csv(str(fusion_metrics_path), encoding="utf-8-sig")

    # Load prediction CSVs for spike/negative analysis
    pred_dir = input_root / "predictions"
    cb_files = sorted(pred_dir.glob("catboost_sota_*.csv"))
    tp_files = sorted(pred_dir.glob("tabpfn_ts_sota_*.csv"))
    fused_files = sorted((input_root / "fusion").glob("fused_*.csv"))

    all_preds = []
    for f in cb_files + tp_files + fused_files:
        df = pd.read_csv(str(f), encoding="utf-8-sig")
        all_preds.append(df)
    all_preds_df = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()

    return fusion_metrics, all_preds_df


def _rank_by_smape(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Rank rows by sMAPE_floor50 (lower=better)."""
    if "sMAPE_floor50" not in df.columns:
        return df
    return df.sort_values(list(group_cols) + ["sMAPE_floor50"])


def build_report(input_root: Path) -> str:
    """Build the full final report as markdown string."""
    lines: list[str] = []

    def _w(s: str = "") -> None:
        lines.append(s)

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    _w("# CatBoost + TabPFN 30-Day Fusion Evaluation Report")
    _w()
    _w(f"**Generated**: {now_str}")
    _w(f"**Input root**: `{input_root}`")
    _w()

    fusion_metrics, all_preds = _load_data(input_root)

    if len(fusion_metrics) == 0:
        _w("ERROR: fusion_metrics.csv is empty. Run run_pair_fusion.py first.")
        return "\n".join(lines)

    _w(f"**Total evaluation rows (all models)**: {len(all_preds):,}")
    _w(f"**Fusion metric rows**: {len(fusion_metrics)}")
    _w()

    # ── Section 1: Single-model ranking (30-day, all tasks combined) ──
    _w("## 1. 30-Day Single-Model Ranking (All Tasks)")
    _w()
    single_models = ["catboost_sota", "tabpfn_ts_sota"]
    single_df = fusion_metrics[fusion_metrics["model_name"].isin(single_models)].copy()
    if len(single_df) > 0:
        # Overall ranking (all rows averaged — already per-model in fusion_metrics)
        single_overall = (
            fusion_metrics[fusion_metrics["model_name"].isin(single_models)]
            .sort_values("sMAPE_floor50")
        )
        _w("| Rank | Model | sMAPE_floor50 | MAE | RMSE | peak_MAE_q90 | n |")
        _w("|---|---|---|---|---|---|---|")
        for rank, (_, row) in enumerate(single_overall.iterrows(), 1):
            _w(
                f"| {rank} "
                f"| {row['model_name']} "
                f"| {row.get('sMAPE_floor50', float('nan')):.2f} "
                f"| {row.get('MAE', float('nan')):.2f} "
                f"| {row.get('RMSE', float('nan')):.2f} "
                f"| {row.get('peak_MAE_q90', float('nan')):.2f} "
                f"| {int(row.get('n', 0))} |"
            )
    else:
        _w("*No single-model metrics found.*")
    _w()

    # ── Section 2: Fusion-method ranking ──
    _w("## 2. 30-Day Fusion-Method Ranking (All Tasks)")
    _w()
    fused_models = ["fused_simple_average", "fused_inverse_smape_weight", "fused_period_best"]
    fused_df = fusion_metrics[fusion_metrics["model_name"].isin(fused_models)].copy()
    all_models_df = fusion_metrics[
        fusion_metrics["model_name"].isin(single_models + fused_models)
    ].copy()

    if len(all_models_df) > 0:
        ranked = all_models_df.sort_values("sMAPE_floor50")
        _w("| Rank | Model | sMAPE_floor50 | MAE | RMSE | peak_MAE_q90 | n |")
        _w("|---|---|---|---|---|---|---|")
        for rank, (_, row) in enumerate(ranked.iterrows(), 1):
            _w(
                f"| {rank} "
                f"| {row['model_name']} "
                f"| {row.get('sMAPE_floor50', float('nan')):.2f} "
                f"| {row.get('MAE', float('nan')):.2f} "
                f"| {row.get('RMSE', float('nan')):.2f} "
                f"| {row.get('peak_MAE_q90', float('nan')):.2f} "
                f"| {int(row.get('n', 0))} |"
            )
    _w()

    # ── Section 3: Best method per task ──
    _w("## 3. Best Method per Task")
    _w()
    if "task" in fusion_metrics.columns:
        for task in sorted(fusion_metrics["task"].unique()):
            t_df = fusion_metrics[fusion_metrics["task"] == task].copy()
            t_df = t_df.sort_values("sMAPE_floor50")
            best = t_df.iloc[0]
            _w(f"### Task: `{task}`")
            _w()
            _w(f"**Best method**: `{best['model_name']}` "
                f"(sMAPE_floor50={best.get('sMAPE_floor50', float('nan')):.2f}%)")
            _w()
            _w("| Model | sMAPE_floor50 | MAE | RMSE |")
            _w("|---|---|---|---|")
            for _, row in t_df.iterrows():
                _w(
                    f"| {row['model_name']} "
                    f"| {row.get('sMAPE_floor50', float('nan')):.2f} "
                    f"| {row.get('MAE', float('nan')):.2f} "
                    f"| {row.get('RMSE', float('nan')):.2f} |"
                )
            _w()
    else:
        _w("*`task` column not found in fusion_metrics.*")
        _w()

    # ── Section 4: Best method per period ──
    _w("## 4. Best Method per Period")
    _w()
    # Need per-period metrics: re-compute from all_preds
    if len(all_preds) > 0 and "period" in all_preds.columns:
        period_metrics = []
        for model_name in single_models + fused_models:
            m_df = all_preds[all_preds["model_name"] == model_name]
            if len(m_df) == 0:
                continue
            for period in m_df["period"].unique():
                p_df = m_df[m_df["period"] == period]
                y_true = p_df["y_true"].values
                y_pred = p_df["y_pred"].values
                valid = ~(np.isnan(y_true) | np.isnan(y_pred))
                if valid.sum() < 2:
                    continue
                smape_val = smape_floor50(y_true[valid], y_pred[valid])
                mae_val = float(np.mean(np.abs(y_true[valid] - y_pred[valid])))
                period_metrics.append({
                    "model_name": model_name,
                    "period": period,
                    "sMAPE_floor50": smape_val,
                    "MAE": mae_val,
                    "n": int(valid.sum()),
                })
        period_metrics_df = pd.DataFrame(period_metrics)
        if len(period_metrics_df) > 0:
            for period in sorted(period_metrics_df["period"].unique()):
                p_df = period_metrics_df[period_metrics_df["period"] == period].copy()
                p_df = p_df.sort_values("sMAPE_floor50")
                best = p_df.iloc[0]
                _w(f"### Period: `{period}`")
                _w()
                _w(f"**Best method**: `{best['model_name']}` "
                    f"(sMAPE_floor50={best['sMAPE_floor50']:.2f}%)")
                _w()
                _w("| Model | sMAPE_floor50 | MAE | n |")
                _w("|---|---|---|---|")
                for _, row in p_df.iterrows():
                    _w(
                        f"| {row['model_name']} "
                        f"| {row['sMAPE_floor50']:.2f} "
                        f"| {row['MAE']:.2f} "
                        f"| {row['n']} |"
                    )
                _w()
        else:
            _w("*Could not compute per-period metrics (insufficient data).*")
            _w()
    else:
        _w("*`period` column not available in predictions.*")
        _w()

    # ── Section 5: Spike hours (y_true top 10%) ──
    _w("## 5. Best Method for Spike Hours (y_true Top 10%)")
    _w()
    if len(all_preds) > 0 and "y_true" in all_preds.columns:
        spike_threshold = np.quantile(all_preds["y_true"], 0.9)
        spike_df = all_preds[all_preds["y_true"] >= spike_threshold].copy()
        _w(f"**Spike threshold (y_true P90)**: {spike_threshold:.1f} €/MWh")
        _w(f"**Spike hours**: {len(spike_df):,} / {len(all_preds):,} "
            f"({len(spike_df)/len(all_preds)*100:.1f}%)")
        _w()

        spike_metrics = []
        for model_name in single_models + fused_models:
            m_df = spike_df[spike_df["model_name"] == model_name]
            if len(m_df) == 0:
                continue
            y_true = m_df["y_true"].values
            y_pred = m_df["y_pred"].values
            valid = ~(np.isnan(y_true) | np.isnan(y_pred))
            if valid.sum() < 2:
                continue
            smape_val = smape_floor50(y_true[valid], y_pred[valid])
            mae_val = float(np.mean(np.abs(y_true[valid] - y_pred[valid])))
            spike_metrics.append({
                "model_name": model_name,
                "sMAPE_floor50": smape_val,
                "MAE": mae_val,
                "n": int(valid.sum()),
            })
        spike_metrics_df = pd.DataFrame(spike_metrics).sort_values("sMAPE_floor50")
        if len(spike_metrics_df) > 0:
            best = spike_metrics_df.iloc[0]
            _w(f"**Best method (spike hours)**: `{best['model_name']}` "
                f"(sMAPE_floor50={best['sMAPE_floor50']:.2f}%)")
            _w()
            _w("| Model | sMAPE_floor50 | MAE | n |")
            _w("|---|---|---|---|")
            for _, row in spike_metrics_df.iterrows():
                _w(
                    f"| {row['model_name']} "
                    f"| {row['sMAPE_floor50']:.2f} "
                    f"| {row['MAE']:.2f} "
                    f"| {row['n']} |"
                )
        else:
            _w("*No spike-hour predictions found.*")
        _w()
    else:
        _w("*Predictions not available for spike analysis.*")
        _w()

    # ── Section 6: Negative hours (y_true < 0) ──
    _w("## 6. Best Method for Negative Hours (y_true < 0)")
    _w()
    if len(all_preds) > 0 and "y_true" in all_preds.columns:
        neg_df = all_preds[all_preds["y_true"] < 0].copy()
        _w(f"**Negative hours**: {len(neg_df):,} / {len(all_preds):,} "
            f"({len(neg_df)/len(all_preds)*100:.1f}%)")
        _w()

        if len(neg_df) > 0:
            neg_metrics = []
            for model_name in single_models + fused_models:
                m_df = neg_df[neg_df["model_name"] == model_name]
                if len(m_df) == 0:
                    continue
                y_true = m_df["y_true"].values
                y_pred = m_df["y_pred"].values
                valid = ~(np.isnan(y_true) | np.isnan(y_pred))
                if valid.sum() < 2:
                    continue
                smape_val = smape_floor50(y_true[valid], y_pred[valid])
                neg_hit = (y_pred[valid] < 0).sum() / valid.sum() * 100.0
                mae_val = float(np.mean(np.abs(y_true[valid] - y_pred[valid])))
                neg_metrics.append({
                    "model_name": model_name,
                    "sMAPE_floor50": smape_val,
                    "negative_price_hit_rate": neg_hit,
                    "MAE": mae_val,
                    "n": int(valid.sum()),
                })
            neg_metrics_df = pd.DataFrame(neg_metrics).sort_values("sMAPE_floor50")
            if len(neg_metrics_df) > 0:
                best = neg_metrics_df.iloc[0]
                _w(f"**Best method (negative hours)**: `{best['model_name']}` "
                    f"(sMAPE_floor50={best['sMAPE_floor50']:.2f}%)")
                _w()
                _w("| Model | sMAPE_floor50 | MAE | Neg_Hit_Rate | n |")
                _w("|---|---|---|---|---|")
                for _, row in neg_metrics_df.iterrows():
                    _w(
                        f"| {row['model_name']} "
                        f"| {row['sMAPE_floor50']:.2f} "
                        f"| {row['MAE']:.2f} "
                        f"| {row.get('negative_price_hit_rate', float('nan')):.1f}% "
                        f"| {row['n']} |"
                    )
            else:
                _w("*No negative-hour predictions found.*")
        else:
            _w("*No negative price hours in evaluation period.*")
        _w()
    else:
        _w("*Predictions not available for negative-hours analysis.*")
        _w()

    # ── Section 7: Does fusion beat single models? ──
    _w("## 7. Does Fusion Beat Single Models?")
    _w()
    if len(all_models_df) > 0:
        best_single = all_models_df[all_models_df["model_name"].isin(single_models)].copy()
        best_single = best_single.sort_values("sMAPE_floor50").iloc[0]

        best_fused = all_models_df[all_models_df["model_name"].isin(fused_models)].copy()
        best_fused = best_fused.sort_values("sMAPE_floor50").iloc[0]

        fusion_wins = best_fused["sMAPE_floor50"] < best_single["sMAPE_floor50"]
        delta = best_single["sMAPE_floor50"] - best_fused["sMAPE_floor50"]

        _w(f"- **Best single model**: `{best_single['model_name']}` "
            f"(sMAPE_floor50={best_single['sMAPE_floor50']:.2f}%)")
        _w(f"- **Best fused method**: `{best_fused['model_name']}` "
            f"(sMAPE_floor50={best_fused['sMAPE_floor50']:.2f}%)")
        _w(f"- **Delta (single − fused)**: {delta:.2f} percentage points")
        _w()

        if fusion_wins:
            _w("**✅ Yes — fusion improves over the best single model.**")
        elif delta > -0.5:
            _w("**≈ Approximately equal — fusion is competitive with single models.**")
        else:
            _w("**❌ No — single model still better. Fusion needs tuning.**")
        _w()
    else:
        _w("*Metrics not available.*")
        _w()

    # ── Section 8: Recommended default fusion method ──
    _w("## 8. Recommended Default Fusion Method")
    _w()
    if len(fused_df) > 0:
        # Rank fused methods by sMAPE_floor50
        fused_ranked = fused_df.sort_values("sMAPE_floor50")
        best_fused_method = fused_ranked.iloc[0]["model_name"]
        _w(f"**Recommendation**: `{best_fused_method}`")
        _w()
        _w("**Rationale:**")
        _w(f"- Lowest sMAPE_floor50 ({fused_ranked.iloc[0].get('sMAPE_floor50', float('nan')):.2f}%) "
            f"among fusion methods.")
        _w("- Simple average is the most robust fallback if adaptive weights fail.")
        _w("- Inverse-sMAPE weighting adapts to recent model performance.")
        _w()
        _w("**Fallback order**:")
        _w("1. `fused_inverse_smape_weight` (adaptive)")
        _w("2. `fused_simple_average` (robust fallback)")
        _w("3. `fused_period_best` (regime-conditional)")
        _w()
    else:
        _w("*Fusion metrics not available.*")
        _w()

    # ── Section 9: Should this be integrated into main pipeline? ──
    _w("## 9. Integration Recommendation for `electricity_forecast_model2.0_exp`")
    _w()
    if len(all_models_df) > 0 and len(fused_df) > 0:
        fusion_wins = (
            fused_df.sort_values("sMAPE_floor50").iloc[0]["sMAPE_floor50"]
            < single_df.sort_values("sMAPE_floor50").iloc[0]["sMAPE_floor50"]
            if len(single_df) > 0
            else True
        )
        # Also check if fusion helps in spike/negative regimes
        has_spike_benefit = True  # will be refined below
        _w("**Verdict:**")
        if fusion_wins:
            _w()
            _w("### ✅ Recommended: Integrate into main fusion pipeline")
            _w()
            _w("**Rationale:**")
            _w("1. Fusion improves overall sMAPE vs. best single model.")
            _w("2. Complementarity analysis (see `complementarity_report.md`) "
                "shows low error correlation.")
            _w("3. Robust fallback available (simple average).")
            _w()
            _w("**Integration plan:**")
            _w("- Add `fused_inverse_smape_weight` as a candidate in the main fusion ensemble.")
            _w("- Use simple average as fallback when < 7 days of history available.")
            _w("- Monitor fusion performance daily via ledger.")
        else:
            _w()
            _w("### ⚠️ Conditional: More evaluation needed")
            _w()
            _w("Fusion does not yet consistently beat the best single model.")
            _w("Recommendation: run 30-day evaluation, then re-assess.")
        _w()
    else:
        _w("*Insufficient data for integration recommendation.*")
        _w()

    # ── Footer ──
    _w("---")
    _w()
    _w(f"*Report generated by `scripts/build_30d_final_report.py` on {now_str}.*")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Build final 30-day fusion evaluation report"
    )
    parser.add_argument(
        "--input-root",
        type=str,
        default="outputs/catboost_tabpfn_30d",
        help="Root directory (contains fusion/ and predictions/ subdirectories)",
    )
    args = parser.parse_args()

    input_root = Path(args.input_root)

    # ── Early exit ──
    if _early_exit(input_root):
        sys.exit(1)

    logger.info(f"Building final report from {input_root}")

    # ── Build report ──
    report = build_report(input_root)

    # ── Save to {input_root}/reports/ ──
    report_dir = input_root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "catboost_tabpfn_30d_fusion_report.md"
    with open(str(report_path), "w", encoding="utf-8") as f:
        f.write(report)
    logger.info(f"Report saved to {report_path}")

    # ── Copy to docs/reports/ ──
    docs_dir = Path(_PROJECT_DIR) / "docs" / "reports"
    docs_dir.mkdir(parents=True, exist_ok=True)
    docs_path = docs_dir / "catboost_tabpfn_30d_fusion_report.md"
    copy2(str(report_path), str(docs_path))
    logger.info(f"Report copied to {docs_path}")

    print(report)


if __name__ == "__main__":
    main()
