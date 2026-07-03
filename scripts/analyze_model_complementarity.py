"""
analyze_model_complementarity.py — CatBoost vs TabPFN error complementarity analysis.

Usage:
    python scripts/analyze_model_complementarity.py ^
        --input-root outputs/catboost_tabpfn_30d

Output:
    {input_root}/reports/complementarity_report.md

If 30d predictions are not found yet, prints:
    "30d predictions not found yet. Run walk-forward first."
and exits with code 1.

Report sections (7 total):
1. Absolute error correlation (CatBoost vs TabPFN).
2. Prediction error correlation.
3. Per-task and per-period: which model is stronger.
4. Spike hours (y_true top 10%): which model is stronger.
5. Negative hours (y_true < 0): which model is stronger.
6. Disagreement analysis: does |CB - TP| predict high error?
7. Fusion value verdict: is fusion worthwhile?
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ── Ensure src on path ──
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_DIR)

from src.common.metrics import compute_all_metrics, smape_floor50

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

MERGE_KEYS = ["ds", "task", "target_day", "hour_business", "period", "y_true"]


def _load_predictions(input_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load CatBoost and TabPFN predictions from predictions/ directory."""
    pred_dir = input_root / "predictions"
    if not pred_dir.exists():
        return pd.DataFrame(), pd.DataFrame()

    cb_files = sorted(pred_dir.glob("catboost_sota_*.csv"))
    tp_files = sorted(pred_dir.glob("tabpfn_ts_sota_*.csv"))

    if len(cb_files) == 0 or len(tp_files) == 0:
        return pd.DataFrame(), pd.DataFrame()

    cb_dfs = [pd.read_csv(str(f), encoding="utf-8-sig") for f in cb_files]
    tp_dfs = [pd.read_csv(str(f), encoding="utf-8-sig") for f in tp_files]

    cb_all = pd.concat(cb_dfs, ignore_index=True)
    tp_all = pd.concat(tp_dfs, ignore_index=True)
    return cb_all, tp_all


def _align(cb_df: pd.DataFrame, tp_df: pd.DataFrame) -> pd.DataFrame:
    """Align CatBoost and TabPFN on common keys."""
    available = [c for c in MERGE_KEYS if c in cb_df.columns and c in tp_df.columns]
    if "ds" not in available:
        # Fallback: merge on all common columns except model-specific ones
        common = [c for c in cb_df.columns if c in tp_df.columns
                  and c not in ("y_pred", "model_name", "source", "run_mode", "created_at")]
        available = common
    merged = cb_df.merge(tp_df, on=available, suffixes=("_cb", "_tp"), how="inner")
    logger.info(f"Aligned: {len(merged)} rows (keys: {available[:6]})")
    return merged


def _abs_error_corr(merged: pd.DataFrame) -> float:
    """Correlation of absolute errors between CB and TP."""
    ae_cb = np.abs(merged["y_true"] - merged["y_pred_cb"])
    ae_tp = np.abs(merged["y_true"] - merged["y_pred_tp"])
    valid = ~(np.isnan(ae_cb) | np.isnan(ae_tp))
    if valid.sum() < 3:
        return np.nan
    return float(np.corrcoef(ae_cb[valid], ae_tp[valid])[0, 1])


def _error_corr(merged: pd.DataFrame) -> float:
    """Correlation of raw errors (y_true - y_pred) between CB and TP."""
    e_cb = merged["y_true"] - merged["y_pred_cb"]
    e_tp = merged["y_true"] - merged["y_pred_tp"]
    valid = ~(np.isnan(e_cb) | np.isnan(e_tp))
    if valid.sum() < 3:
        return np.nan
    return float(np.corrcoef(e_cb[valid], e_tp[valid])[0, 1])


def _smape_series(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Per-row sMAPE (raw, not floored)."""
    denom = np.abs(y_true) + np.abs(y_pred)
    denom = np.where(denom < 1e-8, 1e-8, denom)
    return 200.0 * np.abs(y_true - y_pred) / denom


def _format_val(v: float) -> str:
    if pd.isna(v) or v is None:
        return "N/A"
    return f"{v:.4f}"


def build_report(cb_df: pd.DataFrame, tp_df: pd.DataFrame, input_root: Path) -> str:
    """Build the full complementarity report as markdown string."""
    lines: list[str] = []

    def _w(s: str = "") -> None:
        lines.append(s)

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    _w(f"# Complementarity Analysis: CatBoost vs TabPFN-TS")
    _w()
    _w(f"**Generated**: {now_str}")
    if len(cb_df) > 0 and "target_day" in cb_df.columns:
        try:
            start_day = pd.to_datetime(cb_df["target_day"]).min().strftime("%Y-%m-%d")
            end_day = pd.to_datetime(cb_df["target_day"]).max().strftime("%Y-%m-%d")
            _w(f"**Evaluation Period**: {start_day} → {end_day}")
        except Exception:
            _w(f"**Evaluation Period**: {len(cb_df)} rows")
    else:
        _w(f"**Evaluation Period**: {len(cb_df)} rows")
    _w()

    # ── Align ──
    merged = _align(cb_df, tp_df)
    if len(merged) == 0:
        _w("ERROR: Could not align CatBoost and TabPFN predictions. Check CSV schemas.")
        return "\n".join(lines)

    _w(f"**Aligned rows**: {len(merged)}")
    _w()

    # Add derived columns
    merged["abs_err_cb"] = np.abs(merged["y_true"] - merged["y_pred_cb"])
    merged["abs_err_tp"] = np.abs(merged["y_true"] - merged["y_pred_tp"])
    merged["cb_better"] = merged["abs_err_cb"] < merged["abs_err_tp"]
    merged["tp_better"] = merged["abs_err_tp"] < merged["abs_err_cb"]
    merged["cb_win_rate"] = merged["cb_better"].mean() * 100.0
    merged["tp_win_rate"] = merged["tp_better"].mean() * 100.0

    # ── Section 1: Absolute error correlation ──
    _w("## 1. Absolute Error Correlation")
    _w()
    ae_corr = _abs_error_corr(merged)
    _w(f"- **Absolute error correlation (Pearson)**: {_format_val(ae_corr)}")
    if pd.notna(ae_corr):
        if abs(ae_corr) < 0.3:
            _w("- Interpretation: **Low correlation** — errors are largely independent → strong complementarity.")
        elif abs(ae_corr) < 0.6:
            _w("- Interpretation: **Moderate correlation** — some overlap in error patterns.")
        else:
            _w("- Interpretation: **High correlation** — models make similar errors → limited complementarity.")
    _w()

    # ── Section 2: Prediction error correlation ──
    _w("## 2. Prediction Error Correlation (Raw Errors)")
    _w()
    e_corr = _error_corr(merged)
    _w(f"- **Raw error correlation (Pearson)**: {_format_val(e_corr)}")
    if pd.notna(e_corr):
        if abs(e_corr) < 0.3:
            _w("- Low correlation: model errors diverge → fusion highly beneficial.")
        elif abs(e_corr) < 0.6:
            _w("- Moderate correlation: partial error overlap.")
        else:
            _w("- High correlation: similar error patterns → fusion benefit limited.")
    _w()

    # Win-rate summary
    _w(f"- **CatBoost wins**: {merged['cb_win_rate'].iloc[0]:.1f}% of hours")
    _w(f"- **TabPFN wins**: {merged['tp_win_rate'].iloc[0]:.1f}% of hours")
    _w(f"- **Ties**: {100.0 - merged['cb_win_rate'].iloc[0] - merged['tp_win_rate'].iloc[0]:.1f}% of hours")
    _w()

    # ── Section 3: Per-task and per-period strength ──
    _w("## 3. Per-Task / Per-Period Strength")
    _w()

    # Per-task
    _w("### 3.1 Per Task")
    _w()
    for task in sorted(merged["task"].unique()):
        t_df = merged[merged["task"] == task]
        cb_smape = np.nanmean(_smape_series(t_df["y_true"].values, t_df["y_pred_cb"].values))
        tp_smape = np.nanmean(_smape_series(t_df["y_true"].values, t_df["y_pred_tp"].values))
        better = (
            "CatBoost" if cb_smape < tp_smape else
            "TabPFN" if tp_smape < cb_smape else
            "Tie"
        )
        _w(f"- **{task}**: CatBoost sMAPE={cb_smape:.2f}%, TabPFN sMAPE={tp_smape:.2f}% → **{better}**")
    _w()

    # Per-period
    if "period" in merged.columns:
        _w("### 3.2 Per Period")
        _w()
        for period in sorted(merged["period"].unique()):
            p_df = merged[merged["period"] == period]
            cb_smape = np.nanmean(_smape_series(p_df["y_true"].values, p_df["y_pred_cb"].values))
            tp_smape = np.nanmean(_smape_series(p_df["y_true"].values, p_df["y_pred_tp"].values))
            better = (
                "CatBoost" if cb_smape < tp_smape else
                "TabPFN" if tp_smape < cb_smape else
                "Tie"
            )
            _w(f"- **Period {period}**: CatBoost sMAPE={cb_smape:.2f}%, TabPFN sMAPE={tp_smape:.2f}% → **{better}**")
        _w()

    # ── Section 4: Spike hours (y_true top 10%) ──
    _w("## 4. Spike Hours (y_true Top 10%)")
    _w()
    spike_threshold = np.quantile(merged["y_true"], 0.9)
    spike_mask = merged["y_true"] >= spike_threshold
    spike_df = merged[spike_mask]
    nonspike_df = merged[~spike_mask]
    _w(f"- **Spike threshold (y_true P90)**: {spike_threshold:.1f} €/MWh")
    _w(f"- **Spike hours**: {spike_mask.sum()} / {len(merged)} ({spike_mask.mean()*100:.1f}%)")
    _w()

    for label, df_ in [("Spike (top 10%)", spike_df), ("Non-spike", nonspike_df)]:
        if len(df_) == 0:
            _w(f"**{label}**: no data")
            continue
        cb_smape = np.nanmean(_smape_series(df_["y_true"].values, df_["y_pred_cb"].values))
        tp_smape = np.nanmean(_smape_series(df_["y_true"].values, df_["y_pred_tp"].values))
        cb_mae = np.nanmean(np.abs(df_["y_true"].values - df_["y_pred_cb"].values))
        tp_mae = np.nanmean(np.abs(df_["y_true"].values - df_["y_pred_tp"].values))
        better = (
            "CatBoost" if cb_smape < tp_smape else
            "TabPFN" if tp_smape < cb_smape else
            "Tie"
        )
        _w(f"**{label}** ({len(df_)} hours):")
        _w(f"  - CatBoost: sMAPE={cb_smape:.2f}%, MAE={cb_mae:.2f}")
        _w(f"  - TabPFN:  sMAPE={tp_smape:.2f}%, MAE={tp_mae:.2f}")
        _w(f"  - **Better: {better}**")
        _w()

    # ── Section 5: Negative hours (y_true < 0) ──
    _w("## 5. Negative Price Hours (y_true < 0)")
    _w()
    neg_mask = merged["y_true"] < 0
    neg_count = neg_mask.sum()
    _w(f"- **Negative hours**: {neg_count} / {len(merged)} ({neg_mask.mean()*100:.1f}%)")
    _w()

    if neg_count > 0:
        neg_df = merged[neg_mask]
        pos_df = merged[~neg_mask]
        for label, df_ in [("Negative (y_true < 0)", neg_df), ("Non-negative (y_true ≥ 0)", pos_df)]:
            if len(df_) == 0:
                continue
            cb_smape = np.nanmean(_smape_series(df_["y_true"].values, df_["y_pred_cb"].values))
            tp_smape = np.nanmean(_smape_series(df_["y_true"].values, df_["y_pred_tp"].values))
            cb_hit = (df_["y_pred_cb"] < 0).sum() / len(df_) * 100.0
            tp_hit = (df_["y_pred_tp"] < 0).sum() / len(df_) * 100.0
            better = (
                "CatBoost" if cb_smape < tp_smape else
                "TabPFN" if tp_smape < cb_smape else
                "Tie"
            )
            _w(f"**{label}** ({len(df_)} hours):")
            _w(f"  - CatBoost: sMAPE={cb_smape:.2f}%, negative hit rate={cb_hit:.1f}%")
            _w(f"  - TabPFN:  sMAPE={tp_smape:.2f}%, negative hit rate={tp_hit:.1f}%")
            _w(f"  - **Better: {better}**")
            _w()
    else:
        _w("- No negative price hours in the evaluation period.")
        _w()

    # ── Section 6: Disagreement as risk signal ──
    _w("## 6. Disagreement as Risk Signal")
    _w()
    merged["disagreement"] = np.abs(merged["y_pred_cb"] - merged["y_pred_tp"])
    # Actual error of simple average
    merged["avg_pred"] = 0.5 * merged["y_pred_cb"] + 0.5 * merged["y_pred_tp"]
    merged["actual_error_avg"] = np.abs(merged["y_true"] - merged["avg_pred"])

    dis_corr = merged["disagreement"].corr(merged["actual_error_avg"])
    _w(f"- **Disagreement–Error correlation**: {_format_val(dis_corr)}")
    if pd.notna(dis_corr):
        if dis_corr > 0.3:
            _w("- **Strong signal**: high disagreement → high fusion error. Disagreement can flag risky hours.")
        elif dis_corr > 0.1:
            _w("- **Moderate signal**: some correlation between disagreement and error.")
        else:
            _w("- **Weak signal**: disagreement does not reliably indicate prediction risk.")
    _w()

    # Top-10 disagreement hours
    _w("**Top-10 highest disagreement hours:**")
    _w()
    _w("| ds | task | y_true | CatBoost | TabPFN | Disagreement | Avg_Error |")
    _w("|---|---|---|---|---|---|---|")
    top_dis = merged.nlargest(min(10, len(merged)), "disagreement")
    for _, row in top_dis.iterrows():
        ds_str = str(row.get("ds", "N/A"))
        task_str = str(row.get("task", "N/A"))
        _w(
            f"| {ds_str} | {task_str} "
            f"| {row['y_true']:.1f} "
            f"| {row['y_pred_cb']:.1f} "
            f"| {row['y_pred_tp']:.1f} "
            f"| {row['disagreement']:.1f} "
            f"| {row['actual_error_avg']:.1f} |"
        )
    _w()

    # ── Section 7: Fusion value verdict ──
    _w("## 7. Fusion Value Verdict")
    _w()

    ae_corr_val = ae_corr if pd.notna(ae_corr) else 1.0
    cb_wr = merged["cb_win_rate"].iloc[0]
    tp_wr = merged["tp_win_rate"].iloc[0]
    dis_corr_val = dis_corr if pd.notna(dis_corr) else 0.0

    has_complementarity = abs(ae_corr_val) < 0.5 and cb_wr > 20 and tp_wr > 20
    has_disagreement_signal = dis_corr_val > 0.15

    if has_complementarity:
        _w("### ✅ High fusion value")
        _w()
        _w("**Rationale:**")
        _w(f"1. Absolute error correlation = {ae_corr_val:.3f} — errors are low-correlated.")
        _w(f"2. CatBoost win rate = {cb_wr:.1f}%, TabPFN win rate = {tp_wr:.1f}% — neither model dominates.")
        if has_disagreement_signal:
            _w(f"3. Disagreement–error correlation = {dis_corr_val:.3f} — disagreement flags risky hours.")
        _w()
        _w("**Recommendation:**")
        _w("- **Primary**: `inverse_smape_weight` (adaptive per task/period)")
        _w("- **Fallback**: `simple_average` (if weight computation fails)")
        _w("- **Consider**: using disagreement > threshold to flag low-confidence hours.")
    elif abs(ae_corr_val) < 0.6:
        _w("### ⚠️ Moderate fusion value")
        _w()
        _w(f"- Absolute error correlation = {ae_corr_val:.3f}")
        _w(f"- CatBoost win rate = {cb_wr:.1f}%, TabPFN win rate = {tp_wr:.1f}%")
        _w("- Fusion may help in specific regimes (spikes, negative prices).")
        _w("- Consider regime-conditional fusion.")
    else:
        _w("### ❌ Low fusion value")
        _w()
        _w(f"- Absolute error correlation = {ae_corr_val:.3f} — errors are highly correlated.")
        _w("- Fusion unlikely to improve over the better single model.")
        _w("- Recommendation: use the better single model per task (see Section 3).")
    _w()

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="CatBoost vs TabPFN complementarity analysis"
    )
    parser.add_argument(
        "--input-root",
        type=str,
        default="outputs/catboost_tabpfn_30d",
        help="Root directory (contains predictions/ subdirectory)",
    )
    args = parser.parse_args()

    input_root = Path(args.input_root)
    pred_dir = input_root / "predictions"

    # ── Early exit ──
    if not pred_dir.exists():
        print("30d predictions not found yet. Run walk-forward first.")
        logger.error(f"Predictions directory not found: {pred_dir}")
        sys.exit(1)

    cb_df, tp_df = _load_predictions(input_root)
    if len(cb_df) == 0 or len(tp_df) == 0:
        print("30d predictions not found yet. Run walk-forward first.")
        logger.error("Missing CatBoost or TabPFN prediction files.")
        sys.exit(1)

    logger.info(f"CatBoost: {len(cb_df)} rows, TabPFN: {len(tp_df)} rows")

    # ── Build report ──
    report = build_report(cb_df, tp_df, input_root)

    # ── Save ──
    report_dir = input_root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "complementarity_report.md"
    with open(str(report_path), "w", encoding="utf-8") as f:
        f.write(report)

    logger.info(f"Complementarity report saved to {report_path}")
    print(report)


if __name__ == "__main__":
    main()
