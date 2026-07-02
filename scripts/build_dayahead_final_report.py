"""
build_dayahead_final_report.py — Day-ahead final report generator.

Reads diagnosis and fusion results, outputs a comprehensive final report.

Usage:
    python scripts/build_dayahead_final_report.py ^
        --diagnosis-root outputs/dayahead_diagnosis ^
        --fusion-root outputs/dayahead_fusion_30d ^
        --output docs/reports/dayahead_final_report.md
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path so `import src` works
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _find_root(cli_path: str, fallback_subdirs: list[str]) -> Path | None:
    p = Path(cli_path)
    if p.exists():
        return p
    for sub in fallback_subdirs:
        cand = Path(sub)
        if cand.exists():
            logger.info(f"Using fallback: {cand}")
            return cand
    return None


def _load_metrics(fusion_root: Path) -> pd.DataFrame | None:
    metrics_path = fusion_root / "fusion" / "fusion_metrics.csv"
    if not metrics_path.exists():
        logger.warning(f"fusion_metrics.csv not found at {metrics_path}")
        return None
    try:
        df = pd.read_csv(str(metrics_path), encoding="utf-8-sig")
        return df
    except Exception as e:
        logger.warning(f"Failed to load fusion_metrics.csv: {e}")
        return None


def _load_diagnosis(diagnosis_root: Path) -> dict:
    """
    Load diagnosis report and extract key facts.
    Returns dict with diagnosis info.
    """
    report_path = diagnosis_root / "reports" / "dayahead_error_diagnosis.md"
    if not report_path.exists():
        logger.warning(f"Diagnosis report not found: {report_path}")
        return {}

    with open(report_path, "r", encoding="utf-8") as f:
        content = f.read()

    info = {}
    # Extract overall sMAPE
    for line in content.split("\n"):
        if "Overall day-ahead sMAPE" in line:
            try:
                val = line.split("`")[1].replace("%", "")
                info["overall_smape"] = float(val)
            except Exception:
                pass
        if "Gap to 8% target" in line:
            try:
                val = line.split("`")[1].replace("pp", "").strip()
                info["gap_to_8"] = float(val)
            except Exception:
                pass
        if "Worst period" in line and "```" not in line:
            info["worst_period_line"] = line.strip()
        if "Worst hour" in line and "```" not in line:
            info["worst_hour_line"] = line.strip()

    return info


# ── Build report ───────────────────────────────────────────────────────────


def _build_report(metrics_df: pd.DataFrame | None, diagnosis_info: dict, output_path: Path) -> None:
    """
    Build the final markdown report.
    """
    lines = []
    lines.append("# Day-Ahead Final Report")
    lines.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ── Section 1: Model rankings ─────────────────────────────────────────
    if metrics_df is not None:
        # Filter to dayahead task
        if "task" in metrics_df.columns:
            mdf = metrics_df[metrics_df["task"] == "dayahead"].copy()
        else:
            mdf = metrics_df.copy()

        # Separate base models vs specialists vs fusion
        base_models = [n for n in mdf["model_name"].values if "fused" not in n and "specialist" not in n and "tuned" not in n]
        specialist_models = [n for n in mdf["model_name"].values if "specialist" in n or "tuned" in n]
        fusion_models = [n for n in mdf["model_name"].values if "fused" in n]

        lines.append("## 1. Single Model Rankings (by sMAPE_floor50)")
        lines.append("")
        if len(base_models) > 0:
            base_df = mdf[mdf["model_name"].isin(base_models)].sort_values("sMAPE_floor50")
            lines.append("**Base models:**")
            lines.append("")
            lines.append("| model_name | sMAPE_floor50 | MAE | RMSE | n |")
            lines.append("|------------|---------------|-----|------|---|")
            for _, row in base_df.iterrows():
                lines.append(f"| {row['model_name']} | {row['sMAPE_floor50']:.2f}% | {row['MAE']:.2f} | {row['RMSE']:.2f} | {int(row['n'])} |")
            lines.append("")
        else:
            lines.append("*No base model results found.*")
            lines.append("")

        if len(specialist_models) > 0:
            spec_df = mdf[mdf["model_name"].isin(specialist_models)].sort_values("sMAPE_floor50")
            lines.append("**Specialist models:**")
            lines.append("")
            lines.append("| model_name | sMAPE_floor50 | MAE | RMSE | n |")
            lines.append("|------------|---------------|-----|------|---|")
            for _, row in spec_df.iterrows():
                lines.append(f"| {row['model_name']} | {row['sMAPE_floor50']:.2f}% | {row['MAE']:.2f} | {row['RMSE']:.2f} | {int(row['n'])} |")
            lines.append("")
        else:
            lines.append("*No specialist model results found.*")
            lines.append("")

        # ── Section 2: Fusion method rankings ──────────────────────────────
        lines.append("## 2. Fusion Method Rankings (by sMAPE_floor50)")
        lines.append("")
        if len(fusion_models) > 0:
            fusion_df = mdf[mdf["model_name"].isin(fusion_models)].sort_values("sMAPE_floor50")
            lines.append("| model_name | sMAPE_floor50 | MAE | RMSE | n |")
            lines.append("|------------|---------------|-----|------|---|")
            for _, row in fusion_df.iterrows():
                lines.append(f"| {row['model_name']} | {row['sMAPE_floor50']:.2f}% | {row['MAE']:.2f} | {row['RMSE']:.2f} | {int(row['n'])} |")
            lines.append("")
        else:
            lines.append("*No fusion results found.*")
            lines.append("")

        # ── Section 3: Target check ───────────────────────────────────────
        best_smape = mdf["sMAPE_floor50"].min() if len(mdf) > 0 else np.nan
        lines.append("## 3. Target Check (sMAPE_floor50)")
        lines.append("")
        lines.append(f"- **Best model sMAPE:** `{best_smape:.2f}%`" if not np.isnan(best_smape) else "- Best model sMAPE: N/A")
        lines.append(f"- **Below 12%:** {'✅ Yes' if not np.isnan(best_smape) and best_smape < 12.0 else '❌ No'}")
        lines.append(f"- **Below 10%:** {'✅ Yes' if not np.isnan(best_smape) and best_smape < 10.0 else '❌ No'}")
        lines.append(f"- **Below 8%:** {'✅ Yes' if not np.isnan(best_smape) and best_smape < 8.0 else '❌ No'}")
        lines.append("")

        # ── Section 4: Worst days/hours ───────────────────────────────────
        lines.append("## 4. Worst-Case Analysis")
        lines.append("")
        if "overall_smape" in diagnosis_info:
            lines.append(f"- **Overall sMAPE (from diagnosis):** `{diagnosis_info['overall_smape']:.2f}%`")
            lines.append(f"- **Gap to 8%:** `{diagnosis_info.get('gap_to_8', 'N/A')}` pp")
        if "worst_period_line" in diagnosis_info:
            lines.append(f"- {diagnosis_info['worst_period_line']}")
        if "worst_hour_line" in diagnosis_info:
            lines.append(f"- {diagnosis_info['worst_hour_line']}")
        lines.append("")

        # Per-hour from metrics if available
        if "hour_business" in mdf.columns or True:  # Always show what we have
            pass  # Will be filled from diagnosis

    else:
        lines.append("*No fusion metrics found. Run fusion first.*")
        lines.append("")

    # ── Section 5: Spike / negative / SF window ─────────────────────────
    lines.append("## 5. Spike & Negative Price Performance")
    lines.append("")
    lines.append("*See diagnosis report for detailed spike/negative/SF analysis.*")
    if "overall_smape" in diagnosis_info:
        lines.append(f"Refer to the diagnosis report for spike hours, negative hours, and Spring Festival window performance.")
    lines.append("")

    # ── Section 6: Recommendations ─────────────────────────────────────────
    lines.append("## 6. Recommendations")
    lines.append("")

    if metrics_df is not None:
        if "task" in metrics_df.columns:
            mdf = metrics_df[metrics_df["task"] == "dayahead"].copy()
        else:
            mdf = metrics_df.copy()

        best_model = mdf.loc[mdf["sMAPE_floor50"].idxmin()] if len(mdf) > 0 else None
        best_fusion = mdf[mdf["model_name"].str.contains("fused", na=False)].sort_values("sMAPE_floor50").iloc[0] if len(mdf[mdf["model_name"].str.contains("fused", na=False)]) > 0 else None

        if best_model is not None:
            lines.append(f"**Recommended day-ahead main model:** `{best_model['model_name']}` (sMAPE = `{best_model['sMAPE_floor50']:.2f}%`)")
        if best_fusion is not None:
            lines.append(f"**Recommended fusion method:** `{best_fusion['model_name']}` (sMAPE = `{best_fusion['sMAPE_floor50']:.2f}%`)")
            if best_model is not None:
                fusion_better = best_fusion["sMAPE_floor50"] < best_model["sMAPE_floor50"]
                lines.append(f"**Fusion improves over best single model:** {'✅ Yes' if fusion_better else '❌ No'}")
        lines.append("")

        # Integration recommendation
        best_smape = mdf["sMAPE_floor50"].min() if len(mdf) > 0 else np.nan
        if not np.isnan(best_smape) and best_smape < 10.0:
            lines.append(f"✅ **Recommend integrating into `electricity_forecast_model2.0_exp`** — best sMAPE = `{best_smape:.2f}%` < 10%.")
        elif not np.isnan(best_smape) and best_smape < 12.0:
            lines.append(f"⚠️ **Consider integrating** — best sMAPE = `{best_smape:.2f}%` < 12%, but target is 8%. More tuning needed.")
        else:
            lines.append(f"❌ **Not ready for integration** — best sMAPE = `{best_smape:.2f}%`, still above 12% target.")
    else:
        lines.append("*Run fusion first to get recommendations.*")
    lines.append("")

    # ── Section 7: Next steps ─────────────────────────────────────────────
    lines.append("## 7. Next Steps")
    lines.append("")
    if metrics_df is not None:
        best_smape = metrics_df["sMAPE_floor50"].min() if len(metrics_df) > 0 else np.nan
        if not np.isnan(best_smape):
            if best_smape > 8.0:
                lines.append(f"1. Tune `{best_model['model_name'] if best_model is not None else 'best model'}` further (target transform, more features, hour specialists).")
                lines.append(f"2. Try spike correction for top-decile hours.")
                lines.append(f"3. If Spring Festival window is a major drag, add `is_spring_festival_window` as a feature or do expost correction.")
                lines.append(f"4. Re-run fusion after specialist models are added.")
            else:
                lines.append(f"1. ✅ Target reached! Integrate best model into `electricity_forecast_model2.0_exp`.")
                lines.append(f"2. Run backtest on holdout period to confirm generalization.")
    lines.append("")

    # ── Write report ───────────────────────────────────────────────────────
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info(f"Report written to {output_path}")


# ── CLI ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Day-ahead final report generator")
    parser.add_argument("--diagnosis-root", type=str, default="outputs/dayahead_diagnosis", help="Diagnosis output root")
    parser.add_argument("--fusion-root", type=str, default="outputs/dayahead_fusion_30d", help="Fusion output root")
    parser.add_argument("--output", type=str, default="docs/reports/dayahead_final_report.md", help="Output report path")
    args = parser.parse_args()

    diagnosis_root = _find_root(args.diagnosis_root, ["outputs/dayahead_diagnosis"])
    fusion_root = _find_root(args.fusion_root, ["outputs/dayahead_fusion_30d"])

    if diagnosis_root is None:
        logger.warning("Diagnosis root not found, report will be partial")
        diagnosis_info = {}
    else:
        diagnosis_info = _load_diagnosis(diagnosis_root)

    if fusion_root is None:
        logger.warning("Fusion root not found, report will be partial")
        metrics_df = None
    else:
        metrics_df = _load_metrics(fusion_root)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Also save to fusion_root/reports/ for self-contained output
    if fusion_root is not None:
        alt_path = fusion_root / "reports" / "dayahead_final_report.md"
        alt_path.parent.mkdir(parents=True, exist_ok=True)
        # Build report twice (once to each path)
        _build_report(metrics_df, diagnosis_info, alt_path)
        logger.info(f"Also saved to {alt_path}")

    _build_report(metrics_df, diagnosis_info, output_path)

    print(f"\nReport generated: {output_path}")
    if fusion_root is not None:
        print(f"Also saved to: {fusion_root / 'reports' / 'dayahead_final_report.md'}")


if __name__ == "__main__":
    main()
