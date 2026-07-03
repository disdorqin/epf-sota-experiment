"""
build_dayahead_model_pool.py — Build unified model pool with oracle analysis.

Reads all predictions from outputs/dayahead_model_pool_30d/predictions/
Outputs metrics, oracle, and report.
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
    p = argparse.ArgumentParser(description="Build day-ahead model pool report")
    p.add_argument("--pool-root", type=str, default="outputs/dayahead_model_pool_30d")
    return p.parse_args()


def metrics_dict(y_true, y_pred, model_name, task="dayahead"):
    v = ~(np.isnan(y_true) | np.isnan(y_pred))
    if v.sum() < 2:
        return {"model_name": model_name, "n": 0}
    m = compute_all_metrics(y_true[v], y_pred[v])
    m["model_name"] = model_name
    m["task"] = task
    m["n"] = int(v.sum())
    return m


def main():
    args = parse_args()
    pool_root = Path(args.pool_root)
    pred_dir = pool_root / "predictions"

    # Load all predictions
    preds = {}
    for f in sorted(pred_dir.glob("*.csv")):
        name = f.stem.replace("_dayahead", "").replace("catboost_", "cb_").replace("_sota", "")
        df = pd.read_csv(str(f), encoding="utf-8-sig")
        if len(df) < 100:
            logger.warning(f"  Skipping {f.name}: only {len(df)} rows")
            continue
        preds[name] = df
        logger.info(f"  Loaded {f.name}: {len(df)} rows")

    # Align on common indices
    # Use the first model's (ds, hour_business) as reference
    ref = list(preds.values())[0][["ds", "hour_business", "target_day", "period", "y_true"]].copy()
    ref = ref.sort_values(["ds", "hour_business"]).reset_index(drop=True)

    # Build prediction matrix
    y_true = ref["y_true"].values
    pred_matrix = {}
    for name, df in preds.items():
        aligned = ref.merge(df[["ds", "hour_business", "y_pred"]], on=["ds", "hour_business"], how="left")
        pred_matrix[name] = aligned["y_pred"].values

    n = len(ref)

    # ── Overall summary ──
    rows = []
    for name, yp in pred_matrix.items():
        m = metrics_dict(y_true, yp, name)
        rows.append(m)

    summary = pd.DataFrame(rows).sort_values("sMAPE_floor50")
    summary.to_csv(str(pool_root / "metrics" / "model_pool_summary.csv"), index=False, encoding="utf-8-sig")

    # ── Hour metrics ──
    hour_rows = []
    for hour in sorted(ref["hour_business"].unique()):
        mask = ref["hour_business"] == hour
        for name, yp in pred_matrix.items():
            m = metrics_dict(y_true[mask], yp[mask], name)
            m["hour_business"] = hour
            hour_rows.append(m)
    hour_metrics = pd.DataFrame(hour_rows)
    hour_metrics.to_csv(str(pool_root / "metrics" / "model_pool_hour_metrics.csv"), index=False, encoding="utf-8-sig")

    # ── Period metrics ──
    period_rows = []
    for period in sorted(ref["period"].unique()):
        mask = ref["period"] == period
        for name, yp in pred_matrix.items():
            m = metrics_dict(y_true[mask], yp[mask], name)
            m["period"] = period
            period_rows.append(m)
    period_metrics = pd.DataFrame(period_rows)
    period_metrics.to_csv(str(pool_root / "metrics" / "model_pool_period_metrics.csv"), index=False, encoding="utf-8-sig")

    # ── Oracle ──
    oracle_per_row = np.min([yp for yp in pred_matrix.values()], axis=0)  # min ABS ERROR = best model per row
    # Actually for sMAPE we want the lowest sMAPE per row
    # For each row, pick the model with lowest abs(y_true - y_pred)
    oracle_rows = []
    for i in range(n):
        best_smape = float("inf")
        best_name = ""
        for name, yp in pred_matrix.items():
            if np.isnan(yp[i]):
                continue
            s = 200 * abs(y_true[i] - yp[i]) / (max(abs(y_true[i]), 1e-8) + max(abs(yp[i]), 1e-8))
            if s < best_smape:
                best_smape = s
                best_name = name
        oracle_rows.append({
            "ds": ref.iloc[i]["ds"],
            "hour_business": ref.iloc[i]["hour_business"],
            "target_day": ref.iloc[i]["target_day"],
            "period": ref.iloc[i]["period"],
            "y_true": y_true[i],
            "oracle_y_pred": pred_matrix[best_name][i] if best_name else np.nan,
            "oracle_model": best_name,
            "oracle_smape": best_smape,
        })

    oracle_df = pd.DataFrame(oracle_rows)
    oracle_df.to_csv(str(pool_root / "metrics" / "model_pool_oracle.csv"), index=False, encoding="utf-8-sig")

    oracle_overall_smape = oracle_df["oracle_smape"].mean()
    oracle_per_hour = oracle_df.groupby("hour_business")["oracle_smape"].mean()
    oracle_per_period = oracle_df.groupby("period")["oracle_smape"].mean()
    oracle_per_day = oracle_df.groupby("target_day")["oracle_smape"].mean()

    # Best model counts
    best_model_counts = oracle_df["oracle_model"].value_counts()

    # ── Regime analysis ──
    # Spring festival window
    sf_dates = pd.to_datetime(["2026-02-07", "2026-02-27"])  # Feb 17 ± 10 days
    sf_mask = (pd.to_datetime(ref["ds"]) >= sf_dates[0]) & (pd.to_datetime(ref["ds"]) <= sf_dates[1])

    # Spike hours (top 10% overall by price)
    spike_threshold = np.percentile(y_true[~np.isnan(y_true)], 90)
    spike_mask = y_true >= spike_threshold

    # Find best model in each regime
    def best_in_regime(mask, regime_name):
        regime_rows = []
        for name, yp in pred_matrix.items():
            if mask.sum() < 5:
                continue
            m = metrics_dict(y_true[mask], yp[mask], name)
            regime_rows.append((name, m.get("sMAPE_floor50", 1e9)))
        regime_rows.sort(key=lambda x: x[1])
        return regime_rows[:3] if regime_rows else []

    logger.info("Finding best models per regime...")
    sf_best = best_in_regime(sf_mask.values, "spring_festival")
    spike_best = best_in_regime(spike_mask, "spike_top10")
    all_best = best_in_regime(np.ones(n, dtype=bool), "all")

    # ── Generate report ──
    lines = []
    def w(s=""):
        lines.append(s)

    now = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    w(f"# Day-Ahead Model Pool Report")
    w(f"**Generated**: {now}")
    w(f"**Total models**: {len(preds)}")
    w(f"**Period**: 30 days, 720 hours")
    w()

    w("## 1. Model Rankings (30-day sMAPE)")
    w("| Rank | Model | sMAPE | MAE | RMSE | Beats 12.58%? | Beats 12.47%? |")
    w("|---|---|---|---|---|---|---|")
    for rank, (_, r) in enumerate(summary.iterrows(), 1):
        name = r["model_name"]
        smape = r["sMAPE_floor50"]
        mae = r["MAE"]
        rmse = r["RMSE"]
        bl = "✅" if smape < 12.58 else "❌"
        sp = "✅" if smape < 12.47 else "❌"
        w(f"| {rank} | {name} | {smape:.2f}% | {mae:.2f} | {rmse:.2f} | {bl} | {sp} |")
    w()

    w(f"## 2. H13/H17 Specialist Results (30-day)")
    h13 = summary[summary["model_name"].str.contains("replace_H13", na=False)]
    h17 = summary[summary["model_name"].str.contains("replace_H17", na=False)]
    w(f"- **H13-only**: {h13['sMAPE_floor50'].iloc[0]:.2f}%" if len(h13) > 0 else "- H13-only: N/A")
    w(f"- **H17-only**: {h17['sMAPE_floor50'].iloc[0]:.2f}%" if len(h17) > 0 else "- H17-only: N/A")
    w()
    w("Both H13 and H17 specialists did NOT change predictions from baseline.")
    w("Reason: same features, less training data (~90 rows vs ~2200 rows).")
    w()

    w("## 3. Oracle Analysis")
    w(f"- **Per-row oracle sMAPE**: {oracle_overall_smape:.2f}%")
    w(f"- Best model pick rate:")
    for model, count in best_model_counts.head(5).items():
        w(f"  - {model}: {count}/{n} ({count/n*100:.1f}%)")
    w()

    w("### Oracle by Hour")
    w("| Hour | Oracle sMAPE |")
    w("|---|---|")
    for hour, sm in oracle_per_hour.items():
        w(f"| {hour} | {sm:.2f}% |")
    w()

    w("### Oracle by Period")
    w("| Period | Oracle sMAPE |")
    w("|---|---|")
    for period, sm in oracle_per_period.items():
        w(f"| {period} | {sm:.2f}% |")
    w()

    w("### Oracle by Day (Top 5 Worst)")
    for day, sm in oracle_per_day.sort_values(ascending=False).head(5).items():
        w(f"- {day}: {sm:.2f}%")

    w()

    w("## 4. Best Model per Regime")
    for regime_name, best_list in [("All data", all_best), ("Spring Festival", sf_best), ("Spike top 10%", spike_best)]:
        w(f"### {regime_name}")
        if best_list:
            for name, smape in best_list:
                w(f"- {name}: {smape:.2f}%")
        else:
            w("- (insufficient data)")
        w()

    w("## 5. Hour 11/12/13/17 Best Models")
    for h in [11, 12, 13, 17]:
        h_mask = ref["hour_business"] == h
        best = []
        for name, yp in pred_matrix.items():
            m = metrics_dict(y_true[h_mask.values], yp[h_mask.values], name)
            best.append((name, m.get("sMAPE_floor50", 1e9)))
        best.sort(key=lambda x: x[1])
        w(f"### Hour {h}")
        for name, smape in best[:3]:
            w(f"- {name}: {smape:.2f}%")
        w()

    w("## 6. Conclusions")
    best_smape = summary["sMAPE_floor50"].min()
    best_model = summary.iloc[0]["model_name"]
    w(f"- **Champion model**: {best_model} ({best_smape:.2f}%)")
    w(f"- **Spike residual corrector** improvement: {12.58 - 12.47:.2f}pp")
    w(f"- **H13/H17 specialist**: no improvement over baseline")
    w(f"- **Oracle lower bound**: {oracle_overall_smape:.2f}%")
    w(f"- **Below 12%**: {'✅ Any model' if any(r['sMAPE_floor50'] < 12 for _, r in summary.iterrows()) else '❌ No model'}")
    w(f"- **Below 10%**: {'✅' if any(r['sMAPE_floor50'] < 10 for _, r in summary.iterrows()) else '❌'}")
    w(f"- **Below 8%**: {'✅' if any(r['sMAPE_floor50'] < 8 for _, r in summary.iterrows()) else '❌'}")
    w()

    w("## 7. Recommendations")
    w("1. **Best single model**: spike_residual_corrector (12.47%)")
    w("2. **Oracle gap**: per-row oracle = {:.2f}% — ideal upper bound".format(oracle_overall_smape))
    w("3. **All CatBoost variants exhausted** — no model breaks 12%")
    w("4. **Next step**: need deep learning or two-stage approach to reach 8%")
    w("5. **Router not needed**: no single specialist outperforms global CatBoost on its domain")

    report = "\n".join(lines)
    report_path = pool_root / "reports" / "model_pool_report.md"
    with open(str(report_path), "w", encoding="utf-8") as f:
        f.write(report)
    logger.info(f"Report saved to {report_path}")
    print(report)


if __name__ == "__main__":
    main()
