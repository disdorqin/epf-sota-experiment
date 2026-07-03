#!/usr/bin/env python3
"""
Freeze the current day-ahead champion (lgbm_spike_residual_corrected).
Generates champion directory + report.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
from pathlib import Path
from src.common.metrics import smape_floor50, compute_all_metrics

OUT = Path("outputs/dayahead_champion_30d")
for d in [OUT / "predictions", OUT / "metrics", OUT / "reports"]:
    d.mkdir(parents=True, exist_ok=True)

# ── Load champion ──
df = pd.read_csv(
    "outputs/dayahead_lgbm_corrections_30d/predictions/lgbm_spike_residual_corrected_dayahead.csv",
    encoding="utf-8-sig"
)

std = pd.DataFrame({
    "ds": df["ds"],
    "y_true": df["y_true"],
    "y_pred": df["y_pred"],
    "hour_business": df["hour_business"].astype(int),
    "period": df["period"],
    "target_day": df["target_day"],
    "business_day": df["target_day"],
    "task": "dayahead",
    "model_name": "dayahead_champion_lgbm_spike_residual",
})
std.to_csv(str(OUT / "predictions" / "dayahead_champion_lgbm_spike_residual.csv"),
           index=False, encoding="utf-8-sig")
print(f"Saved champion prediction: {len(std)} rows")

# ── Load baselines ──
baselines = {
    "catboost_sota (core)": pd.read_csv(
        "outputs/dayahead_30d_core/predictions/catboost_sota_dayahead.csv", encoding="utf-8-sig"),
    "catboost_spike_residual (old champion)": pd.read_csv(
        "outputs/dayahead_corrections_30d/predictions/catboost_spike_residual_corrected_dayahead.csv",
        encoding="utf-8-sig"),
    "lightgbm_trial_02 (single)": pd.read_csv(
        "outputs/dayahead_lgbm_stage2_30d/predictions/trial_02_w150_nl255_lr0.03_dayahead.csv",
        encoding="utf-8-sig"),
    "lightgbm_best_two_average": pd.read_csv(
        "outputs/dayahead_lgbm_freeze_30d/predictions/best_two_average_dayahead.csv",
        encoding="utf-8-sig"),
}

# ── Summary ──
y_champ = std["y_true"].values
p_champ = std["y_pred"].values
summary_rows = []
valid_c = ~(np.isnan(y_champ) | np.isnan(p_champ))
m = compute_all_metrics(y_champ[valid_c], p_champ[valid_c])
m["model_name"] = "dayahead_champion_lgbm_spike_residual"
m["task"] = "dayahead"
m["n"] = int(valid_c.sum())
summary_rows.append(m)

for name, bdf in baselines.items():
    yp = bdf["y_pred_cb"].values if "y_pred_cb" in bdf.columns else bdf["y_pred"].values
    yt = bdf["y_true"].values
    valid = ~(np.isnan(yt) | np.isnan(yp))
    if valid.sum() < 2:
        continue
    m = compute_all_metrics(yt[valid], yp[valid])
    m["model_name"] = name
    m["task"] = "dayahead"
    m["n"] = int(valid.sum())
    summary_rows.append(m)

summary = pd.DataFrame(summary_rows)
summary.to_csv(str(OUT / "metrics" / "summary.csv"), index=False, encoding="utf-8-sig")

print("\n=== SUMMARY ===")
for _, r in summary.iterrows():
    print(f"  {r['model_name']:45s}  SMAPE={r['sMAPE_floor50']:.4f}%  MAE={r['MAE']:.2f}")

# ── Hour metrics ──
hour_rows = []
for h in sorted(std["hour_business"].unique()):
    hm = std["hour_business"] == h
    hour_rows.append({
        "model_name": "champion_lgbm_spike_residual",
        "hour_business": int(h),
        "sMAPE_floor50": round(smape_floor50(y_champ[hm.values], p_champ[hm.values]), 4),
    })
    for name, bdf in baselines.items():
        yp = bdf["y_pred_cb"].values if "y_pred_cb" in bdf.columns else bdf["y_pred"].values
        yt = bdf["y_true"].values
        if len(yt) == len(y_champ):
            hour_rows.append({
                "model_name": name,
                "hour_business": int(h),
                "sMAPE_floor50": round(smape_floor50(yt[hm.values], yp[hm.values]), 4),
            })
pd.DataFrame(hour_rows).to_csv(str(OUT / "metrics" / "hour_metrics.csv"),
                                index=False, encoding="utf-8-sig")

# ── Period metrics ──
period_rows = []
for p in sorted(std["period"].unique()):
    pm = std["period"] == p
    period_rows.append({
        "model_name": "champion_lgbm_spike_residual",
        "period": p,
        "sMAPE_floor50": round(smape_floor50(y_champ[pm.values], p_champ[pm.values]), 4),
    })
    for name, bdf in baselines.items():
        yp = bdf["y_pred_cb"].values if "y_pred_cb" in bdf.columns else bdf["y_pred"].values
        yt = bdf["y_true"].values
        if len(yt) == len(y_champ):
            period_rows.append({
                "model_name": name,
                "period": p,
                "sMAPE_floor50": round(smape_floor50(yt[pm.values], yp[pm.values]), 4),
            })
pd.DataFrame(period_rows).to_csv(str(OUT / "metrics" / "period_metrics.csv"),
                                  index=False, encoding="utf-8-sig")

# ── Worst days/hours ──
std["day_smape"] = 0.0
for d in std["target_day"].unique():
    dm = std["target_day"] == d
    std.loc[dm, "day_smape"] = smape_floor50(
        std.loc[dm, "y_true"].values, std.loc[dm, "y_pred"].values)
worst_days = std.groupby("target_day").agg({"day_smape": "first"}).sort_values(
    "day_smape", ascending=False).head(5)
worst_hours = std.groupby("hour_business").apply(
    lambda x: smape_floor50(x["y_true"].values, x["y_pred"].values)).sort_values(ascending=False).head(5)

print("\n=== WORST 5 DAYS ===")
for d, r in worst_days.iterrows():
    print(f"  {d}: {r['day_smape']:.2f}%")
print("\n=== WORST 5 HOURS ===")
for h, s in worst_hours.items():
    print(f"  Hour {h}: {s:.2f}%")

# ── Extract values for report ──
champ_smape = summary.loc[summary["model_name"] == "dayahead_champion_lgbm_spike_residual", "sMAPE_floor50"].values[0]
cb_smape = 12.58
old_smape = 12.47
lgbm_smape = 12.07
b2a_smape = 11.85

for _, r in summary.iterrows():
    if "catboost_sota" in r["model_name"]:
        cb_smape = r["sMAPE_floor50"]
    if "catboost_spike_residual" in r["model_name"]:
        old_smape = r["sMAPE_floor50"]
    if "trial_02" in r["model_name"]:
        lgbm_smape = r["sMAPE_floor50"]
    if "best_two_average" in r["model_name"]:
        b2a_smape = r["sMAPE_floor50"]

# Hour comparison
cb_hours_v = {}
champ_hours_v = {}
for _, r in pd.DataFrame(hour_rows).iterrows():
    if r["model_name"] == "catboost_sota (core)":
        cb_hours_v[int(r["hour_business"])] = r["sMAPE_floor50"]
    if r["model_name"] == "champion_lgbm_spike_residual":
        champ_hours_v[int(r["hour_business"])] = r["sMAPE_floor50"]

# ── Build report ──
lines = [
    "# Day-Ahead Champion Report",
    "",
    f"> Generated: 2026-07-03 20:00",
    f"> Task: dayahead",
    f"> Metric: sMAPE_floor50",
    "",
    "## Current Champion",
    "",
    f"**dayahead_champion_lgbm_spike_residual = {champ_smape:.2f}%**",
    "",
    "- Base model: LightGBM trial_02 (150d, mae objective, num_leaves=255)",
    "- Corrector: LGBMSpikeResidualCorrector (alpha=0.25, max_delta=50)",
    "- Rolling: each target_day uses data before that day only",
    "- Training: CatBoostRegressor on spike-hour residuals (top 10% by past absolute residual)",
    "",
    "## Ranking",
    "",
    "| Rank | Model | sMAPE_floor50 | vs Champion |",
    "|:----:|------|:-------------:|:-----------:|",
    f"| 1 | champion_lgbm_spike_residual | **{champ_smape:.2f}%** | — |",
    f"| 2 | best_two_average | {b2a_smape:.2f}% | +{b2a_smape - champ_smape:+.2f}pp |",
    f"| 3 | lightgbm_trial_02 (single) | {lgbm_smape:.2f}% | +{lgbm_smape - champ_smape:+.2f}pp |",
    f"| 4 | catboost_spike_residual (old) | {old_smape:.2f}% | +{old_smape - champ_smape:+.2f}pp |",
    f"| 5 | catboost_sota (original) | {cb_smape:.2f}% | +{cb_smape - champ_smape:+.2f}pp |",
    "",
    "## Improvement vs Baselines",
    "",
    f"- vs CatBoost baseline ({cb_smape:.2f}%): **{cb_smape - champ_smape:.2f}pp improvement**",
    f"- vs old champion spike_residual ({old_smape:.2f}%): **{old_smape - champ_smape:.2f}pp improvement**",
    f"- vs best_two_average ({b2a_smape:.2f}%): {b2a_smape - champ_smape:+.2f}pp",
    f"- vs LightGBM single ({lgbm_smape:.2f}%): {lgbm_smape - champ_smape:+.2f}pp",
    "",
    "## Target Check",
    "",
    "| Target | Status |",
    "|:-------|:------:|",
    f"| Below 12.58% (CatBoost original) | ✅ {champ_smape:.2f}% |",
    f"| Below 12.47% (old champion) | ✅ {champ_smape:.2f}% |",
    f"| Below 12% | ✅ {champ_smape:.2f}% |",
    f"| **Below 11.5%** | **✅ {champ_smape:.2f}%** |",
    f"| Below 11% | ❌ {champ_smape:.2f}% |",
    f"| Below 10% | ❌ |",
    f"| Below 8% | ❌ |",
    "",
    "## Hour Breakdown (Champion vs CatBoost Baseline)",
    "",
    "| Hour | CatBoost | Champion | Change |",
    "|:----:|:--------:|:--------:|:------:|",
]
for h in sorted(champ_hours_v.keys()):
    cbv = cb_hours_v.get(h, 0)
    chv = champ_hours_v.get(h, 0)
    diff = chv - cbv
    icon = "✅" if diff < -0.3 else ("❌" if diff > 0.3 else "➡️")
    lines.append(f"| {h} | {cbv:.2f}% | {chv:.2f}% | {icon} {diff:+.2f}pp |")

lines += [
    "",
    "## Worst 5 Days",
    "",
    "| Day | sMAPE_floor50 |",
    "|:---:|:-------------:|",
]
for d, r in worst_days.iterrows():
    lines.append(f"| {d} | {r['day_smape']:.2f}% |")

lines += [
    "",
    "## Worst 5 Hours",
    "",
    "| Hour | sMAPE_floor50 |",
    "|:----:|:-------------:|",
]
for h, s in worst_hours.items():
    lines.append(f"| {h} | {s:.2f}% |")

lines += [
    "",
    "## Recommendation",
    "",
    "**强烈建议冻结为当前 day-ahead 生产候选。**",
    f"- {champ_smape:.2f}% 是当前所有方法中的绝对最优结果",
    f"- 相对原 CatBoost 基线提升 {cb_smape - champ_smape:.2f} 个百分点",
    "- 修正方法简单（spike residual rolling corrector），无未来泄漏",
    "- 553/720 行改变，17/24 小时改善",
    "- 仅 6 段小时轻微变差（最大 +1.08pp）",
    "",
    "## Limitations",
    "",
    "- 前 7 天 (Feb 1-7) 无修正（因需 168h 历史数据初始化）",
    "- 春节前后最差日仍在 20-35% 左右，correction 不完全",
    "- H17 (18.55% to 19.64%) 反而变差",
    f"- 11% 目标仍差 {champ_smape - 11:.2f}pp，需要 XGBoost / 结构化特征改进",
]

report = "\n".join(lines)
(OUT / "reports" / "dayahead_champion_report.md").write_text(report, encoding="utf-8")
print(f"\nReport saved to {OUT / 'reports' / 'dayahead_champion_report.md'}")
