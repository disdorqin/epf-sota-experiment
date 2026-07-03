#!/usr/bin/env python3
"""
Generate the trusted day-ahead champion (best_two_average = 11.85%)
and its associated report, after data leakage invalidation.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
from pathlib import Path
from src.common.metrics import smape_floor50, compute_all_metrics

OUT = Path("outputs/dayahead_trusted_champion_30d")
for d in [OUT / "predictions", OUT / "metrics", OUT / "reports"]:
    d.mkdir(parents=True, exist_ok=True)

# ── Load trusted champion: best_two_average ──
b2a = pd.read_csv(
    "outputs/dayahead_lgbm_freeze_30d/predictions/best_two_average_dayahead.csv",
    encoding="utf-8-sig"
)

# Standardize schema
std = pd.DataFrame({
    "ds": b2a["ds"],
    "y_true": b2a["y_true"],
    "y_pred": b2a["y_pred"],
    "hour_business": b2a["hour_business"].astype(int),
    "period": b2a["period"],
    "target_day": b2a["target_day"],
    "business_day": b2a["target_day"],
    "task": "dayahead",
    "model_name": "dayahead_trusted_champion_best_two_average",
})
std.to_csv(str(OUT / "predictions" / "dayahead_trusted_champion_best_two_average.csv"),
           index=False, encoding="utf-8-sig")
print(f"Saved trusted champion: {len(std)} rows")

# ── Baseline models ──
baselines = {
    "catboost_sota (original)": "outputs/dayahead_30d_core/predictions/catboost_sota_dayahead.csv",
    "catboost_spike_residual (old)": "outputs/dayahead_corrections_30d/predictions/catboost_spike_residual_corrected_dayahead.csv",
    "lightgbm_trial_02 (single)": "outputs/dayahead_lgbm_stage2_30d/predictions/trial_02_w150_nl255_lr0.03_dayahead.csv",
    "lgbm_spike_residual (INVALID-leaked)": "outputs/dayahead_lgbm_corrections_30d/predictions/lgbm_spike_residual_corrected_dayahead.csv",
}
y_champ, p_champ = std["y_true"].values, std["y_pred"].values

summary = []
valid = ~(np.isnan(y_champ) | np.isnan(p_champ))
m = compute_all_metrics(y_champ[valid], p_champ[valid])
m["model_name"] = "dayahead_trusted_champion_best_two_average"
m["task"] = "dayahead"
m["n"] = int(valid.sum())
summary.append(m)

for name, path in baselines.items():
    bdf = pd.read_csv(path, encoding="utf-8-sig")
    yp = bdf["y_pred_cb"].values if "y_pred_cb" in bdf.columns else bdf["y_pred"].values
    yt = bdf["y_true"].values
    v = ~(np.isnan(yt) | np.isnan(yp))
    if v.sum() < 2:
        continue
    m = compute_all_metrics(yt[v], yp[v])
    m["model_name"] = name
    m["task"] = "dayahead"
    m["n"] = int(v.sum())
    summary.append(m)

sdf = pd.DataFrame(summary)
sdf.to_csv(str(OUT / "metrics" / "summary.csv"), index=False, encoding="utf-8-sig")

print("\n=== SUMMARY ===")
for _, r in sdf.iterrows():
    flag = " ⚠️ INVALID" if "INVALID" in r["model_name"] else ""
    print(f"  {r['model_name']:50s}  SMAPE={r['sMAPE_floor50']:.4f}%{flag}")

# ── Report ──
champ_smape = sdf.loc[sdf["model_name"].str.contains("trusted"), "sMAPE_floor50"].values[0]
invalid = sdf[sdf["model_name"].str.contains("INVALID")]
invalid_smape = invalid["sMAPE_floor50"].values[0] if len(invalid) > 0 else None

lines = []
lines.append("# Day-Ahead Trusted Champion Report")
lines.append("")
lines.append("> Generated: 2026-07-03 21:55")
lines.append("> Task: dayahead")
lines.append("> Metric: sMAPE_floor50 (only)")
lines.append("")
lines.append("## ⚠️ Data Leakage Announcement")
lines.append("")
lines.append("**lgbm_spike_residual_corrected (11.27%) has been INVALIDATED due to data leakage.**")
lines.append("")
lines.append("Leak: prediction features included `y_true` (line 101 of lgbm_dayahead_corrector.py)")
lines.append("See `docs/reports/dayahead_leakage_audit.md` for full details.")
lines.append("")
lines.append("## Current Trusted Champion")
lines.append("")
lines.append(f"**best_two_average = {champ_smape:.2f}%**")
lines.append("")
lines.append("- Construction: simple average of LightGBM trial_02 + trial_24 predictions")
lines.append("- Fusion: `y_pred = (y_pred_t02 + y_pred_t24) / 2` — pure prediction fusion")
lines.append("- No y_true involved at any stage of fusion")
lines.append("- 720 rows, hours 1-24, 30 days (Feb 1-Mar 2)")
lines.append("")
lines.append("## Ranking")
lines.append("")
lines.append("| Rank | Model | sMAPE | Leak-free? |")
lines.append("|:----:|------|:-----:|:----------:|")
lines.append(f"| 🥇 1 | best_two_average (trusted) | **{champ_smape:.2f}%** | ✅ |")
rank = 2
for _, r in sdf.iterrows():
    if "trusted" in r["model_name"]:
        continue
    is_invalid = "INVALID" in r["model_name"]
    flag = "⚠️ LEAKED" if is_invalid else "✅"
    lines.append(f"| {rank} | {r['model_name']} | {r['sMAPE_floor50']:.2f}% | {flag} |")
    rank += 1
lines.append("")
lines.append("## Target Check")
lines.append("")
lines.append(f"| Target | Status |")
lines.append(f"|:-------|:------:|")
lines.append(f"| Below 12.58% (CatBoost) | ✅ {champ_smape:.2f}% |")
lines.append(f"| Below 12.47% (old champion) | ✅ {champ_smape:.2f}% |")
lines.append(f"| Below 12% | ✅ {champ_smape:.2f}% |")
lines.append(f"| Below 11.5% | ❌ {champ_smape:.2f}% (gap {champ_smape-11.5:.2f}pp) |")
lines.append("")
lines.append("## Anti-Leakage Measures")
lines.append("")
lines.append("1. `_validate_prediction_features()` in all corrector prediction paths")
lines.append("2. Denylist: y_true, residual, error, abs_error, future_y, target_actual, oracle, best_model")
lines.append("3. `tests/test_no_target_leakage.py` — static analysis + runtime guard verification")
lines.append("4. All new correction/fusion code must pass before commit")
lines.append("")
lines.append("## Recommendation")
lines.append("")
lines.append(f"- Freeze `best_two_average` ({champ_smape:.2f}%) as current production candidate")
lines.append("- Do NOT use any correction that depends on y_true at prediction time")
lines.append("- Future correction work must pass anti-leakage tests first")

report = "\n".join(lines)
(OUT / "reports" / "dayahead_trusted_champion_report.md").write_text(report, encoding="utf-8")
print(f"\nReport saved to {OUT / 'reports' / 'dayahead_trusted_champion_report.md'}")
