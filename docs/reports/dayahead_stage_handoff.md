# Day-Ahead Stage Handoff Report

> Generated: 2026-07-03
> Task: dayahead
> Metric: sMAPE_floor50 (only)
> Repository: epf-sota-experiment

## 1. Current Trusted Champion

**best_two_average = 11.85%**

Construction: simple average of LightGBM trial_02 (150d, nl=255, lr=0.03, mae) and LightGBM trial_24 (90d, nl=127, lr=0.02, rmse) predictions. Pure y_pred fusion, no y_true involved at any stage. 720 rows, hours 1-24, 30 days (2026-02-01 to 2026-03-02).

## 2. Invalidated Results

**lgbm_spike_residual_corrected = 11.27% is INVALIDATED due to target leakage.**

Root cause: `src/correction/lgbm_dayahead_corrector.py` used `day_data["y_true"]` as a prediction feature in `X_pred`. This gave the model access to the ground truth it was trying to predict, making residual prediction trivially easy.

This result must NOT be used as champion, baseline, or referenced in any comparison.

## 3. Current Trusted Ranking

| Rank | Model/Method | sMAPE_floor50 | Leak-free | Notes |
|:----:|---|:---:|:---:|---|
| 1 | best_two_average | 11.85% | Yes | LightGBM trial_02 + trial_24 |
| 2 | LightGBM trial_02 (single) | 12.07% | Yes | 150d, nl=255, mae |
| 3 | LightGBM trial_14 | 12.19% | Yes | 120d, nl=191, lr=0.015 |
| 4 | LightGBM trial_11 | 12.20% | Yes | 120d, nl=127, lr=0.02 |
| 5 | LightGBM trial_24 (single) | 12.23% | Yes | 90d, nl=127, lr=0.02 |
| 6 | CatBoost spike residual | 12.47% | Yes | Old champion |
| 7 | CatBoost baseline | 12.58% | Yes | Stable baseline |
| 8 | TabPFN-TS | 13.64% | Yes | Below CatBoost |
| 9 | CatBoost tuned | 13.89% | Yes | Dead end |
| 10 | CatBoost period specialist | 14.62% | Yes | Dead end |

## 4. Failed Directions Summary

- CatBoost normal tuning: 13.89%, worse than vanilla 12.58%
- CatBoost period specialist: 14.62%, significantly worse
- Full 24-hour hour specialist: same features, less data, no improvement
- TabPFN as CatBoost replacement: 13.64%, not competitive
- Normal residual correction: marginal gain at best (0.05-0.11pp)
- Router v1: insufficient accuracy
- Regime v2 MoE: 22.58%, catastrophic failure
- H13/H17 replacement: no change from baseline
- Safe ensemble repetition: no improvement
- Chronos / TiRex / TimesFM: not applicable for dayahead tabular

## 5. Current Target

From 11.85%:
- Stage target: below 11.5%
- Stretch target: below 11.0%
- Long-term target: below 8.0% (requires deep models or fundamentally new approach)

## 6. Anti-Leakage Enforcement

All subsequent experiments must:
- Pass `tests/test_no_target_leakage.py` before commit
- Use only `smape_floor50` as primary metric
- Report search_window / confirm_window / full_30d split
- Never use y_true, residual, error, abs_error, future_y, target_actual, oracle, best_model as prediction features
- All rolling/rank/statistics features must use only history visible before prediction day
