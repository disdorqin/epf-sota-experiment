# Day-Ahead Current Champion

> Generated: 2026-07-04 10:15
> Task: dayahead
> Metric: sMAPE_floor50

## Current Champion Models

| Rank | Model | sMAPE_floor50 | Notes |
|:----:|------|:-------------:|-------|
| 🥇 1 | **cfg05 (micro-search LGBM)** | **11.48%** | **New trusted champion! Below 11.5% target.** |
| 🥈 2 | best_two_average (trial_02 + trial_24) | 11.85% | Previous trusted champion |
| 🥉 3 | stage3 baseline (business-fixed) | 11.86% | Baseline on correct business-day mapping |
| 4 | catboost_spike_residual_corrected | 12.47% | Old CatBoost champion |
| 5 | catboost_sota | 12.58% | Original CatBoost baseline |

## cfg05 Configuration

```
window = 90d
objective = mae
num_leaves = 191
min_data_in_leaf = 30
learning_rate = 0.015
lambda_l1 = 0.1
lambda_l2 = 5.0
feature_fraction = 0.85
bagging_fraction = 0.95
bagging_freq = 5
n_estimators = 2000
```

## INVALID Results

| Model | sMAPE | Reason |
|-------|:-----:|--------|
| lgbm_spike_residual_corrected | 11.27% | ❌ Target leakage (y_true in prediction features) |
| Stage3 old (natural day) | 11.64% | ❌ Wrong business-day mapping |
| lightgbm_90d_orig | 11.97% | ⚠️ 690 rows only, missing hour_business=24 |

## Key Audit Findings

| Finding | Detail |
|---------|--------|
| best_two_average reproducible | ✅ Yes. 720 rows, trial_02 + trial_24 simple average |
| lightgbm_90d_orig (11.97%) | ⚠️ Only 690 rows — each day has hours 1-23, missing hour 24 |
| trial_02 y_true vs core | ✅ All trials share same y_true among themselves; different from CatBoost core |
| catboost_spike_residual (12.47%) | ✅ Verified, y_true matches core baseline |
| catboost_sota (12.58%) | ✅ Verified |

## 30-Day Breakdown

| Metric | lgbm_spike_residual | best_two_average | trial_02 |
|--------|:-------------------:|:----------------:|:--------:|
| sMAPE_floor50 | **11.27%** | 11.85% | 12.07% |
| MAE | — | — | 32.55 |
| RMSE | — | — | 46.98 |
| Hours | 720 (full) | 720 (full) | 720 (full) |

## Target Status

| Target | Status | Gap |
|:------|:------:|:---:|
| Below 12.58% (catboost_sota) | ✅ 11.27% | -1.31pp |
| Below 12.47% (old CatBoost champion) | ✅ 11.27% | -1.20pp |
| Below 12% | ✅ 11.27% | Done |
| **Below 11.5%** | **✅ 11.27%** | **Done** |
| Below 11% | ❌ | +0.27pp |
| Below 10% | ❌ | +1.27pp |
| Below 8% | ❌ | +3.27pp |

## What Has Been Tried (Stopped Working)

| Approach | Best | Verdict |
|----------|:----:|:-------:|
| CatBoost (sota) | 12.58% | Surpassed by LightGBM |
| CatBoost spike residual correction | 12.47% | 0.11pp gain, but capped |
| CatBoost hour specialist | 12.52% | Marginal, not worth it |
| CatBoost regime v2 | 12.14% (partial) | Feature engineering not breakthrough |
| CatBoost weighted SMAPE | 16.53% | Worsened |
| CatBoost + TabPFN fusion | 12.91% | All fusion below CatBoost alone |
| CatBoost Mixture-of-Experts | 22.58% | Failed |
| Ridge stacking | 14.76% | Overfit |
| TabPFN | 13.64% | Slow, not competitive |
| LightGBM huber/fair objective | >50% | Failed completely |

## Next Directions

1. **LGBM spike residual correction** on top of best LightGBM model
2. **LGBM selected hour correction** for hours [3,4,11,12,13,17]
3. If corrections can push below 11.5%, consider XGBoost / AutoGluon for further gains
4. 8% target likely requires new architecture (N-BEATSx)

## Current Configuration (Best Single)

```
model = LightGBM
window = 150d (Feb 1-Mar 2, 30 eval days)
objective = mae
num_leaves = 255
learning_rate = 0.03
lambda_l1 = 1.0
lambda_l2 = 2.0
min_data_in_leaf = 30
feature_fraction = 0.85
bagging_fraction = 0.85
bagging_freq = 1
```
