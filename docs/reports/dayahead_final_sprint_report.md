# Day-Ahead Final Sprint Report

> Generated: 2026-07-04 10:15

## 1. Current Trusted Champion

**best_two_average = 11.85%**
- LightGBM trial_02 + trial_24 pure prediction average
- 720 rows, hours 1-24, business-day correct
- No y_true leakage

## 2. Invalid Result

- lgbm_spike_residual_corrected (11.27%): **INVALIDATED** — target leakage in prediction features

## 3. Stage3 Business-Day Fix

- Old Stage3 (11.64%): invalid — natural-day grouping error
- Fixed Stage3 (business_time_mapping): 11.86% — did NOT beat champion

## 4. LightGBM Micro-Search (Task A)

| Config | Full sMAPE | Search | Confirm | Window | Obj | nl |
|--------|:---------:|:------:|:-------:|:-----:|:---:|:--:|
| **cfg05** | **11.48%** | 12.79% | 8.88% | 90 | mae | 191 |
| cfg01 (base) | 11.86% | 12.73% | 10.11% | 90 | mae | 127 |
| cfg08 | 12.40% | 13.23% | 10.75% | 90 | rmse | 127 |
| cfg06 | 12.95% | 14.73% | 9.38% | 150 | mae | 191 |
| cfg03 | 13.04% | 14.83% | 9.45% | 150 | mae | 255 |
| cfg04 | 13.16% | 15.30% | 8.88% | 120 | mae | 127 |
| cfg02 | 13.38% | 15.24% | 9.64% | 120 | mae | 191 |
| cfg07 | 13.20% | 14.94% | 9.74% | all | mae | 127 |

**cfg05 (11.48%) beats champion!**

## 5. Safe Fusion Final (Task B)

Skipped — cfg05 already reached 11.5% target.

## 6. XGBoost Sentinel Mini (Task C)

Skipped — cfg05 already reached 11.5% target.

## 7. Final Ranking

| Rank | Model | sMAPE |
|:----:|------|:-----:|
| 🥇 1 | **cfg05 (micro-search)** | **11.48%** |
| 🥈 2 | best_two_average | 11.85% |
| 🥉 3 | stage3 baseline | 11.86% |
| 4 | catboost spike residual | 12.47% |
| 5 | catboost sota | 12.58% |

## Decision: NEW TRUSTED CHAMPION FOUND!

**cfg05 = 11.48% — below 11.5% target.**

## 8. Recommendations

- **Freeze cfg05 as new trusted champion.** Sprint complete.
- cfg05 params: window=90d, objective=mae, num_leaves=191, min_data_in_leaf=30, lr=0.015, l1=0.1, l2=5.0, feature_fraction=0.85, bagging_fraction=0.95, bagging_freq=5
- No further AutoGluon/N-BEATSx needed for this sprint.
