# LightGBM Stage-3 Day-Ahead Report (v3 Features + Optuna)
**Generated**: 2026-07-03 22:38
**Features**: v3 (base + extended + volatility + ranks + change + interactions)
**Search**: 1 Optuna trials + 1 baseline
**Search window**: Feb 1-20 (20 days)
**Confirm window**: Feb 21-Mar 2 (10 days)
**Full 30d**: Feb 1-Mar 2 (30 days)

## 1. Baseline Confirmation
- Full sMAPE: 11.6399%
- Search: 12.6650%
- Confirm: 9.5898%

## 2. Top 10 Configurations
| Config | Full sMAPE | Search | Confirm | Window | Obj | nl | lr |
|---|---|---|---|---|---|---|---|
| stage3_baseline_90d_mae | 11.64% | 12.66% | 9.59% | 90 | mae | 127 | 0.02 |
| optuna_01 | 12.09% | 12.87% | 10.53% | 90 | rmse | 191 | 0.02 |

## 3. Target Check
- **Best**: stage3_baseline_90d_mae (11.6399%)
- Current champion (best_two_average): 11.85%
- Below 11.85%? YES
- Below 11.5%? NO
- Below 11.0%? NO

## 4. Overfitting Check
- stage3_baseline_90d_mae: search=12.66% confirm=9.59% gap=-3.08pp [WARNING]
- optuna_01: search=12.87% confirm=10.53% gap=-2.34pp [WARNING]

## 5. Feature Importance (Top 15)
(See debug/feature_importance.json)

## 6. Comparison with Stage-2
| Metric | Stage-2 Best | Stage-3 Best | Delta |
|---|---|---|---|
| sMAPE_floor50 | 12.07% | 11.64% | -0.43pp |

## 7. Recommendations
YES - Stage-3 has beaten the 11.85% champion.

### Next steps
- Proceed to XGBoost sentinel experiment
- Evaluate safe fusion with Stage-3 best
- Consider AutoGluon if fusion also fails
