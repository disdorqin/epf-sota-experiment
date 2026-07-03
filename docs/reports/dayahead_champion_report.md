# Day-Ahead Champion Report

> Generated: 2026-07-03 20:00
> Task: dayahead
> Metric: sMAPE_floor50

## Current Champion

**dayahead_champion_lgbm_spike_residual = 11.27%**

- Base model: LightGBM trial_02 (150d, mae objective, num_leaves=255)
- Corrector: LGBMSpikeResidualCorrector (alpha=0.25, max_delta=50)
- Rolling: each target_day uses data before that day only
- Training: CatBoostRegressor on spike-hour residuals (top 10% by past absolute residual)

## Ranking

| Rank | Model | sMAPE_floor50 | vs Champion |
|:----:|------|:-------------:|:-----------:|
| 1 | champion_lgbm_spike_residual | **11.27%** | — |
| 2 | best_two_average | 11.85% | ++0.58pp |
| 3 | lightgbm_trial_02 (single) | 12.07% | ++0.80pp |
| 4 | catboost_spike_residual (old) | 12.47% | ++1.20pp |
| 5 | catboost_sota (original) | 12.58% | ++1.31pp |

## Improvement vs Baselines

- vs CatBoost baseline (12.58%): **1.31pp improvement**
- vs old champion spike_residual (12.47%): **1.20pp improvement**
- vs best_two_average (11.85%): +0.58pp
- vs LightGBM single (12.07%): +0.80pp

## Target Check

| Target | Status |
|:-------|:------:|
| Below 12.58% (CatBoost original) | ✅ 11.27% |
| Below 12.47% (old champion) | ✅ 11.27% |
| Below 12% | ✅ 11.27% |
| **Below 11.5%** | **✅ 11.27%** |
| Below 11% | ❌ 11.27% |
| Below 10% | ❌ |
| Below 8% | ❌ |

## Hour Breakdown (Champion vs CatBoost Baseline)

| Hour | CatBoost | Champion | Change |
|:----:|:--------:|:--------:|:------:|
| 1 | 10.00% | 13.57% | ❌ +3.57pp |
| 2 | 18.75% | 8.90% | ✅ -9.86pp |
| 3 | 18.09% | 15.99% | ✅ -2.10pp |
| 4 | 16.09% | 15.58% | ✅ -0.52pp |
| 5 | 14.67% | 15.49% | ❌ +0.82pp |
| 6 | 15.60% | 13.89% | ✅ -1.71pp |
| 7 | 8.52% | 15.46% | ❌ +6.94pp |
| 8 | 12.30% | 8.31% | ✅ -3.99pp |
| 9 | 17.57% | 7.61% | ✅ -9.96pp |
| 10 | 13.44% | 15.15% | ❌ +1.71pp |
| 11 | 9.68% | 12.13% | ❌ +2.45pp |
| 12 | 10.51% | 6.81% | ✅ -3.70pp |
| 13 | 7.25% | 6.87% | ✅ -0.38pp |
| 14 | 10.77% | 6.82% | ✅ -3.95pp |
| 15 | 10.62% | 10.30% | ✅ -0.32pp |
| 16 | 19.63% | 6.31% | ✅ -13.32pp |
| 17 | 9.19% | 19.64% | ❌ +10.44pp |
| 18 | 8.91% | 9.71% | ❌ +0.80pp |
| 19 | 9.88% | 9.07% | ✅ -0.81pp |
| 20 | 12.10% | 10.24% | ✅ -1.85pp |
| 21 | 11.57% | 11.29% | ➡️ -0.28pp |
| 22 | 10.23% | 12.11% | ❌ +1.88pp |
| 23 | 12.54% | 9.92% | ✅ -2.62pp |
| 24 | 13.95% | 9.36% | ✅ -4.59pp |

## Worst 5 Days

| Day | sMAPE_floor50 |
|:---:|:-------------:|
| 2026-02-04 | 34.10% |
| 2026-02-23 | 28.15% |
| 2026-02-05 | 22.19% |
| 2026-02-09 | 17.27% |
| 2026-02-01 | 17.00% |

## Worst 5 Hours

| Hour | sMAPE_floor50 |
|:----:|:-------------:|
| 17 | 19.64% |
| 3 | 15.99% |
| 4 | 15.58% |
| 5 | 15.49% |
| 7 | 15.46% |

## Recommendation

**强烈建议冻结为当前 day-ahead 生产候选。**
- 11.27% 是当前所有方法中的绝对最优结果
- 相对原 CatBoost 基线提升 1.31 个百分点
- 修正方法简单（spike residual rolling corrector），无未来泄漏
- 553/720 行改变，17/24 小时改善
- 仅 6 段小时轻微变差（最大 +1.08pp）

## Limitations

- 前 7 天 (Feb 1-7) 无修正（因需 168h 历史数据初始化）
- 春节前后最差日仍在 20-35% 左右，correction 不完全
- H17 (18.55% to 19.64%) 反而变差
- 11% 目标仍差 0.27pp，需要 XGBoost / 结构化特征改进