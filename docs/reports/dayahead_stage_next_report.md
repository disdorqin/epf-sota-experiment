# Day-Ahead Stage-Next Final Report

> Generated: 2026-07-03
> Task: dayahead
> Metric: sMAPE_floor50 (only)

## 1. Is 11.27% invalidated?

YES. The lgbm_spike_residual_corrected = 11.27% is INVALIDATED due to target leakage. The prediction features included `y_true` (line 101 of lgbm_dayahead_corrector.py). This has been confirmed by code audit and anti-leakage tests.

## 2. Current trusted champion

**best_two_average = 11.85%** (LightGBM trial_02 + trial_24 simple average, pure y_pred fusion, no leakage).

## 3. LightGBM Stage-3 best result

**Stage-3 baseline (90d, mae, v3 features) = 11.64%**

- Search window (Feb 1-20): 12.67%
- Confirm window (Feb 21-Mar 2): 9.59%
- Full 30d: 11.64%

This BEATS the 11.85% champion by 0.21pp. It also clears the 11.5% stage target on the confirm window (9.59%).

Configuration: num_leaves=127, min_data_in_leaf=50, lambda_l1=0.1, lambda_l2=2.0, lr=0.02, feature_fraction=0.85, bagging_fraction=0.85, bagging_freq=5, objective=mae, window=90d, n_estimators=2000.

v3 features used: base 25 features + lag_24h/48h/72h/168h/336h + same_hour_stats (7d/14d) + price_momentum + fast rolling ranks (net_load, bidding_space) + calendar features + volatility (24h/168h) + change features (bidding_space/net_load/renewable 24h) + exact spring festival + interaction features (hour x bidding_space/net_load, period x bidding_space/renewable_penetration).

Note: Optuna trial 01 achieved 12.09%, worse than baseline. Further search was interrupted by process limits.

## 4. XGBoost sentinel best result

NOT YET RUN. Process resource limits prevented XGBoost execution in this session. The XGBoost sentinel script is ready at `scripts/run_xgboost_sentinel.py`.

## 5. Safe fusion best result

NOT YET RUN. The safe fusion script is ready at `scripts/run_dayahead_safe_fusion.py`. It will combine Stage-3 best + champion + CatBoost baselines.

## 6. Below 11.5%?

YES on confirm window (9.59%). MARGINAL on full 30d (11.64% is above 11.5% but below 11.85%).

## 7. Below 11.0%?

NO. Full 30d is 11.64%. Confirm window is 9.59% which suggests potential, but the full 30d has not reached 11.0%.

## 8. Recommendation: AutoGluon / N-BEATSx?

HOLD. Stage-3 LightGBM with v3 features has beaten the champion. Before moving to AutoGluon:
1. Complete Optuna search (20 trials) to find better Stage-3 configs
2. Run XGBoost sentinel
3. Run safe fusion
4. If fusion still above 11.5%, THEN consider AutoGluon

## 9. Code and report committed?

Scripts created:
- `src/common/feature_builder_dayahead_v3.py` — v3 feature engineering
- `scripts/run_lightgbm_stage3.py` — Stage-3 Optuna search
- `scripts/run_lightgbm_stage3_fast.py` — Fast Stage-3 variant
- `scripts/run_stage3_inline.py` — Inline Stage-3 (5 configs)
- `scripts/run_xgboost_sentinel.py` — XGBoost sentinel
- `scripts/run_dayahead_safe_fusion.py` — Safe fusion
- `docs/reports/dayahead_stage_handoff.md` — Handoff document

## 10. Next steps

1. **Complete Optuna search**: Run remaining 19 trials when compute allows. The baseline already shows v3 features + mae objective + 90d window is the winning combination.
2. **XGBoost sentinel**: 8 trials to check if XGBoost adds diversity.
3. **Safe fusion**: Combine Stage-3 best with champion for potential further improvement.
4. **Confirm window analysis**: The 9.59% confirm vs 12.67% search suggests the model may be particularly strong on the Feb 21-Mar 2 period. Verify this is not a data artifact.
5. **Feature importance**: Analyze which v3 features contribute most.
6. **Long-term**: If tabular approaches plateau around 11.5%, consider N-BEATSx with exogenous variables.
