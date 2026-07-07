# P1.1 Candidate Comparison Report (same four hard months)

All numbers evaluated on **2025-11, 2025-12, 2026-01, 2026-02** (n=120 days). Easy-window numbers excluded.

| Model | sMAPE_floor50 (%) | vs trusted champ (15.04) | vs faithful 2.5 (21.87) | decision |
|---|---|---|---|---|
| cfg05 | 14.68 | -0.36 | -7.19 | shadow |
| cfg05_180d | 14.25 | -0.79 | -7.62 | shadow |
| xgboost_rich | 14.70 | -0.34 | -7.17 | shadow |
| ensemble_rich | 14.54 | -0.50 | -7.33 | shadow |
| baseline_lgbm25 (faithful 2.5) | 22.84 | 7.80 | — | reference |
| **trusted champion best_two_average (same-window)** | **15.04** | — | -6.83 | baseline |
