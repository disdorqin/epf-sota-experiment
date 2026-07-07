# P1.1 Window Ablation Report (cfg05 90d vs 180d)

Both on the same four hard months. Rich feature frame.

| Model | sMAPE_floor50 (%) | 1_8 | 9_16 | 17_24 | spike | normal | train+infer (s) |
|---|---|---|---|---|---|---|---|
| cfg05 | 14.68 | 13.91 | 16.01 | 14.12 | 13.51 | 14.81 | 505.3 |
| cfg05_180d | 14.25 | 13.97 | 15.33 | 13.45 | 13.64 | 14.32 | 1841.4 |

**Conclusion:** cfg05 180d (14.25%) improves cfg05 90d (14.68%); richer/longer window helps.
