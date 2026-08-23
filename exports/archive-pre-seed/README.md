# Archived: pre-seed (unpinned) TENCON data — superseded 2026-07-03

These files backed the **original TENCON submission numbers** but were generated
**before BehaviorSpace random seeds were pinned**. They are one particular
un-reproducible draw and are **superseded by the seeded deposit** in `../seeded/`.
Kept for provenance only — do **not** cite or re-analyse for the camera-ready.

Why archived (see `experiments/EXP-CM-S004-sensitivity-sweep.md` resume note):
a fresh rerun of the canonical H001 gives 23% cascade at 0%-AI vs the 10% in the
deposited draw here — seeds were never fixed, so these numbers do not regenerate.
`setup` now calls `random-seed behaviorspace-run-number`; the seeded re-deposit
(`../seeded/`) regenerates identically and backs the revised paper.

| File | Was | Superseded by |
|------|-----|---------------|
| `...H001_batch1_...-spreadsheet.csv` | H001 exploratory (unpinned) | `../seeded/H001_batch2_...-seeded.csv` |
| `...H001_batch2_...-spreadsheet.csv` | H001 full, primary deposit (unpinned) | `../seeded/H001_batch2_...-seeded.csv` |
| `...H002_batch3_...-spreadsheet.csv` | H002 repeated stress (unpinned) | `../seeded/H002_batch3_...-seeded.csv` |
| `...H003_batch4_...-spreadsheet.csv` | H003 mixed regime (unpinned) | `../seeded/H003_batch4_...-seeded.csv` |
| `H001_fresh_rerun-table.csv` | diagnostic that exposed the seed bug | — |
| `S-H004_..._robustness-table.csv` | sensitivity sweep at n=30 (unpinned) | `../seeded/S-H004_..._robustness-seeded.csv` (n=100) |
| `S-H005_..._robustness-table.csv` | fatigue sweep at n=30 (unpinned) | `../seeded/S-H005_..._robustness-seeded.csv` (n=100) |

Note: the legacy parser `notebooks/h003_analysis.py` still has an in-file demo
default pointing at the old H003 spreadsheet name; it is not on any live path
(the reproducibility gate `notebooks/validate_baseline.py` reads `../seeded/` directly).
