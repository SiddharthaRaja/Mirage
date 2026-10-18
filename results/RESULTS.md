# Mirage results (real runs, Kaggle T4)

## 01 · Pre-training blind spot (CPU) — 2026-10-19
Subset: car_1, concert_singer, flight_2, pets_1, world_1

| database | gold | mutants | pass original | caught by suite |
|---|---|---|---|---|
| car_1 | 52 | 269 | 56 | 11 |
| concert_singer | 26 | 112 | 42 | 11 |
| flight_2 | 40 | 158 | 100 | 5 |
| pets_1 | 23 | 115 | 29 | 9 |
| world_1 | 65 | 386 | 102 | 14 |
| **total** | 206 | 1,040 | 329 (31.6%) | 50 (15.2% of passing) |

Single-database check accepts 31.6% of near-miss mutants; suites prove
at least 50/1,040 (4.8%) wrong-but-rewarded. The remaining passing mutants
are unresolved (many likely semantically equivalent, e.g. DISTINCT on a
unique column). flight_2 is an outlier, probably from empty gold results.

Full 166-database run: TODO

## 01 · Model spike — 100 val questions (3 databases)

| model | EX | suite EX | pass@8 | mixed groups | s/question |
|---|---|---|---|---|---|
| Qwen2.5-Coder-0.5B-Instruct | 41.0% | 37.0% | 72.0% | 64.0% | 2.94 |
| Qwen2.5-0.5B-Instruct | 13.0% | 12.0% | 34.0% | 34.0% | 2.23 |
| Qwen2.5-Coder-1.5B-Instruct | 32.0% | 28.0% | 72.0% | 69.0% | 6.12 |

Decision: Qwen2.5-Coder-0.5B-Instruct, no SFT warm-up (baseline inside the
15–50% band, 64% mixed groups). Small sample (3 databases): gaps are rough.

## Environment notes
- Kaggle image ships torchao 0.10, which PEFT rejects: `pip uninstall -y torchao` after setup.
- Smoke test passed on T4 after that fix.