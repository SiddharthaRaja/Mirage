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

Full 166-database run: see "Run 2" below.

## Run 2 · full study, seed 0 (Kaggle T4, 8.9 h) — 2026-10-20
Notebook: `kaggle/run2_full_study.ipynb` (filter script: `scripts/filter_subset.py`).

### Blind spot, all databases
| scope | DBs | mutants | original DB accepts | suites catch (of accepted) | provably wrong-but-rewarded (of all) |
|---|---|---|---|---|---|
| all | 164 | 23,585 | 49.4% | 16.5% | >= 8.2% |
| excluding 7 empty-table DBs | 157 | 17,881 | 33.2% | 32.4% | >= 10.8% |

Empty-table DBs (every query returns nothing, so every mutant "matches"):
academic, geo, imdb, music_2, restaurants, scholar, yelp. The second row is the headline.

### Pass-rate filter
1,500 training questions chosen by hash (sha256('mirage-filter-v1:'+qid)), k=8 samples each:
kept 925 (attempts disagree under either reward), 468 all-wrong, 113 all-right.
Mean pass: single 33.4%, suite 31.6%.

### GRPO training (Qwen2.5-Coder-0.5B-Instruct + LoRA r=16, 300 steps, seed 0)
| arm | rollouts | match single | match suite | false-positive share [95% CI] | drift | groups with signal |
|---|---|---|---|---|---|---|
| single | 19,200 | 69.8% | 67.0% | 4.0% [2.9, 5.2] | -0.1 pts (n.s.) | 49.1% |
| suite | 19,200 | 69.2% | 66.9% | 3.4% [2.5, 4.6] | -0.5 pts (n.s.) | 47.8% |

Reading: the single-DB reward paid for provably wrong SQL in ~4% of rewarded rollouts,
but over 300 steps the 0.5B model did not learn to exploit it (no significant drift;
the two arms are indistinguishable). The suite arm still shows ~3.4% because the audit's
strict check uses variants the training reward does not; suites are not perfect.
One seed only; dev accuracy pending (`kaggle/08_dev_eval.ipynb`).

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
- Run long jobs as committed versions (Save & Run All); interactive sessions die and wipe /kaggle/working.
- Training speed on a T4: ~45 s per GRPO step (8 questions x 8 rollouts).