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
One seed only; dev accuracy below.

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
## Dev evaluation · seed 0 (Kaggle T4, 1.5 h) — 2026-10-20
Notebook: `kaggle/08_dev_eval.ipynb`. Spider dev, 1,034 questions on 20 databases never seen in training,
touched once. Greedy decoding for EX; 8 samples (T = 0.8) for pass@8 and majority vote. 95% bootstrap CIs.

| model | EX | suite EX | pass@1 | pass@8 | maj EX |
|---|---|---|---|---|---|
| base (Qwen2.5-Coder-0.5B-Instruct) | 42.0% [39.1, 44.8] | 40.1% [37.2, 42.9] | 33.3% | 65.2% | 56.0% |
| GRPO single-DB reward | **56.8%** [54.0, 59.8] | 54.2% [51.3, 57.2] | 52.8% | 69.7% | 60.5% |
| GRPO suite reward | 55.6% [52.5, 58.6] | 53.1% [50.1, 56.1] | 52.2% | 70.1% | 60.3% |

Paired comparisons (bootstrap over questions + McNemar):

| comparison | EX | suite EX | gained / lost (EX) |
|---|---|---|---|
| single − base | **+14.8 pts** [12.0, 17.6], p < 0.001 | +14.0 [11.1, 16.7] | 207 / 54 |
| suite − base | **+13.6 pts** [10.7, 16.6], p < 0.001 | +13.0 [10.2, 15.9] | 206 / 65 |
| suite − single | −1.2 [−2.6, 0.3], p = 0.13 (no detectable difference) | −1.1 [−2.4, 0.3] | 21 / 33 |

Reading:
- GRPO with an execution reward improves the 0.5B model by ~14 points on unseen databases, mostly by
  fixing schema linking (wrong column/table names: 410 → 193 failures).
- The tighter suite reward does not beat the standard one at this scale, consistent with the training
  audit (the standard reward lies on ~4% of rewarded rollouts and the policy does not exploit it).
- Every model loses ~2 points from EX to suite EX: answers right only on the original data.
- Full report (charts, failure taxonomy, false-positive examples): `runs/report/REPORT.md` in the Kaggle output.

Caveat: the report flags that the two arms' configs differ beyond `reward.mode`. The only other difference
is the session time budget passed to each arm (`budget.max_hours`); it never triggered (both arms ran all
300 steps). Seed 1 will pass identical budgets so the manifests match exactly.

## Gallery and write-up (2026-10-21)
- `results/GALLERY.md`: real false positives from dev and training, re-executed on the suite databases.
- `results/WRITEUP.md`: the 840-word write-up of seed 0.
- `results/run2/`: report, charts, dev scores, training ledgers (gzipped), filter, logs.

Two findings from reading the cases:
1. **20.3% of the rewards paid during training (single arm) went to questions whose gold result is
   empty**; 92% of those come from the 7 empty-table databases (geo alone: 1,462). Any query that runs
   and returns nothing is paid. The suite cannot catch this (98.6% of those rollouts also pass it).
   The pass-rate filter selected these questions (failed executions make attempts disagree).
2. **Suite-flagged false positives are about half real.** Of 27 on dev (single arm), 14 are genuinely
   wrong queries right by coincidence; 13 are defensible answers the suite rejects (6 ties:
   `= MAX()` vs `LIMIT 1`; 7 duplicate names: id vs name, DISTINCT).

## Seed 1 · single arm (Kaggle T4, 3.6 h) — 2026-10-21
Notebook: `kaggle/09_seed1.ipynb` (seed 0's suites and filter reused; only the seed changes).
First attempt hit CUDA out-of-memory at step 9 (fragmentation: 3.08 GiB reserved-but-unallocated);
rerun with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` completed all 300 steps without retries.

| arm | seed | rollouts | match single | match suite | false-positive share [95% CI] | drift | groups with signal |
|---|---|---|---|---|---|---|---|
| single | 0 | 19,200 | 69.8% | 67.0% | 4.0% [2.9, 5.2] | -0.1 pts (n.s.) | 49.1% |
| single | 1 | 19,200 | 68.9% | 66.0% | 4.2% [2.9, 5.6] | +1.7 pts (n.s.) | 50.2% |

Seed 1 replicates seed 0 within noise: the false-positive share and the absence of drift both hold.
Seed-1 suite arm and the dev evaluation of both seed-1 arms: see the final run below.

## Final run · seed-1 suite arm, seed-1 dev, official evaluator (2026-10-26)
Notebook: `kaggle/12_final.ipynb` (generated by `kaggle/build_final.py`). All tables: `results/final/FINAL.md`.

- **v1** (Kaggle T4 x2, 4.25 h): GPU 0 trained the seed-1 suite arm (300 steps, 230 min) while GPU 1 generated the
  seed-1 single arm's dev predictions; both arms scored. Then two bugs: the audit cell could not import
  `mirage.analysis` (the kernel's working directory put the repo folder `/kaggle/working/mirage` on `sys.path`,
  shadowing the package in `src/`), and the official-evaluator step picked Mirage's own suites folder as the
  official databases. Fixed: `src/` first on `sys.path` and the import checked at setup; the database check
  requires the official layout (`<db>/<db>.sqlite` plus variants).
- **Rerun** (CPU only, v1's output attached, 7.6 min): reused everything, downloaded the official test suites,
  scored five models, audited four arms.

| arm | seed | false-positive share [95% CI] | drift | dev EX | suite EX | official test-suite EX |
|---|---|---|---|---|---|---|
| base | - | - | - | 42.0% | 40.1% | 36.8% |
| single | 0 | 4.0% [2.9, 5.2] | -0.1 (n.s.) | 56.8% | 54.2% | 47.5% |
| suite | 0 | 3.4% [2.5, 4.6] | -0.5 (n.s.) | 55.6% | 53.1% | 46.7% |
| single | 1 | 4.2% [2.9, 5.6] | +1.7 (n.s.) | 57.3% | 54.5% | 47.9% |
| suite | 1 | 3.9% [2.7, 5.1] | +1.1 (n.s.) | 56.0% | 53.5% | 46.7% |

Paired on dev EX: GRPO single s1 − base +15.3 [12.5, 18.1]; suite s1 − base +14.0 [11.0, 17.0]; suite − single
-1.3 [-2.7, +0.2] (p 0.12) for seed 1, -1.2 for seed 0; seed 1 − seed 0 +0.5 (single) / +0.4 (suite), n.s.
Everything found on seed 0 replicates. Mirage is complete; no further runs planned.
