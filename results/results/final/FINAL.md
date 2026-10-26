# Mirage · final results (seeds 0 and 1)

Kaggle T4 · repo d96dd9b Seed 1 single results; seed-1 dev-eval and official test-suite notebooks · seed-1 suite arm trained in run 12 v1 (4.25 h, T4 x2); this report from a CPU rerun on its output (0.1 h) · Qwen2.5-Coder-0.5B-Instruct + LoRA r16, GRPO 300 steps per arm.
Spider dev: 1,034 questions on 20 databases never seen in training, touched once per model. 95% bootstrap CIs.
If an interval for a difference includes zero, it is not called a difference.

## 1 · Training audit: how often the standard reward paid for wrong SQL

| arm | seed | steps | rollouts | match single | match suite | false-positive share [95% CI] | drift last−first quarter [95% CI] | groups with signal |
|---|---|---|---|---|---|---|---|---|
| single | 0 | 300 | 19,200 | 69.8% | 67.0% | 4.0% [2.9%, 5.2%] | -0.1 pts [-2.1, +2.0] (n.s.) | 49.1% |
| suite | 0 | 300 | 19,200 | 69.2% | 66.9% | 3.4% [2.5%, 4.6%] | -0.5 pts [-2.4, +1.3] (n.s.) | 47.8% |
| single | 1 | 300 | 19,200 | 68.9% | 66.0% | 4.2% [2.9%, 5.6%] | +1.7 pts [-0.6, +4.0] (n.s.) | 50.2% |
| suite | 1 | 300 | 19,200 | 68.6% | 66.0% | 3.9% [2.7%, 5.1%] | +1.1 pts [-0.9, +3.2] (n.s.) | 51.0% |

## 2 · Dev accuracy

| model | EX [95% CI] | suite EX | official test-suite EX | pass@1 | pass@8 | majority EX |
|---|---|---|---|---|---|---|
| base | 42.0% [39.1, 44.8] | 40.1% | 36.8% | 33.3% | 65.2% | 56.0% |
| GRPO single s0 | 56.8% [54.0, 59.8] | 54.2% | 47.5% | 52.8% | 69.7% | 60.5% |
| GRPO suite s0 | 55.6% [52.5, 58.6] | 53.1% | 46.7% | 52.2% | 70.1% | 60.3% |
| GRPO single s1 | 57.3% [54.3, 60.3] | 54.5% | 47.9% | 53.5% | 70.4% | 61.2% |
| GRPO suite s1 | 56.0% [53.0, 59.0] | 53.5% | 46.7% | 52.4% | 68.6% | 59.7% |

Official = taoyds/test-suite-sql-eval on its released distilled test suites, same greedy predictions, `--etype exec`.

## 3 · Paired comparisons (same 1,034 questions; bootstrap 95% CI + exact McNemar)

| comparison | metric | difference [95% CI] | gained / lost | McNemar p | significant |
|---|---|---|---|---|---|
| GRPO single s0 − base | ex | +14.8 [+12.0, +17.6] | 207 / 54 | 2.72e-22 | yes |
| GRPO single s0 − base | suite_ex | +14.0 [+11.1, +16.7] | 199 / 54 | 1.07e-20 | yes |
| GRPO suite s0 − base | ex | +13.6 [+10.7, +16.6] | 206 / 65 | 2.99e-18 | yes |
| GRPO suite s0 − base | suite_ex | +13.0 [+10.2, +15.9] | 196 / 62 | 2.23e-17 | yes |
| GRPO single s1 − base | ex | +15.3 [+12.5, +18.1] | 204 / 46 | 6.3e-25 | yes |
| GRPO single s1 − base | suite_ex | +14.4 [+11.6, +17.2] | 197 / 48 | 1.29e-22 | yes |
| GRPO suite s1 − base | ex | +14.0 [+11.0, +17.0] | 204 / 59 | 7.07e-20 | yes |
| GRPO suite s1 − base | suite_ex | +13.3 [+10.4, +16.2] | 195 / 57 | 7.56e-19 | yes |
| GRPO suite s0 − GRPO single s0 | ex | -1.2 [-2.6, +0.3] | 21 / 33 | 0.134 | no |
| GRPO suite s0 − GRPO single s0 | suite_ex | -1.1 [-2.4, +0.3] | 20 / 31 | 0.161 | no |
| GRPO suite s1 − GRPO single s1 | ex | -1.3 [-2.7, +0.2] | 23 / 36 | 0.117 | no |
| GRPO suite s1 − GRPO single s1 | suite_ex | -1.1 [-2.5, +0.3] | 21 / 32 | 0.169 | no |
| GRPO single s1 − GRPO single s0 | ex | +0.5 [-1.0, +1.9] | 31 / 26 | 0.597 | no |
| GRPO single s1 − GRPO single s0 | suite_ex | +0.4 [-1.0, +1.7] | 28 / 24 | 0.678 | no |
| GRPO suite s1 − GRPO suite s0 | ex | +0.4 [-1.0, +1.6] | 25 / 21 | 0.659 | no |
| GRPO suite s1 − GRPO suite s0 | suite_ex | +0.4 [-0.9, +1.6] | 23 / 19 | 0.644 | no |

## 4 · Reading (generated from the tables)

- The single-database reward paid for provably wrong SQL in 3.4%–4.2% of rewarded rollouts across 4 arms.
- No arm shows a significant drift in that share over training: no evidence the policy learned to exploit it.
- GRPO vs base (EX): GRPO single s0 +14.8 pts (significant); GRPO suite s0 +13.6 pts (significant); GRPO single s1 +15.3 pts (significant); GRPO suite s1 +14.0 pts (significant).
- Suite reward − single reward (EX): seed 0 -1.2 pts (not distinguishable); seed 1 -1.3 pts (not distinguishable); mean over seeds -1.2 pts.
- Seed 1 vs seed 0, same arm (EX): single +0.5 pts (replicates); suite +0.4 pts (replicates).
- Official test-suite EX: base 36.8%; GRPO suite s0 46.7%; GRPO single s0 47.5%; GRPO single s1 47.9%; GRPO suite s1 46.7%.

## 5 · What this does not establish

- Two seeds per arm is the minimum for a replication, not a distribution; seed-to-seed spread is only roughly known.
- One model size (0.5B) and 300 steps: exploitation may appear with a larger model or longer training.
- Mirage's suites approximate the official test suites; about half of suite-flagged false positives on dev were defensible answers (results/GALLERY.md).
- 20.3% of seed-0 training rewards went to questions with empty gold results (7 empty-table databases); the suites cannot catch that leak.

## 6 · Run status

| stage | state | at (h) | detail |
|---|---|---|---|
| filter | ok | 0.05 | {"kept": 925} |
| suites | reused | 0.05 | {"databases": 166} |
| train_suite | reused | 0.05 |  |
| dev_suite | reused | 0.05 |  |
| train_single | reused | 0.05 |  |
| dev_single | reused | 0.05 |  |
| ts_download | ok | 0.05 | {"rc": 0, "minutes": 0.2} |
| official_dbs | ok | 0.07 | {"path": "/kaggle/working/ts_raw/database", "databases": 29} |
| official_base | ok | 0.09 | {"rc": 0, "minutes": 0.9} |
| official_GRPO_suite_s0 | ok | 0.09 | {"rc": 0, "minutes": 1.0} |
| official_GRPO_single_s0 | ok | 0.09 | {"rc": 0, "minutes": 1.0} |
| official_GRPO_single_s1 | ok | 0.1 | {"rc": 0, "minutes": 0.7} |
| official_GRPO_suite_s1 | ok | 0.11 | {"rc": 0, "minutes": 1.5} |
| lanes | done | 0.11 |  |
| audit | ok | 0.12 | {"arms": ["single-s0", "suite-s0", "single-s1", "suite-s1"]} |
| report_seed1 | ok | 0.12 | {"rc": 0, "minutes": 0.1} |
