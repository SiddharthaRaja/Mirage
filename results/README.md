# Mirage

**How often does the execution reward lie?** A reward-fidelity study for GRPO
text-to-SQL on a free GPU, with the result priced as cost per correct query.

Training a model to write SQL with reinforcement learning needs a reward, and
the standard one is *execution match*: run the model's query and the gold
query on the database and compare the rows. On Spider's databases, which often
have a handful of rows per table, a wrong query frequently returns the right
rows by coincidence. The reward pays for it anyway.

Mirage measures that, three ways:

1. **Before training** — how many near-miss queries (one plausible mistake away
   from gold) the single-database check accepts, and how many a few perturbed
   copies of the database catch.
2. **During training** — two GRPO arms from identical starting weights, one
   rewarded on the single database, one on the database *plus* its perturbed
   copies. Every rollout in both arms is scored by both checks and written to a
   ledger, so the false-positive rate of the standard reward is measured *while
   training on it*, and whether the policy learns to exploit it is a drift test
   with a confidence interval, not an anecdote.
3. **After training** — execution accuracy and suite accuracy on Spider dev,
   pass@k, majority vote, a failure taxonomy, and, through
   [Vantage](https://github.com/SiddharthaRaja/vantage), **cost per 1,000
   correct answers** for the base and both trained models served from one vLLM
   process.

```bash
pip install -e ".[dev]" && mirage demo      # no GPU, no download, ~10 seconds
```

## Results (2 seeds × 2 arms, free Kaggle T4s)

Qwen2.5-Coder-0.5B-Instruct + LoRA, 300 GRPO steps per arm, two seeds per arm, Spider dev (1,034 questions,
20 databases never seen in training), evaluated once per model. Full write-up: [results/WRITEUP.md](results/WRITEUP.md);
every final table: [results/final/FINAL.md](results/final/FINAL.md).

| | value |
|---|---|
| near-miss SQL the standard check accepts (157 non-empty DBs) | **33.2%**; at least 10.8% provably wrong |
| rewarded training attempts the suite rejects | 3.4–4.2% across 4 arms, no significant drift in any |
| ... of which genuinely wrong (dev cases, hand-checked) | 14 of 27; the rest are ties / duplicate names |
| training rewards paid on empty-result questions | **20.3%** (7 empty-table databases; invisible to the suite) |
| dev EX, base → GRPO (single reward), seed 0 / seed 1 | 42.0% → **56.8% / 57.3%** (+14.8 / +15.3, both p < 0.001) |
| official test-suite EX (taoyds), base → GRPO single | 36.8% → **47.5% / 47.9%** |
| dev EX, suite reward vs single reward | −1.2 (seed 0), −1.3 (seed 1); both intervals include zero |
| seed 1 vs seed 0, same arm | +0.5 / +0.4 points, not distinguishable: replicates |

In one line: the execution reward lies, mostly through empty databases rather than near-miss SQL;
a 0.5B policy did not learn to exploit it in 300 steps, RL still added ~15 points on unseen
databases (~11 on the official test suites), and the stricter reward bought nothing measurable.
All of it replicated on a second seed.
Cases: [results/GALLERY.md](results/GALLERY.md). Numbers and run notes: [results/RESULTS.md](results/RESULTS.md).


`mirage demo` runs the entire study end to end on a bundled synthetic fixture
with a simulated policy and writes a report with seven charts. Everything in
that report is stamped **SIMULATED**: it proves the measurements work, not
anything about language models. The GPU runs are what produce results.

---

## What is (and is not) new here

GRPO with execution rewards for text-to-SQL is well covered (Reasoning-SQL,
SQL-R1, Arctic-Text2SQL-R1 and others, mostly at 7B+). Test-suite evaluation
(Zhong et al., 2020) exists precisely because single-database execution
accuracy has false positives. In a literature check I did not find a
head-to-head training comparison of single-database versus multi-database
reward that measures reward hacking directly, at small scale. That is absence
of evidence from a handful of searches, so this repo says "I didn't find prior
work", never "novel".

The framing is a **measurement study under a tiny compute budget**: Qwen2.5
0.5B–1.5B, LoRA, one free Kaggle T4.

## The pieces

| | what it does | where |
|---|---|---|
| Verifier | one definition of "correct", shared by reward and evaluation: safe read-only execution, Spider's result-equivalence rules (multiset rows, column permutations, order only when gold orders) | `reward/verifier.py`, `sql/` |
| Suites | perturbed copies of each database (row shuffles, within-column permutation, numeric jitter, gold-literal boundary planting, row edits), selected greedily to reject the most near-miss mutants | `reward/perturb.py`, `reward/suite.py` |
| Mutants | ten mutation operators over a sqlglot AST: `>`→`>=`, dropped condition, DISTINCT toggled, MIN↔MAX, ORDER BY dropped… | `sql/mutate.py` |
| Rewards | binary or shaped; `single` or `suite`; a TRL-compatible callable that writes the ledger and logs audit metrics through TRL's `log_metric` | `reward/rewards.py` |
| Training | SFT warm-up (optional), pass-rate filter, GRPO per arm via TRL; T4-safe defaults, auto-resume, session-budget stop | `train/` |
| Audit | false-positive share over training with question-clustered bootstrap intervals, first-vs-last-quarter drift test, hack-shape counts | `analysis/audit.py`, `sql/hacks.py` |
| Evaluation | generate on GPU, score on CPU; EX, suite EX, pass@k, majority vote, failure taxonomy, paired bootstrap + McNemar | `eval/` |
| Cost | Vantage workload export with the exact training prompt; cost per correct join | `integrations/vantage.py`, `integrations/vantage/` |
| Report | generated findings, seven charts, a "what this does not establish" section | `analysis/report.py` |

## Running it for real

The GPU side runs as five Kaggle notebooks (`kaggle/`), each a thin shell
around CLI commands. Full walkthrough in [docs/KAGGLE.md](docs/KAGGLE.md).

```bash
mirage data check  $SPIDER_DIR                       # dev must share no databases with train
mirage suites build --spider $SPIDER_DIR             # + the pre-training blind spot (CPU)
mirage spike  -c configs/spike.yaml --n 100          # which model, warm-up or not
mirage sft    -c configs/sft.yaml                    # only if the spike says so
mirage filter -c configs/grpo_base.yaml              # questions where attempts disagree
mirage grpo   -c configs/grpo_single.yaml            # arm A
mirage grpo   -c configs/grpo_suite.yaml             # arm B
mirage audit  runs/arms/single/ledger.jsonl
mirage generate -c configs/grpo_base.yaml --split dev --adapter runs/arms/single/adapter
mirage score  runs/eval/.../predictions.jsonl --spider $SPIDER_DIR --suites runs/suites
mirage report study.yaml
```

## Methodology in brief

Full version: [docs/METHODOLOGY.md](docs/METHODOLOGY.md).

- **One variable.** The two arms share a base config and differ only in
  `reward.mode`; `mirage report` compares their manifests (config hash and
  starting-weights fingerprint) and refuses to call it a comparison otherwise.
- **Same start, right reference.** An SFT adapter is merged into the base
  before GRPO adds a fresh adapter, so TRL's KL reference (adapters disabled)
  is the SFT model, not the raw base.
- **Dev is touched once.** Every tuning decision uses a validation split carved
  from Spider train *by database*, hashed so it is stable across runs.
- **One prompt.** Baseline, SFT, both arms, evaluation and the Vantage workload
  all build messages through one function.
- **Honest intervals.** Accuracies are bootstrapped over questions; rollout
  rates over *questions* too (eight attempts at one question are not eight
  observations). If an interval for a difference includes zero, the report does
  not call it a difference.
- **Generate once, score forever.** Raw completions are saved; scoring is CPU
  only, so every past run can be rescored if a rule changes.

## Layout

```
src/mirage/
  data/          Spider loader, database-level splits, schema rendering, the prompt, fixture
  sql/           extraction, safe executor, result equivalence, mutants, hack detectors
  reward/        perturbation, suites, verifier, TRL reward + ledger
  train/         SFT, pass-rate filter, GRPO, resume/budget handling
  eval/          generation backends, scoring, stats, failure taxonomy, official evaluator hook
  analysis/      ledger audit, simulator, report
  integrations/  Vantage export and cost per correct
configs/         base + spike + sft + two arms + ablations + example study
kaggle/          five notebooks (generated by build_notebooks.py)
integrations/vantage/   backend configs, sweep, and a 13-line Vantage patch
tests/           70 tests, none needing a GPU
scripts/smoke_train.py  every training/generation call on a tiny random model (2 min)
```

## Status

- Measurement side (data, execution, suites, rewards, audit, evaluation,
  report): built and tested — 70 tests plus the full simulated study in CI.
- Real runs on a Kaggle T4 (2026-10-19 to 21): suites on all databases, model
  spike, pass-rate filter, both GRPO arms (seed 0), training audit, dev
  evaluation, gallery and write-up. Outputs in `results/run2/`; notebooks in
  `kaggle/` (`run2_full_study`, `08_dev_eval`, `09_seed1`).
- Done (2026-10-26): both arms × 2 seeds, official test-suite numbers. Not pursued: a clean run without
  empty-gold questions, a 1.5B check, cost per correct query through Vantage.

## Licence

MIT. Spider is CC BY-SA 4.0 and is not redistributed here.
