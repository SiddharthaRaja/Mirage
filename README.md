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
- Training side: `scripts/smoke_train.py` runs SFT, merge, filter, both GRPO
  arms, resume, generation and scoring on a tiny random model with torch 2.14,
  transformers 5.19, TRL 1.15.0 and PEFT 0.21.2. Passes on CPU. Not yet run on
  a GPU: no real-model results exist yet.
- See [HANDOUT.md](HANDOUT.md) Part 9 for exactly what has and has not been
  executed.

## Licence

MIT. Spider is CC BY-SA 4.0 and is not redistributed here.
