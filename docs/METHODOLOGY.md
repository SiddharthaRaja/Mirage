# Methodology

Each section states a choice, the reason for it, and what would go wrong
without it.

## 1. The question

> When a small model is trained with GRPO on a single-database execution
> reward, how often is that reward wrong, does the policy learn to exploit it,
> and does a stricter multi-database reward change the outcome at equal cost?

The study is built so that each part of that question has its own measurement
and can come out null without the project failing. A null result ("the reward
is wrong 4% of the time and that does not grow") is reported as such.

## 2. Data and splits

- **Spider 1.0** (CC BY-SA 4.0): `train_spider.json` + `train_others.json`
  for training, `dev.json` for the final number. Train and dev share no
  databases; `mirage data check` asserts it.
- **Validation** is carved from train **by database** with a salted hash
  (`data.val_fraction`, `data.split_salt`). Dev generalises to unseen
  schemas, so validation must too. A hash, not a shuffle, so the split is the
  same on every machine and after adding data.
- **Dev is used once per model**, after all decisions. Model choice, warm-up,
  KL, reward shape and checkpoint selection are all made on validation.
- **Question ids are content hashes** of (database, question, gold SQL) so
  ledgers and predictions join across re-exports.

## 3. Correctness

One `Verifier` decides correctness for rewards and evaluation alike.

- **Extraction** (`sql/extract.py`): first fenced block, else first line
  starting SELECT/WITH up to a blank line; first statement only. Fixed for all
  models; how the SQL was found is logged.
- **Execution** (`sql/executor.py`): `mode=ro` connection, an authorizer that
  denies writes/ATTACH/PRAGMA, a 2-second progress-handler timeout, a row cap.
  Failures are values with a kind, never exceptions.
- **Equivalence** (`sql/compare.py`), following the test-suite evaluator:
  multiset of rows (order only when gold has a top-level ORDER BY), any column
  permutation, 1 == 1.0, floats to 6 significant digits, strings exact. Errors
  never match.

## 4. Suites

For each database:

1. Mutate every gold query on it (`sql/mutate.py`, 10 operators, ≤24 mutants
   per query, deduplicated, provably-equivalent DISTINCT cases skipped).
2. Record which mutants the original database already rejects.
3. Generate 12 candidate variants (`reward/perturb.py`). Primary/foreign keys
   and every literal appearing in a gold query are preserved; non-key columns
   are permuted, numbers jittered, gold numeric literals planted at boundaries,
   rows added (fresh integer keys, valid foreign keys) and removed (only from
   unreferenced tables), and row order shuffled.
4. Reject any variant on which a gold query errors.
5. Greedily keep up to 4 variants that reject the most not-yet-rejected
   mutants.

`suite.json` per database records the blind-spot rate (share of mutants the
original accepts) and catch rate (share of those the suite rejects), overall
and per operator.

**Leakage.** Train suites use train gold queries, which the reward already
uses. Dev suites use dev gold queries, as distilled test suites do; they are
an evaluation instrument and never touch training.

**Limits.** This is simpler than Zhong et al.'s fuzzing-and-distillation. The
official test-suite evaluator is supported for dev (`mirage score
--official`) and should be reported next to Mirage's suite EX.

## 5. Training

- **Model**: chosen by the spike — baseline EX roughly 15–50% and a high share
  of questions with mixed outcomes over 8 samples. Lead candidate
  Qwen2.5-Coder-0.5B-Instruct.
- **SFT warm-up**, only if needed: ≤2,000 examples, 1 epoch, completion-only
  loss, LoRA r=16.
- **Starting point**: the SFT adapter is merged into the base; GRPO trains a
  fresh LoRA on the merged model. TRL uses "adapters disabled" as the KL
  reference, which is then the SFT model.
- **Pass-rate filter**: 8 samples per training question from the starting
  model; keep a question if attempts disagree under *either* reward. Same file
  for both arms.
- **GRPO** (TRL 1.15): 8 generations, β=0.04 (set explicitly; TRL's default is
  0), lr 1e-5, temperature 0.9, 160 completion tokens, 8 questions per
  optimizer step, `loss_type=dapo` and `scale_rewards=group` stated explicitly
  so a library upgrade cannot change them silently. fp16, not bf16 (T4).
- **Arms** differ only in `reward.mode`. Checked from manifests by the report.
- **Kaggle survival**: checkpoint every 25 steps; self-stop 20 minutes before
  the configured budget; auto-resume; ledger trimmed to the resumed step.

## 6. The audit

Every rollout row: arm, step, question, reward, SQL, both verdicts, error
kind, hack flags.

- **False-positive share** = rewarded-by-single ∧ rejected-by-suite, divided
  by rewarded-by-single. Ten step bins per arm.
- **Intervals** by bootstrap over questions within each bin.
- **Drift**: last-quarter minus first-quarter false-positive share, with a
  bootstrap over questions in each quarter. Exploitation predicts a positive,
  significant drift in the single arm and not in the suite arm.
- **Hack shapes** (`sql/hacks.py`): empty match, LIMIT without ORDER BY,
  dropped filter, SELECT *, constant answer, tautology, DISTINCT mismatch,
  boundary nudge. Heuristic labels on rewarded rollouts; the suite verdict is
  what decides wrong.
- **Groups with signal**: share of (step, question) groups where rewards
  differ. GRPO learns nothing from the rest.

## 7. Evaluation

- Greedy decoding for EX and suite EX; 8 samples at T=0.8 for pass@k
  (unbiased estimator) and majority vote by execution result.
- Paired bootstrap over questions for differences, plus exact McNemar.
- Failure taxonomy on wrong greedy answers, first structural difference wins.
- Difficulty labels recomputed with Spider's rule on a sqlglot parse
  (approximate; the official evaluator's labels are authoritative).

## 8. Cost (part D)

Vantage serves the start model and both adapters from one `vllm serve
--enable-lora` process with the exact training prompt (requires the included
13-line Vantage patch). Cost per 1,000 correct = Vantage's $/1,000 requests ÷
accuracy, with accuracy's interval mapped through. Reported against both EX
and suite EX.

## 9. What would change the conclusions

- A second seed per arm. If the arm difference is inside the seed spread, it is
  not established. Budget for at least two seeds per arm before claiming one.
- Official test-suite accuracy disagreeing with Mirage suite EX.
- A different starting accuracy: the hacking question may have different
  answers at 10% and at 60% baseline EX.
