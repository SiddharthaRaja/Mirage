# Mirage: the whole thing explained from scratch

You asked me to build this in one go. This document is the "read it
afterwards" part, written for someone who has never done RL post-training and
(by your own account) is not yet good at SQL. It is ordered so you can stop at
any point and still have something useful.

- **Part 1** — what this project is, and why it is shaped this way
- **Part 2** — just enough SQL to read every query in this repo
- **Part 3** — every ML and statistics concept the project uses, from zero
- **Part 4** — what the code is, file by file
- **Part 5** — running it: the demo, then Kaggle
- **Part 6** — what the results could look like, and what each outcome means
- **Part 7** — what to do next, in order
- **Part 8** — how to talk about it, and the questions you will get
- **Part 9** — what I could not do, and what is still unverified

---

# Part 1 — What this is, and why

## The one-line version

Mirage trains a small model to write SQL using reinforcement learning, and
measures something most people skip: **how often the reward used for training
is wrong**, whether the model learns to take advantage of that, and what a
stricter reward changes — then prices the trained model as **cost per correct
answer** using Vantage.

## Where it sits in your portfolio

| project | what it shows |
|---|---|
| Vectora | you can build an application on top of models (retrieval, citations, refusal) |
| Vantage | you can measure a serving system honestly (load, latency, cost, quality gate) |
| **Mirage** | you can **train** a model, and you understand what can go wrong with the training signal |

Applications, systems, training. And Mirage is not RAG on purpose, as you
asked, so the three don't look like one idea three times.

## Why "how often does the reward lie" is the interesting part

The standard recipe for RL on text-to-SQL is: the model writes a query, you
run it and the correct ("gold") query on the database, and if they return the
same rows the model gets reward 1. It's simple, cheap and automatic.

The catch: Spider's databases are tiny. A table might have six rows. On six
rows, a *wrong* query often returns the same rows as the right one by
coincidence. Example straight from the bundled fixture:

```sql
-- the question: "What is the name of the oldest singer?"
SELECT name FROM singer ORDER BY age DESC LIMIT 1   -- correct
SELECT name FROM singer LIMIT 1                     -- wrong: just takes the first row
```

In that table, the first row happens to be the oldest singer. Both queries
return `Mara Quell`. The reward pays for the wrong one. A model being trained
on that reward is being *told* that "just take the first row" is correct.

Researchers know execution matching has false positives — that is why Spider's
authors built "test suites" for *evaluation*. What I could not find was a
clean study of what happens when you *train* on the leaky reward versus a
stricter one, measured directly, at small scale. That is Mirage's question, and
it is a good one for you because:

- the answer is a measurement, not a leaderboard number, so you don't need a
  big GPU to have something real to say;
- a null result ("the reward is wrong 4% of the time and the model doesn't
  exploit it") is still a result you can defend;
- it shows exactly the kind of skepticism about your own numbers that Vantage
  showed, now applied to training.

---

# Part 2 — Just enough SQL

You said you're not strong at SQL. This is the subset that appears in Spider,
which is also the subset that appears in most interviews.

A **table** is rows and columns. `singer(singer_id, name, country, age)` has
one row per singer.

```sql
SELECT name, age FROM singer                       -- pick columns
SELECT name FROM singer WHERE age > 40             -- filter rows
SELECT name FROM singer WHERE age > 40 AND country = 'France'
SELECT count(*) FROM singer                        -- aggregate: count, sum, avg, min, max
SELECT country, count(*) FROM singer GROUP BY country          -- one row per country
SELECT country FROM singer GROUP BY country HAVING count(*) > 1 -- filter the groups
SELECT name FROM singer ORDER BY age DESC LIMIT 1  -- sort, then keep the first row
SELECT DISTINCT country FROM singer                -- remove duplicate rows
```

**Joins** combine tables through a shared key. `singer_in_concert` has
`(concert_id, singer_id)` pairs:

```sql
SELECT T1.name
FROM singer AS T1
JOIN singer_in_concert AS T2 ON T1.singer_id = T2.singer_id
JOIN concert AS T3 ON T2.concert_id = T3.concert_id
WHERE T3.concert_name = 'Summer Lights'
```

Read it as: for each singer, find their appearances, find the concert of each
appearance, keep the ones at Summer Lights. `T1`, `T2` are just short names.

A **primary key** uniquely identifies a row (`singer_id`). A **foreign key**
is a column that points at another table's primary key
(`singer_in_concert.singer_id → singer.singer_id`). Joins almost always go
foreign key → primary key.

**Subqueries** put one query inside another:

```sql
SELECT name FROM singer
WHERE singer_id NOT IN (SELECT singer_id FROM singer_in_concert)   -- singers who never performed
```

That's all of it. The fastest way to get good: run `mirage demo`, open
`runs/demo/eval/base/scored_items.jsonl`, and for each wrong answer, work out
*why* it's wrong. The failure taxonomy tells you which kind of mistake to look
for.

---

# Part 3 — Every concept, from zero

## 3.1 Execution accuracy (EX)

Instead of comparing the model's SQL text to the gold SQL text (many different
queries are correct), run both and compare results. Same rows → correct.
This is Spider's standard metric. Mirage calls it **EX** and computes it with
the same rules as the official evaluator:

- rows are compared as a **multiset** (order ignored, duplicates counted) —
  unless the gold query has an ORDER BY, then order matters;
- columns may come back in any order (`SELECT name, age` = `SELECT age, name`);
- 1 equals 1.0; strings must match exactly.

## 3.2 Test-suite accuracy, and Mirage's suites

The fix for coincidental matches: check the query on **several** databases
that have the same tables but different data. A query that is right by luck
on one database is rarely right by luck on five.

The official version (Zhong et al., 2020) generates many random databases and
keeps a small "distilled" set that best separates correct queries from close
mistakes. Mirage builds its own suites with a simpler version of the same
idea, for every database including the *training* ones (which the official
release doesn't cover, and which the training reward needs):

1. Generate **mutants** of every gold query — the gold query with one small,
   plausible mistake (`>` vs `>=`, a dropped condition, DISTINCT
   added/removed, MIN vs MAX, ORDER BY removed). This is "mutation testing",
   borrowed from software engineering.
2. Make perturbed copies of the database: shuffle row order (breaks LIMIT
   without ORDER BY), shuffle values within a column (breaks coincidental
   correlations), jitter numbers, plant the gold query's own numbers at
   boundaries (breaks `>` vs `>=`), add and remove rows (breaks missing
   DISTINCT, COUNT vs COUNT DISTINCT). Keys and the gold query's literal values
   are never touched, so the gold answer still makes sense.
3. Keep the 1–4 copies that reject the most mutants.

**Suite EX** = correct on the original *and* every copy.

## 3.3 SFT versus RL

**SFT (supervised fine-tuning):** show the model the right answer, train it to
reproduce it. The gold SQL is the target.

**RL:** the model is never shown the right answer. It writes its own
attempts, a reward scores them, and the model is nudged toward attempts that
scored well. The gold SQL is only used inside the reward, to check results.

RL can discover answers the dataset didn't spell out, and it optimises the
thing you actually care about (correct results) rather than "copy this text".
It can also optimise a *flawed* reward perfectly — which is the whole subject
of this project.

## 3.4 GRPO, with numbers

GRPO (Group Relative Policy Optimization) is the RL algorithm used for most
recent "reasoning" models. One step:

1. Take a question. Sample **8 attempts** from the model (that's the "group").
2. Score each: say 3 are right (reward 1), 5 wrong (reward 0).
3. The group's average is 0.375. Each attempt's **advantage** is how much
   better than average it did: right attempts ≈ +1.29, wrong ≈ −0.77 after
   dividing by the group's standard deviation.
4. Increase the probability of the tokens in positive-advantage attempts,
   decrease it for negative ones.
5. Repeat over thousands of questions.

Using the group average as the baseline is what lets GRPO skip the separate
"critic" network older methods (PPO) needed. Two consequences matter here:

- **All-right or all-wrong groups teach nothing.** If all 8 score the same,
  every advantage is zero. That's why Mirage filters training questions to ones
  where the starting model's attempts disagree, and logs "groups with signal"
  during training.
- **The model can only reinforce what it already sometimes does.** A model at
  0% has nothing to reinforce, which is why there's an optional SFT warm-up.

## 3.5 The KL penalty (β)

A pull back toward the starting model, so the policy can't drift into
degenerate text. Too weak: collapse. Too strong: no learning. Note that TRL's
current default is **β = 0** (no pull at all); Mirage sets 0.04 explicitly.

A subtle trap Mirage avoids: with LoRA, TRL uses "the model with the adapter
turned off" as the reference. If you trained an SFT adapter and then continue
GRPO on the *same* adapter, "adapter off" is the raw base model, which can't
write SQL — the KL term would pull your model toward incompetence. Mirage
merges the SFT adapter into the weights first, then trains a fresh GRPO
adapter, so "adapter off" means "the SFT model".

## 3.6 LoRA

Instead of updating all 500M weights, freeze them and train two small
matrices per layer (rank 16 here), well under 1% of the parameters. That's
what makes training fit on a free 16 GB T4.

## 3.7 Reward hacking and false positives

**Reward hacking:** the model finds a way to score without doing the task.
For SQL, Mirage tracks named shapes:

| shape | example |
|---|---|
| `limit_without_order` | `SELECT name FROM singer LIMIT 1` instead of ordering by age |
| `empty_match` | gold returns nothing, the model's query also returns nothing |
| `dropped_filter` | the WHERE clause didn't matter on this data |
| `distinct_mismatch` | DISTINCT added/removed, equal only because there are no duplicates |
| `boundary` | `>=` instead of `>`, equal only because no row sits on the boundary |
| `constant_answer` | `SELECT 6` — guessing the number |

A **false positive** is one rollout that the single-database reward paid for
and the suite rejected. The **false-positive share** is the fraction of all
paid-for rollouts that were false positives. If it *rises* during training in
the single-reward arm, the model is learning the loophole.

## 3.8 Why two arms

To know whether the stricter reward *causes* anything, you need two training
runs that are identical except for the reward. Same starting weights, same
data, same seed, same hyperparameters. Mirage enforces this: both arm configs
inherit one base file, and the report checks a hash of each run's full config
(minus the reward) and a fingerprint of its starting weights before it will
compare them.

## 3.9 The statistics

- **Bootstrap confidence intervals:** resample your questions with replacement
  2,000 times, recompute the accuracy each time, the middle 95% is the
  interval. On Spider dev (~1,000 questions) a typical interval is about ±3
  points — bigger than many published "improvements".
- **Cluster bootstrap:** the 8 attempts at one question are not 8 independent
  observations. When computing a rate over rollouts (like false-positive
  share), Mirage resamples *questions*, keeping their attempts together.
  Resampling individual rollouts gives intervals that are far too narrow.
- **Paired comparison:** both models answered the same questions, so resample
  the per-question *differences*. Removes the noise from some questions being
  harder. Plus an exact **McNemar test**, which looks only at questions where
  the two models disagree.
- **pass@k:** chance that at least one of k attempts is right, computed with
  the unbiased estimator from the Codex paper rather than by literally
  sampling k.
- **The rule, carried over from Vantage:** if the interval for a difference
  includes zero, the report doesn't call it a difference.

## 3.10 Cost per correct query

```
cost per 1,000 correct = (cost per 1,000 requests) ÷ accuracy
```

Vantage measures the first part under real load; Mirage measures the second.
A model that's 10% cheaper to serve but 20% less accurate costs *more* per
correct answer. Computed with suite EX too, because "correct only on the
original data" answers aren't really correct.

---

# Part 4 — What the code is

```
src/mirage/
  data/      spider.py (loader), splits.py (database-level val split), schema.py,
             prompts.py (THE prompt), hardness.py, fixtures.py (synthetic Spider)
  sql/       extract.py, executor.py, compare.py, mutate.py, hacks.py
  reward/    perturb.py, suite.py, verifier.py, rewards.py
  train/     data.py, common.py (resume, budget, merge), sft.py, filter.py, grpo.py
  eval/      generate.py, evaluate.py, stats.py, failures.py, official.py
  analysis/  audit.py, simulate.py, report.py
  integrations/vantage.py
  config.py, store.py, demo.py, cli.py
```

The files worth reading first, in order:

1. **`sql/compare.py`** — the definition of "correct". Everything depends on it.
2. **`reward/verifier.py`** — one function used by both the reward and the
   evaluation. If they used two implementations, a bug in one would look like
   a training effect.
3. **`reward/perturb.py` + `reward/suite.py`** — how the suites are built.
4. **`reward/rewards.py`** — the TRL reward function and the ledger. Its
   signature matches TRL 1.15 (dataset columns as keyword arguments,
   `trainer_state`, `log_metric`), checked against TRL's source.
5. **`train/grpo.py`** — under 200 lines because the logic lives elsewhere.
   Read the T4 notes at the top.
6. **`analysis/audit.py`** — turns the ledger into the false-positive curve and
   the drift test.
7. **`analysis/simulate.py`** — the fake model that makes the demo possible.
   Read its docstring carefully; Part 9 explains why it must never be quoted
   as a result.

Design choices to be able to explain:

- **Generate once, score forever.** `mirage generate` (GPU) writes raw
  completions; `mirage score` (CPU, seconds) turns them into numbers. Change a
  scoring rule and you can rescore every past run without touching a GPU.
- **Failures are values.** The executor never raises on bad SQL; it returns an
  error kind that gets counted.
- **The executor is locked down.** Read-only connection, an authorizer that
  blocks writes/ATTACH/PRAGMA, a timeout, a row cap. A policy that can modify
  the database it's graded against has found the ultimate hack.
- **Content-hash ids.** Question ids are hashes of the content, so a ledger
  row from last week still joins to today's data.
- **Unknown config keys are errors.** `num_generation: 4` (typo) fails loudly
  instead of silently running the default.

---

# Part 5 — Running it

## The demo (5 minutes, no GPU)

```powershell
cd path\to\mirage
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
pytest                     # 70 tests
mirage demo                # ~10 seconds
```

Open `runs\demo\report\report.html`. You'll see the report a real run
produces, with every number about the simulator. Also look at:

- `runs/demo/suites/<db>/suite.json` — the blind spot per database;
- `runs/demo/arms/single/ledger.jsonl` — every simulated rollout, both verdicts;
- `runs/demo/eval/base/scored_items.jsonl` — every question, every failure.

## On Kaggle

[docs/KAGGLE.md](docs/KAGGLE.md) has the full walkthrough. In short: push the
repo, attach Spider, run `kaggle/01_spike.ipynb` … `05_final_eval.ipynb` in
order, carrying `runs/` forward between notebooks.

---

# Part 6 — What the results could look like

Decide now how you'll read each outcome, so you don't rationalise after the fact.

**The pre-training blind spot (notebook 01, CPU only).** This is a result
by itself, before any training. On the fixture ~32% of mutants pass the
single-database check (13% excluding DISTINCT toggles, which are often
equivalent). On real Spider it will be different. Whatever it is, it's a
number nobody can take away from you.

**Then training. Four possible stories:**

| single arm FP drift | suite arm vs single on dev suite EX | what it means |
|---|---|---|
| rises, significant | suite arm better | the clean story: the leaky reward was exploited, and fixing it helped |
| rises, significant | no difference | exploitation is real but cheap: it costs little accuracy at this scale |
| flat | no difference | the reward is noisy but not exploited by a 0.5B model in a few hundred steps |
| flat | suite arm worse | the stricter reward is sparser and slowed learning; check groups-with-signal |

Every row is reportable. The third is the most likely for a small model and
short training, and it's still a defensible finding: "measured directly, the
standard reward's error rate was X% and did not grow; at this scale it isn't
the bottleneck". What would *not* be defensible is claiming the first row from
one seed with overlapping intervals.

---

# Part 7 — What to do next, in order

1. **Read this, run the demo, read the report.** Change something (e.g.
   `--steps 40`) and watch what moves. An hour.
2. **Push to GitHub** as `mirage` (public), so Kaggle can clone it. Ten
   minutes. I didn't push it: publishing a repo under your name should be your
   call.
3. **Notebook 01 on real Spider.** Data check, suites, blind spot, model
   spike. The blind-spot number is your first real result. Half a day.
4. **Decide model and warm-up from the spike**, on val. Write the decision
   and the numbers in the repo.
5. **Notebooks 02–04.** SFT (if needed), filter, both arms. Several sessions.
   Keep an eye on "groups with signal" in the logs: if it's below ~30%, the
   filter or the starting point is wrong, and more steps won't help.
6. **Notebook 05, once.** Dev numbers for start, single, suite.
7. **A second seed per arm**, if quota allows. This is what turns "an
   interesting run" into "a result".
8. **Part D** through Vantage. Best on an hour of a rented GPU with the load
   generator on another machine. You *can* run `vllm serve` and Vantage in the
   same Kaggle session, but then the client shares a CPU with the server it's
   measuring; if you do, say so next to the numbers.
9. **Write the 600-word post.** Blind spot chart, fidelity chart, accuracy
   table, the most interesting false-positive example, and what you'd do next.

DSA and applications keep going in parallel. The project is a conversation
starter in interviews you already have.

---

# Part 8 — How to talk about it

## Resume bullets (fill the brackets with real numbers only)

> **Reward-fidelity study for RL text-to-SQL** — Trained Qwen2.5-Coder-0.5B
> with GRPO (TRL, LoRA, free T4) on Spider using execution-based rewards;
> built perturbed database suites via mutation testing and found the standard
> single-database reward accepts [X]% of near-miss queries.

> Ran a controlled two-arm comparison (single- vs multi-database reward,
> identical init/data/seed) logging both verdicts on [N]k rollouts; measured
> the training reward's false-positive rate at [Y]% with [no significant /
> a +Z-point] drift over training.

> Improved Spider dev execution accuracy from [A]% to [B]% ([CI]), and joined
> the evaluation with my serving benchmark (Vantage) to report cost per 1,000
> correct queries.

## Questions you'll be asked

**"Why GRPO and not PPO?"** — No critic network: the group average is the
baseline. Less memory, which matters on a T4, and it's the standard for
verifiable-reward training now.

**"Why not just SFT on Spider?"** — SFT teaches copying gold SQL text. RL
optimises executing correctly, which allows different-but-correct queries. I
used a short SFT warm-up only so GRPO had correct attempts to reinforce.

**"What's a false positive here?"** — A query the single-database check
rewards but the perturbed suite rejects. Example: `LIMIT 1` without `ORDER BY`
when the first row happens to be the answer.

**"How do you know the two arms differ only in the reward?"** — Both configs
inherit one base file; the report hashes each run's resolved config minus the
reward and fingerprints the starting weights, and refuses to compare if they
differ.

**"Why cluster the bootstrap by question?"** — Eight attempts at the same
question are correlated. Treating them as independent makes intervals too
narrow, so you'd call noise a finding.

**"What was the KL reference?"** — The SFT model. I merged the SFT adapter
before adding the GRPO adapter, because with LoRA, TRL's reference is "adapter
disabled", which would otherwise be the raw base.

**"Isn't your suite just test-suite accuracy?"** — Same idea, simpler
construction, and built for the training databases, which the official release
doesn't cover and the reward needs. I report the official evaluator's number
for dev alongside mine.

**"What would you do with more compute?"** — More seeds per arm first. Then a
1.5B model to see whether exploitation grows with capability, and BIRD, whose
dirtier values make coincidental matches likelier.

**"What did you get wrong?"** — A real one from the build: the first version of
the hack detector flagged *every* rewarded query as a "constant answer",
because the parser library had renamed the FROM clause's key. The audit looked
plausible until I read the examples. Lesson: read the examples, not just the
counts.

## The one-sentence pitch

> "I trained a small model to write SQL with RL and measured how often the
> reward it learns from is wrong — then compared against a stricter reward in a
> controlled two-arm experiment, and priced the result as cost per correct query
> through my serving benchmark."

---

# Part 9 — What I could not do, and what is open

Said plainly so you don't discover it in an interview.

**Nothing here has run on a GPU.** I had no GPU. What I *could* do, and did:
install the real stack (torch 2.14.1, transformers 5.19, TRL 1.15.0, PEFT
0.21.2) and run `scripts/smoke_train.py`, which pushes a ~1M-parameter random
Qwen2 model through every training and generation command: SFT, adapter merge,
pass-rate filter, both GRPO arms, resume-from-checkpoint, generation with an
adapter, scoring and the audit. It passes on CPU. That run caught one real bug
(transformers 5 removed `warmup_ratio`; fixed with a compatibility shim) and
confirmed that the reward function's audit metrics land in TRL's own logs.

What that does **not** cover:

- **fp16 on a real T4.** On CPU the tiny model trains in fp32. Mixed precision
  on the T4 is the most likely first-run surprise; the smoke test with
  `--device cuda` (first cell of notebook 01) checks it in two minutes.
- **TRL's Triton kernels.** TRL 1.15 computes the loss with a GPU-only kernel;
  on CPU the smoke test swaps in a plain-PyTorch reference
  (`train/cpu_shim.py`, only ever active on a machine without CUDA). On Kaggle
  the real kernel runs.
- **vLLM.** Not installed here, so the `vllm` generation backend and
  `grpo.use_vllm` are untested. Start with the default transformers backend.
- **Anything about speed or memory** at 0.5B. The spike measures that.

**Everything that decides whether a number is right has also run**: data,
execution, comparison, mutants, suites, verifier, reward, ledger, audit,
evaluation, statistics, report, Vantage join. 70 tests, plus the full
simulated study in CI.

**The demo is simulated, all of it.** The "model" is a softmax over a fixed
list of candidate queries per question. Its training is a real
group-relative policy gradient, and nothing about hacking is scripted in, but
its behaviour comes from how I built it. If you ever quote a demo number as a
result about language models, the whole project loses credibility. The report
says SIMULATED in its first lines for that reason.

**The demo's cost numbers come from Vantage's simulator.** I ran the real
Vantage (with the included patch) on Mirage's exported workload against its
simulated backend. Real numbers need a real GPU serving the models.

**I did not download Spider.** The workspace couldn't reach it. Everything
is tested on the synthetic fixture, which is shaped like Spider. Check the
Spider download source and licence yourself (it's CC BY-SA 4.0 per its Hugging
Face card).

**Suites are an approximation** of the official test suites. The official
evaluator hook is there; it's untested here because its databases are a
Google Drive download.

**Difficulty labels are recomputed**, not official.

**Kaggle quota and session limits** — I don't know your current numbers.
Check them before planning runs.

**I didn't push to GitHub.** Your call.

## If something breaks

1. `pytest` first. If it passes, the measurement code is fine and the problem
   is configuration or the GPU side. Then `python scripts/smoke_train.py
   --device cuda`: if that passes, the libraries are fine too.
2. `mirage data check $SPIDER_DIR` — most Spider problems show up here.
3. GRPO out of memory: halve `per_device_train_batch_size`, double
   `gradient_accumulation_steps`.
4. A TRL error naming an argument: the library changed. Check its current
   `GRPOConfig` and adjust `train/grpo.py`.

Bring the error and the run folder next session and we'll go through it.
