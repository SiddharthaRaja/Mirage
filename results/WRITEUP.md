# Does the SQL reward lie? Reward fidelity in GRPO text-to-SQL on a free GPU

*Seed 0 of 2. Qwen2.5-Coder-0.5B-Instruct, LoRA, one Kaggle T4. Data and code: this repository.*

## The question

Text-to-SQL is a favourite task for reinforcement learning because the reward looks free: run the
model's query and the reference query on the database and pay 1 if the rows match. On Spider's
small databases that check can be fooled. A query with `>` instead of `>=`, or a missing join, often
returns the right rows by coincidence. This project measures how often the reward pays for wrong SQL,
whether a policy trained on it learns to exploit that, and whether a stricter reward helps.

## Setup

- **Data.** Spider: 8,651 training questions over 146 databases (7,106 after holding out a
  validation split by database), and the 1,034-question dev set on 20 databases never used in
  training. Dev was evaluated once.
- **Two rewards.** *Single*: match on the original database (the standard). *Suite*: match on the
  original **and** on 2–4 perturbed copies of it, chosen to break near-miss queries.
- **Two arms**, identical except for the reward: same 925 training questions, same starting
  weights, seed and 300 GRPO steps (8 questions × 8 attempts per step, LoRA r = 16, β = 0.04).
- **A ledger** scores every training attempt with both checks, so the standard reward's errors are
  measured while training on it.

## Result 1: before training, the standard check is easy to fool

For every reference query, ten mutation operators produced 23,585 near-miss variants. The original
database accepted 49.4% of them. That number is inflated by 7 databases whose tables are empty, so
every query returns nothing and every mutant "matches"; without them the check accepts **33.2%**
of near misses, and the perturbed copies prove at least **10.8% of all mutants** wrong but rewarded.
Some accepted mutants are genuinely equivalent (adding `DISTINCT` to a unique column), so the
true rate lies between those two numbers.

![blind spot by mutation operator](run2/report/charts/blind_spot.png)

## Result 2: during training, the reward lied, but the model did not learn to exploit it

Of the attempts the standard reward paid for, the suite rejected **4.0%** [2.9, 5.2] in the single
arm and 3.4% [2.5, 4.6] in the suite arm. Between the first and last quarter of training that share
moved by −0.1 and −0.5 points, with intervals spanning zero. Over 300 steps, a 0.5B policy did not
drift toward the loophole.

![false-positive share over training](run2/report/charts/fidelity.png)

Reading the cases behind that 4% changed its meaning. Of 27 such answers on dev, 14 are genuinely
wrong (a `min()` on the wrong column, a `GROUP BY` that never joins the table it should count) and 13
are defensible answers the suite punishes: returning every tied row instead of one arbitrary row, or
excluding students by id rather than by name. Suite-flagged false positives are about half real.

The larger leak was elsewhere. **20% of all rewards** went to questions whose correct answer is
empty: on those, any query that runs and returns nothing earns full reward. 92% of them come from the
7 empty databases, and the pass-rate filter actively selected those questions, because attempts that
fail to execute make the group's rewards disagree. The suite cannot see this (a perturbed empty table
is still empty). Examples of paid answers are in the [gallery](GALLERY.md).

## Result 3: training still worked, and the stricter reward did not matter

| model | dev EX | suite EX | pass@1 | pass@8 |
|---|---|---|---|---|
| base | 42.0% [39.1, 44.8] | 40.1% | 33.3% | 65.2% |
| GRPO, single reward | **56.8%** [54.0, 59.8] | 54.2% | 52.8% | 69.7% |
| GRPO, suite reward | 55.6% [52.5, 58.6] | 53.1% | 52.2% | 70.1% |

GRPO improved execution accuracy on unseen databases by **+14.8 points** [12.0, 17.6] (207 questions
fixed, 54 broken, McNemar p < 0.001). The suite arm gained +13.6. The difference between arms,
−1.2 [−2.6, 0.3], is not distinguishable from zero. Most of the gain is schema linking: wrong table or
column names fell from 410 to 193 errors. pass@8 moved far less than pass@1 (+4.5 vs +19.5 points):
RL mostly made the model reliably produce answers it could already reach sometimes.

![accuracy](run2/report/charts/accuracy.png)

## What this does and does not show

- At this scale the standard execution reward is good enough: it lies on a few percent of rewarded
  attempts, the policy does not exploit them, and a stricter reward buys nothing measurable.
- The serious problem is not near-miss SQL but **degenerate data**: empty result sets reward any
  query that runs. Dropping empty-gold questions before filtering is the obvious fix and the next run.
- One seed. A second seed of both arms is the next run; an arm difference of 1.2 points cannot be
  separated from seed noise with one pair.
- The suites are an approximation of the official distilled test suites, and as the gallery shows they
  are sometimes stricter than the question. The official test-suite evaluator number is still to come.
- A 0.5B model trained for 300 steps is small and short. Exploitation may appear with larger models,
  longer training or a leakier reward; the ledger and audit are built to detect it if it does.
- Spider is old and its schemas are small; BIRD-style dirty data would stress the reward more.
