# When the reward lies: a gallery from real runs

Run 2, seed 0. Everything below comes from `results/run2/`: the trained model's answers on Spider
dev (`eval/grpo-single-dev/scored_items.jsonl`) and the training ledgers (`arms/*/ledger.jsonl.gz`).
Each dev case was re-executed on the perturbed suite databases (`runs/suites/<db>/<db>__s*.sqlite`)
to see what actually differs.

Two findings, in order of importance:

1. **One in five training rewards came from questions whose correct answer is empty.** On those,
   *any* query that runs and returns nothing is paid. 92% of them come from the 7 Spider databases
   whose tables have no rows. The suite cannot see this hole (98.6% of those rollouts also pass it).
2. **A "false positive" flagged by the suite is wrong only about half the time.** Of the 27 dev
   answers the suite rejected after the original database accepted them, 14 are genuinely wrong
   queries that got lucky; 13 are defensible answers the suite punishes for ties or duplicate names.

## 1. The empty-table hole (training ledgers)

| | single arm | suite arm |
|---|---|---|
| rollouts | 19,200 | 19,200 |
| rewarded by the single-DB check | 13,397 | 13,290 |
| of those, gold returns **no rows** | **20.3%** (2,725) | **20.8%** |
| of those, also passed by the suite | 98.6% | 98.7% |
| suite-flagged false positives (the 4.0% / 3.4%) | 530 | 453 |

- 2,509 of the 2,725 empty-gold rewards come from geo (1,462), imdb, yelp, restaurants, academic and
  music_2: databases shipped with empty tables. 198 of the 925 training questions (21%) are from them.
- The pass-rate filter *selected* these questions: some attempts fail to execute, so the 8 attempts
  disagree, which is exactly what the filter keeps. On them the reward means "write SQL that runs".
- Rewarded examples (all paid 1.0):

```sql
-- geo: gold returns nothing, so does this
SELECT state_name FROM state WHERE capital = (SELECT highest_point FROM highlow
  WHERE state_name = (SELECT state_name FROM state WHERE capital = (SELECT highest_point FROM highlow)))
-- academic: searching titles for an author's name
SELECT title FROM publication WHERE title LIKE '%H. V. Jagadish%'
```

- Why the measured false-positive share (4%) misses it: a suite variant is a perturbed copy of an
  empty database, which is still empty, so gold and the rollout still both return nothing.
- Why dev accuracy still rose: none of the 20 dev databases is empty, and the other 79% of rewards
  were informative. The hole wasted roughly a fifth of the training signal rather than reversing it.
- Fix for future runs: drop questions whose gold result is empty (or the 7 databases) before the
  pass-rate filter. Seed 1 keeps the seed-0 setup so the two seeds replicate each other; the clean
  variant is a separate experiment.

## 2. Dev false positives, classified (GRPO single arm, greedy)

27 of the 587 EX-correct answers fail suite EX (base model: 19 of 434; suite arm: 26 of 575).

| class | count | what it is | is the answer wrong? |
|---|---|---|---|
| A. right by coincidence | 14 | wrong logic that happens to give gold's rows on the original data | yes: the reward lied |
| B. ties | 6 | `WHERE x = (SELECT MAX(x) ...)` vs gold's `ORDER BY x DESC LIMIT 1` | no: returns every tied row; gold returns one arbitrary row |
| C. duplicates / id vs name | 7 | `NOT IN` by id vs gold's `EXCEPT` on names; `DISTINCT` toggles | usually no: differs only when two rows share a name |

So the honest reading of "4% false positives" is: about half are real lies, about half are the
suite being stricter than the question. Suite EX is a better check than EX, not ground truth.

### A. Right by coincidence (the reward lied)

**A1 · battle_death — wrong column.** *What is the maximum and minimum death toll caused each time?*
```sql
gold: SELECT max(killed), min(killed) FROM death
pred: SELECT max(killed), min(injured) FROM death
```
On the original data the smallest `killed` and smallest `injured` are both 0. A perturbed copy gives
gold (29, 0) and the prediction (29, −2).

**A2 · flight_2 — no join, right by row order.** *Which airline has the most flights?*
```sql
gold: SELECT T1.Airline FROM airlines T1 JOIN flights T2 ON T1.uid = T2.Airline
      GROUP BY T1.Airline ORDER BY count(*) DESC LIMIT 1
pred: SELECT Airline FROM airlines GROUP BY Airline ORDER BY COUNT(*) DESC LIMIT 1
```
The prediction never looks at flights: every airline counts 1, and `LIMIT 1` returns whichever row
comes first, which on the original file is United Airlines. Reordered copy: United vs JetBlue.

**A3 · world_1 — wrong ordering column.** *Name, population and life expectancy of the largest Asian
country by land?* The prediction orders by `Population` instead of `SurfaceArea`. On real data the
answer is the same country either way; on a perturbed copy it is not.

**A4 · world_1 — counts the wrong thing.** *Which continent has the most diverse languages?* The
prediction counts countries per continent; gold counts country–language pairs. Same winner on the
original data, different on a perturbed copy (Europe vs Africa). Appears twice (paraphrased question).

**A5 · car_1 — OR instead of AND.** *Years with cars weighing less than 4000 and more than 3000?*
`WHERE Weight < 4000 OR Weight > 3000` is true for every car. Original data: same years as
`BETWEEN 3000 AND 4000`. Perturbed copy: 19 years vs 17.

**A6 · pets_1 — empty match.** *Students who have both a cat and a dog?* No student has both in the
original database, so gold returns nothing, and the prediction (`PetType = 'Cat'`, capitalised, which
matches no row) also returns nothing. The suite plants a student with both: gold returns Tracy, the
prediction still nothing.

Also in class A: museum_visit (sums a visitor's tickets instead of the largest single visit),
student_transcripts_tracking (counts degree programmes, never joins enrolments), world_1 (drops the
continent filter in the outer query), and concert_singer ×2 (adds an unasked condition: the singer
must also appear in a concert).

### B. Ties: the suite punishes the better answer

**B1 · concert_singer.** *Name and capacity of the stadium with the highest average attendance?*
```sql
gold: SELECT name, capacity FROM stadium ORDER BY average DESC LIMIT 1
pred: SELECT Name, Capacity FROM stadium WHERE Average = (SELECT MAX(Average) FROM stadium)
```
The suite plants a tie: gold returns one of the two tied stadiums, the prediction returns both.
The question does not say what to do with ties; the prediction is at least as correct.

### C. Duplicates: id vs name

**C1 · network_1.** *Names of students who have no friends?* Gold removes by *name* (`EXCEPT` on
names); the prediction removes by *id* (`WHERE ID NOT IN (SELECT student_id FROM Friend)`). With two
students both named Gabriel, one with friends and one without, gold drops both; the prediction keeps
the friendless one. The prediction is the correct one.

## What this changes in the write-up

- Report the false-positive share as "suite-flagged", with the class A share alongside: about half.
- Report the empty-gold share of rewards (20%) as the larger, separate leak, invisible to the suite.
- EX − suite EX (≈2.6 points for the trained models) overstates "right for the wrong reason" by
  roughly half, for the same reason.
- None of this changes the dev result: +14.8 EX on databases that are not empty.
