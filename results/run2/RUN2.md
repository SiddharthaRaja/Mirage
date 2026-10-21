# Run 2 summary

ran 8.92 h · commit 152ddd2 Record 01 results: blind spot on 5 DBs, model spike; skip SFT

## Stage status
{
  "suites": "ok",
  "filter": "ok",
  "grpo_single": "ok",
  "grpo_suite": "ok",
  "audit_single": "ok",
  "audit_suite": "ok"
}

## Blind spot (all databases)
blind spot: original DB accepts 49.2% of mutants; suites catch 16.6% of those. 
all: 164 DBs, 23585 mutants: original accepts 49.4%; suites catch 16.5% of those; provably wrong-but-rewarded >= 8.2% of all mutants
excluding 7 degenerate DBs (academic, geo, imdb, music_2, restaurants, scholar, yelp): 157 DBs, 17881 mutants: original accepts 33.2%; suites catch 32.4% of those; provably wrong-but-rewarded >= 10.8% of all mutants

## Filter
{
  "n": 1500,
  "kept": 925,
  "all_wrong": 468,
  "all_right_single": 113,
  "mean_pass_single": 0.3335,
  "mean_pass_suite": 0.3160833333333333,
  "subset_of": 7106,
  "k": 8,
  "subset_rule": "sha256('mirage-filter-v1:'+qid), first N"
}

## Training reached
single: step 300 · suite: step 300  (target 300, seed 0; if the two differ, resume the shorter arm before comparing)

## Audit
grpo-single: 19200 rollouts, match single 69.8%, suite 67.0%, false-positive 
share 4.0% [2.9%, 5.2%]; drift -0.1 pts (not significant); groups with signal 
49.1%

grpo-suite: 19200 rollouts, match single 69.2%, suite 66.9%, false-positive 
share 3.4% [2.5%, 4.6%]; drift -0.5 pts (not significant); groups with signal 
47.8%

