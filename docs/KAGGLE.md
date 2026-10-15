# Running Mirage on free Kaggle

Everything that needs a GPU runs in five notebooks under `kaggle/`. Each is a
thin shell around `mirage` commands, so the logic is the tested package, not
notebook cells.

## Before you spend any GPU time

1. **Check your quota.** Kaggle's free GPU allowance is weekly and sessions
   have a maximum length; both have changed over time, so read the current
   numbers on your account page. `budget.max_hours` in `configs/base.yaml`
   must be below your session limit.
2. **Get Spider.** Download Spider 1.0 from its project page (it is CC BY-SA
   4.0) and upload it as a private Kaggle dataset, or attach an existing public
   Kaggle copy after checking it has `train_spider.json`, `train_others.json`,
   `dev.json`, `tables.json` and `database/<db_id>/<db_id>.sqlite`. Set
   `SPIDER_DIR` in the first notebook cell to that folder.
3. **Push this repo to GitHub** so the notebooks can clone it (the setup cell
   clones `github.com/SiddharthaRaja/mirage`; edit `REPO` in
   `kaggle/build_notebooks.py` and rerun it if the name differs).
4. **Notebook settings:** Accelerator *GPU T4 x2* (Mirage uses one), Internet
   *On* (pip and the model download).

## The order

| notebook | GPU time (rough) | what it decides |
|---|---|---|
| `01_spike` | under 1 h | which model; whether to warm up. Also builds suites and prints the blind spot (CPU). |
| `02_sft_and_filter` | 1–2 h | warm-up adapter (optional); which training questions carry signal |
| `03_grpo_single` | one or more sessions | arm A |
| `04_grpo_suite` | one or more sessions | arm B, same steps as A |
| `05_final_eval` | 1–2 h | the dev numbers, once |

The times are planning guesses, not measurements. The spike notebook measures
seconds per question; multiply it out before starting a long run.

## Carrying state between notebooks

Outputs are written to `/kaggle/working/runs`. When a notebook finishes,
**Save Version** keeps that folder. In the next notebook, *Add Input* → your
previous notebook's output, and uncomment the `cp` line in the second cell.
GRPO resumes from the newest `checkpoint-*` it finds in its output folder, and
trims its ledger to that step so the audit never counts a step twice.

## Things that will probably need a nudge on first run

- **Run the smoke test first** (`python scripts/smoke_train.py --device
  cuda`, the first cell after setup in notebook 01). It passed on CPU with
  torch 2.14 / transformers 5.19 / TRL 1.15.0 / PEFT 0.21.2; Kaggle's image may
  pair different versions, and fp16 on the T4 has not been exercised yet.
- **vLLM on a T4** is optional (`eval.backend: vllm`, `grpo.use_vllm: true`).
  Start with the transformers backend; switch only after a small test.
- **Memory.** If GRPO runs out of memory, halve
  `grpo.per_device_train_batch_size` and double
  `gradient_accumulation_steps` (the product must stay divisible by
  `num_generations`), or lower `max_completion_length`.

## Writing it up

Copy `configs/study.example.yaml`, point it at your runs, and run
`mirage report study.yaml`. Set `simulated: false`. Keep the caveats section.
