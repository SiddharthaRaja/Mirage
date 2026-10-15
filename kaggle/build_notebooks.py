"""Generate the Kaggle notebooks from one source of truth.

    python kaggle/build_notebooks.py

Each notebook is a thin shell around `mirage` CLI commands, so the logic
lives in the tested package and the notebooks never drift from it. They
assume:

* Accelerator: GPU T4 x2 (or P100). Internet: ON (to pip install and pull the
  model from the Hub).
* Spider attached as a Kaggle dataset, or uploaded by you. Set SPIDER_DIR in
  the first cell to the folder that contains train_spider.json, dev.json,
  tables.json and database/.
* Outputs go to /kaggle/working/runs, which Kaggle keeps when you "Save
  Version". Add a previous notebook's output as an input to continue from it.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).parent
REPO = "https://github.com/SiddharthaRaja/mirage"

SETUP = f"""\
# --- setup: run once per session -------------------------------------------
import os, subprocess, sys
SPIDER_DIR = "/kaggle/input/spider/spider"   # <- EDIT: the folder with train_spider.json and database/
RUNS = "/kaggle/working/runs"
os.environ["SPIDER_DIR"], os.environ["MIRAGE_RUNS"] = SPIDER_DIR, RUNS
os.chdir("/kaggle/working")
if not os.path.exists("mirage"):
    subprocess.run(["git", "clone", "--depth", "1", "{REPO}"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", "mirage[train]"], check=True)
os.chdir("/kaggle/working/mirage")
!nvidia-smi --query-gpu=name,memory.total --format=csv
"""

PREV = """\
# If you are continuing from an earlier notebook, copy its runs folder in:
# !cp -r /kaggle/input/<your-previous-notebook-output>/runs/* /kaggle/working/runs/ 2>/dev/null || true
"""

NOTEBOOKS = {
    "01_spike.ipynb": [
        ("md", "# 01 · Feasibility spike\n\nCheck the data, build the suites, measure the reward's blind spot "
               "**before any training**, and baseline candidate models. Everything here is cheap; it decides "
               "the rest of the plan."),
        ("code", SETUP),
        ("md", "## Two-minute smoke test\nRuns every TRL/PEFT call in the training code on a tiny random model. "
               "If a library update broke something, you find out now, not two hours into a run."),
        ("code", "!python scripts/smoke_train.py --device cuda --out /kaggle/working/smoke"),
        ("code", "!mirage data check $SPIDER_DIR"),
        ("md", "## The blind spot, before training (CPU only)\nBuilding suites for all ~200 Spider databases takes "
               "a while on CPU; start with a subset to see the numbers, then build all of them once."),
        ("code", "!mirage suites build --spider $SPIDER_DIR --out $MIRAGE_RUNS/suites --only concert_singer,pets_1,car_1,flight_2,world_1"),
        ("code", "# all databases (train + dev); run once, keep the output\n!mirage suites build --spider $SPIDER_DIR --out $MIRAGE_RUNS/suites"),
        ("md", "## Baseline the candidates on a val slice\nWant: EX roughly 15–50%, and many questions where the "
               "8 attempts disagree ('mixed groups'). Near 0% means an SFT warm-up first."),
        ("code", "!mirage spike -c configs/spike.yaml --n 100"),
        ("md", "Write the decision down: which model, and whether to warm up. Then edit `configs/base.yaml` "
               "(`model.name`) and, if skipping SFT, remove `init_adapter` from `configs/grpo_base.yaml`."),
    ],
    "02_sft_and_filter.ipynb": [
        ("md", "# 02 · SFT warm-up (optional) and pass-rate filter\n\nSkip the SFT cell if the spike baseline already "
               "had enough mixed groups."),
        ("code", SETUP),
        ("code", PREV),
        ("code", "!mirage sft -c configs/sft.yaml"),
        ("md", "Check the warm-up on **val** (never dev):"),
        ("code", "!mirage generate -c configs/grpo_base.yaml --split val --label start\n"
                 "!mirage score $MIRAGE_RUNS/base/eval/start-val/predictions.jsonl --spider $SPIDER_DIR --suites $MIRAGE_RUNS/suites"),
        ("md", "## Pass-rate filter\nKeep training questions where the starting model's attempts disagree under "
               "either reward. Both arms use this same file."),
        ("code", "!mirage filter -c configs/grpo_base.yaml --k 8"),
    ],
    "03_grpo_single.ipynb": [
        ("md", "# 03 · GRPO arm A: single-database reward\n\nCheckpoints every 25 steps and stops itself before the "
               "session limit. Re-run this notebook (with its previous output attached) to resume."),
        ("code", SETUP),
        ("code", PREV),
        ("code", "!mirage grpo -c configs/grpo_single.yaml"),
        ("code", "!mirage audit $MIRAGE_RUNS/arms/single/ledger.jsonl"),
    ],
    "04_grpo_suite.ipynb": [
        ("md", "# 04 · GRPO arm B: suite reward\n\nIdentical to arm A except the reward. `mirage report` will refuse "
               "to compare the two if anything else differs."),
        ("code", SETUP),
        ("code", PREV),
        ("code", "!mirage grpo -c configs/grpo_suite.yaml"),
        ("code", "!mirage audit $MIRAGE_RUNS/arms/suite/ledger.jsonl"),
    ],
    "05_final_eval.ipynb": [
        ("md", "# 05 · Final evaluation on Spider dev\n\n**This is the only notebook that touches dev.** Run it once "
               "per model, after every decision has been made on val."),
        ("code", SETUP),
        ("code", PREV),
        ("code", "!mirage generate -c configs/grpo_base.yaml --split dev --label start --out $MIRAGE_RUNS/eval/start-dev\n"
                 "!mirage generate -c configs/grpo_base.yaml --split dev --adapter $MIRAGE_RUNS/arms/single/adapter --label grpo-single --out $MIRAGE_RUNS/eval/grpo-single-dev\n"
                 "!mirage generate -c configs/grpo_base.yaml --split dev --adapter $MIRAGE_RUNS/arms/suite/adapter --label grpo-suite --out $MIRAGE_RUNS/eval/grpo-suite-dev"),
        ("code", "for name in ['start', 'grpo-single', 'grpo-suite']:\n"
                 "    !mirage score $MIRAGE_RUNS/eval/{name}-dev/predictions.jsonl --spider $SPIDER_DIR --suites $MIRAGE_RUNS/suites --label {name}"),
        ("md", "Optional: official test-suite accuracy. Clone `taoyds/test-suite-sql-eval`, download its test-suite "
               "databases, and pass `--official <repo> --official-db <databases>` to `mirage score`."),
        ("code", "!cp configs/study.example.yaml $MIRAGE_RUNS/../study.yaml && sed -i 's#\\.\\./runs#runs#g' $MIRAGE_RUNS/../study.yaml\n"
                 "!cd $MIRAGE_RUNS/.. && mirage report study.yaml"),
    ],
}


def build() -> list[Path]:
    out = []
    for name, cells in NOTEBOOKS.items():
        nb = nbf.v4.new_notebook()
        nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
        nb.cells = [nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src) for kind, src in cells]
        p = HERE / name
        nbf.write(nb, p)
        out.append(p)
    return out


if __name__ == "__main__":
    for p in build():
        print("wrote", p)
