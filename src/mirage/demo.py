"""`mirage demo`: the whole study, end to end, with no GPU and no downloads.

Every stage is the real code path except the policy, which is the
candidate-pool simulator in `analysis/simulate.py`:

1. build the synthetic Spider-shaped fixture;
2. build perturbed suites for every database and measure the blind spot;
3. split train/val by database; dev is the fixture's held-out databases;
4. train two arms (single-db reward, suite reward) from the same start, with
   the real `ExecutionReward` writing the real ledger;
5. generate + score base, single and suite on dev through the real evaluation;
6. join with a Vantage run for cost per correct query (a bundled Vantage
   *simulator* run unless you pass a real one);
7. write the report.

The report is stamped SIMULATED in its first lines.
"""

from __future__ import annotations

import json
import shutil
from importlib import resources
from pathlib import Path

import yaml

from mirage.analysis.report import build_report
from mirage.analysis.simulate import TabularPolicy, train_tabular_grpo
from mirage.data.fixtures import build_mini_spider
from mirage.data.spider import Spider
from mirage.data.splits import split_by_database
from mirage.eval.evaluate import generate_predictions, save_scores, score_predictions
from mirage.integrations.vantage import cost_per_correct, export_workload
from mirage.reward.rewards import ExecutionReward, Ledger, RewardConfig
from mirage.reward.suite import SuiteIndex, build_suite
from mirage.reward.verifier import Verifier
from mirage.store import write_manifest


def run_demo(out: str | Path = "runs/demo", steps: int = 150, seed: int = 0, vantage_run: str | None = None, log=print) -> Path:
    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    log("[1/7] building the synthetic fixture")
    root = build_mini_spider(out / "mini_spider")
    sp = Spider(root)

    log("[2/7] building perturbed suites")
    pool = sp.load("train")
    dev = sp.load("dev")
    by_db: dict[str, list[str]] = {}
    for e in pool + dev:
        by_db.setdefault(e.db_id, []).append(e.gold_sql)
    for db, qs in by_db.items():
        build_suite(db, sp.db_path(db), qs, out / "suites", seed=seed + 1)
    suites = SuiteIndex(out / "suites")

    log("[3/7] splitting by database")
    train, val = split_by_database(pool, val_fraction=0.2)
    paths = {d: str(sp.db_path(d)) for d in by_db}
    everything = train + val + dev
    base = TabularPolicy(everything, seed=seed)

    shared_cfg = {
        "policy": "simulated candidate-pool softmax",
        "steps": steps,
        "questions_per_step": 8,
        "group_size": 8,
        "lr": 0.04,
        "beta": 0.02,
        "seed": seed,
        "n_train": len(train),
    }
    policies = {}
    for arm in ("single", "suite"):
        log(f"[4/7] training the {arm} arm ({steps} steps)")
        arm_dir = out / "arms" / arm
        arm_dir.mkdir(parents=True)
        write_manifest(
            arm_dir,
            "grpo-simulated",
            {**shared_cfg, "name": arm, "reward": {"mode": arm, "shape": "binary"}},
            init_fingerprint=f"sim:{seed}",
        )
        reward = ExecutionReward(
            Verifier(paths, suites), RewardConfig(mode=arm), Ledger(arm_dir / "ledger.jsonl"), arm=arm
        )
        policies[arm] = train_tabular_grpo(base.copy(), train, reward, steps=steps, seed=seed)
        (arm_dir / "weights.json").write_text(json.dumps(policies[arm].weights, indent=1), encoding="utf-8")

    log("[5/7] evaluating base, single and suite on dev")
    evals = {}
    for name, pol in (("base", base), ("grpo-single", policies["single"]), ("grpo-suite", policies["suite"])):
        d = out / "eval" / name
        pred = generate_predictions(dev, pol, paths, d / "predictions.jsonl", n_samples=8, temperature=1.0)
        scores = score_predictions(pred, Verifier(paths, suites), label=name)
        save_scores(scores, d)
        evals[name] = scores

    log("[6/7] joining with a Vantage run for cost per correct query")
    export_workload(dev, paths, out / "vantage" / "workload.mirage-dev.jsonl")
    if vantage_run:
        vdir = Path(vantage_run)
    else:
        vdir = out / "vantage" / "bundled-sim-run"
        vdir.mkdir(parents=True, exist_ok=True)
        src = resources.files("mirage").joinpath("assets/demo_vantage_summaries.json")
        (vdir / "summaries.json").write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    mapping = json.loads(resources.files("mirage").joinpath("assets/demo_vantage_mapping.json").read_text(encoding="utf-8"))
    rows = cost_per_correct(vdir, {name: {"vantage": mapping[name], "scores": evals[name]} for name in evals})
    (out / "vantage" / "cost.json").write_text(json.dumps(rows, indent=1, default=list), encoding="utf-8")

    log("[7/7] writing the report")
    study = {
        "title": "Mirage demo: reward fidelity on the synthetic fixture",
        "simulated": True,
        "eval_set": f"fixture dev ({len(dev)} questions, {len({e.db_id for e in dev})} unseen databases)",
        "suites": "suites",
        "arms": {"single": "arms/single", "suite": "arms/suite"},
        "evals": {"base": "eval/base", "grpo-single": "eval/grpo-single", "grpo-suite": "eval/grpo-suite"},
        "cost": "vantage/cost.json",
        "out": "report",
    }
    (out / "study.yaml").write_text(yaml.safe_dump(study, sort_keys=False), encoding="utf-8")
    return build_report(out / "study.yaml")
