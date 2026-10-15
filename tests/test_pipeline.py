import json
from pathlib import Path

import pytest

from mirage.config import load_config, parse_overrides
from mirage.errors import ConfigError
from mirage.integrations.vantage import cost_per_correct, export_workload
from mirage.store import arms_comparable

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


# ------------------------------------------------------------------- config


def test_arms_differ_only_in_reward():
    a = load_config(CONFIGS / "grpo_single.yaml").model_dump()
    b = load_config(CONFIGS / "grpo_suite.yaml").model_dump()
    diff = {k for k in a if a[k] != b[k]}
    assert diff == {"name", "out", "reward"}
    assert {k for k in a["reward"] if a["reward"][k] != b["reward"][k]} == {"mode"}
    ok, problems = arms_comparable([{"config": a, "init_fingerprint": "x"}, {"config": b, "init_fingerprint": "x"}])
    assert ok, problems
    ok, _ = arms_comparable([{"config": a, "init_fingerprint": "x"}, {"config": b, "init_fingerprint": "y"}])
    assert not ok


def test_every_shipped_config_loads():
    for p in CONFIGS.rglob("*.yaml"):
        if p.name.startswith("study"):
            continue
        load_config(p)


def test_unknown_keys_are_errors(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("name: x\nspider: s\nout: o\ngrpo: {num_generation: 4}\n")
    with pytest.raises(ConfigError):
        load_config(p)


def test_overrides_and_env(tmp_path, monkeypatch):
    monkeypatch.setenv("SPIDER_DIR", "/data/spider")
    c = load_config(CONFIGS / "grpo_single.yaml", parse_overrides(["grpo.beta=0.1", "reward.shape=shaped"]))
    assert c.spider == "/data/spider" and c.grpo.beta == 0.1 and c.reward.shape == "shaped"


def test_t4_defaults():
    c = load_config(CONFIGS / "base.yaml")
    assert c.model.dtype == "float16"
    assert c.grpo.beta > 0  # TRL's default is 0.0; this project sets the KL term on purpose
    assert (c.grpo.per_device_train_batch_size * c.grpo.gradient_accumulation_steps) % c.grpo.num_generations == 0


def test_suite_arm_refuses_to_train_without_suites(spider, tmp_path):
    from mirage.train.grpo import build_reward

    cfg = load_config(CONFIGS / "grpo_suite.yaml", {"suites": str(tmp_path / "none")})
    with pytest.raises(ConfigError):
        build_reward(cfg, {"concert_hall": str(spider.db_path("concert_hall"))}, None)


# ------------------------------------------------------------------- vantage (part D)


def test_vantage_workload_carries_the_exact_prompt(spider, db_paths, tmp_path):
    from mirage.data.prompts import USER_TEMPLATE

    dev = spider.load("dev")
    p = export_workload(dev, db_paths, tmp_path / "w.jsonl")
    rows = [json.loads(x) for x in p.read_text().splitlines()]
    assert len(rows) == len(dev)
    assert rows[0]["prompt"].startswith(USER_TEMPLATE.split("{")[0])
    assert {"id", "question", "context", "prompt", "reference_answer"} <= rows[0].keys()


def test_cost_per_correct_arithmetic(tmp_path):
    (tmp_path / "summaries.json").write_text(json.dumps([
        {"config_id": "m[c=8]", "achieved_qps": 10.0, "cost": {"usd_per_1k_requests": 0.02},
         "total": {"p95": 0.5}, "ttft": {"p95": 0.1}, "goodput_ratio": 1.0},
    ]))
    scores = {"overall": {"ex": {"mean": 0.5, "ci95": (0.4, 0.6)}, "suite_ex": {"mean": 0.4, "ci95": (0.3, 0.5)}}}
    (row,) = cost_per_correct(tmp_path, {"m": {"vantage": "m[c=8]", "scores": scores}})
    assert row["usd_per_1k_correct_ex"] == pytest.approx(0.04)
    assert row["usd_per_1k_correct_suite_ex"] == pytest.approx(0.05)
    lo, hi = row["usd_per_1k_correct_suite_ex_ci95"]
    assert lo == pytest.approx(0.04) and hi == pytest.approx(0.02 / 0.3)
    assert row["correct_per_gpu_hour_suite_ex"] == pytest.approx(10 * 3600 * 0.4)


# ------------------------------------------------------------------- end to end


def test_demo_end_to_end(tmp_path):
    from mirage.demo import run_demo

    rep = run_demo(tmp_path / "demo", steps=30, log=lambda *_: None)
    md = (rep / "REPORT.md").read_text()
    assert "SIMULATED" in md.splitlines()[2]
    assert "What this run does not establish" in md
    data = json.loads((rep / "report.json").read_text())
    assert data["arms_comparable"], data["arm_problems"]
    assert set(data["audits"]) == {"single", "suite"}
    assert {"base", "grpo-single", "grpo-suite"} == set(data["evals"])
    assert (rep / "report.html").stat().st_size > 50_000  # charts embedded
    assert (rep / "charts" / "fidelity.png").exists()
