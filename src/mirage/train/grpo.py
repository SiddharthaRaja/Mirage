"""GRPO training, one arm per run.

    mirage grpo -c configs/grpo_single.yaml
    mirage grpo -c configs/grpo_suite.yaml

The two configs `extend` one base file and differ in `reward.mode` only; the
report checks this from the manifests before comparing them.

Per step, TRL samples `num_generations` completions per question, calls
`ExecutionReward`, normalises rewards within each group (GRPO's baseline),
and updates the LoRA weights. The reward function also writes every rollout
to `ledger.jsonl` with both verdicts, which `mirage audit` reads.

T4 specifics, all set explicitly because the defaults are wrong for it:

* `fp16=True, bf16=False` — TRL's GRPOConfig defaults to bf16, which the
  T4 does not support.
* `beta` is set explicitly (TRL 1.15 defaults to 0.0, i.e. no KL penalty and
  no reference model). With PEFT, the reference is the model with adapters
  disabled; `common.resolve_start` makes sure that is the SFT model.
* Generation uses transformers by default. vLLM colocate mode can be switched
  on with `grpo.use_vllm: true`; check memory on a 16 GB T4 before trusting it.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from mirage.config import ExperimentCfg
from mirage.errors import ConfigError, MissingDependency
from mirage.reward.rewards import ExecutionReward, Ledger, RewardConfig
from mirage.reward.suite import SuiteIndex
from mirage.reward.verifier import Verifier
from mirage.store import write_manifest
from mirage.train.common import (
    fingerprint,
    latest_checkpoint,
    make_budget_callback,
    resolve_start,
    trim_ledger,
)
from mirage.train.data import grpo_records, load_sets, prompt_spec


def build_reward(cfg: ExperimentCfg, db_paths: dict[str, str], ledger_path: Path | None) -> ExecutionReward:
    suites = SuiteIndex(cfg.suites)
    needed = set(db_paths)
    if cfg.reward.mode == "suite":
        missing = sorted(needed - {d for d in needed if d in suites})
        if missing:
            raise ConfigError(
                f"reward.mode=suite but {len(missing)} training databases have no suite "
                f"(e.g. {missing[:5]}). Run `mirage suites build` first."
            )
    elif not len(suites):
        print("[grpo] warning: no suites found, so the audit cannot measure false positives")
    verifier = Verifier(db_paths, suites)
    rc = RewardConfig(**cfg.reward.model_dump())
    return ExecutionReward(verifier, rc, Ledger(ledger_path), arm=cfg.name)


def run_grpo(cfg: ExperimentCfg) -> Path:
    try:
        from datasets import Dataset
        from peft import LoraConfig
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as e:
        raise MissingDependency("trl") from e

    from mirage.train.cpu_shim import maybe_install

    maybe_install()
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    g = cfg.grpo
    if (g.per_device_train_batch_size * g.gradient_accumulation_steps) % g.num_generations:
        raise ConfigError(
            "per_device_train_batch_size * gradient_accumulation_steps must be divisible by num_generations"
        )

    sets = load_sets(cfg)
    train_paths = {e.db_id: str(sets.spider.db_path(e.db_id)) for e in sets.train}
    records = grpo_records(sets.train, train_paths, prompt_spec(cfg))
    ds = Dataset.from_list(records)

    ledger_path = out / "ledger.jsonl"
    resume = latest_checkpoint(out)
    if resume is not None:
        step = int(resume.name.split("-")[1])
        removed = trim_ledger(ledger_path, step)
        print(f"[grpo] resuming from {resume} (ledger trimmed by {removed} rows past step {step})")

    start = resolve_start(cfg)
    reward = build_reward(cfg, train_paths, ledger_path)
    fp16 = cfg.model.dtype == "float16"
    args = GRPOConfig(
        output_dir=str(out),
        learning_rate=g.learning_rate,
        num_generations=g.num_generations,
        max_completion_length=g.max_completion_length,
        temperature=g.temperature,
        beta=g.beta,
        loss_type=g.loss_type,
        scale_rewards=g.scale_rewards,
        per_device_train_batch_size=g.per_device_train_batch_size,
        gradient_accumulation_steps=g.gradient_accumulation_steps,
        max_steps=g.max_steps,
        logging_steps=g.logging_steps,
        save_steps=g.save_steps,
        save_total_limit=g.save_total_limit,
        fp16=fp16,
        bf16=cfg.model.dtype == "bfloat16",
        gradient_checkpointing=True,
        use_vllm=g.use_vllm,
        vllm_mode="colocate",
        report_to="none",
        seed=g.seed,
        log_completions=False,
        model_init_kwargs={"dtype": cfg.model.dtype},
        chat_template_kwargs=cfg.model.chat_template_kwargs or None,
    )
    peft = LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=cfg.lora.target_modules,
        task_type="CAUSAL_LM",
    )
    write_manifest(
        out,
        "grpo",
        cfg.model_dump(),
        start_model=start,
        init_fingerprint=fingerprint(start),
        n_train=len(sets.train),
        n_train_databases=len(train_paths),
        n_suites=len(SuiteIndex(cfg.suites)),
        resumed_from=str(resume) if resume else None,
    )
    trainer = GRPOTrainer(
        model=start,
        reward_funcs=[reward],
        args=args,
        train_dataset=ds,
        peft_config=peft,
        callbacks=[make_budget_callback(cfg.budget.max_hours, cfg.budget.margin_minutes)],
    )
    t0 = time.time()
    trainer.train(resume_from_checkpoint=str(resume) if resume else None)
    adapter = out / "adapter"
    trainer.save_model(str(adapter))
    (out / "train_log.json").write_text(json.dumps(trainer.state.log_history, indent=1), encoding="utf-8")
    (out / "timing.json").write_text(
        json.dumps(
            {
                "train_seconds": time.time() - t0,
                "global_step": trainer.state.global_step,
                "executor": reward.verifier.ex.stats(),
            }
        ),
        encoding="utf-8",
    )
    return adapter
