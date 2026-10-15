"""The SFT warm-up.

Only run if the baseline spike says it is needed: when the untouched model
gets so few training questions right that most GRPO groups have no correct
attempt, there is no gradient. A short supervised pass on Spider's
(question, gold SQL) pairs gets the policy into the range where groups are
mixed. It is a warm-up — capped at `sft.max_examples` and one epoch by default
— not a full fine-tune, because a policy that already solves almost
everything also has no gradient (all-correct groups).

The warm-up uses the same prompt as everything else and computes loss on the
answer only. It trains on the *train* split; val databases never appear.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from mirage.config import ExperimentCfg
from mirage.errors import MissingDependency
from mirage.store import write_manifest
from mirage.train.common import fingerprint, latest_checkpoint, make_budget_callback, warmup_kwargs
from mirage.train.data import load_sets, prompt_spec, sft_records


def run_sft(cfg: ExperimentCfg) -> Path:
    try:
        from datasets import Dataset
        from peft import LoraConfig
        from trl import SFTConfig, SFTTrainer
    except ImportError as e:
        raise MissingDependency("trl") from e

    from mirage.train.cpu_shim import maybe_install

    maybe_install()
    out = Path(cfg.out)
    sets = load_sets(cfg)
    train = sets.train
    if cfg.sft.max_examples:
        import random

        rng = random.Random(cfg.sft.seed)
        train = list(train)
        rng.shuffle(train)
        train = train[: cfg.sft.max_examples]
    ds = Dataset.from_list(sft_records(train, sets.db_paths(), prompt_spec(cfg)))
    fp16 = cfg.model.dtype == "float16"
    args = SFTConfig(
        output_dir=str(out),
        num_train_epochs=cfg.sft.epochs,
        learning_rate=cfg.sft.learning_rate,
        per_device_train_batch_size=cfg.sft.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.sft.gradient_accumulation_steps,
        max_length=cfg.sft.max_length,
        **warmup_kwargs(SFTConfig, cfg.sft.warmup_ratio),
        logging_steps=cfg.sft.logging_steps,
        save_steps=cfg.sft.save_steps,
        save_total_limit=2,
        fp16=fp16,
        bf16=cfg.model.dtype == "bfloat16",
        gradient_checkpointing=True,
        completion_only_loss=True,
        report_to="none",
        seed=cfg.sft.seed,
        model_init_kwargs={"dtype": cfg.model.dtype},
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
        "sft",
        cfg.model_dump(),
        n_train=len(train),
        n_train_databases=len({e.db_id for e in train}),
        init_fingerprint=fingerprint(cfg.model.name),
    )
    trainer = SFTTrainer(
        model=cfg.model.name,
        args=args,
        train_dataset=ds,
        peft_config=peft,
        callbacks=[make_budget_callback(cfg.budget.max_hours, cfg.budget.margin_minutes)],
    )
    t0 = time.time()
    trainer.train(resume_from_checkpoint=str(latest_checkpoint(out)) if latest_checkpoint(out) else None)
    adapter = out / "adapter"
    trainer.save_model(str(adapter))
    (out / "train_log.json").write_text(json.dumps(trainer.state.log_history, indent=1), encoding="utf-8")
    (out / "timing.json").write_text(json.dumps({"train_seconds": time.time() - t0}), encoding="utf-8")
    return adapter
