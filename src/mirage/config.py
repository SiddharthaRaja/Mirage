"""Experiment configuration.

Every run is described by one YAML file, validated here. Unknown keys are an
error, not a warning: a typo like `num_generation: 4` that is silently ignored
produces a run that is not the experiment you think it is, and you find out
from a chart a week later.

The resolved config is written into every run's `manifest.json`, together
with the prompt version, git commit and library versions, so a result can
always be traced to exactly what produced it.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from mirage.errors import ConfigError


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())


class ModelCfg(_Strict):
    name: str = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    dtype: Literal["float16", "bfloat16", "float32"] = "float16"  # T4: float16
    chat_template_kwargs: dict[str, Any] = Field(default_factory=dict)


class PromptCfg(_Strict):
    sample_values: int = 0


class DataCfg(_Strict):
    split: str = "train"  # "train" (train_spider.json) or "train_all" (+ train_others.json)
    val_fraction: float = 0.1
    split_salt: str = "mirage-v1"
    max_examples: int | None = None
    filtered_file: str | None = None  # output of `mirage filter`; restricts training questions
    seed: int = 0


class LoraCfg(_Strict):
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = Field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )


class RewardCfg(_Strict):
    mode: Literal["single", "suite"] = "single"
    shape: Literal["binary", "shaped"] = "binary"
    credit_extracted: float = 0.1
    credit_executed: float = 0.2


class SFTCfg(_Strict):
    epochs: float = 1.0
    learning_rate: float = 1e-4
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 2
    max_length: int = 2048
    warmup_ratio: float = 0.03
    logging_steps: int = 10
    save_steps: int = 200
    max_examples: int | None = 2000  # a warm-up, not a full fine-tune
    seed: int = 0


class GRPOCfg(_Strict):
    num_generations: int = 8
    beta: float = 0.04  # TRL's default is 0.0 (no KL term); this project sets it on purpose
    learning_rate: float = 1e-5
    temperature: float = 0.9
    max_completion_length: int = 160
    per_device_train_batch_size: int = 16  # completions, not questions: 16 / 8 = 2 questions
    gradient_accumulation_steps: int = 4
    max_steps: int = 300
    loss_type: str = "dapo"  # TRL 1.15 default, stated so a TRL upgrade cannot change it silently
    scale_rewards: str = "group"
    logging_steps: int = 5
    save_steps: int = 25
    save_total_limit: int = 3
    use_vllm: bool = False
    seed: int = 0


class BudgetCfg(_Strict):
    max_hours: float = 11.0  # stop and checkpoint before Kaggle's session limit
    margin_minutes: float = 20.0


class EvalCfg(_Strict):
    split: Literal["val", "dev"] = "val"
    n_samples: int = 8
    temperature: float = 0.8
    max_examples: int | None = None
    backend: Literal["hf", "vllm", "openai", "mock"] = "hf"
    batch_size: int = 16
    max_new_tokens: int = 192
    base_url: str | None = None


class ExperimentCfg(_Strict):
    name: str
    spider: str
    out: str
    suites: str | None = None
    model: ModelCfg = Field(default_factory=ModelCfg)
    init_adapter: str | None = None  # LoRA to start from (merged before training)
    init_model: str | None = None  # an already-merged checkpoint to start from
    prompt: PromptCfg = Field(default_factory=PromptCfg)
    data: DataCfg = Field(default_factory=DataCfg)
    lora: LoraCfg = Field(default_factory=LoraCfg)
    reward: RewardCfg = Field(default_factory=RewardCfg)
    sft: SFTCfg = Field(default_factory=SFTCfg)
    grpo: GRPOCfg = Field(default_factory=GRPOCfg)
    budget: BudgetCfg = Field(default_factory=BudgetCfg)
    eval: EvalCfg = Field(default_factory=EvalCfg)
    candidates: list[str] = Field(default_factory=list)  # baseline spike: models to compare
    notes: str = ""


_ENV = re.compile(r"\$\{([A-Z0-9_]+)(?::-([^}]*))?\}")


def _expand(v):
    if isinstance(v, str):
        return _ENV.sub(lambda m: os.environ.get(m.group(1), m.group(2) or ""), v)
    if isinstance(v, dict):
        return {k: _expand(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_expand(x) for x in v]
    return v


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> ExperimentCfg:
    """Load a YAML config. `extends: other.yaml` inherits from another file,
    which is how the two GRPO arms share everything but the reward."""
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"config not found: {p}")
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    parent = raw.pop("extends", None)
    if parent:
        base = load_config(p.parent / parent).model_dump()
        raw = _merge(base, raw)
    raw = _merge(raw, overrides or {})
    try:
        return ExperimentCfg(**_expand(raw))
    except Exception as e:
        raise ConfigError(f"invalid config {p}:\n{e}") from e


def parse_overrides(pairs: list[str]) -> dict[str, Any]:
    """`--set grpo.beta=0.02 --set reward.mode=suite` -> nested dict."""
    out: dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ConfigError(f"override must look like key.sub=value, got {pair!r}")
        key, val = pair.split("=", 1)
        cur = out
        parts = key.split(".")
        for part in parts[:-1]:
            cur = cur.setdefault(part, {})
        cur[parts[-1]] = yaml.safe_load(val)
    return out
