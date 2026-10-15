"""Pieces shared by SFT and GRPO: surviving Kaggle, and starting points.

Kaggle sessions end (12 hours at most, often sooner if the tab idles), and a
weekly GPU quota runs out. Training therefore:

* checkpoints every `save_steps`, keeping the last few;
* stops itself, cleanly and with a final checkpoint, `margin_minutes` before
  `max_hours` — a run killed mid-step loses everything since the last save;
* resumes automatically from the newest checkpoint in its output folder;
* trims the rollout ledger back to the resumed step, so the audit never
  double-counts steps that were recomputed after a resume.

The starting point matters for the comparison. Both GRPO arms must start from
the *same* weights. If they start from an SFT adapter, that adapter is merged
into the base model first and GRPO trains a fresh adapter on top. That way
the KL reference — which TRL takes to be "the model with the adapter
disabled" — is the SFT model, not the raw base model. Getting this wrong
means the KL term pulls the policy back towards a model that cannot write SQL.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

from mirage.errors import MissingDependency


def latest_checkpoint(out_dir: str | Path) -> Path | None:
    out = Path(out_dir)
    if not out.exists():
        return None
    cks = [(int(m.group(1)), p) for p in out.glob("checkpoint-*") if (m := re.match(r"checkpoint-(\d+)$", p.name))]
    return max(cks)[1] if cks else None


def trim_ledger(ledger: Path, max_step: int) -> int:
    """Drop ledger rows after `max_step`. Returns how many were removed."""
    if not ledger.exists():
        return 0
    lines = ledger.read_text(encoding="utf-8").splitlines()
    keep = [ln for ln in lines if ln.strip() and json.loads(ln)["step"] <= max_step]
    ledger.write_text("\n".join(keep) + ("\n" if keep else ""), encoding="utf-8")
    return len(lines) - len(keep)


def fingerprint(path_or_name: str) -> str:
    """Identify a starting point. For a local folder, hash the weight files'
    names and sizes (cheap, and enough to catch "different checkpoint")."""
    p = Path(path_or_name)
    if p.exists():
        h = hashlib.sha256()
        for f in sorted(p.rglob("*")):
            if f.is_file() and f.suffix in (".safetensors", ".bin", ".json"):
                h.update(f.name.encode())
                h.update(str(f.stat().st_size).encode())
        return f"local:{h.hexdigest()[:12]}"
    return f"hub:{path_or_name}"


def make_budget_callback(max_hours: float, margin_minutes: float):
    try:
        from transformers import TrainerCallback
    except ImportError as e:
        raise MissingDependency("transformers") from e

    deadline = time.time() + max_hours * 3600 - margin_minutes * 60

    class BudgetCallback(TrainerCallback):
        """Save and stop before the session is killed."""

        def on_step_end(self, args, state, control, **kwargs):
            if time.time() > deadline:
                print(f"[budget] {max_hours}h budget nearly used; saving at step {state.global_step} and stopping")
                control.should_save = True
                control.should_training_stop = True
            return control

    return BudgetCallback()


def merge_adapter(base: str, adapter: str, out_dir: str | Path, dtype: str = "float16") -> Path:
    """Fold a LoRA adapter into its base model and save a standalone model."""
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        raise MissingDependency("peft") from e
    out = Path(out_dir)
    td = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]
    model = AutoModelForCausalLM.from_pretrained(base, dtype=td)
    model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    model.save_pretrained(out, safe_serialization=True)
    AutoTokenizer.from_pretrained(adapter if (Path(adapter) / "tokenizer_config.json").exists() else base).save_pretrained(out)
    (out / "mirage_merge.json").write_text(json.dumps({"base": base, "adapter": adapter}), encoding="utf-8")
    return out


def resolve_start(cfg) -> str:
    """The model GRPO/SFT/eval should load: an explicit merged model, an
    adapter merged on the fly into `<out>/../_merged/<name>`, or the base."""
    if cfg.init_model:
        return cfg.init_model
    if cfg.init_adapter:
        target = Path(cfg.init_adapter).parent / (Path(cfg.init_adapter).name + "-merged")
        if not (target / "config.json").exists():
            print(f"[init] merging {cfg.init_adapter} into {cfg.model.name} -> {target}")
            merge_adapter(cfg.model.name, cfg.init_adapter, target, cfg.model.dtype)
        return str(target)
    return cfg.model.name


def warmup_kwargs(config_cls, ratio: float) -> dict:
    """transformers 5 folded `warmup_ratio` into `warmup_steps` (a float in
    [0, 1) is read as a ratio); 4.x has both. Pass whichever exists."""
    import inspect

    params = inspect.signature(config_cls.__init__).parameters
    if "warmup_ratio" in params:
        return {"warmup_ratio": ratio}
    return {"warmup_steps": ratio}
