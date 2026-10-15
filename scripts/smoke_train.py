"""Two-minute end-to-end check of the GPU-side code, on a tiny random model.

    python scripts/smoke_train.py [--device cpu|cuda] [--out runs/smoke]

Builds the synthetic fixture and its suites, creates a ~1M-parameter Qwen2
model with a tokenizer trained on the fixture's own text, and then runs the
real commands against it: SFT warm-up, merge, pass-rate filter, both GRPO
arms (a few steps each, with the real reward and ledger), generation with an
adapter, scoring, and the audit.

The model is random noise and learns nothing; that is not the point. The point
is that every TRL/PEFT/transformers call in `train/` and `eval/generate.py`
executes with the installed library versions. Run it on Kaggle at the start of
a session before committing hours to a real run: if a library update broke an
argument, this fails in two minutes instead of two hours in.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


def build_tiny_model(out: Path, corpus: list[str]) -> Path:
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    specials = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(
        corpus,
        trainers.BpeTrainer(vocab_size=1200, special_tokens=specials, initial_alphabet=pre_tokenizers.ByteLevel.alphabet()),
    )
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        eos_token="<|im_end|>",
        pad_token="<|endoftext|>",
        bos_token=None,
    )
    fast.chat_template = (
        "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
        "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
    )
    cfg = Qwen2Config(
        vocab_size=len(fast),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=4096,
        tie_word_embeddings=True,
        eos_token_id=fast.eos_token_id,
        pad_token_id=fast.pad_token_id,
    )
    model = Qwen2ForCausalLM(cfg)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    fast.save_pretrained(out)
    return out


def sh(*args: str) -> None:
    print("\n$", " ".join(args), flush=True)
    subprocess.run(args, check=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/smoke")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    a = ap.parse_args()
    out = Path(a.out).resolve()

    from mirage.data.fixtures import build_mini_spider
    from mirage.data.prompts import SYSTEM
    from mirage.data.schema import render_schema
    from mirage.data.spider import Spider
    from mirage.reward.suite import build_suite

    spider = Spider(build_mini_spider(out / "spider"))
    by_db = defaultdict(list)
    exs = spider.load("train") + spider.load("dev")
    for e in exs:
        by_db[e.db_id].append(e.gold_sql)
    for db, qs in by_db.items():
        build_suite(db, spider.db_path(db), qs, out / "suites", seed=1)
    corpus = [SYSTEM] + [render_schema(str(spider.db_path(d))) for d in by_db] + [e.question + " " + e.gold_sql for e in exs]
    model_dir = build_tiny_model(out / "tiny-model", corpus)

    dtype = "float32" if a.device == "cpu" else "float16"
    cfg = {
        "name": "smoke",
        "spider": str(out / "spider"),
        "suites": str(out / "suites"),
        "out": str(out / "base"),
        "model": {"name": str(model_dir), "dtype": dtype},
        "data": {"split": "train", "val_fraction": 0.2},
        "lora": {"r": 4, "alpha": 8, "target_modules": ["q_proj", "v_proj"]},
        "sft": {"epochs": 1, "per_device_train_batch_size": 4, "gradient_accumulation_steps": 1, "max_examples": 16,
                "max_length": 1024, "logging_steps": 1, "save_steps": 1000},
        "grpo": {"num_generations": 4, "per_device_train_batch_size": 4, "gradient_accumulation_steps": 2,
                 "max_steps": 2, "max_completion_length": 24, "save_steps": 1, "logging_steps": 1, "beta": 0.04},
        "eval": {"backend": "hf", "n_samples": 2, "batch_size": 8, "max_new_tokens": 24, "max_examples": 6},
        "budget": {"max_hours": 1.0, "margin_minutes": 1},
    }
    import yaml

    cdir = out / "configs"
    cdir.mkdir(exist_ok=True)
    (cdir / "base.yaml").write_text(yaml.safe_dump(cfg))
    (cdir / "sft.yaml").write_text(yaml.safe_dump({"extends": "base.yaml", "name": "sft", "out": str(out / "sft")}))
    (cdir / "grpo_base.yaml").write_text(
        yaml.safe_dump({"extends": "base.yaml", "name": "grpo", "init_adapter": str(out / "sft" / "adapter")})
    )
    for arm in ("single", "suite"):
        (cdir / f"grpo_{arm}.yaml").write_text(
            yaml.safe_dump({"extends": "grpo_base.yaml", "name": f"grpo-{arm}", "out": str(out / "arms" / arm),
                            "reward": {"mode": arm}})
        )

    import os

    if a.device == "cpu":
        os.environ["MIRAGE_CPU_KERNEL_SHIM"] = "1"  # TRL 1.15 kernels are Triton-only
    m = [sys.executable, "-m", "mirage.cli"]
    sh(*m, "sft", "-c", str(cdir / "sft.yaml"))
    sh(*m, "filter", "-c", str(cdir / "grpo_base.yaml"), "--k", "2", "--out", str(out / "filter" / "keep.jsonl"))
    for arm in ("single", "suite"):
        sh(*m, "grpo", "-c", str(cdir / f"grpo_{arm}.yaml"))
    # resume path: one more step on top of the saved checkpoint
    sh(*m, "grpo", "-c", str(cdir / "grpo_single.yaml"), "--set", "grpo.max_steps=3")
    sh(*m, "generate", "-c", str(cdir / "grpo_base.yaml"), "--split", "val",
       "--adapter", str(out / "arms" / "single" / "adapter"), "--label", "grpo-single", "--out", str(out / "eval" / "single"))
    sh(*m, "score", str(out / "eval" / "single" / "predictions.jsonl"), "--spider", str(out / "spider"),
       "--suites", str(out / "suites"))
    sh(*m, "audit", str(out / "arms" / "single" / "ledger.jsonl"))

    rows = [json.loads(x) for x in (out / "arms" / "single" / "ledger.jsonl").read_text().splitlines()]
    steps = sorted({r["step"] for r in rows})
    print(f"\nledger: {len(rows)} rollouts over steps {steps}")
    assert rows and all(set(r) >= {"match_single", "match_suite", "false_positive"} for r in rows)
    assert len(steps) == len(set(steps)) and steps == sorted(steps)
    print("SMOKE TEST PASSED: every training and generation call ran with the installed libraries.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
