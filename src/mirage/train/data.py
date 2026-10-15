"""Building the training and evaluation sets from a config.

Everything goes through `load_sets`, which applies the database-level split
and checks disjointness before anything is trained. The returned `val` set is
for every tuning decision; `dev` is returned only when explicitly asked for
(the final evaluation), so it is hard to use by accident.
"""

from __future__ import annotations

from dataclasses import dataclass

from mirage.config import ExperimentCfg
from mirage.data.prompts import PromptSpec, build_messages, format_answer
from mirage.data.schema import render_schema
from mirage.data.spider import Example, Spider, read_jsonl
from mirage.data.splits import assert_disjoint, split_by_database
from mirage.errors import DataError


@dataclass
class Sets:
    spider: Spider
    train: list[Example]
    val: list[Example]
    dev: list[Example]

    def db_paths(self) -> dict[str, str]:
        ids = {e.db_id for e in self.train + self.val + self.dev}
        return {d: str(self.spider.db_path(d)) for d in ids}


def load_sets(cfg: ExperimentCfg, include_dev: bool = False) -> Sets:
    sp = Spider(cfg.spider)
    pool = sp.load(cfg.data.split)
    train, val = split_by_database(pool, cfg.data.val_fraction, cfg.data.split_salt)
    dev = sp.load("dev") if include_dev else []
    assert_disjoint(("train", train), ("val", val), *([("dev", dev)] if dev else []))
    if cfg.data.filtered_file:
        keep = {r["qid"] for r in read_jsonl(cfg.data.filtered_file) if r.get("keep")}
        before = len(train)
        train = [e for e in train if e.qid in keep]
        if not train:
            raise DataError(f"the filter file {cfg.data.filtered_file} kept no training questions")
        print(f"[data] pass-rate filter kept {len(train)}/{before} training questions")
    if cfg.data.max_examples:
        train = train[: cfg.data.max_examples]
    return Sets(sp, train, val, dev)


def prompt_spec(cfg: ExperimentCfg) -> PromptSpec:
    return PromptSpec(sample_values=cfg.prompt.sample_values)


def grpo_records(examples: list[Example], db_paths: dict[str, str], spec: PromptSpec) -> list[dict]:
    """Rows for TRL's GRPOTrainer. `prompt` is chat-format; every other column
    is handed to the reward function as a keyword argument."""
    return [
        {
            "prompt": build_messages(render_schema(db_paths[e.db_id], spec.sample_values), e.question, spec),
            "db_id": e.db_id,
            "gold_sql": e.gold_sql,
            "qid": e.qid,
            "hardness": e.hardness,
        }
        for e in examples
    ]


def sft_records(examples: list[Example], db_paths: dict[str, str], spec: PromptSpec) -> list[dict]:
    """Conversational prompt-completion rows for TRL's SFTTrainer; loss is
    computed on the completion only (TRL's default for this format)."""
    return [
        {
            "prompt": build_messages(render_schema(db_paths[e.db_id], spec.sample_values), e.question, spec),
            "completion": [{"role": "assistant", "content": format_answer(e.gold_sql)}],
        }
        for e in examples
    ]
