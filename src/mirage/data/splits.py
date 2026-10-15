"""Who gets to see what.

Three sets, and the rules that keep them honest:

* **train** — GRPO and the SFT warm-up learn from these questions.
* **val** — carved out of Spider *train*, by database. Every tuning decision
  (which model, warm-up or not, KL coefficient, which checkpoint) is made on
  this. Never on dev.
* **dev** — Spider's official dev set. Touched once per arm, at the end, for
  the reported number. If dev is used to pick anything, the reported number is
  optimistic by an unknown amount and the project loses its main claim.

The val split is by **database**, not by question, because dev is by database:
Spider's dev databases never appear in training. A question-level split would
put sibling questions about the same tables on both sides and measure
memorised schema, not generalisation to a new schema.

The split is a hash of the database name, not a shuffle, so it survives
re-runs, new data and a different machine: a database in val today is in val
next month.
"""

from __future__ import annotations

import hashlib
from collections import Counter

from mirage.data.spider import Example


def _bucket(db_id: str, salt: str) -> float:
    h = hashlib.sha256(f"{salt}:{db_id}".encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def split_by_database(
    examples: list[Example], val_fraction: float = 0.1, salt: str = "mirage-v1"
) -> tuple[list[Example], list[Example]]:
    """Return (train, val), with every database wholly on one side."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")
    train, val = [], []
    for e in examples:
        (val if _bucket(e.db_id, salt) < val_fraction else train).append(e)
    if not val or not train:
        # Tiny datasets (the fixture) can land every database on one side;
        # fall back to the database with the lowest hash going to val.
        dbs = sorted({e.db_id for e in examples}, key=lambda d: _bucket(d, salt))
        n_val = max(1, round(len(dbs) * val_fraction))
        val_dbs = set(dbs[:n_val])
        train = [e for e in examples if e.db_id not in val_dbs]
        val = [e for e in examples if e.db_id in val_dbs]
    for e in train:
        e.split = "train"
    for e in val:
        e.split = "val"
    return train, val


def assert_disjoint(*named: tuple[str, list[Example]]) -> None:
    """Raise if any two sets share a database. Called before every training run."""
    seen: dict[str, str] = {}
    for name, exs in named:
        for db in {e.db_id for e in exs}:
            if db in seen and seen[db] != name:
                raise AssertionError(
                    f"database '{db}' appears in both '{seen[db]}' and '{name}'. "
                    "Splits must be disjoint by database."
                )
            seen[db] = name


def describe(examples: list[Example]) -> dict:
    return {
        "n": len(examples),
        "n_databases": len({e.db_id for e in examples}),
        "by_hardness": dict(sorted(Counter(e.hardness for e in examples).items())),
    }
