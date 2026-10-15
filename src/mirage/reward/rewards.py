"""The reward functions GRPO trains on, and the ledger that audits them.

Two arms, one variable:

* `single` — reward 1 if the query matches gold on the original database.
* `suite`  — reward 1 only if it matches on the original **and** every
  perturbed variant.

Everything else (initial checkpoint, data, prompt, seed, group size, KL,
learning rate) is identical between arms. The reward *shape* is a separate
switch: `binary`, or `shaped` partial credit for producing parseable SQL and
for executing without error. Shaped rewards help a cold policy find any signal
at all, and they are also an obvious thing to exploit (always emit some
harmless query that runs), so the shape is recorded and ablated, never silent.

The ledger
----------
Every rollout, in both arms, is written to `ledger.jsonl` with both verdicts,
the training step, the extracted SQL and any hack flags. That file is the raw
evidence for the project's central chart: the false-positive rate of the
single-database reward over the course of training, in each arm. If the
policy is learning to exploit the single-database reward, the single arm's
false-positive rate rises while the suite arm's does not.

The function also reports per-batch metrics through TRL's `log_metric` hook
(available from TRL 1.15), so they appear in the trainer's logs next to loss
and KL without any extra plumbing.
"""

from __future__ import annotations

import json
import threading
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mirage.reward.verifier import Verdict, Verifier
from mirage.sql.extract import completion_text


@dataclass
class RewardConfig:
    mode: str = "single"  # "single" | "suite"
    shape: str = "binary"  # "binary" | "shaped"
    credit_extracted: float = 0.1
    credit_executed: float = 0.2

    def __post_init__(self):
        if self.mode not in ("single", "suite"):
            raise ValueError(f"reward mode must be 'single' or 'suite', got {self.mode!r}")
        if self.shape not in ("binary", "shaped"):
            raise ValueError(f"reward shape must be 'binary' or 'shaped', got {self.shape!r}")


def reward_value(v: Verdict, cfg: RewardConfig) -> float:
    if cfg.mode == "suite":
        correct = bool(v.match_suite) if v.match_suite is not None else v.match_single
    else:
        correct = v.match_single
    if correct:
        return 1.0
    if cfg.shape == "binary":
        return 0.0
    if v.executed:
        return cfg.credit_executed
    if v.sql:
        return cfg.credit_extracted
    return 0.0


class Ledger:
    """Append-only JSONL, one row per rollout. Safe to resume into."""

    def __init__(self, path: str | Path | None):
        self.path = Path(path) if path else None
        self._lock = threading.Lock()
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, rows: list[dict[str, Any]]) -> None:
        if not self.path or not rows:
            return
        with self._lock, self.path.open("a", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")


class ExecutionReward:
    """A TRL-compatible reward function (`reward_funcs=[ExecutionReward(...)]`).

    TRL calls it with the batch's prompts and completions plus every dataset
    column as a keyword list, and (since 1.15) `trainer_state` and
    `log_metric`. It must return one float per completion.
    """

    def __init__(self, verifier: Verifier, cfg: RewardConfig, ledger: Ledger | None = None, arm: str = ""):
        self.verifier = verifier
        self.cfg = cfg
        self.ledger = ledger or Ledger(None)
        self.arm = arm or cfg.mode
        # TRL names the logged reward column after the function's __name__
        self.__name__ = f"exec_{cfg.mode}"
        self._calls = 0

    def __call__(
        self,
        prompts=None,
        completions=None,
        db_id=None,
        gold_sql=None,
        qid=None,
        hardness=None,
        trainer_state=None,
        log_metric=None,
        **_: Any,
    ) -> list[float]:
        self._calls += 1
        step = int(getattr(trainer_state, "global_step", self._calls - 1) or 0)
        n = len(completions)
        hardness = hardness or ["unknown"] * n
        rewards: list[float] = []
        rows: list[dict] = []
        groups: dict[str, list[float]] = defaultdict(list)
        n_single = n_suite = n_fp = n_exec = n_sql = 0
        for i in range(n):
            text = completion_text(completions[i])
            v = self.verifier.verify(db_id[i], gold_sql[i], text)
            r = reward_value(v, self.cfg)
            rewards.append(r)
            groups[qid[i]].append(r)
            n_single += v.match_single
            n_suite += bool(v.match_suite)
            n_fp += v.false_positive
            n_exec += v.executed
            n_sql += v.sql is not None
            rows.append(
                {
                    "arm": self.arm,
                    "step": step,
                    "qid": qid[i],
                    "db_id": db_id[i],
                    "hardness": hardness[i],
                    "reward": r,
                    "sql": v.sql,
                    "how": v.extracted_how,
                    "executed": v.executed,
                    "error_kind": v.error_kind,
                    "match_single": v.match_single,
                    "match_suite": v.match_suite,
                    "false_positive": v.false_positive,
                    "gold_empty": v.gold_empty,
                    "hacks": v.hacks,
                    "completion_chars": len(text),
                }
            )
        self.ledger.write(rows)

        if log_metric is not None:
            mixed = sum(1 for g in groups.values() if len(set(g)) > 1) / max(1, len(groups))
            log_metric("audit/match_single", n_single / n)
            log_metric("audit/match_suite", n_suite / n)
            log_metric("audit/false_positive_share", n_fp / max(1, n_single))
            log_metric("audit/executed", n_exec / n)
            log_metric("audit/extracted", n_sql / n)
            log_metric("audit/groups_with_signal", mixed)
        return rewards
