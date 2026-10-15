"""A GPU-free stand-in for the model, for tests and `mirage demo`.

This is *not* a language model and nothing it produces is a result about
language models. It exists so that every measurement in the pipeline — the
verifier, both rewards, the ledger, the audit, evaluation, the statistics and
the report — runs end to end on a laptop and in CI, on real SQL executed
against real (fixture) databases.

How it works, honestly stated:

* For every question, a fixed **candidate pool**: the gold query (only for
  questions "within reach" — fewer of the hard ones), its near-miss mutants (from `sql/mutate.py`), and a few broken attempts (a
  syntax error, a wrong column, no SQL at all).
* The policy is a **linear softmax over candidate features**: one shared
  weight per candidate type (gold, each mutation operator, each kind of
  breakage), plus fixed per-candidate noise and a difficulty offset. Because
  the weights are shared across questions, what it learns on training
  questions transfers to unseen dev questions — the property that lets a
  training effect show up in evaluation.
* Training is **GRPO in miniature**: sample a group of candidates per
  question, score them with the real `ExecutionReward` (so the real ledger is
  written), compute group-normalised advantages, take a policy-gradient step
  on the shared weights, and shrink towards the starting weights (a KL-like
  pull).

Nothing about reward hacking is scripted in. If the single-database reward
pays out for a mutation type often enough, the algorithm raises that type's
weight, and the audit sees it; under the suite reward it does not get paid.
Whether that happens on the fixture is an outcome of the data and the
algorithm, which is the point of having the demo — but it says nothing about
whether a 0.5B model on Spider does the same. Only the GPU runs answer that.
"""

from __future__ import annotations

import math
import random
import re
from collections import defaultdict
from dataclasses import dataclass, field

from mirage.data.spider import Example
from mirage.reward.rewards import ExecutionReward
from mirage.sql.mutate import mutants

_HARDNESS_OFFSET = {"easy": 0.6, "medium": 0.0, "hard": -0.7, "extra": -1.1, "unknown": 0.0}
# Chance that the correct query is within the policy's reach at all. A real
# small model cannot solve some questions however it is trained; on those, the
# only reward available under a leaky check is a near miss that matches by luck.
_REACHABLE = {"easy": 0.95, "medium": 0.8, "hard": 0.55, "extra": 0.4, "unknown": 0.8}

BASE_WEIGHTS = {
    "gold": 0.0,
    "broken_syntax": 0.3,
    "broken_schema": 0.5,
    "no_sql": -0.6,
    "cmp_flip": 0.2,
    "drop_conjunct": 0.1,
    "drop_where": 0.0,
    "distinct_toggle": 0.1,
    "agg_swap": 0.0,
    "drop_order": 0.3,
    "flip_order": 0.0,
    "limit_bump": -0.3,
    "literal_bump": 0.0,
    "not_in_flip": 0.1,
}


@dataclass
class Candidate:
    kind: str
    text: str
    sql: str | None
    noise: float


@dataclass
class _State:
    step: int = 0

    @property
    def global_step(self) -> int:
        return self.step


def _broken(gold: str, rng: random.Random) -> list[Candidate]:
    out = []
    syn = re.sub(r"\bSELECT\b", "SELEC", gold, count=1, flags=re.I)
    out.append(Candidate("broken_syntax", f"```sql\n{syn}\n```", syn, rng.gauss(0, 0.5)))
    m = re.search(r"\bSELECT\s+(?:DISTINCT\s+)?([A-Za-z_][A-Za-z0-9_]*\.)?([A-Za-z_][A-Za-z0-9_]*)", gold, re.I)
    if m and m.group(2).lower() not in ("count", "avg", "sum", "min", "max", "distinct"):
        bad = gold[: m.start(2)] + m.group(2) + "_name" + gold[m.end(2) :]
        out.append(Candidate("broken_schema", f"```sql\n{bad}\n```", bad, rng.gauss(0, 0.5)))
    out.append(Candidate("no_sql", "I am not sure which table holds this information.", None, rng.gauss(0, 0.3)))
    return out


@dataclass
class TabularPolicy:
    examples: list[Example]
    seed: int = 0
    weights: dict[str, float] = field(default_factory=lambda: dict(BASE_WEIGHTS))

    def __post_init__(self):
        rng = random.Random(self.seed)
        self.pools: dict[str, list[Candidate]] = {}
        self.by_question: dict[str, Example] = {}
        for e in self.examples:
            pool = []
            if rng.random() < _REACHABLE.get(e.hardness, 0.8):
                pool.append(Candidate("gold", f"```sql\n{e.gold_sql}\n```", e.gold_sql, rng.gauss(0, 0.4)))
            for op, m in mutants(e.gold_sql, 16):
                pool.append(Candidate(op, f"```sql\n{m}\n```", m, rng.gauss(0, 0.7)))
            pool.extend(_broken(e.gold_sql, rng))
            self.pools[e.qid] = pool
            self.by_question[e.question] = e
        self.hard = {e.qid: _HARDNESS_OFFSET.get(e.hardness, 0.0) for e in self.examples}
        self._rng = random.Random(self.seed + 1)

    def copy(self) -> TabularPolicy:
        p = TabularPolicy.__new__(TabularPolicy)
        p.examples, p.seed, p.weights = self.examples, self.seed, dict(self.weights)
        p.pools, p.by_question, p.hard = self.pools, self.by_question, self.hard
        p._rng = random.Random(self.seed + 2)
        return p

    # -- policy ------------------------------------------------------------ #

    def logits(self, qid: str) -> list[float]:
        out = []
        for c in self.pools[qid]:
            z = self.weights.get(c.kind, 0.0) + c.noise
            if c.kind == "gold":
                z += self.hard[qid]
            out.append(z)
        return out

    def probs(self, qid: str, temperature: float = 1.0) -> list[float]:
        z = [x / max(temperature, 1e-6) for x in self.logits(qid)]
        m = max(z)
        e = [math.exp(x - m) for x in z]
        s = sum(e)
        return [x / s for x in e]

    def sample(self, qid: str, n: int, temperature: float = 1.0) -> list[int]:
        p = self.probs(qid, temperature)
        return self._rng.choices(range(len(p)), weights=p, k=n)

    def greedy(self, qid: str) -> int:
        z = self.logits(qid)
        return max(range(len(z)), key=z.__getitem__)

    # -- Generator protocol (used by evaluation) --------------------------- #

    def _qid_for(self, messages: list[dict]) -> str:
        user = messages[-1]["content"]
        q = user.rsplit("Question:", 1)[-1].strip()
        return self.by_question[q].qid

    def generate(self, prompts: list[list[dict]], n: int, temperature: float) -> list[list[str]]:
        out = []
        for msgs in prompts:
            qid = self._qid_for(msgs)
            pool = self.pools[qid]
            if temperature <= 0:
                out.append([pool[self.greedy(qid)].text])
            else:
                out.append([pool[i].text for i in self.sample(qid, n, temperature)])
        return out


def train_tabular_grpo(
    policy: TabularPolicy,
    train: list[Example],
    reward: ExecutionReward,
    steps: int = 120,
    questions_per_step: int = 8,
    group_size: int = 8,
    lr: float = 0.04,
    beta: float = 0.02,
    temperature: float = 1.0,
    seed: int = 0,
) -> TabularPolicy:
    """Group-relative policy gradient on the shared candidate-type weights."""
    rng = random.Random(seed)
    init = dict(policy.weights)
    state = _State()
    for step in range(steps):
        state.step = step
        batch = rng.sample(train, min(questions_per_step, len(train)))
        completions, db_ids, golds, qids, hard, picks = [], [], [], [], [], []
        for e in batch:
            idx = policy.sample(e.qid, group_size, temperature)
            pool = policy.pools[e.qid]
            for i in idx:
                completions.append(pool[i].text)
                db_ids.append(e.db_id)
                golds.append(e.gold_sql)
                qids.append(e.qid)
                hard.append(e.hardness)
                picks.append((e.qid, i))
        rewards = reward(
            prompts=[None] * len(completions),
            completions=completions,
            db_id=db_ids,
            gold_sql=golds,
            qid=qids,
            hardness=hard,
            trainer_state=state,
        )
        by_q: dict[str, list[int]] = defaultdict(list)
        for j, q in enumerate(qids):
            by_q[q].append(j)
        grad: dict[str, float] = defaultdict(float)
        for q, js in by_q.items():
            rs = [rewards[j] for j in js]
            mu = sum(rs) / len(rs)
            sd = (sum((r - mu) ** 2 for r in rs) / len(rs)) ** 0.5
            if sd == 0:
                continue  # no signal in this group, exactly as in real GRPO
            p = policy.probs(q, temperature)
            pool = policy.pools[q]
            for j in js:
                adv = (rewards[j] - mu) / (sd + 1e-6)
                chosen = picks[j][1]
                # d log p(chosen) / d w_k = 1[kind(chosen)=k] - sum_{c: kind=k} p_c
                grad[pool[chosen].kind] += adv
                for ci, c in enumerate(pool):
                    grad[c.kind] -= adv * p[ci]
        n = len(completions)
        for k in policy.weights:
            policy.weights[k] += lr * grad.get(k, 0.0) / max(1, n) * group_size
            policy.weights[k] -= beta * (policy.weights[k] - init[k])
    return policy
