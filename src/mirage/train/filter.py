"""Pass-rate filtering: train only on questions that can teach something.

GRPO's advantage is (reward - group mean). If every attempt at a question
gets the same reward — all right or all wrong — the advantage is zero and
that question contributes nothing but rollout time, which is the scarce
resource on a T4. One published negative result found most of its groups had
identical rewards once its starting model was strong.

So before training, sample `k` attempts per training question from the
starting model and keep a question if its attempts *disagree* under either
reward:

    keep  if  0 < pass_single < 1   or   0 < pass_suite < 1

The same filtered set is used by both arms. Filtering on only one reward
would hand that arm a dataset tuned to it, and the comparison would no longer
change a single variable.

The filter is computed once, from the starting weights, and its output file
records every question's pass rates so the choice can be inspected.
"""

from __future__ import annotations

from pathlib import Path

from mirage.data.prompts import PromptSpec
from mirage.data.spider import Example, write_jsonl
from mirage.eval.evaluate import build_prompts
from mirage.reward.verifier import Verifier


def pass_rate_filter(
    examples: list[Example],
    generator,
    verifier: Verifier,
    spec: PromptSpec,
    out_path: str | Path,
    k: int = 8,
    temperature: float = 0.9,
    chunk: int = 32,
) -> dict:
    rows = []
    for i in range(0, len(examples), chunk):
        part = examples[i : i + chunk]
        samples = generator.generate(build_prompts(part, verifier.db_paths, spec), k, temperature)
        for e, s in zip(part, samples, strict=True):
            vs = [verifier.verify(e.db_id, e.gold_sql, t) for t in s]
            ps = sum(v.match_single for v in vs) / len(vs)
            pu = sum(bool(v.match_suite) for v in vs) / len(vs)
            rows.append(
                {
                    "qid": e.qid,
                    "db_id": e.db_id,
                    "hardness": e.hardness,
                    "pass_single": ps,
                    "pass_suite": pu,
                    "keep": (0 < ps < 1) or (0 < pu < 1),
                }
            )
    write_jsonl(out_path, rows)
    n = len(rows)
    return {
        "n": n,
        "kept": sum(r["keep"] for r in rows),
        "all_wrong": sum(r["pass_single"] == 0 for r in rows),
        "all_right_single": sum(r["pass_single"] == 1 for r in rows),
        "mean_pass_single": sum(r["pass_single"] for r in rows) / n if n else 0.0,
        "mean_pass_suite": sum(r["pass_suite"] for r in rows) / n if n else 0.0,
    }
