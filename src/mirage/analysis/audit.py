"""Reading the ledger: how often did the reward lie, and did it get worse?

Input is the `ledger.jsonl` a GRPO run writes (one row per rollout, both
verdicts on every row). Output is everything the fidelity part of the report
needs, per arm:

* match rates over training (single-db and suite) — the learning curves;
* **false-positive share** over training: of the rollouts the single-database
  check rewarded, what fraction did the suite reject — with intervals
  clustered by question;
* the **drift test**: false-positive share in the last quarter of training
  minus the first quarter, with a cluster-bootstrap interval. A positive,
  significant drift in the `single` arm and not in the `suite` arm is the
  evidence of exploitation. Anything else is reported as what it is;
* which hack shapes the false positives take, early vs late;
* the share of question groups that carried any learning signal (GRPO learns
  nothing from a group where all attempts score the same).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from mirage.data.spider import read_jsonl
from mirage.eval.stats import cluster_rate_ci
from mirage.sql.hacks import HACK_KINDS


def _bins(steps: list[int], n_bins: int) -> list[tuple[int, int]]:
    lo, hi = min(steps), max(steps)
    if hi == lo:
        return [(lo, hi)]
    edges = np.linspace(lo, hi + 1, n_bins + 1)
    return [(int(edges[i]), int(edges[i + 1]) - 1) for i in range(n_bins)]


def _rate_block(rows: list[dict]) -> dict:
    n = len(rows)
    single = [float(r["match_single"]) for r in rows]
    suite = [float(bool(r["match_suite"])) for r in rows]
    fp = [float(r["false_positive"]) for r in rows]
    q = [r["qid"] for r in rows]
    fp_share = cluster_rate_ci(fp, single, q)
    m_single = cluster_rate_ci(single, [1.0] * n, q)
    m_suite = cluster_rate_ci(suite, [1.0] * n, q)
    return {
        "n_rollouts": n,
        "n_questions": len(set(q)),
        "match_single": m_single,
        "match_suite": m_suite,
        "false_positive_share": fp_share,
        "executed": sum(r["executed"] for r in rows) / n if n else float("nan"),
        "mean_reward": sum(r["reward"] for r in rows) / n if n else float("nan"),
    }


def _signal_share(rows: list[dict]) -> float:
    g: dict[tuple, set] = defaultdict(set)
    for r in rows:
        g[(r["step"], r["qid"])].add(r["reward"])
    return sum(len(v) > 1 for v in g.values()) / len(g) if g else float("nan")


def _drift(rows: list[dict], seed: int = 0, n_boot: int = 1000) -> dict:
    """FP share in the last quarter of steps minus the first quarter,
    bootstrapping questions within each quarter independently."""
    steps = sorted({r["step"] for r in rows})
    if len(steps) < 4:
        return {"diff": float("nan"), "ci95": (float("nan"), float("nan")), "significant": False}
    q1 = steps[len(steps) // 4]
    q3 = steps[(3 * len(steps)) // 4]
    early = [r for r in rows if r["step"] <= q1]
    late = [r for r in rows if r["step"] >= q3]

    def per_q(rs):
        d: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
        for r in rs:
            d[r["qid"]][0] += r["false_positive"]
            d[r["qid"]][1] += r["match_single"]
        return np.asarray(list(d.values()), dtype=float)

    a, b = per_q(early), per_q(late)
    if a.size == 0 or b.size == 0 or a[:, 1].sum() == 0 or b[:, 1].sum() == 0:
        return {"diff": float("nan"), "ci95": (float("nan"), float("nan")), "significant": False}
    point = b[:, 0].sum() / b[:, 1].sum() - a[:, 0].sum() / a[:, 1].sum()
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        ia = rng.integers(0, a.shape[0], a.shape[0])
        ib = rng.integers(0, b.shape[0], b.shape[0])
        da, db = a[ia, 1].sum(), b[ib, 1].sum()
        if da and db:
            diffs.append(b[ib, 0].sum() / db - a[ia, 0].sum() / da)
    lo, hi = (float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))) if diffs else (float("nan"),) * 2
    return {
        "diff": float(point),
        "ci95": (lo, hi),
        "significant": bool(diffs) and (lo > 0 or hi < 0),
        "early_steps": (steps[0], q1),
        "late_steps": (q3, steps[-1]),
    }


def audit_ledger(path: str | Path, n_bins: int = 10) -> dict:
    rows = read_jsonl(path)
    if not rows:
        return {"arms": {}}
    by_arm: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_arm[r["arm"]].append(r)

    out: dict = {"ledger": str(path), "arms": {}}
    for arm, rs in by_arm.items():
        steps = [r["step"] for r in rs]
        curve = []
        for lo, hi in _bins(steps, n_bins):
            b = [r for r in rs if lo <= r["step"] <= hi]
            if not b:
                continue
            blk = _rate_block(b)
            blk["steps"] = (lo, hi)
            blk["groups_with_signal"] = _signal_share(b)
            curve.append(blk)

        fps = [r for r in rs if r["false_positive"]]
        matches = [r for r in rs if r["match_single"]]
        mid = (min(steps) + max(steps)) / 2

        def hack_counts(subset):
            c = Counter(h for r in subset for h in r["hacks"])
            return {k: c.get(k, 0) for k in HACK_KINDS}

        examples = []
        seen_q = set()
        for r in sorted(fps, key=lambda r: -r["step"]):
            if r["qid"] in seen_q or not r.get("sql"):
                continue
            seen_q.add(r["qid"])
            examples.append({k: r[k] for k in ("step", "qid", "db_id", "sql", "hacks")})
            if len(examples) >= 12:
                break

        out["arms"][arm] = {
            "overall": {**_rate_block(rs), "groups_with_signal": _signal_share(rs)},
            "curve": curve,
            "drift": _drift(rs),
            "hacks_in_false_positives": {
                "early": hack_counts([r for r in fps if r["step"] <= mid]),
                "late": hack_counts([r for r in fps if r["step"] > mid]),
            },
            "hacks_in_all_matches": hack_counts(matches),
            "n_false_positives": len(fps),
            "error_kinds": dict(Counter(r["error_kind"] for r in rs if not r["executed"]).most_common()),
            "extraction": dict(Counter(r["how"] for r in rs)),
            "false_positive_examples": examples,
        }
    return out
