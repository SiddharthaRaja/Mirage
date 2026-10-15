"""Distilled suites: a few perturbed databases that actually discriminate.

Generating random variants is easy; most of them are useless (every query,
right or wrong, gives the same answer on them). A suite is chosen from a pool
of candidate variants by asking which ones tell gold queries apart from their
near-miss mutants:

1. For every gold query on this database, generate mutants.
2. Find which mutants the original database already rejects.
3. Generate `n_candidates` variants. Drop any on which a gold query errors.
4. Greedily keep the variant that rejects the most *not-yet-rejected* mutants,
   until `n_keep` are chosen or no variant adds anything.

The outcome for each database is recorded in `suite.json`, including the
number that matters before any training happens:

    of the near-miss mutants the original database lets through,
    how many does the suite catch?

That is the size of the single-database reward's blind spot on this data.

Design note: suites for training databases are built from training gold
queries. That uses no information the reward does not already use (the reward
compares against those same gold queries). Suites for *dev* databases are
built from dev gold queries, as Zhong et al. do; they are an evaluation
instrument, never a training signal. If the official Spider test-suite
databases are available, dev is also scored with those (see `eval/official.py`).
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

from mirage.reward.perturb import PerturbConfig, gold_literals, make_variant
from mirage.sql.compare import order_matters, results_match
from mirage.sql.executor import Executor
from mirage.sql.mutate import mutants


@dataclass
class SuiteStats:
    db_id: str
    variants: list[str]
    n_gold: int
    n_gold_failed_original: int
    n_mutants: int
    killed_by_original: int
    survivors: int  # mutants the original database cannot tell from gold
    survivors_killed_by_suite: int
    n_candidates: int
    n_candidates_rejected: int
    by_operator: dict[str, dict[str, int]] = field(default_factory=dict)
    seconds: float = 0.0

    @property
    def blind_spot_rate(self) -> float:
        """Fraction of mutants the single-database reward would pay out for."""
        return self.survivors / self.n_mutants if self.n_mutants else 0.0

    @property
    def suite_catch_rate(self) -> float:
        return self.survivors_killed_by_suite / self.survivors if self.survivors else 0.0


def build_suite(
    db_id: str,
    db_path: str | Path,
    gold_queries: list[str],
    out_dir: str | Path,
    n_candidates: int = 12,
    n_keep: int = 4,
    seed: int = 0,
    max_mutants_per_query: int = 24,
    perturb: PerturbConfig | None = None,
    executor: Executor | None = None,
) -> SuiteStats:
    t0 = time.perf_counter()
    ex = executor or Executor(timeout_s=2.0)
    db_path = Path(db_path)
    out = Path(out_dir) / db_id
    out.mkdir(parents=True, exist_ok=True)

    golds = list(dict.fromkeys(gold_queries))
    gold_res = {g: ex.execute(db_path, g) for g in golds}
    good_golds = [g for g in golds if gold_res[g].ok]

    # mutant id -> (gold, operator, sql)
    muts: list[tuple[str, str, str]] = []
    for g in good_golds:
        for op, m in mutants(g, max_mutants_per_query):
            muts.append((g, op, m))

    def killed_on(path: Path, gold_results: dict) -> set[int]:
        k = set()
        for i, (g, _op, m) in enumerate(muts):
            mr = ex.execute(path, m)
            if not results_match(mr, gold_results[g], order_matters(g)):
                k.add(i)
        return k

    killed_orig = killed_on(db_path, gold_res)
    survivors = set(range(len(muts))) - killed_orig

    protected = gold_literals(good_golds)
    cands: list[tuple[Path, set[int], float]] = []
    rejected = 0
    for c in range(n_candidates):
        vpath = out / f"_cand{c}.sqlite"
        make_variant(db_path, vpath, seed=seed * 1000 + c, protected=protected, cfg=perturb)
        vres = {g: ex.execute(vpath, g) for g in good_golds}
        if any(not r.ok for r in vres.values()):
            rejected += 1
            ex.close()
            vpath.unlink(missing_ok=True)
            continue
        nonempty = sum(bool(r.rows) for r in vres.values()) / max(1, len(vres))
        cands.append((vpath, killed_on(vpath, vres) & survivors, nonempty))

    chosen: list[tuple[Path, set[int]]] = []
    covered: set[int] = set()
    pool = list(cands)
    while pool and len(chosen) < n_keep:
        pool.sort(key=lambda c: (len(c[1] - covered), c[2]), reverse=True)
        best = pool.pop(0)
        if chosen and not (best[1] - covered):
            break
        chosen.append((best[0], best[1]))
        covered |= best[1]

    ex.close()  # release file handles before renaming/deleting on Windows
    final: list[str] = []
    for i, (p, _k) in enumerate(chosen):
        dst = out / f"{db_id}__s{i}.sqlite"
        dst.unlink(missing_ok=True)
        p.rename(dst)
        final.append(str(dst))
    for p, _k, _ne in pool:
        p.unlink(missing_ok=True)

    by_op: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "survive_original": 0, "caught_by_suite": 0})
    for i, (_g, op, _m) in enumerate(muts):
        by_op[op]["n"] += 1
        if i in survivors:
            by_op[op]["survive_original"] += 1
            if i in covered:
                by_op[op]["caught_by_suite"] += 1

    stats = SuiteStats(
        db_id=db_id,
        variants=final,
        n_gold=len(golds),
        n_gold_failed_original=len(golds) - len(good_golds),
        n_mutants=len(muts),
        killed_by_original=len(killed_orig),
        survivors=len(survivors),
        survivors_killed_by_suite=len(covered),
        n_candidates=n_candidates,
        n_candidates_rejected=rejected,
        by_operator=dict(by_op),
        seconds=round(time.perf_counter() - t0, 2),
    )
    (out / "suite.json").write_text(
        json.dumps(
            {**asdict(stats), "blind_spot_rate": stats.blind_spot_rate, "suite_catch_rate": stats.suite_catch_rate},
            indent=1,
        ),
        encoding="utf-8",
    )
    return stats


class SuiteIndex:
    """db_id -> list of variant database paths, read from a suites folder."""

    def __init__(self, root: str | Path | None):
        self.root = Path(root) if root else None
        self._map: dict[str, list[str]] = {}
        if self.root and self.root.exists():
            for sj in self.root.glob("*/suite.json"):
                d = json.loads(sj.read_text(encoding="utf-8"))
                # resolve relative to the folder actually on disk, so a suites
                # folder built on Kaggle still works after being downloaded
                self._map[d["db_id"]] = [str(sj.parent / Path(v).name) for v in d["variants"]]

    def variants(self, db_id: str) -> list[str]:
        return self._map.get(db_id, [])

    def __contains__(self, db_id: str) -> bool:
        return db_id in self._map

    def __len__(self) -> int:
        return len(self._map)

    def summary(self) -> list[dict]:
        out = []
        if not self.root:
            return out
        for sj in sorted(self.root.glob("*/suite.json")):
            out.append(json.loads(sj.read_text(encoding="utf-8")))
        return out
