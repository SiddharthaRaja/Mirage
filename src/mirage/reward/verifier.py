"""The verifier: one function that decides whether an attempt is right.

Training rewards and evaluation both call `Verifier.verify`. If they used two
implementations, a difference between them would be indistinguishable from a
training effect. Every attempt gets both verdicts:

* `match_single` — same result as gold on the original database (what most
  execution-reward training uses, and what Spider's EX metric measures).
* `match_suite`  — same result on the original database **and** on every
  perturbed variant (closer to test-suite accuracy).

The training arm decides which verdict becomes the reward. The audit uses both
on every rollout, in both arms, which is how the false-positive rate of the
single-database reward is measured *while training on it*.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

from mirage.reward.suite import SuiteIndex
from mirage.sql.compare import order_matters, results_match
from mirage.sql.executor import Executor, GoldCache
from mirage.sql.extract import extract_sql
from mirage.sql.hacks import detect


@dataclass
class Verdict:
    sql: str | None
    extracted_how: str
    executed: bool
    error_kind: str
    match_single: bool
    match_suite: bool | None  # None when no suite exists for this database
    n_variants: int
    pred_empty: bool
    gold_empty: bool
    hacks: list[str] = field(default_factory=list)
    n_rows: int = 0

    @property
    def false_positive(self) -> bool:
        """Rewarded by the single-database check, rejected by the suite."""
        return self.match_single and self.match_suite is False

    def to_dict(self) -> dict:
        d = asdict(self)
        d["false_positive"] = self.false_positive
        return d


class Verifier:
    def __init__(
        self,
        db_paths: dict[str, str],
        suites: SuiteIndex | None = None,
        timeout_s: float = 2.0,
        executor: Executor | None = None,
    ):
        self.db_paths = db_paths
        self.suites = suites or SuiteIndex(None)
        self.ex = executor or Executor(timeout_s=timeout_s)
        self.gold = GoldCache(self.ex)

    @classmethod
    def from_spider(cls, spider, db_ids, suites_dir: str | Path | None = None, **kw) -> Verifier:
        return cls({d: str(spider.db_path(d)) for d in set(db_ids)}, SuiteIndex(suites_dir), **kw)

    def verify(self, db_id: str, gold_sql: str, completion: str, sql: str | None = None) -> Verdict:
        if sql is None:
            ext = extract_sql(completion)
            sql, how = ext.sql, ext.how
        else:
            how = "given"
        db = self.db_paths[db_id]
        gold = self.gold.get(db, gold_sql)
        ordered = order_matters(gold_sql)
        pred = self.ex.execute(db, sql)
        single = results_match(pred, gold, ordered)

        variants = self.suites.variants(db_id)
        suite: bool | None = None
        if variants:
            if not single:
                suite = False  # fails on the original, so it fails the suite
            else:
                suite = True
                for v in variants:
                    gv = self.gold.get(v, gold_sql)
                    if not gv.ok:  # a broken variant must never decide a verdict
                        continue
                    if not results_match(self.ex.execute(v, sql), gv, ordered):
                        suite = False
                        break

        hacks = detect(sql, gold_sql, gold.empty) if single and sql else []
        return Verdict(
            sql=sql,
            extracted_how=how,
            executed=pred.ok,
            error_kind=pred.error_kind,
            match_single=single,
            match_suite=suite,
            n_variants=len(variants),
            pred_empty=pred.empty,
            gold_empty=gold.empty,
            hacks=hacks,
            n_rows=len(pred.rows) if pred.ok else 0,
        )
