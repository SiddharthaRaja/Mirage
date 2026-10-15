"""Named shapes of "right for the wrong reason".

A false positive is a query the single-database reward pays for that the
suite rejects. That number says *how often* the reward lies; these detectors
say *how*. Each is a cheap structural check on the predicted query against the
gold query, run on every rewarded rollout so the report can show which
patterns grow during training — the difference between "the reward is noisy"
and "the policy learned to exploit it".

They are heuristics and are labelled as such. A flag means "this rewarded
query has the shape of a known shortcut", not "this query is wrong"; the
suite verdict is what decides wrong.
"""

from __future__ import annotations

from functools import lru_cache

import sqlglot
from sqlglot import exp

HACK_KINDS = (
    "empty_match",  # gold returns no rows and so does the prediction
    "limit_without_order",  # LIMIT with no ORDER BY where gold orders: "first row" by luck
    "dropped_filter",  # gold filters, prediction does not, rows happen to agree
    "select_star",  # SELECT * on a table narrow enough to match
    "constant_answer",  # no FROM clause: the answer is a literal the model guessed
    "tautology",  # a WHERE condition that is always true
    "distinct_mismatch",  # DISTINCT added or dropped; equal only while the data has no duplicates
    "boundary",  # same query shape with a comparison or literal nudged (> vs >=, 20 vs 21)
)


@lru_cache(maxsize=100_000)
def _parse(sql: str):
    try:
        return sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return None


def _top_select(tree):
    node = tree
    while isinstance(node, (exp.Union, exp.Intersect, exp.Except)):
        node = node.left
    return node if isinstance(node, exp.Select) else None


def detect(pred_sql: str, gold_sql: str, gold_empty: bool) -> list[str]:
    flags: list[str] = []
    if gold_empty:
        flags.append("empty_match")
    p, g = _parse(pred_sql), _parse(gold_sql)
    if p is None or g is None:
        return flags
    ps, gs = _top_select(p), _top_select(g)
    if ps is None:
        return flags
    if ps.args.get("limit") is not None and p.args.get("order") is None and ps.args.get("order") is None:
        if gs is not None and (gs.args.get("order") is not None or g.args.get("order") is not None):
            flags.append("limit_without_order")
    if gs is not None and gs.args.get("where") is not None and ps.args.get("where") is None:
        flags.append("dropped_filter")
    if any(isinstance(e, exp.Star) for e in ps.expressions):
        flags.append("select_star")
    if (ps.args.get("from_") or ps.args.get("from")) is None:  # key renamed in newer sqlglot
        flags.append("constant_answer")
    if gs is not None and bool(ps.args.get("distinct")) != bool(gs.args.get("distinct")):
        flags.append("distinct_mismatch")
    try:
        pn = [type(n).__name__ for n in p.walk() if not isinstance(n, exp.Literal)]
        gn = [type(n).__name__ for n in g.walk() if not isinstance(n, exp.Literal)]
        cmp_kinds = (exp.GT, exp.GTE, exp.LT, exp.LTE, exp.EQ, exp.NEQ)
        same_tables = {t.name for t in p.find_all(exp.Table)} == {t.name for t in g.find_all(exp.Table)}
        if same_tables and len(pn) == len(gn) and pn != gn:
            diff = [(a, b) for a, b in zip(pn, gn, strict=True) if a != b]
            if diff and all(a in {c.__name__ for c in cmp_kinds} and b in {c.__name__ for c in cmp_kinds} for a, b in diff):
                flags.append("boundary")
        elif pn == gn and sorted(lit.this for lit in p.find_all(exp.Literal)) != sorted(lit.this for lit in g.find_all(exp.Literal)):
            flags.append("boundary")
    except Exception:
        pass
    where = ps.args.get("where")
    if where is not None:
        for cmp in where.find_all(exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE):
            if isinstance(cmp.this, exp.Literal) and isinstance(cmp.expression, exp.Literal):
                flags.append("tautology")
                break
        else:
            if any(isinstance(b, exp.Boolean) and b.this for b in where.find_all(exp.Boolean)):
                flags.append("tautology")
    return flags
