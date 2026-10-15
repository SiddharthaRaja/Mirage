"""When do two result sets count as the same answer?

This is the definition of "correct" for the whole project, so it follows the
official Spider test-suite evaluator's semantics rather than inventing new ones:

* **Rows are a multiset** unless the gold query has a top-level ORDER BY, in
  which case order matters. (Duplicates count: `SELECT name` and
  `SELECT DISTINCT name` differ when names repeat.)
* **Column order does not matter.** `SELECT name, age` answers the same
  question as `SELECT age, name`; the evaluator accepts any permutation of
  columns that makes the results equal.
* **Values compare after light normalisation**: 1 and 1.0 are equal, floats
  are compared to 1e-6 relative precision, bytes are decoded. Strings are
  compared exactly — 'Rock' and 'rock' are different answers.

Both sides must have executed successfully; an error never matches anything,
including another error.
"""

from __future__ import annotations

import itertools
import math
import re
from collections import Counter
from functools import lru_cache

import sqlglot
from sqlglot import exp

from mirage.sql.executor import ExecResult

_MAX_PERMUTATIONS = 5040  # 7 columns; beyond that only identity and column-multiset matches are tried


def _norm(v):
    if isinstance(v, bytes):
        try:
            return v.decode("utf-8")
        except UnicodeDecodeError:
            return v.hex()
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, float):
        if math.isnan(v):
            return "NaN"
        if math.isinf(v):
            return "inf" if v > 0 else "-inf"
        if v.is_integer():
            return int(v)
        return float(f"{v:.6g}")
    return v


def _key(v):
    """A sortable, hashable key so mixed-type columns can be compared."""
    return (type(v).__name__ if not isinstance(v, (int, float)) else "num", v)


def normalize_rows(rows: list[tuple]) -> list[tuple]:
    return [tuple(_norm(v) for v in r) for r in rows]


@lru_cache(maxsize=65536)
def order_matters(gold_sql: str) -> bool:
    """True when the gold query's outermost SELECT has an ORDER BY."""
    try:
        tree = sqlglot.parse_one(gold_sql, read="sqlite")
    except Exception:
        return bool(re.search(r"\border\s+by\b", gold_sql, re.I))
    node = tree
    if isinstance(node, (exp.Union, exp.Intersect, exp.Except)):
        return node.args.get("order") is not None
    return node.args.get("order") is not None


def _rows_equal(a: list[tuple], b: list[tuple], ordered: bool) -> bool:
    if ordered:
        return a == b
    return Counter(a) == Counter(b)


def results_match(pred: ExecResult, gold: ExecResult, ordered: bool) -> bool:
    if not (pred.ok and gold.ok):
        return False
    p = normalize_rows(pred.rows)
    g = normalize_rows(gold.rows)
    if len(p) != len(g):
        return False
    if not g:  # both empty: a match, and one of the hacks the audit tracks
        return True
    ncol = len(g[0])
    if any(len(r) != ncol for r in p):
        return False
    if _rows_equal(p, g, ordered):
        return True
    if ncol == 1:
        return False

    # Column permutations: only try permutations where each predicted column's
    # multiset of values equals the gold column it is mapped to.
    gcols = [Counter(r[i] for r in g) for i in range(ncol)]
    pcols = [Counter(r[i] for r in p) for i in range(ncol)]
    candidates = [[j for j in range(ncol) if pcols[i] == gcols[j]] for i in range(ncol)]
    if any(not c for c in candidates):
        return False
    tried = 0
    for perm in itertools.product(*candidates):
        if len(set(perm)) != ncol:
            continue
        tried += 1
        if tried > _MAX_PERMUTATIONS:
            return False
        # predicted column i is gold column perm[i]; reorder pred into gold order
        inv = [0] * ncol
        for i, j in enumerate(perm):
            inv[j] = i
        reordered = [tuple(r[inv[j]] for j in range(ncol)) for r in p]
        if _rows_equal(reordered, g, ordered):
            return True
    return False
