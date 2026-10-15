"""Spider's easy / medium / hard / extra difficulty labels.

Spider's evaluation script assigns every gold query a difficulty from how many
SQL components it uses. The released JSON does not carry the label, so it has
to be recomputed. This module re-implements the published rule on top of a
sqlglot parse instead of Spider's own parser.

It is an approximation and is labelled as one everywhere it is reported: the
counting rules are the same, but the parsers differ on edge cases (Spider's
parser folds some constructs differently). For headline numbers, take the
difficulty breakdown from the official `test-suite-sql-eval` script, which
`mirage eval --official` can call when that repo is present.

The rule, from Spider's `evaluation.py`:

    component1 = WHERE, GROUP BY, ORDER BY, LIMIT, each extra joined table, OR, LIKE
    component2 = INTERSECT/UNION/EXCEPT, nested subqueries
    others     = >1 aggregate, >1 selected column, >1 WHERE condition, >1 GROUP BY column

    easy   : c1 <= 1 and others == 0 and c2 == 0
    medium : (others <= 2 and c1 <= 1 and c2 == 0) or (c1 <= 2 and others < 2 and c2 == 0)
    hard   : (others > 2 and c1 <= 2 and c2 == 0) or (2 < c1 <= 3 and others <= 2 and c2 == 0)
             or (c1 <= 1 and others == 0 and c2 <= 1)
    extra  : anything else
"""

from __future__ import annotations

from functools import lru_cache

import sqlglot
from sqlglot import exp

HARDNESS_LEVELS = ["easy", "medium", "hard", "extra"]


def _strip_subqueries(node: exp.Expression, root: exp.Expression) -> list[exp.Expression]:
    """All descendants of `node` that are not inside a nested SELECT."""
    out = []
    for n in node.walk():
        cur = n.parent
        nested = False
        while cur is not None and cur is not node:
            if isinstance(cur, exp.Select) and cur is not root:
                nested = True
                break
            cur = cur.parent
        if not nested:
            out.append(n)
    return out


def components(sql: str) -> tuple[int, int, int] | None:
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return None
    if tree is None:
        return None

    c2 = 0
    main = tree
    while isinstance(main, (exp.Union, exp.Intersect, exp.Except)):
        c2 += 1
        main = main.left
    if isinstance(main, exp.Subquery):
        main = main.this
    if not isinstance(main, exp.Select):
        return None

    c1 = 0
    where = main.args.get("where")
    having = main.args.get("having")
    group = main.args.get("group")
    if where is not None:
        c1 += 1
    if group is not None:
        c1 += 1
    if main.args.get("order") is not None:
        c1 += 1
    if main.args.get("limit") is not None:
        c1 += 1
    c1 += len(main.args.get("joins") or [])

    cond_nodes: list[exp.Expression] = []
    for clause in (where, having):
        if clause is not None:
            cond_nodes.extend(_strip_subqueries(clause, main))
    c1 += sum(isinstance(n, exp.Or) for n in cond_nodes)
    c1 += sum(isinstance(n, exp.Like) for n in cond_nodes)

    # nested selects anywhere below the main query (FROM, WHERE, HAVING)
    for n in main.walk():
        if isinstance(n, exp.Select) and n is not main:
            c2 += 1

    others = 0
    sel = main.expressions
    n_agg = 0
    for e in sel:
        n_agg += sum(isinstance(n, exp.AggFunc) for n in _strip_subqueries(e, main))
    if n_agg > 1:
        others += 1
    if len(sel) > 1:
        others += 1
    if where is not None:
        wn = _strip_subqueries(where, main)
        n_conds = 1 + sum(isinstance(n, (exp.And, exp.Or)) for n in wn)
        if n_conds > 1:
            others += 1
    if group is not None and len(group.expressions) > 1:
        others += 1
    return c1, c2, others


@lru_cache(maxsize=65536)
def classify_hardness(sql: str) -> str:
    comp = components(sql)
    if comp is None:
        return "unknown"
    c1, c2, others = comp
    if c1 <= 1 and others == 0 and c2 == 0:
        return "easy"
    if (others <= 2 and c1 <= 1 and c2 == 0) or (c1 <= 2 and others < 2 and c2 == 0):
        return "medium"
    if (
        (others > 2 and c1 <= 2 and c2 == 0)
        or (2 < c1 <= 3 and others <= 2 and c2 == 0)
        or (c1 <= 1 and others == 0 and c2 <= 1)
    ):
        return "hard"
    return "extra"
