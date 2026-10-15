"""Why a wrong query is wrong.

A single accuracy number says nothing about what training changed. This
module puts each wrong prediction into one category, by comparing its parse
with the gold query's, so the report can show e.g. "GRPO removed most schema
linking errors but join errors barely moved".

Categories are assigned in priority order — the first difference found wins —
so every wrong query is counted exactly once:

    no_sql          nothing extractable (format failure)
    syntax          SQLite could not parse it
    schema_link     referenced a table/column that does not exist
    exec_other      timeout or other runtime error
    wrong_tables    joined a different set of tables
    wrong_select    selected different columns or a different number of them
    aggregation     different aggregate functions
    grouping        different GROUP BY / HAVING
    nesting         different set operations or subquery structure
    filter          different WHERE structure (columns or operators)
    values          same structure, different literal values
    ordering        different ORDER BY / LIMIT
    other           parses the same at this level of detail, still differs

This is structural and approximate. Read the examples in the report — they
are the point of the exercise, and the fastest way to get better at SQL.
"""

from __future__ import annotations

from collections import Counter
from functools import lru_cache

import sqlglot
from sqlglot import exp

FAILURE_KINDS = (
    "no_sql",
    "syntax",
    "schema_link",
    "exec_other",
    "wrong_tables",
    "wrong_select",
    "aggregation",
    "grouping",
    "nesting",
    "filter",
    "values",
    "ordering",
    "other",
)


@lru_cache(maxsize=50_000)
def _features(sql: str):
    try:
        t = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return None
    if t is None:
        return None
    tables = frozenset(x.name.lower() for x in t.find_all(exp.Table))
    top = t
    while isinstance(top, (exp.Union, exp.Intersect, exp.Except)):
        top = top.left
    sel_cols = ()
    group: tuple = ()
    having = False
    order: tuple = ()
    limit = None
    if isinstance(top, exp.Select):
        sel_cols = tuple(
            sorted(c.name.lower() for e in top.expressions for c in e.find_all(exp.Column))
        ) + (("*",) if any(isinstance(e, exp.Star) or e.find(exp.Star) for e in top.expressions) else ())
        n_sel = len(top.expressions)
        g = top.args.get("group")
        group = tuple(sorted(e.sql().lower() for e in g.expressions)) if g else ()
        having = top.args.get("having") is not None
        o = top.args.get("order") or t.args.get("order")
        order = tuple((e.this.sql().lower(), bool(e.args.get("desc"))) for e in o.expressions) if o else ()
        lim = top.args.get("limit") or t.args.get("limit")
        limit = lim.expression.sql() if lim is not None and lim.expression is not None else None
    else:
        n_sel = 0
    aggs = tuple(sorted(type(a).__name__ for a in t.find_all(exp.AggFunc)))
    setops = tuple(sorted(type(s).__name__ for s in t.find_all(exp.Union, exp.Intersect, exp.Except)))
    n_sub = sum(1 for s in t.find_all(exp.Select)) - 1
    where_shape = []
    for w in t.find_all(exp.Where):
        for c in w.find_all(exp.Predicate):
            cols = tuple(sorted(x.name.lower() for x in c.find_all(exp.Column)))
            where_shape.append((type(c).__name__, cols))
    lits = tuple(sorted(lit.this for lit in t.find_all(exp.Literal)))
    return {
        "tables": tables,
        "sel": (n_sel, sel_cols),
        "aggs": aggs,
        "group": (group, having),
        "nest": (setops, n_sub),
        "where": tuple(sorted(where_shape)),
        "lits": lits,
        "order": (order, limit),
    }


def classify_failure(pred_sql: str | None, gold_sql: str, executed: bool, error_kind: str) -> str:
    if not pred_sql:
        return "no_sql"
    if not executed:
        if error_kind == "syntax":
            return "syntax"
        if error_kind in ("no_such_table", "no_such_column", "ambiguous_column"):
            return "schema_link"
        return "exec_other"
    p, g = _features(pred_sql), _features(gold_sql)
    if p is None or g is None:
        return "other"
    for key, kind in (
        ("tables", "wrong_tables"),
        ("sel", "wrong_select"),
        ("aggs", "aggregation"),
        ("group", "grouping"),
        ("nest", "nesting"),
        ("where", "filter"),
        ("lits", "values"),
        ("order", "ordering"),
    ):
        if p[key] != g[key]:
            return kind
    return "other"


def summarize(kinds: list[str]) -> dict[str, int]:
    c = Counter(kinds)
    return {k: c.get(k, 0) for k in FAILURE_KINDS}
