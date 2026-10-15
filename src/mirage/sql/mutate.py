"""Near-miss versions of a gold query.

A mutant is the gold query with one small, plausible mistake: `>` instead of
`>=`, a dropped condition, a missing DISTINCT, MIN instead of MAX, ORDER BY
removed but LIMIT kept. These are exactly the mistakes a small model makes, and
they are the queries most likely to return the right rows *by accident* on a
small database.

Mutants are used twice:

1. **Building suites.** A perturbed database is useful only if it tells the
   gold query apart from its near misses. Suite construction keeps the variants
   that "kill" the most mutants (the idea behind Zhong et al.'s distilled test
   suites, here with a simpler mutation set).
2. **Measuring the reward's blind spot directly**, before any training: what
   fraction of near-miss queries does the single-database reward pay out for?
   That number needs no GPU and is one of the project's headline figures.

Some mutants are semantically equivalent to the gold query (dropping DISTINCT
when the column is already unique, for instance). No database can kill those,
so "mutants killed" is a lower bound on a suite's strength, and the report
says so.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import sqlglot
from sqlglot import exp

_CMP_SWAPS: dict[type, list[type]] = {
    exp.GT: [exp.GTE, exp.LT],
    exp.GTE: [exp.GT],
    exp.LT: [exp.LTE, exp.GT],
    exp.LTE: [exp.LT],
    exp.EQ: [exp.NEQ],
    exp.NEQ: [exp.EQ],
}
_AGG_SWAPS: dict[type, list[type]] = {
    exp.Max: [exp.Min],
    exp.Min: [exp.Max],
    exp.Avg: [exp.Sum],
    exp.Sum: [exp.Avg],
}


def _parse(sql: str) -> exp.Expression | None:
    try:
        return sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return None


def _nodes(tree: exp.Expression, kind) -> list[exp.Expression]:
    return [n for n in tree.walk() if isinstance(n, kind)]


def _each(tree: exp.Expression, kind, fn: Callable[[exp.Expression], exp.Expression | None]) -> Iterator[exp.Expression]:
    """For each node of `kind`, yield a copy of the tree with only that node changed."""
    n = len(_nodes(tree, kind))
    for i in range(n):
        t = tree.copy()
        target = _nodes(t, kind)[i]
        new = fn(target)
        if new is None:
            continue
        if new is not target:
            target.replace(new)
        yield t


def _cmp_flip(tree):
    for kind, swaps in _CMP_SWAPS.items():
        for new_kind in swaps:
            yield from _each(tree, kind, lambda n, nk=new_kind: nk(this=n.this.copy(), expression=n.expression.copy()))


def _drop_conjunct(tree):
    for side in ("this", "expression"):
        yield from _each(tree, exp.And, lambda n, s=side: n.args[s].copy())


def _drop_where(tree):
    def fn(sel):
        if sel.args.get("where") is None:
            return None
        sel.set("where", None)
        return sel

    yield from _each(tree, exp.Select, fn)


def _distinct_toggle(tree):
    def fn(sel):
        if not sel.args.get("distinct"):
            # Adding DISTINCT is provably a no-op in two common shapes; skip them
            # so they do not inflate the "mutants no database can catch" count.
            if isinstance(sel.parent, (exp.In, exp.Subquery)) and isinstance(sel.parent.parent, exp.In):
                return None
            if isinstance(sel.parent, exp.In):
                return None
            aggregate_only = sel.args.get("group") is None and all(
                isinstance(e.unalias() if hasattr(e, "unalias") else e, exp.AggFunc) for e in sel.expressions
            )
            if aggregate_only:
                return None
        sel.set("distinct", None if sel.args.get("distinct") else exp.Distinct())
        return sel

    yield from _each(tree, exp.Select, fn)


def _agg_swap(tree):
    for kind, swaps in _AGG_SWAPS.items():
        for new_kind in swaps:
            yield from _each(tree, kind, lambda n, nk=new_kind: nk(this=n.this.copy()))

    def count_distinct(n):
        arg = n.this
        if isinstance(arg, (exp.Star, exp.Distinct)):
            return None
        return exp.Count(this=exp.Distinct(expressions=[arg.copy()]))

    yield from _each(tree, exp.Count, count_distinct)

    def count_plain(n):
        arg = n.this
        if isinstance(arg, exp.Distinct) and len(arg.expressions) == 1:
            return exp.Count(this=arg.expressions[0].copy())
        return None

    yield from _each(tree, exp.Count, count_plain)


def _drop_order(tree):
    def fn(node):
        if node.args.get("order") is None:
            return None
        node.set("order", None)
        return node

    yield from _each(tree, exp.Select, fn)


def _flip_order(tree):
    def fn(o):
        o.set("desc", not o.args.get("desc"))
        return o

    yield from _each(tree, exp.Ordered, fn)


def _limit_bump(tree):
    def fn(lim):
        e = lim.expression
        if isinstance(e, exp.Literal) and e.is_int:
            lim.set("expression", exp.Literal.number(int(e.this) + 1))
            return lim
        return None

    yield from _each(tree, exp.Limit, fn)


def _literal_bump(tree):
    def fn(lit):
        if not lit.is_number or isinstance(lit.parent, exp.Limit):
            return None
        try:
            v = float(lit.this)
        except ValueError:
            return None
        return exp.Literal.number(int(v) + 1 if v.is_integer() else v + 1)

    yield from _each(tree, exp.Literal, fn)


def _not_in_flip(tree):
    def fn(n):
        return n.this.copy() if isinstance(n.this, exp.In) else None

    yield from _each(tree, exp.Not, fn)


OPERATORS: dict[str, Callable] = {
    "cmp_flip": _cmp_flip,
    "drop_conjunct": _drop_conjunct,
    "drop_where": _drop_where,
    "distinct_toggle": _distinct_toggle,
    "agg_swap": _agg_swap,
    "drop_order": _drop_order,
    "flip_order": _flip_order,
    "limit_bump": _limit_bump,
    "literal_bump": _literal_bump,
    "not_in_flip": _not_in_flip,
}


def mutants(sql: str, max_mutants: int = 24) -> list[tuple[str, str]]:
    """Return up to `max_mutants` (operator, mutant_sql) pairs, deduplicated
    and excluding anything textually equal to the gold query."""
    tree = _parse(sql)
    if tree is None:
        return []
    try:
        gold_norm = tree.sql(dialect="sqlite")
    except Exception:
        return []
    seen = {gold_norm}
    out: list[tuple[str, str]] = []
    # round-robin across operators so one prolific operator cannot crowd out the rest
    gens = {name: op(tree) for name, op in OPERATORS.items()}
    while gens and len(out) < max_mutants:
        for name in list(gens):
            try:
                t = next(gens[name])
            except StopIteration:
                del gens[name]
                continue
            except Exception:
                del gens[name]
                continue
            try:
                s = t.sql(dialect="sqlite")
            except Exception:
                continue
            if s not in seen:
                seen.add(s)
                out.append((name, s))
                if len(out) >= max_mutants:
                    break
    return out
