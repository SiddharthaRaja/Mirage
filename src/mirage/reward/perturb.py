"""Perturbed copies of a database.

The single-database reward asks: does the policy's query return the same rows
as the gold query *on this one database*? On a table of six rows that question
is often answered "yes" for the wrong reason. A perturbed copy keeps the schema
and the referential structure but changes the data, so a query that was right
by coincidence usually stops being right.

What a variant changes, and why each change catches something:

* **Row order shuffled.** Catches `LIMIT` without `ORDER BY`, and any query
  that relied on insertion order to pick "the first" row.
* **Non-key values permuted within their column.** Breaks the coincidental
  correlations a wrong filter or wrong column can ride on, while keeping every
  value that exists (so `WHERE country = 'France'` still has rows to find).
* **Numbers jittered, and gold literals planted at boundaries.** Separates `>` from `>=`, MIN from MAX on ties, and AVG
  from SUM when counts are small.
* **Rows added and removed.** Separates COUNT from COUNT(DISTINCT), catches a
  missing DISTINCT, and changes which entity is "the most".

What a variant never changes:

* **Primary and foreign key columns.** Joins stay meaningful; a join on the
  wrong key still returns different rows from a join on the right one, but a
  correct join never becomes meaningless.
* **Literals that appear in gold queries for this database.** If the gold
  query filters on 'Summer Lights', that value stays in the data, otherwise the
  gold answer becomes empty and every wrong query that also returns nothing
  would match.
"""

from __future__ import annotations

import random
import shutil
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import sqlglot
from sqlglot import exp

from mirage.data.schema import TableInfo, _q, read_schema


@dataclass
class PerturbConfig:
    permute_prob: float = 0.6  # chance a non-key column is shuffled
    jitter_prob: float = 0.35  # chance a numeric non-key column is jittered
    jitter_scale: float = 0.25  # jitter as a fraction of the column's range
    boundary_prob: float = 0.5  # chance a numeric column gets a gold literal planted in it
    max_added_rows: int = 3
    max_dropped_rows: int = 2


@dataclass
class _Table:
    info: TableInfo
    columns: list[str]
    rows: list[list]
    key_cols: set[str] = field(default_factory=set)
    int_pk: str | None = None
    referenced: bool = False


def gold_literals(queries: list[str]) -> set:
    """Every literal value in a set of gold queries, as SQLite would store it."""
    out: set = set()
    for q in queries:
        try:
            tree = sqlglot.parse_one(q, read="sqlite")
        except Exception:
            continue
        for lit in tree.find_all(exp.Literal):
            if lit.is_string:
                out.add(lit.this)
            else:
                try:
                    v = float(lit.this)
                    out.add(int(v) if v.is_integer() else v)
                except ValueError:
                    pass
    return out


def _load(db_path: Path) -> list[_Table]:
    infos = read_schema(db_path)
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.text_factory = lambda b: b.decode("utf-8", errors="replace")
    try:
        tables = []
        for info in infos:
            cols = [c.name for c in info.columns]
            rows = [list(r) for r in con.execute(f"SELECT * FROM {_q(info.name)}")]
            pks = [c.name for c in info.columns if c.pk]
            t = _Table(info=info, columns=cols, rows=rows, key_cols=set(pks))
            if len(pks) == 1:
                col = next(c for c in info.columns if c.name == pks[0])
                if "int" in col.type.lower():
                    t.int_pk = pks[0]
            for fk in info.foreign_keys:
                t.key_cols.add(fk.column)
            tables.append(t)
    finally:
        con.close()
    by_name = {t.info.name.lower(): t for t in tables}
    for t in tables:
        for fk in t.info.foreign_keys:
            parent = by_name.get(fk.ref_table.lower())
            if parent is not None:
                parent.referenced = True
                if fk.ref_column:
                    parent.key_cols.add(fk.ref_column)
                else:
                    parent.key_cols.update(c.name for c in parent.info.columns if c.pk)
    return tables


def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _perturb_table(t: _Table, rng: random.Random, cfg: PerturbConfig, protected: set) -> None:
    n = len(t.rows)
    if n == 0:
        return
    for ci, col in enumerate(t.columns):
        if col in t.key_cols:
            continue
        vals = [r[ci] for r in t.rows]
        if n > 1 and rng.random() < cfg.permute_prob:
            rng.shuffle(vals)
        nums = [v for v in vals if _is_num(v)]
        if nums and len(nums) == len([v for v in vals if v is not None]) and rng.random() < cfg.jitter_prob:
            lo, hi = min(nums), max(nums)
            span = (hi - lo) or abs(hi) or 1
            is_int = all(isinstance(v, int) for v in nums)
            new = []
            for v in vals:
                if v is None or v in protected:
                    new.append(v)
                    continue
                d = rng.uniform(-1, 1) * cfg.jitter_scale * span
                if is_int:
                    d = int(round(d)) or rng.choice((-1, 1))
                    new.append(int(v + d))
                else:
                    new.append(round(v + d, 2))
            vals = new
        # Boundary planting: put a value from a gold query's literals into the
        # column, so `price < 5000` and `price <= 5000` stop agreeing. Random
        # jitter almost never lands exactly on a boundary; this is what catches
        # off-by-one comparison mistakes.
        num_lits = sorted(v for v in protected if _is_num(v))
        if nums and num_lits and rng.random() < cfg.boundary_prob:
            lo, hi = min(nums), max(nums)
            span = (hi - lo) or abs(hi) or 1
            near = [v for v in num_lits if lo - span <= v <= hi + span]
            if near:
                idx = rng.randrange(n)
                if vals[idx] not in protected:
                    vals[idx] = rng.choice(near)
        for r, v in zip(t.rows, vals, strict=True):
            r[ci] = v

    # drop rows from tables nothing references (keeps foreign keys valid)
    if not t.referenced and n > 2 and cfg.max_dropped_rows:
        k = rng.randint(0, min(cfg.max_dropped_rows, n - 2))
        for _ in range(k):
            t.rows.pop(rng.randrange(len(t.rows)))

    # add rows with fresh integer primary keys; other values drawn from existing ones
    if t.int_pk and cfg.max_added_rows:
        pk_i = t.columns.index(t.int_pk)
        existing = [r[pk_i] for r in t.rows if isinstance(r[pk_i], int)]
        nxt = (max(existing) if existing else 0) + 1
        for _ in range(rng.randint(0, cfg.max_added_rows)):
            src = rng.choice(t.rows) if t.rows else None
            if src is None:
                break
            row = []
            for ci, col in enumerate(t.columns):
                if ci == pk_i:
                    row.append(nxt)
                elif col in t.key_cols:
                    row.append(src[ci])  # foreign key: an existing, valid value
                else:
                    row.append(rng.choice(t.rows)[ci])
            nxt += 1
            t.rows.append(row)

    rng.shuffle(t.rows)


def make_variant(
    db_path: str | Path,
    out_path: str | Path,
    seed: int,
    protected: set | None = None,
    cfg: PerturbConfig | None = None,
) -> Path:
    """Write one perturbed copy of `db_path` to `out_path`."""
    db_path, out_path = Path(db_path), Path(out_path)
    cfg = cfg or PerturbConfig()
    rng = random.Random(seed)
    tables = _load(db_path)
    originals = {t.info.name: [list(r) for r in t.rows] for t in tables}
    for t in tables:
        _perturb_table(t, rng, cfg, protected or set())

    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(db_path, out_path)  # keeps DDL, constraints, indexes exactly
    con = sqlite3.connect(out_path)
    try:
        con.execute("PRAGMA foreign_keys = OFF")
        for t in tables:
            name = _q(t.info.name)
            con.execute(f"DELETE FROM {name}")
            ph = ",".join("?" * len(t.columns))
            orig_rows = originals[t.info.name]
            for i, row in enumerate(t.rows):
                try:
                    con.execute(f"INSERT INTO {name} VALUES ({ph})", row)
                except sqlite3.IntegrityError:
                    # a jittered value collided with a UNIQUE constraint; fall
                    # back to an original row if this position had one
                    if i < len(orig_rows):
                        try:
                            con.execute(f"INSERT INTO {name} VALUES ({ph})", orig_rows[i])
                        except sqlite3.IntegrityError:
                            pass
        con.commit()
        con.execute("VACUUM")
    finally:
        con.close()
    return out_path
