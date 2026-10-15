"""Turning a database into the text the model sees.

The schema is read from the SQLite file itself, not from `tables.json`, so
what the model is told about a table is exactly what the executor will run
against. It is rendered as compact `CREATE TABLE` statements, which is the
format code models have seen most of, with primary and foreign keys spelled
out because joins are the most common thing small models get wrong.

Optionally a few example values per column are appended as SQL comments.
Values help with questions like "...in the 'Rock' genre" where the model must
know whether the stored string is 'Rock' or 'rock'. They also lengthen the
prompt, which costs rollout time on a T4, so it is a config switch and the
choice is recorded with every run.

Whatever is chosen here must be identical at SFT, GRPO, evaluation and when
serving through Vantage. A model trained on one prompt format and evaluated
with another loses most of what it learned.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class Column:
    name: str
    type: str
    pk: bool


@dataclass(frozen=True)
class ForeignKey:
    table: str
    column: str
    ref_table: str
    ref_column: str


@dataclass(frozen=True)
class TableInfo:
    name: str
    columns: tuple[Column, ...]
    foreign_keys: tuple[ForeignKey, ...]


def _q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def read_schema(db_path: str | Path) -> list[TableInfo]:
    con = sqlite3.connect(f"file:{Path(db_path)}?mode=ro", uri=True)
    try:
        names = [
            r[0]
            for r in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY rowid"
            )
        ]
        tables = []
        for t in names:
            cols = tuple(
                Column(name=r[1], type=(r[2] or "").strip(), pk=bool(r[5]))
                for r in con.execute(f"PRAGMA table_info({_q(t)})")
            )
            fks = tuple(
                ForeignKey(table=t, column=r[3], ref_table=r[2], ref_column=r[4] or "")
                for r in con.execute(f"PRAGMA foreign_key_list({_q(t)})")
            )
            tables.append(TableInfo(name=t, columns=cols, foreign_keys=fks))
        return tables
    finally:
        con.close()


def _sample_values(con: sqlite3.Connection, table: str, column: str, k: int) -> list[str]:
    try:
        rows = con.execute(
            f"SELECT DISTINCT {_q(column)} FROM {_q(table)} WHERE {_q(column)} IS NOT NULL LIMIT {int(k)}"
        ).fetchall()
    except sqlite3.Error:
        return []
    out = []
    for (v,) in rows:
        if isinstance(v, bytes):
            continue
        s = repr(v) if isinstance(v, str) else str(v)
        out.append(s if len(s) <= 40 else s[:37] + "...")
    return out


@lru_cache(maxsize=512)
def render_schema(db_path: str, sample_values: int = 0) -> str:
    """CREATE TABLE text for a database. Cached: GRPO renders the same
    schema thousands of times."""
    tables = read_schema(db_path)
    con = sqlite3.connect(f"file:{Path(db_path)}?mode=ro", uri=True) if sample_values else None
    try:
        blocks = []
        for t in tables:
            lines = []
            pks = [c.name for c in t.columns if c.pk]
            for c in t.columns:
                line = f"  {c.name} {c.type}".rstrip()
                if con is not None:
                    vals = _sample_values(con, t.name, c.name, sample_values)
                    if vals:
                        line += f",  -- e.g. {', '.join(vals)}"
                        lines.append(line)
                        continue
                lines.append(line + ",")
            if pks:
                lines.append(f"  PRIMARY KEY ({', '.join(pks)}),")
            for fk in t.foreign_keys:
                lines.append(f"  FOREIGN KEY ({fk.column}) REFERENCES {fk.ref_table}({fk.ref_column}),")
            # tidy the trailing comma on the last real line
            if lines:
                last = lines[-1]
                if "  -- e.g." in last:
                    head, _, tail = last.partition(",  -- e.g.")
                    lines[-1] = head + "  -- e.g." + tail
                else:
                    lines[-1] = last.rstrip(",")
            blocks.append(f"CREATE TABLE {t.name} (\n" + "\n".join(lines) + "\n);")
        return "\n\n".join(blocks)
    finally:
        if con is not None:
            con.close()


def tables_json_entry(db_id: str, db_path: str | Path) -> dict:
    """A Spider-style `tables.json` entry built from a SQLite file, for
    datasets (like the fixture, or your own databases) that do not ship one."""
    tables = read_schema(db_path)
    column_names: list[list] = [[-1, "*"]]
    column_types: list[str] = ["text"]
    primary_keys: list[int] = []
    index: dict[tuple[str, str], int] = {}
    for ti, t in enumerate(tables):
        for c in t.columns:
            index[(t.name.lower(), c.name.lower())] = len(column_names)
            column_names.append([ti, c.name])
            ty = c.type.lower()
            column_types.append(
                "number" if any(k in ty for k in ("int", "real", "num", "float", "double", "dec")) else "text"
            )
            if c.pk:
                primary_keys.append(index[(t.name.lower(), c.name.lower())])
    foreign_keys = []
    for t in tables:
        for fk in t.foreign_keys:
            a = index.get((t.name.lower(), fk.column.lower()))
            b = index.get((fk.ref_table.lower(), fk.ref_column.lower()))
            if a is not None and b is not None:
                foreign_keys.append([a, b])
    return {
        "db_id": db_id,
        "table_names_original": [t.name for t in tables],
        "table_names": [t.name.replace("_", " ").lower() for t in tables],
        "column_names_original": column_names,
        "column_names": [[i, n.replace("_", " ").lower()] for i, n in column_names],
        "column_types": column_types,
        "primary_keys": primary_keys,
        "foreign_keys": foreign_keys,
    }
