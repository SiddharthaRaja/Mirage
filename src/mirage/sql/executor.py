"""Running model-written SQL safely and fast.

During GRPO every rollout is executed — 8 per question, thousands of
questions, and with the suite reward several databases per rollout. So this
has to be cheap, and it has to survive anything a policy can produce:

* **Read-only.** Databases are opened with `mode=ro`, and an authorizer
  refuses ATTACH, PRAGMA and every write action. A policy that discovers it can
  modify the database it is graded against has found the ultimate reward hack.
* **Time-boxed.** A progress handler aborts any query past its deadline. A
  6-way cross join on a 5,000-row table is easy to write by accident and would
  otherwise stall the whole training step.
* **Row-capped.** Results larger than `max_rows` are an error, not a silently
  truncated comparison.
* **Connection-pooled.** Opening SQLite is cheap but not free across ~100k
  executions; connections are reused per file.

Failures are values, never exceptions: an `ExecResult` with `ok=False` and an
`error_kind` the failure taxonomy can count.
"""

from __future__ import annotations

import re
import sqlite3
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_DENY = {
    sqlite3.SQLITE_ATTACH,
    sqlite3.SQLITE_DETACH,
    sqlite3.SQLITE_PRAGMA,
    sqlite3.SQLITE_INSERT,
    sqlite3.SQLITE_UPDATE,
    sqlite3.SQLITE_DELETE,
    sqlite3.SQLITE_DROP_TABLE,
    sqlite3.SQLITE_CREATE_TABLE,
    sqlite3.SQLITE_ALTER_TABLE,
    sqlite3.SQLITE_CREATE_INDEX,
    sqlite3.SQLITE_DROP_INDEX,
    sqlite3.SQLITE_CREATE_TRIGGER,
    sqlite3.SQLITE_CREATE_VIEW,
    sqlite3.SQLITE_TRANSACTION,
}


def _authorizer(action, *_args):
    return sqlite3.SQLITE_DENY if action in _DENY else sqlite3.SQLITE_OK


@dataclass
class ExecResult:
    ok: bool
    rows: list[tuple] = field(default_factory=list)
    n_cols: int = 0
    error_kind: str = ""
    error: str = ""
    elapsed_s: float = 0.0

    @property
    def empty(self) -> bool:
        return self.ok and not self.rows


ERROR_KINDS = (
    "empty_sql",
    "syntax",
    "no_such_table",
    "no_such_column",
    "ambiguous_column",
    "function",
    "timeout",
    "row_limit",
    "not_allowed",
    "other",
)


def classify_error(msg: str) -> str:
    m = msg.lower()
    if "interrupted" in m:
        return "timeout"
    if "no such table" in m:
        return "no_such_table"
    if "no such column" in m:
        return "no_such_column"
    if "ambiguous column" in m:
        return "ambiguous_column"
    if "no such function" in m or "wrong number of arguments" in m or "misuse of aggregate" in m:
        return "function"
    if "not authorized" in m or "readonly" in m or "read-only" in m:
        return "not_allowed"
    if "syntax error" in m or "incomplete input" in m or "unrecognized token" in m or "near " in m:
        return "syntax"
    return "other"


_WRITE_START = re.compile(r"^\s*(insert|update|delete|drop|create|alter|attach|detach|pragma|replace|vacuum|reindex)\b", re.I)


class Executor:
    def __init__(self, timeout_s: float = 2.0, max_rows: int = 100_000, pool_size: int = 64):
        self.timeout_s = timeout_s
        self.max_rows = max_rows
        self.pool_size = pool_size
        self._pool: OrderedDict[str, sqlite3.Connection] = OrderedDict()
        self._lock = threading.Lock()
        self.n_exec = 0
        self.total_s = 0.0

    def _conn(self, db_path: str) -> sqlite3.Connection:
        with self._lock:
            con = self._pool.get(db_path)
            if con is not None:
                self._pool.move_to_end(db_path)
                return con
            if not Path(db_path).exists():
                raise FileNotFoundError(db_path)
            con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, check_same_thread=False)
            con.text_factory = lambda b: b.decode("utf-8", errors="replace")
            con.set_authorizer(_authorizer)
            self._pool[db_path] = con
            while len(self._pool) > self.pool_size:
                _, old = self._pool.popitem(last=False)
                old.close()
            return con

    def execute(self, db_path: str | Path, sql: str | None) -> ExecResult:
        if sql is None or not sql.strip():
            return ExecResult(ok=False, error_kind="empty_sql", error="no SQL found")
        if _WRITE_START.match(sql):
            return ExecResult(ok=False, error_kind="not_allowed", error="only SELECT is allowed")
        con = self._conn(str(db_path))
        deadline = time.perf_counter() + self.timeout_s

        def _progress():
            return 1 if time.perf_counter() > deadline else 0

        t0 = time.perf_counter()
        con.set_progress_handler(_progress, 2_000)
        try:
            cur = con.execute(sql)
            rows = cur.fetchmany(self.max_rows + 1)
            n_cols = len(cur.description or [])
            cur.close()
        except sqlite3.Warning as e:  # e.g. "You can only execute one statement at a time."
            return self._fail("syntax", str(e), t0)
        except sqlite3.Error as e:
            return self._fail(classify_error(str(e)), str(e), t0)
        except (OverflowError, ValueError, MemoryError) as e:
            return self._fail("other", str(e), t0)
        finally:
            con.set_progress_handler(None, 0)
        if len(rows) > self.max_rows:
            return self._fail("row_limit", f"more than {self.max_rows} rows", t0)
        el = time.perf_counter() - t0
        self.n_exec += 1
        self.total_s += el
        return ExecResult(ok=True, rows=[tuple(r) for r in rows], n_cols=n_cols, elapsed_s=el)

    def _fail(self, kind: str, msg: str, t0: float) -> ExecResult:
        el = time.perf_counter() - t0
        self.n_exec += 1
        self.total_s += el
        return ExecResult(ok=False, error_kind=kind, error=msg[:300], elapsed_s=el)

    def close(self) -> None:
        with self._lock:
            for con in self._pool.values():
                con.close()
            self._pool.clear()

    def stats(self) -> dict[str, Any]:
        return {
            "n_exec": self.n_exec,
            "total_s": round(self.total_s, 3),
            "mean_ms": round(1000 * self.total_s / self.n_exec, 3) if self.n_exec else None,
        }


class GoldCache:
    """Gold results are executed once per (database, query) and reused.

    With 8 rollouts per question and a 5-database suite, this turns ~40 gold
    executions per question per step into 6 per question for the whole run.
    """

    def __init__(self, executor: Executor):
        self.ex = executor
        self._cache: dict[tuple[str, str], ExecResult] = {}

    def get(self, db_path: str | Path, sql: str) -> ExecResult:
        key = (str(db_path), sql)
        r = self._cache.get(key)
        if r is None:
            r = self.ex.execute(db_path, sql)
            self._cache[key] = r
        return r
