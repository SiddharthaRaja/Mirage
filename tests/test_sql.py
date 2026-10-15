import sqlglot

from mirage.sql.compare import order_matters, results_match
from mirage.sql.executor import ExecResult, Executor, GoldCache
from mirage.sql.extract import completion_text, extract_sql
from mirage.sql.hacks import detect
from mirage.sql.mutate import mutants


def R(rows, ok=True):
    return ExecResult(ok=ok, rows=rows, n_cols=len(rows[0]) if rows else 0)


# ---------------------------------------------------------------- extraction


def test_extract_prefers_fenced_block():
    e = extract_sql("Sure!\n```sql\nSELECT a FROM t;\n```\nThis returns a.")
    assert e.sql == "SELECT a FROM t" and e.how == "fence"


def test_extract_bare_and_stops_at_prose():
    e = extract_sql("SELECT a\nFROM t\n\nThis query selects a.")
    assert e.sql == "SELECT a\nFROM t" and e.how == "bare"


def test_extract_takes_first_statement_and_respects_quotes():
    e = extract_sql("```sql\nSELECT 'a;b' FROM t; DROP TABLE t;\n```")
    assert e.sql == "SELECT 'a;b' FROM t" and e.n_statements == 2


def test_extract_nothing():
    assert extract_sql("I don't know.").sql is None
    assert extract_sql("").how == "none"


def test_completion_text_accepts_chat_format():
    assert completion_text([{"role": "assistant", "content": "x"}]) == "x"
    assert completion_text("y") == "y"


# ---------------------------------------------------------------- execution


def test_executor_success_and_error_kinds(db_paths):
    ex = Executor()
    p = db_paths["concert_hall"]
    assert ex.execute(p, "SELECT count(*) FROM singer").rows == [(6,)]
    assert ex.execute(p, "SELEC 1").error_kind == "syntax"
    assert ex.execute(p, "SELECT * FROM nope").error_kind == "no_such_table"
    assert ex.execute(p, "SELECT nope FROM singer").error_kind == "no_such_column"
    assert ex.execute(p, None).error_kind == "empty_sql"


def test_executor_refuses_writes_and_attach(db_paths):
    ex = Executor()
    p = db_paths["concert_hall"]
    for sql in ("DELETE FROM singer", "DROP TABLE singer", "ATTACH DATABASE ':memory:' AS x", "PRAGMA writable_schema=1"):
        r = ex.execute(p, sql)
        assert not r.ok and r.error_kind == "not_allowed", sql
    assert ex.execute(p, "SELECT count(*) FROM singer").rows == [(6,)]


def test_executor_times_out_runaway_queries(db_paths):
    ex = Executor(timeout_s=0.2)
    r = ex.execute(
        db_paths["concert_hall"],
        "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT count(*) FROM c",
    )
    assert r.error_kind == "timeout" and r.elapsed_s < 2


def test_executor_row_cap(db_paths):
    ex = Executor(max_rows=10)
    r = ex.execute(db_paths["concert_hall"], "SELECT * FROM singer, singer AS b")
    assert r.error_kind == "row_limit"


def test_gold_cache_executes_once(db_paths):
    ex = Executor()
    gc = GoldCache(ex)
    for _ in range(5):
        gc.get(db_paths["concert_hall"], "SELECT count(*) FROM singer")
    assert ex.n_exec == 1


# ---------------------------------------------------------------- comparison


def test_order_only_matters_with_gold_order_by():
    a, b = R([(1,), (2,)]), R([(2,), (1,)])
    assert results_match(a, b, ordered=False)
    assert not results_match(a, b, ordered=True)
    assert order_matters("SELECT a FROM t ORDER BY a")
    assert not order_matters("SELECT a FROM t WHERE b IN (SELECT b FROM u ORDER BY b)")


def test_duplicates_count():
    assert not results_match(R([(1,), (1,)]), R([(1,)]), ordered=False)


def test_column_permutation_is_accepted():
    assert results_match(R([(1, "a"), (2, "b")]), R([("a", 1), ("b", 2)]), ordered=False)
    assert not results_match(R([(1, "b"), (2, "a")]), R([("a", 1), ("b", 2)]), ordered=False)


def test_numeric_normalisation():
    assert results_match(R([(1.0,)]), R([(1,)]), ordered=False)
    assert results_match(R([(0.1 + 0.2,)]), R([(0.3,)]), ordered=False)


def test_errors_never_match():
    assert not results_match(ExecResult(ok=False), ExecResult(ok=False), ordered=False)


def test_empty_matches_empty():
    assert results_match(R([]), R([]), ordered=False)


# ---------------------------------------------------------------- mutants and hacks


def test_mutants_are_distinct_valid_and_cover_operators():
    gold = "SELECT name FROM singer WHERE age > 30 AND country = 'France' ORDER BY age DESC LIMIT 1"
    ms = mutants(gold, 50)
    ops = {op for op, _ in ms}
    assert {"cmp_flip", "drop_conjunct", "drop_order", "flip_order", "limit_bump"} <= ops
    sqls = [m for _, m in ms]
    assert len(sqls) == len(set(sqls))
    for m in sqls:
        sqlglot.parse_one(m, read="sqlite")


def test_no_distinct_mutant_for_single_row_aggregates():
    assert not [m for op, m in mutants("SELECT count(*) FROM t") if op == "distinct_toggle"]


def test_hack_detectors():
    assert "limit_without_order" in detect("SELECT a FROM t LIMIT 1", "SELECT a FROM t ORDER BY b LIMIT 1", False)
    assert "constant_answer" in detect("SELECT 6", "SELECT count(*) FROM t", False)
    assert "empty_match" in detect("SELECT a FROM t WHERE b = 9", "SELECT a FROM t WHERE b = 1", True)
    assert "distinct_mismatch" in detect("SELECT DISTINCT a FROM t", "SELECT a FROM t", False)
    assert "boundary" in detect("SELECT a FROM t WHERE x >= 3", "SELECT a FROM t WHERE x > 3", False)
    assert detect("SELECT a FROM t WHERE x > 3", "SELECT a FROM t WHERE x > 3", False) == []
