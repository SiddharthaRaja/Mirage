import json

import pytest

from mirage.data.hardness import classify_hardness
from mirage.data.prompts import build_messages, format_answer
from mirage.data.schema import read_schema, render_schema, tables_json_entry
from mirage.data.spider import Spider, make_qid
from mirage.data.splits import assert_disjoint, split_by_database
from mirage.errors import DataError
from mirage.sql.extract import extract_sql


def test_fixture_loads_and_dev_is_held_out(spider):
    rep = spider.check()
    assert rep["train"]["n_questions"] > 50
    assert rep["dev"]["n_databases"] == 2
    assert rep["train_dev_db_overlap"] == []
    assert not rep["train"]["missing_db_files"]


def test_qid_is_a_content_hash_not_an_index():
    a = make_qid("db", "How many?", "SELECT count(*) FROM t")
    assert a == make_qid("db", "  How many?  ", "SELECT count(*) FROM t ")
    assert a != make_qid("db", "How many?", "SELECT count(*) FROM u")


def test_missing_folder_is_a_clean_error(tmp_path):
    with pytest.raises(DataError):
        Spider(tmp_path / "nope")


def test_split_is_by_database_and_deterministic(spider):
    pool = spider.load("train")
    tr1, va1 = split_by_database(list(pool), 0.2)
    tr2, va2 = split_by_database(list(pool), 0.2)
    assert {e.qid for e in va1} == {e.qid for e in va2}
    assert not ({e.db_id for e in tr1} & {e.db_id for e in va1})
    assert va1 and tr1
    assert_disjoint(("train", tr1), ("val", va1), ("dev", spider.load("dev")))


def test_assert_disjoint_catches_overlap(spider):
    ex = spider.load("train")
    with pytest.raises(AssertionError):
        assert_disjoint(("a", ex[:5]), ("b", ex[:5]))


@pytest.mark.parametrize(
    "sql,level",
    [
        ("SELECT count(*) FROM singer", "easy"),
        ("SELECT name FROM singer WHERE age > 30", "easy"),
        ("SELECT name, age FROM singer WHERE age > 30", "medium"),
        ("SELECT year, count(*) FROM concert GROUP BY year", "medium"),
        ("SELECT name FROM singer WHERE singer_id NOT IN (SELECT singer_id FROM singer_in_concert)", "hard"),
        (
            "SELECT a FROM t1 JOIN t2 ON t1.x = t2.x WHERE b > 1 GROUP BY a HAVING count(*) > 2 ORDER BY a LIMIT 3",
            "extra",
        ),
    ],
)
def test_hardness_follows_spider_rule(sql, level):
    assert classify_hardness(sql) == level


def test_hardness_unparseable_is_unknown():
    assert classify_hardness("SELEC FROM WHERE") in ("unknown", "easy")


def test_schema_rendering_includes_keys(db_paths):
    text = render_schema(db_paths["concert_hall"])
    assert "CREATE TABLE singer" in text
    assert "PRIMARY KEY (singer_id)" in text
    assert "FOREIGN KEY (singer_id) REFERENCES singer(singer_id)" in text
    with_vals = render_schema(db_paths["concert_hall"], 2)
    assert "-- e.g." in with_vals and "'France'" in with_vals


def test_tables_json_entry_resolves_foreign_keys(db_paths):
    t = tables_json_entry("concert_hall", db_paths["concert_hall"])
    assert t["table_names_original"] == ["singer", "concert", "singer_in_concert"]
    assert len(t["foreign_keys"]) == 2
    json.dumps(t)
    assert len(read_schema(db_paths["concert_hall"])) == 3


def test_prompt_and_answer_format_round_trip():
    msgs = build_messages("CREATE TABLE t (a INT);", "How many?")
    assert msgs[0]["role"] == "system" and "Question: How many?" in msgs[1]["content"]
    assert extract_sql(format_answer("SELECT count(*) FROM t")).sql == "SELECT count(*) FROM t"
