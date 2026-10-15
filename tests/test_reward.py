import json
import sqlite3
from pathlib import Path

import pytest

from mirage.data.schema import read_schema
from mirage.reward.perturb import gold_literals, make_variant
from mirage.reward.rewards import ExecutionReward, Ledger, RewardConfig, reward_value
from mirage.reward.suite import SuiteIndex
from mirage.reward.verifier import Verifier
from mirage.sql.executor import Executor

ORACLE = "SELECT name FROM singer ORDER BY age DESC LIMIT 1"
LUCKY = "SELECT name FROM singer LIMIT 1"  # singer 1 is first AND oldest in the fixture


def _fence(sql):
    return f"```sql\n{sql}\n```"


# ------------------------------------------------------------------- perturbation


def test_variant_keeps_schema_keys_and_foreign_keys(db_paths, tmp_path):
    src = db_paths["concert_hall"]
    out = make_variant(src, tmp_path / "v.sqlite", seed=3)
    assert [t.name for t in read_schema(src)] == [t.name for t in read_schema(out)]
    con = sqlite3.connect(out)
    orphans = con.execute(
        "SELECT count(*) FROM singer_in_concert WHERE singer_id NOT IN (SELECT singer_id FROM singer)"
    ).fetchone()[0]
    assert orphans == 0
    ids = [r[0] for r in con.execute("SELECT singer_id FROM singer")]
    assert len(ids) == len(set(ids))


def test_variant_is_deterministic(db_paths, tmp_path):
    a = make_variant(db_paths["library"], tmp_path / "a.sqlite", seed=7)
    b = make_variant(db_paths["library"], tmp_path / "b.sqlite", seed=7)
    q = "SELECT * FROM book ORDER BY book_id"
    assert sqlite3.connect(a).execute(q).fetchall() == sqlite3.connect(b).execute(q).fetchall()


def test_protected_literals_survive(db_paths, tmp_path):
    lits = gold_literals(["SELECT name FROM singer WHERE country = 'France'"])
    assert "France" in lits
    for seed in range(5):
        v = make_variant(db_paths["concert_hall"], tmp_path / f"v{seed}.sqlite", seed=seed, protected=lits)
        n = sqlite3.connect(v).execute("SELECT count(*) FROM singer WHERE country = 'France'").fetchone()[0]
        assert n >= 1


# ------------------------------------------------------------------- suites


def test_suite_stats_are_consistent(suites_dir):
    rows = SuiteIndex(suites_dir).summary()
    assert len(rows) == 8
    for r in rows:
        assert r["killed_by_original"] + r["survivors"] == r["n_mutants"]
        assert 0 <= r["survivors_killed_by_suite"] <= r["survivors"]
        assert r["variants"]


def test_gold_executes_and_matches_itself_on_every_variant(spider, suites_dir):
    idx = SuiteIndex(suites_dir)
    ex = Executor()
    for e in spider.load("train")[:30]:
        for v in idx.variants(e.db_id):
            assert ex.execute(v, e.gold_sql).ok


def test_suite_index_survives_being_moved(suites_dir, tmp_path):
    import shutil

    moved = tmp_path / "moved"
    shutil.copytree(suites_dir, moved)
    idx = SuiteIndex(moved)
    assert all(Path(p).exists() and str(moved) in p for p in idx.variants("concert_hall"))


# ------------------------------------------------------------------- verifier + reward


def test_lucky_query_is_a_false_positive(verifier):
    v = verifier.verify("concert_hall", ORACLE, _fence(LUCKY))
    assert v.match_single is True
    assert v.match_suite is False
    assert v.false_positive
    assert "limit_without_order" in v.hacks


def test_gold_passes_both(verifier):
    v = verifier.verify("concert_hall", ORACLE, _fence(ORACLE))
    assert v.match_single and v.match_suite and not v.false_positive


def test_no_suite_means_unknown_not_false(db_paths):
    v = Verifier(db_paths).verify("concert_hall", ORACLE, _fence(LUCKY))
    assert v.match_suite is None and not v.false_positive


def test_reward_values():
    from mirage.reward.verifier import Verdict

    base = dict(extracted_how="fence", error_kind="", n_variants=2, pred_empty=False, gold_empty=False)
    fp = Verdict(sql="x", executed=True, match_single=True, match_suite=False, **base)
    runs = Verdict(sql="x", executed=True, match_single=False, match_suite=False, **base)
    junk = Verdict(sql="x", executed=False, match_single=False, match_suite=False, **base)
    assert reward_value(fp, RewardConfig(mode="single")) == 1.0
    assert reward_value(fp, RewardConfig(mode="suite")) == 0.0
    assert reward_value(runs, RewardConfig(shape="shaped")) == 0.2
    assert reward_value(junk, RewardConfig(shape="shaped")) == 0.1
    assert reward_value(runs, RewardConfig()) == 0.0
    with pytest.raises(ValueError):
        RewardConfig(mode="both")


def test_execution_reward_speaks_trl(verifier, tmp_path):
    """The call signature TRL 1.15 uses: dataset columns as kwargs, chat-format
    completions, trainer_state and log_metric injected."""
    ledger = tmp_path / "ledger.jsonl"
    metrics = {}

    class State:
        global_step = 7

    fn = ExecutionReward(verifier, RewardConfig(mode="single"), Ledger(ledger), arm="single")
    comps = [[{"role": "assistant", "content": _fence(s)}] for s in (ORACLE, LUCKY, "SELEC x", ORACLE)]
    rewards = fn(
        prompts=[[]] * 4,
        completions=comps,
        db_id=["concert_hall"] * 4,
        gold_sql=[ORACLE] * 4,
        qid=["q1"] * 4,
        hardness=["hard"] * 4,
        trainer_state=State(),
        log_metric=lambda k, v: metrics.__setitem__(k, v),
        completion_ids=None,
    )
    assert rewards == [1.0, 1.0, 0.0, 1.0]
    assert fn.__name__ == "exec_single"
    rows = [json.loads(x) for x in ledger.read_text().splitlines()]
    assert len(rows) == 4 and all(r["step"] == 7 for r in rows)
    assert [r["false_positive"] for r in rows] == [False, True, False, False]
    assert metrics["audit/false_positive_share"] == pytest.approx(1 / 3)
    assert metrics["audit/groups_with_signal"] == 1.0

    suite_fn = ExecutionReward(verifier, RewardConfig(mode="suite"), arm="suite")
    assert suite_fn(completions=comps, db_id=["concert_hall"] * 4, gold_sql=[ORACLE] * 4, qid=["q"] * 4) == [
        1.0, 0.0, 0.0, 1.0,
    ]
