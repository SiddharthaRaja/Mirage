import json
import random

import pytest

from mirage.analysis.audit import audit_ledger
from mirage.analysis.simulate import TabularPolicy
from mirage.eval.evaluate import (
    compare_runs,
    generate_predictions,
    load_scores,
    save_scores,
    score_predictions,
)
from mirage.eval.failures import classify_failure
from mirage.eval.stats import cluster_rate_ci, mcnemar_exact, mean_ci, paired_diff_ci, pass_at_k

# ------------------------------------------------------------------- stats


def test_pass_at_k_matches_known_values():
    assert pass_at_k(8, 0, 1) == 0.0
    assert pass_at_k(8, 8, 1) == 1.0
    assert pass_at_k(8, 4, 1) == pytest.approx(0.5)
    assert pass_at_k(8, 1, 8) == 1.0
    assert pass_at_k(10, 3, 2) == pytest.approx(1 - (7 * 6) / (10 * 9))


def test_mean_ci_brackets_the_mean():
    m, lo, hi = mean_ci([0, 1] * 50)
    assert lo < m < hi and m == 0.5


def test_paired_diff_detects_a_real_gain_and_not_noise():
    a = [0] * 100
    b = [1] * 30 + [0] * 70
    assert paired_diff_ci(a, b)["significant"]
    rng = random.Random(0)
    x = [rng.random() < 0.5 for _ in range(60)]
    assert not paired_diff_ci(x, x)["significant"]


def test_cluster_interval_is_wider_than_naive():
    numer = ([1] * 8 + [0] * 8) * 5
    denom = [1] * 80
    clusters = [f"q{i // 8}" for i in range(80)]  # each question all-right or all-wrong
    _, lo, hi = cluster_rate_ci(numer, denom, clusters)
    _, nlo, nhi = mean_ci(numer)
    assert (hi - lo) >= (nhi - nlo)


def test_mcnemar():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(10, 0) < 0.01


# ------------------------------------------------------------------- failures


@pytest.mark.parametrize(
    "pred,executed,err,kind",
    [
        (None, False, "empty_sql", "no_sql"),
        ("SELEC a", False, "syntax", "syntax"),
        ("SELECT nope FROM t", False, "no_such_column", "schema_link"),
        ("SELECT a FROM u", True, "", "wrong_tables"),
        ("SELECT b FROM t WHERE x > 1", True, "", "wrong_select"),
        ("SELECT a FROM t WHERE x < 1", True, "", "filter"),
        ("SELECT a FROM t WHERE x > 2", True, "", "values"),
        ("SELECT a FROM t WHERE x > 1 ORDER BY a", True, "", "ordering"),
    ],
)
def test_failure_taxonomy(pred, executed, err, kind):
    assert classify_failure(pred, "SELECT a FROM t WHERE x > 1", executed, err) == kind


# ------------------------------------------------------------------- evaluation


def test_generate_score_compare_round_trip(spider, verifier, db_paths, tmp_path):
    dev = spider.load("dev")
    pol = TabularPolicy(dev, seed=0)
    p = generate_predictions(dev, pol, db_paths, tmp_path / "a" / "predictions.jsonl", n_samples=4, temperature=1.0)
    rows = [json.loads(x) for x in p.read_text().splitlines()]
    assert len(rows) == len(dev) and all(len(r["samples"]) == 4 for r in rows)
    # resumable: a second call adds nothing
    generate_predictions(dev, pol, db_paths, p, n_samples=4)
    assert len(p.read_text().splitlines()) == len(dev)

    sc = score_predictions(p, verifier, label="a")
    o = sc["overall"]
    assert o["n"] == len(dev)
    assert 0 <= o["suite_ex"]["mean"] <= o["ex"]["mean"] <= 1
    assert "pass@4" in o and o["pass@4"]["mean"] >= o["pass@1"]["mean"]
    assert set(sc["by_hardness"]) <= {"easy", "medium", "hard", "extra", "unknown"}
    save_scores(sc, tmp_path / "a")
    again = load_scores(tmp_path / "a")
    assert again["overall"]["ex"]["mean"] == o["ex"]["mean"]
    res = compare_runs(again, again, "ex")
    assert res["diff"] == 0 and not res["significant"]


# ------------------------------------------------------------------- audit


def _ledger(path, fp_rate_by_step):
    rng = random.Random(0)
    with open(path, "w") as f:
        for step, rate in enumerate(fp_rate_by_step):
            for q in range(10):
                for _ in range(8):
                    single = rng.random() < 0.6
                    fp = single and rng.random() < rate
                    f.write(json.dumps({
                        "arm": "single", "step": step, "qid": f"q{q}", "db_id": "d", "hardness": "easy",
                        "reward": float(single), "sql": "SELECT 1", "how": "fence", "executed": True,
                        "error_kind": "", "match_single": single, "match_suite": single and not fp,
                        "false_positive": fp, "gold_empty": False, "hacks": ["empty_match"] if fp else [],
                        "completion_chars": 10,
                    }) + "\n")


def test_audit_detects_growing_false_positives(tmp_path):
    _ledger(tmp_path / "l.jsonl", [0.02 + 0.015 * s for s in range(20)])
    a = audit_ledger(tmp_path / "l.jsonl")["arms"]["single"]
    assert a["drift"]["significant"] and a["drift"]["diff"] > 0
    assert len(a["curve"]) == 10
    assert a["hacks_in_false_positives"]["late"]["empty_match"] > a["hacks_in_false_positives"]["early"]["empty_match"]


def test_audit_does_not_invent_drift(tmp_path):
    _ledger(tmp_path / "l.jsonl", [0.1] * 20)
    a = audit_ledger(tmp_path / "l.jsonl")["arms"]["single"]
    assert not a["drift"]["significant"]
