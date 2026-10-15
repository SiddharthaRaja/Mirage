"""Evaluation: generate once, score as often as needed.

`generate_predictions` runs the model and writes raw completions:

    {"qid", "db_id", "hardness", "greedy": "<text>", "samples": ["<text>", ...]}

`score_predictions` turns that file into numbers, on a CPU, with the same
`Verifier` the reward uses. It reports, overall and per difficulty:

* **EX** — greedy answer matches gold on the original database (Spider's
  execution accuracy).
* **Suite EX** — greedy answer matches on the original and every perturbed
  variant. The gap between EX and Suite EX is the evaluation-side false
  positive rate, and it is reported for every model and arm.
* **pass@k** (single and suite) from the sampled completions, with the
  unbiased estimator.
* **Majority vote by execution result** — cluster the samples by what they
  return on the original database and answer with the biggest cluster. This
  is the cheap "test-time compute" baseline that training should beat or at
  least be compared to.
* A **failure taxonomy** of wrong greedy answers.

Comparisons between two scored runs (`compare_runs`) use a paired bootstrap
over questions plus an exact McNemar test.
"""

from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from mirage.data.prompts import PromptSpec, build_messages
from mirage.data.schema import render_schema
from mirage.data.spider import Example, read_jsonl, write_jsonl
from mirage.eval.failures import classify_failure, summarize
from mirage.eval.stats import mcnemar_exact, mean_ci, paired_diff_ci, pass_at_k
from mirage.reward.verifier import Verifier
from mirage.sql.compare import normalize_rows


def build_prompts(examples: list[Example], db_paths: dict[str, str], spec: PromptSpec) -> list[list[dict]]:
    return [
        build_messages(render_schema(db_paths[e.db_id], spec.sample_values), e.question, spec) for e in examples
    ]


def generate_predictions(
    examples: list[Example],
    generator,
    db_paths: dict[str, str],
    out_path: str | Path,
    spec: PromptSpec | None = None,
    n_samples: int = 0,
    temperature: float = 0.8,
    chunk: int = 64,
    progress=None,
) -> Path:
    """Write predictions.jsonl. Resumable: questions already in the file are
    skipped, so a Kaggle session that dies halfway loses nothing."""
    spec = spec or PromptSpec()
    out_path = Path(out_path)
    done: set[str] = set()
    if out_path.exists():
        done = {r["qid"] for r in read_jsonl(out_path)}
    todo = [e for e in examples if e.qid not in done]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for i in range(0, len(todo), chunk):
        part = todo[i : i + chunk]
        prompts = build_prompts(part, db_paths, spec)
        t0 = time.perf_counter()
        greedy = generator.generate(prompts, 1, 0.0)
        samples = generator.generate(prompts, n_samples, temperature) if n_samples else [[] for _ in part]
        dt = time.perf_counter() - t0
        with out_path.open("a", encoding="utf-8") as f:
            for e, g, s in zip(part, greedy, samples, strict=True):
                f.write(
                    json.dumps(
                        {
                            "qid": e.qid,
                            "db_id": e.db_id,
                            "hardness": e.hardness,
                            "question": e.question,
                            "gold_sql": e.gold_sql,
                            "greedy": g[0] if g else "",
                            "samples": s,
                            "gen_s_per_question": round(dt / len(part), 4),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        if progress:
            progress(min(i + chunk, len(todo)), len(todo))
    return out_path


def _signature(verifier: Verifier, db_id: str, sql: str | None):
    if not sql:
        return None
    r = verifier.ex.execute(verifier.db_paths[db_id], sql)
    if not r.ok:
        return None
    rows = normalize_rows(r.rows)
    try:
        return (r.n_cols, tuple(sorted(Counter(map(repr, rows)).items())))
    except TypeError:
        return None


def score_predictions(pred_path: str | Path, verifier: Verifier, label: str = "", ks=(1, 4, 8)) -> dict[str, Any]:
    preds = read_jsonl(pred_path)
    items: list[dict] = []
    for p in preds:
        v = verifier.verify(p["db_id"], p["gold_sql"], p["greedy"])
        item = {
            "qid": p["qid"],
            "db_id": p["db_id"],
            "hardness": p.get("hardness", "unknown"),
            "question": p.get("question", ""),
            "gold_sql": p["gold_sql"],
            "pred_sql": v.sql,
            "ex": float(v.match_single),
            "suite_ex": float(bool(v.match_suite)) if v.match_suite is not None else None,
            "false_positive": v.false_positive,
            "executed": v.executed,
            "error_kind": v.error_kind,
            "hacks": v.hacks,
            "failure": None if v.match_single else classify_failure(v.sql, p["gold_sql"], v.executed, v.error_kind),
        }
        samples = p.get("samples") or []
        if samples:
            sv = [verifier.verify(p["db_id"], p["gold_sql"], s) for s in samples]
            n = len(sv)
            c_single = sum(x.match_single for x in sv)
            c_suite = sum(bool(x.match_suite) for x in sv)
            for k in ks:
                if k <= n:
                    item[f"pass@{k}"] = pass_at_k(n, c_single, k)
                    item[f"suite_pass@{k}"] = pass_at_k(n, c_suite, k)
            # majority vote over execution results on the original database
            sigs = [_signature(verifier, p["db_id"], x.sql) for x in sv]
            counts = Counter(s for s in sigs if s is not None)
            if counts:
                top = counts.most_common(1)[0][0]
                winner = next(i for i, s in enumerate(sigs) if s == top)
                item["maj_ex"] = float(sv[winner].match_single)
                item["maj_suite_ex"] = float(bool(sv[winner].match_suite))
            else:
                item["maj_ex"] = item["maj_suite_ex"] = 0.0
            item["n_samples"] = n
        items.append(item)
    return aggregate(items, label=label, source=str(pred_path))


def _agg_block(items: list[dict]) -> dict[str, Any]:
    out: dict[str, Any] = {"n": len(items)}
    keys = ["ex", "suite_ex", "maj_ex", "maj_suite_ex"] + sorted(
        {k for i in items for k in i if k.startswith(("pass@", "suite_pass@"))}
    )
    for k in keys:
        vals = [i[k] for i in items if i.get(k) is not None]
        if vals:
            m, lo, hi = mean_ci(vals)
            out[k] = {"mean": m, "ci95": (lo, hi), "n": len(vals)}
    rewarded = [i for i in items if i["ex"]]
    out["false_positive_share"] = (
        sum(i["false_positive"] for i in rewarded) / len(rewarded) if rewarded else float("nan")
    )
    out["executed"] = sum(i["executed"] for i in items) / len(items) if items else float("nan")
    return out


def aggregate(items: list[dict], label: str = "", source: str = "") -> dict[str, Any]:
    by_h: dict[str, list[dict]] = defaultdict(list)
    for i in items:
        by_h[i["hardness"]].append(i)
    order = ["easy", "medium", "hard", "extra", "unknown"]
    failures = summarize([i["failure"] for i in items if i["failure"]])
    return {
        "label": label,
        "source": source,
        "overall": _agg_block(items),
        "by_hardness": {h: _agg_block(by_h[h]) for h in order if by_h.get(h)},
        "failures": failures,
        "error_kinds": dict(Counter(i["error_kind"] for i in items if not i["executed"])),
        "items": items,
    }


def compare_runs(a: dict, b: dict, metric: str = "ex") -> dict[str, Any]:
    """Paired comparison of two scored runs on the questions they share."""
    ai = {i["qid"]: i for i in a["items"] if i.get(metric) is not None}
    bi = {i["qid"]: i for i in b["items"] if i.get(metric) is not None}
    common = sorted(ai.keys() & bi.keys())
    x = [ai[q][metric] for q in common]
    y = [bi[q][metric] for q in common]
    res = paired_diff_ci(x, y)
    b_only = sum(1 for p, q in zip(x, y, strict=True) if q > p)
    a_only = sum(1 for p, q in zip(x, y, strict=True) if p > q)
    res["mcnemar_p"] = mcnemar_exact(b_only, a_only)
    res["metric"] = metric
    res["a"] = a.get("label", "a")
    res["b"] = b.get("label", "b")
    return res


def save_scores(scores: dict, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "scored_items.jsonl", scores["items"])
    summary = {k: v for k, v in scores.items() if k != "items"}
    (out / "scores.json").write_text(json.dumps(summary, indent=1, default=list), encoding="utf-8")
    return out / "scores.json"


def load_scores(out_dir: str | Path) -> dict:
    out = Path(out_dir)
    s = json.loads((out / "scores.json").read_text(encoding="utf-8"))
    s["items"] = read_jsonl(out / "scored_items.jsonl")
    return s
