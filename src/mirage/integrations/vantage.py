"""Part D: cost per correct query, measured through Vantage.

Accuracy alone does not say whether a trained model is worth serving.
"Cost per correct query" joins two measurements that are each only half the
story:

* **Vantage** measures what serving costs: throughput, latency percentiles,
  goodput against an SLO, and dollars per 1,000 requests, under real load on
  real hardware. It knows nothing about SQL.
* **Mirage** measures how often the answers are right, on the original
  database and under the suite. It knows nothing about latency.

    cost per 1,000 correct = (cost per 1,000 requests) / accuracy
    correct per GPU-hour   = achieved requests/s * 3600 * accuracy

The interval on accuracy carries through to the cost (cost is monotone in
accuracy, so the interval endpoints map directly). Reported for both EX and
suite EX: a model whose suite accuracy is lower is more expensive per
*actually* correct answer than its EX suggests.

Two things keep the join honest:

1. **Same prompt.** `export_workload` writes Vantage workload rows with the
   exact mirage prompt (`prompt` field) and the sweep sets the mirage system
   prompt. That needs the small Vantage patch in
   `integrations/vantage/vantage-raw-prompt.patch` (adds verbatim-prompt rows
   and a sweep-level `system:` key; Vantage's own test suite passes with it).
2. **Same weights, same process.** Base and trained adapters are served by one
   `vllm serve --enable-lora` process, so the cost comparison has no hardware
   confound. Note that a LoRA adapter served unmerged is slightly slower than
   the base model; that overhead is real serving cost and belongs in the number.
"""

from __future__ import annotations

import json
from pathlib import Path

from mirage.data.prompts import SYSTEM, USER_TEMPLATE, PromptSpec
from mirage.data.schema import render_schema
from mirage.data.spider import Example, write_jsonl
from mirage.errors import DataError


def export_workload(
    examples: list[Example], db_paths: dict[str, str], out_path: str | Path, spec: PromptSpec | None = None
) -> Path:
    spec = spec or PromptSpec()
    rows = []
    for e in examples:
        schema = render_schema(db_paths[e.db_id], spec.sample_values)
        rows.append(
            {
                "id": e.qid,
                "question": e.question,
                "context": schema,
                "prompt": USER_TEMPLATE.format(schema=schema, question=e.question),
                "reference_answer": e.gold_sql,
                "meta": {"db_id": e.db_id, "hardness": e.hardness, "source": "mirage", "prompt_version": spec.version},
            }
        )
    write_jsonl(out_path, rows)
    return Path(out_path)


def system_prompt() -> str:
    return SYSTEM


def _load_summaries(run_dir: Path) -> list[dict]:
    p = run_dir / "summaries.json"
    if not p.exists():
        raise DataError(f"no summaries.json in {run_dir} (point at a Vantage run directory)")
    return json.loads(p.read_text(encoding="utf-8"))


def _pick(summaries: list[dict], key: str) -> dict:
    """Match a Vantage summary by config_id, label or group."""
    for s in summaries:
        if key in (s.get("config_id"), s.get("label"), (s.get("params") or {}).get("group")):
            return s
    raise DataError(f"no Vantage run matching {key!r}; have {[s.get('config_id') for s in summaries]}")


def cost_per_correct(vantage_run_dir: str | Path, pairs: dict[str, dict], gpu_hourly_usd: float | None = None) -> list[dict]:
    """`pairs`: {display name: {"vantage": <config id/label>, "scores": <scores dict>}}."""
    summaries = _load_summaries(Path(vantage_run_dir))
    out = []
    for name, spec in pairs.items():
        s = _pick(summaries, spec["vantage"])
        scores = spec["scores"]["overall"]
        cost = (s.get("cost") or {}).get("usd_per_1k_requests")
        qps = s.get("achieved_qps") or 0.0
        if cost is None and gpu_hourly_usd and qps:
            cost = gpu_hourly_usd / 3600 / qps * 1000
        row = {
            "name": name,
            "vantage_config": s.get("config_id"),
            "achieved_qps": qps,
            "p95_total_s": (s.get("total") or {}).get("p95"),
            "p95_ttft_s": (s.get("ttft") or {}).get("p95"),
            "goodput_ratio": s.get("goodput_ratio"),
            "usd_per_1k_requests": cost,
            "token_counts_estimated": s.get("token_counts_estimated", False),
        }
        for metric in ("ex", "suite_ex"):
            m = scores.get(metric)
            if not m:
                continue
            acc, (lo, hi) = m["mean"], m["ci95"]
            row[f"{metric}"] = acc
            row[f"{metric}_ci95"] = (lo, hi)
            if cost is not None and acc > 0:
                row[f"usd_per_1k_correct_{metric}"] = cost / acc
                row[f"usd_per_1k_correct_{metric}_ci95"] = (
                    cost / hi if hi > 0 else float("inf"),
                    cost / lo if lo > 0 else float("inf"),
                )
            row[f"correct_per_gpu_hour_{metric}"] = qps * 3600 * acc
        out.append(row)
    return out


VLLM_SERVE_HINT = """\
# Serve the starting model and both GRPO adapters from ONE process:
vllm serve {start_model} --port 8000 --dtype half --max-model-len 4096 \\
  --enable-lora --max-lora-rank {rank} \\
  --lora-modules mirage-single={single_adapter} mirage-suite={suite_adapter}
"""
