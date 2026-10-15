"""Optional: score dev with the official Spider test-suite evaluator.

Mirage's own suites are an approximation of test-suite accuracy, built by a
simpler procedure. For the final dev number, the official evaluator
(github.com/taoyds/test-suite-sql-eval) with its released test-suite
databases is the reference, and reporting both is what makes the comparison
credible.

This writes the evaluator's input files, runs it as a subprocess, saves its
full output next to the run, and makes a best-effort attempt to read the
overall execution accuracy from that output. The output format belongs to
that repository and may change, so the raw text is always kept and the parsed
number is labelled as parsed.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

from mirage.data.spider import read_jsonl
from mirage.sql.extract import extract_sql


def run_official(
    predictions: str | Path,
    evaluator_dir: str | Path,
    db_dir: str | Path,
    tables_json: str | Path,
    out_dir: str | Path,
) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    preds = read_jsonl(predictions)
    gold_lines, pred_lines = [], []
    for p in preds:
        gold_lines.append(f"{p['gold_sql']}\t{p['db_id']}")
        sql = extract_sql(p["greedy"]).sql or "SELECT 1"
        pred_lines.append(" ".join(sql.split()))
    (out / "official_gold.txt").write_text("\n".join(gold_lines) + "\n", encoding="utf-8")
    (out / "official_pred.txt").write_text("\n".join(pred_lines) + "\n", encoding="utf-8")
    script = Path(evaluator_dir) / "evaluation.py"
    if not script.exists():
        return {"ok": False, "error": f"{script} not found; clone test-suite-sql-eval there"}
    cmd = [
        sys.executable,
        str(script),
        "--gold", str(out / "official_gold.txt"),
        "--pred", str(out / "official_pred.txt"),
        "--db", str(db_dir),
        "--table", str(tables_json),
        "--etype", "exec",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(evaluator_dir))
    (out / "official_stdout.txt").write_text(proc.stdout + "\n--- stderr ---\n" + proc.stderr, encoding="utf-8")
    parsed = None
    for line in proc.stdout.splitlines():
        if line.strip().lower().startswith("execution"):
            nums = re.findall(r"\d+\.\d+", line)
            if nums:
                parsed = float(nums[-1])  # the 'all' column is last
    return {"ok": proc.returncode == 0, "returncode": proc.returncode, "parsed_all_exec": parsed, "cmd": cmd}
