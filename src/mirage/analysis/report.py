"""The written report: charts, tables, generated findings, and caveats.

`mirage report study.yaml` reads a study file that points at the pieces
(suites, each arm's ledger and manifest, each scored evaluation, optionally
the Vantage cost join) and writes `REPORT.md`, a standalone `report.html`
with the charts embedded, and `report.json` with every number used.

Findings at the top are generated from the numbers, not written by hand, so
they cannot drift from the tables beneath them. Every statement about a
difference checks its interval first; if the interval spans zero the report
says "no detectable difference", whatever the point estimate is.

The report ends with "What this run does not establish". That section is
part of the result.
"""

from __future__ import annotations

import base64
import html
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import yaml  # noqa: E402

from mirage.analysis.audit import audit_ledger  # noqa: E402
from mirage.eval.evaluate import compare_runs, load_scores  # noqa: E402
from mirage.reward.suite import SuiteIndex  # noqa: E402
from mirage.sql.hacks import HACK_KINDS  # noqa: E402
from mirage.store import arms_comparable, read_manifest  # noqa: E402

# Reference categorical palette (light mode), assigned in fixed order by role.
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BASE_GRAY = "#8a8985"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
STYLES = ["-", "--", "-.", ":"]
MARKERS = ["o", "s", "D", "^"]
# EX vs suite EX is a metric distinction, not an arm: neutral inks, shape-coded.
METRIC_EX, METRIC_SUITE = "#a3a29c", INK


def _style():
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK2,
            "axes.titlecolor": INK,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": INK2,
            "ytick.color": INK2,
            "font.size": 10,
            "legend.frameon": False,
            "legend.labelcolor": INK2,
            "lines.linewidth": 2,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
        }
    )


def _pct(x: float | None, digits: int = 1) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{100 * x:.{digits}f}%"


def _ci(ci, digits: int = 1) -> str:
    if not ci or any(c is None or (isinstance(c, float) and math.isnan(c)) for c in ci):
        return ""
    return f"[{100 * ci[0]:.{digits}f}, {100 * ci[1]:.{digits}f}]"


# --------------------------------------------------------------------------- #
# charts


def chart_fidelity(audits: dict[str, dict], path: Path) -> Path | None:
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    drew = False
    for i, (arm, a) in enumerate(audits.items()):
        pts = [c for c in a["curve"] if not math.isnan(c["false_positive_share"][0])]
        if len(pts) < 2:
            continue
        x = [(c["steps"][0] + c["steps"][1]) / 2 for c in pts]
        y = [c["false_positive_share"][0] for c in pts]
        lo = [c["false_positive_share"][1] for c in pts]
        hi = [c["false_positive_share"][2] for c in pts]
        col = SERIES[i % len(SERIES)]
        ax.fill_between(x, lo, hi, color=col, alpha=0.12, linewidth=0)
        ax.plot(x, y, color=col, linestyle=STYLES[i % 4], marker=MARKERS[i % 4], markersize=5, label=f"{arm} arm")
        ax.annotate(f"{arm}", (x[-1], y[-1]), xytext=(6, 0), textcoords="offset points", color=INK2, va="center")
        drew = True
    if not drew:
        plt.close(fig)
        return None
    ax.set_title("How often the single-database reward was wrong, during training")
    ax.set_xlabel("training step")
    ax.set_ylabel("rewarded rollouts the suite rejects")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right")
    fig.savefig(path)
    plt.close(fig)
    return path


def chart_learning(audits: dict[str, dict], path: Path) -> Path | None:
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    drew = False
    for i, (arm, a) in enumerate(audits.items()):
        pts = a["curve"]
        if len(pts) < 2:
            continue
        x = [(c["steps"][0] + c["steps"][1]) / 2 for c in pts]
        col = SERIES[i % len(SERIES)]
        ax.plot(x, [c["match_suite"][0] for c in pts], color=col, marker=MARKERS[i % 4], markersize=5, label=f"{arm}: suite match")
        ax.plot(x, [c["match_single"][0] for c in pts], color=col, linestyle="--", linewidth=1.5, label=f"{arm}: single-db match")
        drew = True
    if not drew:
        plt.close(fig)
        return None
    ax.set_title("Rollout accuracy during training (training questions)")
    ax.set_xlabel("training step")
    ax.set_ylabel("share of rollouts correct")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right", fontsize=8)
    fig.savefig(path)
    plt.close(fig)
    return path


def chart_blind_spot(suite_rows: list[dict], path: Path) -> Path | None:
    agg: dict[str, list[int]] = {}
    for s in suite_rows:
        for op, d in s.get("by_operator", {}).items():
            a = agg.setdefault(op, [0, 0, 0])
            a[0] += d["n"]
            a[1] += d["survive_original"]
            a[2] += d["caught_by_suite"]
    agg = {k: v for k, v in agg.items() if v[0]}
    if not agg:
        return None
    ops = sorted(agg, key=lambda k: agg[k][1] / agg[k][0])
    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(ops) + 1.2))
    y = range(len(ops))
    surv = [agg[o][1] / agg[o][0] for o in ops]
    still = [(agg[o][1] - agg[o][2]) / agg[o][0] for o in ops]
    # Not arm colours: this chart is about the check itself, before any arm exists.
    ax.barh(list(y), surv, color="#c3c2b7", height=0.6, label="accepted by the original database")
    ax.barh(list(y), still, color=INK2, height=0.6, label="still accepted with the suite")
    for i, o in enumerate(ops):
        ax.text(surv[i] + 0.01, i, f"{agg[o][1]}/{agg[o][0]}", va="center", color=INK2, fontsize=8)
    ax.set_yticks(list(y), ops)
    ax.set_xlim(0, 1.08)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title("Near-miss queries the reward pays out for, before any training")
    ax.set_xlabel("share of mutants of gold queries judged correct")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="y", visible=False)
    fig.savefig(path)
    plt.close(fig)
    return path


def chart_accuracy(evals: dict[str, dict], path: Path) -> Path | None:
    if not evals:
        return None
    fig, ax = plt.subplots(figsize=(7.2, 0.6 * len(evals) + 1.4))
    names = list(evals)
    for row, name in enumerate(names):
        o = evals[name]["overall"]
        for k, (metric, col, mk) in enumerate((("ex", METRIC_EX, "o"), ("suite_ex", METRIC_SUITE, "s"))):
            m = o.get(metric)
            if not m:
                continue
            yy = row + (0.15 if k == 0 else -0.15)
            ax.errorbar(
                m["mean"], yy, xerr=[[m["mean"] - m["ci95"][0]], [m["ci95"][1] - m["mean"]]],
                fmt=mk, color=col, markersize=7, capsize=3, linewidth=1.5,
                label=("EX (original database)" if metric == "ex" else "suite EX") if row == 0 else None,
            )
            ax.text(m["ci95"][1] + 0.01, yy, _pct(m["mean"]), va="center", fontsize=8, color=INK2)
    ax.set_yticks(range(len(names)), names)
    ax.set_xlim(0, 1.1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title("Greedy accuracy on the evaluation set, with 95% intervals", pad=26)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=8)
    ax.grid(axis="y", visible=False)
    ax.invert_yaxis()
    fig.savefig(path)
    plt.close(fig)
    return path


def chart_hardness(evals: dict[str, dict], path: Path, metric: str = "suite_ex") -> Path | None:
    levels = [h for h in ("easy", "medium", "hard", "extra") if any(h in e["by_hardness"] for e in evals.values())]
    if not levels or not evals:
        return None
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    w = 0.8 / len(evals)
    for i, (name, e) in enumerate(evals.items()):
        xs, ys = [], []
        for j, h in enumerate(levels):
            b = e["by_hardness"].get(h, {}).get(metric)
            if b:
                xs.append(j - 0.4 + w * (i + 0.5))
                ys.append(b["mean"])
        col = BASE_GRAY if i == 0 else SERIES[(i - 1) % len(SERIES)]
        ax.bar(xs, ys, width=w * 0.92, color=col, label=name, edgecolor=SURFACE, linewidth=1)
    ax.set_xticks(range(len(levels)), [f"{h}\n(n={next(iter(evals.values()))['by_hardness'].get(h, {}).get('n', 0)})" for h in levels])
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylim(0, 1)
    ax.set_title(f"{'Suite EX' if metric == 'suite_ex' else 'EX'} by Spider difficulty (approximate labels)", pad=26)
    ax.legend(fontsize=8, ncol=len(evals), loc="lower left", bbox_to_anchor=(0, 1.0))
    ax.grid(axis="x", visible=False)
    fig.savefig(path)
    plt.close(fig)
    return path


def chart_hacks(audits: dict[str, dict], path: Path) -> Path | None:
    kinds = [k for k in HACK_KINDS if any(a["hacks_in_false_positives"]["early"][k] + a["hacks_in_false_positives"]["late"][k] for a in audits.values())]
    if not kinds:
        return None
    fig, axes = plt.subplots(1, len(audits), figsize=(7.2, 0.45 * len(kinds) + 1.3), sharey=True, squeeze=False)
    for ax, (i, (arm, a)) in zip(axes[0], enumerate(audits.items()), strict=False):
        early = [a["hacks_in_false_positives"]["early"][k] for k in kinds]
        late = [a["hacks_in_false_positives"]["late"][k] for k in kinds]
        y = range(len(kinds))
        ax.barh([v + 0.18 for v in y], early, height=0.34, color=BASE_GRAY, label="first half")
        ax.barh([v - 0.18 for v in y], late, height=0.34, color=SERIES[i % 4], label="second half")
        ax.set_title(f"{arm} arm", fontsize=10)
        ax.set_yticks(list(y), kinds)
        ax.grid(axis="y", visible=False)
        ax.invert_yaxis()
    from matplotlib.patches import Patch

    fig.legend(
        [Patch(color=BASE_GRAY), Patch(facecolor=SURFACE, edgecolor=INK2)],
        ["first half of training", "second half (arm colour)"],
        loc="lower center", bbox_to_anchor=(0.5, 0.0), fontsize=8, frameon=False, ncol=2,
    )
    fig.suptitle("Shapes of false positives (count of rewarded rollouts)", x=0.02, y=1.0, ha="left",
                 fontweight="bold", color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0.08, 1, 0.93))
    fig.savefig(path)
    plt.close(fig)
    return path


def chart_cost(cost_rows: list[dict], path: Path) -> Path | None:
    rows = [r for r in cost_rows if r.get("usd_per_1k_correct_suite_ex") or r.get("usd_per_1k_correct_ex")]
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(7.2, 0.6 * len(rows) + 1.4))
    for i, r in enumerate(rows):
        for k, (metric, col, mk) in enumerate((("ex", METRIC_EX, "o"), ("suite_ex", METRIC_SUITE, "s"))):
            v = r.get(f"usd_per_1k_correct_{metric}")
            if v is None:
                continue
            lo, hi = r[f"usd_per_1k_correct_{metric}_ci95"]
            yy = i + (0.15 if k == 0 else -0.15)
            ax.errorbar(v, yy, xerr=[[v - lo], [min(hi, v * 5) - v]], fmt=mk, color=col, capsize=3, markersize=7,
                        label=("per correct (EX)" if metric == "ex" else "per correct (suite EX)") if i == 0 else None)
            ax.text(min(hi, v * 5), yy, f"  ${v:.3g}", va="center", fontsize=8, color=INK2)
    ax.set_yticks(range(len(rows)), [r["name"] for r in rows])
    ax.set_xlim(left=0)
    ax.set_title("Serving cost per 1,000 correct answers (USD)", pad=26)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=8)
    ax.grid(axis="y", visible=False)
    ax.invert_yaxis()
    fig.savefig(path)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# study loading


def load_study(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    s = yaml.safe_load(p.read_text(encoding="utf-8"))
    base = p.parent

    def rel(x):
        return str((base / x).resolve()) if x and not Path(x).is_absolute() else x

    s["_dir"] = str(base)
    s["suites"] = rel(s.get("suites"))
    s["arms"] = {k: rel(v) for k, v in (s.get("arms") or {}).items()}
    s["evals"] = {k: rel(v) for k, v in (s.get("evals") or {}).items()}
    s["cost"] = rel(s.get("cost"))
    s["out"] = rel(s.get("out", "report"))
    return s


def build_report(study_path: str | Path) -> Path:
    _style()
    st = load_study(study_path)
    out = Path(st["out"])
    charts_dir = out / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    simulated = bool(st.get("simulated"))

    suite_rows = SuiteIndex(st["suites"]).summary() if st.get("suites") else []
    audits: dict[str, dict] = {}
    manifests = []
    for arm, d in st["arms"].items():
        ledger = Path(d) / "ledger.jsonl"
        if ledger.exists():
            a = audit_ledger(ledger)
            audits[arm] = next(iter(a["arms"].values())) if a["arms"] else {}
        manifests.append(read_manifest(d))
    comparable, problems = arms_comparable([m for m in manifests if m]) if len([m for m in manifests if m]) > 1 else (True, [])

    evals = {name: load_scores(d) for name, d in st["evals"].items() if (Path(d) / "scores.json").exists()}
    comparisons = []
    names = list(evals)
    if len(names) > 1:
        base = names[0]
        for other in names[1:]:
            for metric in ("ex", "suite_ex"):
                comparisons.append(compare_runs(evals[base], evals[other], metric))
        if len(names) > 2:
            for metric in ("ex", "suite_ex"):
                comparisons.append(compare_runs(evals[names[1]], evals[names[2]], metric))
    cost_rows = json.loads(Path(st["cost"]).read_text(encoding="utf-8")) if st.get("cost") and Path(st["cost"]).exists() else []

    charts = {
        "blind_spot": chart_blind_spot(suite_rows, charts_dir / "blind_spot.png"),
        "fidelity": chart_fidelity(audits, charts_dir / "fidelity.png"),
        "learning": chart_learning(audits, charts_dir / "learning.png"),
        "hacks": chart_hacks(audits, charts_dir / "hacks.png"),
        "accuracy": chart_accuracy(evals, charts_dir / "accuracy.png"),
        "hardness": chart_hardness(evals, charts_dir / "hardness.png"),
        "cost": chart_cost(cost_rows, charts_dir / "cost.png"),
    }

    data = {
        "study": {k: v for k, v in st.items() if not k.startswith("_")},
        "simulated": simulated,
        "suites": _suite_totals(suite_rows),
        "audits": {k: {kk: vv for kk, vv in v.items() if kk != "curve"} | {"curve": v.get("curve", [])} for k, v in audits.items()},
        "arms_comparable": comparable,
        "arm_problems": problems,
        "evals": {k: {kk: vv for kk, vv in v.items() if kk != "items"} for k, v in evals.items()},
        "comparisons": comparisons,
        "cost": cost_rows,
    }
    (out / "report.json").write_text(json.dumps(data, indent=1, default=list), encoding="utf-8")
    md = _markdown(st, data, charts, evals)
    (out / "REPORT.md").write_text(md, encoding="utf-8")
    (out / "report.html").write_text(_html(md, charts, st.get("title", "Mirage report")), encoding="utf-8")
    return out


def _suite_totals(rows: list[dict]) -> dict:
    if not rows:
        return {}
    n = sum(r["n_mutants"] for r in rows)
    surv = sum(r["survivors"] for r in rows)
    caught = sum(r["survivors_killed_by_suite"] for r in rows)
    nd = sum(r["by_operator"].get("distinct_toggle", {}).get("n", 0) for r in rows)
    sd = sum(r["by_operator"].get("distinct_toggle", {}).get("survive_original", 0) for r in rows)
    cd = sum(r["by_operator"].get("distinct_toggle", {}).get("caught_by_suite", 0) for r in rows)
    return {
        "n_databases": len(rows),
        "n_variants": sum(len(r["variants"]) for r in rows),
        "n_mutants": n,
        "survivors": surv,
        "caught": caught,
        "blind_spot_rate": surv / n if n else float("nan"),
        "catch_rate": caught / surv if surv else float("nan"),
        "blind_spot_rate_excl_distinct": (surv - sd) / (n - nd) if n - nd else float("nan"),
        "catch_rate_excl_distinct": (caught - cd) / (surv - sd) if surv - sd else float("nan"),
    }


# --------------------------------------------------------------------------- #
# prose


def _findings(data: dict) -> list[str]:
    f = []
    s = data["suites"]
    if s:
        f.append(
            f"**Before training**, the single-database check accepts {_pct(s['blind_spot_rate'])} of near-miss "
            f"mutants of gold queries ({s['survivors']} of {s['n_mutants']}, over {s['n_databases']} databases). "
            f"Excluding DISTINCT toggles, which are often semantically equivalent, it accepts "
            f"{_pct(s['blind_spot_rate_excl_distinct'])}. The perturbed suites reject {_pct(s['catch_rate'])} of those "
            f"({_pct(s['catch_rate_excl_distinct'])} excluding DISTINCT toggles)."
        )
    for arm, a in data["audits"].items():
        if not a:
            continue
        fp = a["overall"]["false_positive_share"]
        d = a["drift"]
        drift_txt = (
            f"rose by {100 * d['diff']:.1f} points from the first to the last quarter of training {_ci(d['ci95'])}"
            if d.get("significant") and d["diff"] > 0
            else f"fell by {abs(100 * d['diff']):.1f} points {_ci(d['ci95'])}"
            if d.get("significant")
            else f"showed no detectable drift (change {100 * d['diff']:+.1f} points, interval {_ci(d['ci95'])} spans zero)"
            if not math.isnan(d.get("diff", float("nan")))
            else "has too few steps to test for drift"
        )
        f.append(
            f"**{arm} arm**: {_pct(fp[0])} of the rollouts the single-database check rewarded were rejected by the "
            f"suite {_ci(fp[1:])}; that share {drift_txt}."
        )
    for c in data["comparisons"]:
        name = "EX" if c["metric"] == "ex" else "suite EX"
        if c["significant"]:
            f.append(
                f"**{c['b']} vs {c['a']}**, {name}: {100 * c['diff']:+.1f} points {_ci(c['ci95'])}, "
                f"{c['wins']} questions gained, {c['losses']} lost (McNemar p={c['mcnemar_p']:.3f})."
            )
        else:
            f.append(
                f"**{c['b']} vs {c['a']}**, {name}: no detectable difference ({100 * c['diff']:+.1f} points, "
                f"interval {_ci(c['ci95'])} spans zero)."
            )
    for name, e in data["evals"].items():
        o = e["overall"]
        if o.get("ex") and o.get("suite_ex"):
            gap = o["ex"]["mean"] - o["suite_ex"]["mean"]
            f.append(f"**{name}** loses {100 * gap:.1f} points going from EX to suite EX: answers that were right only on the original data.")
    if data["cost"]:
        for r in data["cost"]:
            if r.get("usd_per_1k_correct_suite_ex"):
                f.append(
                    f"**{r['name']}** costs ${r['usd_per_1k_correct_suite_ex']:.3f} per 1,000 suite-correct answers "
                    f"(${r.get('usd_per_1k_requests', float('nan')):.3f} per 1,000 requests at {r['achieved_qps']:.1f} req/s)."
                )
    if not data["arms_comparable"]:
        f.append("**Warning:** the arms are not a clean comparison: " + "; ".join(data["arm_problems"]))
    return f


def _markdown(st: dict, data: dict, charts: dict, evals: dict) -> str:
    L: list[str] = []
    title = st.get("title", "Mirage report")
    L.append(f"# {title}\n")
    if data["simulated"]:
        L.append(
            "> **SIMULATED.** Generated by `mirage demo`: a scripted candidate-pool policy on the synthetic "
            "fixture databases, not a language model and not Spider. It demonstrates the measurements, "
            "not a result. Every number below is about the simulator.\n"
        )
    L.append("## Findings\n")
    L.extend(f"- {x}" for x in _findings(data))
    L.append("")

    def img(key, caption):
        if charts.get(key):
            L.append(f"![{caption}](charts/{Path(charts[key]).name})\n")

    s = data["suites"]
    if s:
        L.append("## 1. The reward's blind spot, before training\n")
        L.append(
            "Each gold query was mutated into near misses (one small, plausible mistake each). A mutant that returns the "
            "same rows as gold on the original database would be **rewarded** by a single-database execution check. "
            "The suite is a few perturbed copies of each database chosen to reject as many of those as possible.\n"
        )
        L.append("| | value |\n|---|---|")
        L.append(f"| databases / suite variants | {s['n_databases']} / {s['n_variants']} |")
        L.append(f"| mutants tested | {s['n_mutants']} |")
        L.append(f"| accepted by the original database | {s['survivors']} ({_pct(s['blind_spot_rate'])}) |")
        L.append(f"| of those, rejected by the suite | {s['caught']} ({_pct(s['catch_rate'])}) |")
        L.append(f"| same two rates, excluding DISTINCT toggles | {_pct(s['blind_spot_rate_excl_distinct'])} / {_pct(s['catch_rate_excl_distinct'])} |\n")
        img("blind_spot", "blind spot by mutation operator")
        L.append(
            "Some mutants are equivalent to the gold query (adding DISTINCT to an already-unique column), so no "
            "database can reject them; the catch rate is a lower bound on how much the suite tightens the reward.\n"
        )

    if data["audits"]:
        L.append("## 2. Did the reward lie more as training went on?\n")
        L.append(
            "Every rollout in both arms was scored by both checks. A **false positive** is a rollout the single-database "
            "check rewarded and the suite rejected. Intervals are bootstrapped over questions, not rollouts.\n"
        )
        L.append("| arm | rollouts | match (single) | match (suite) | false-positive share | drift, last vs first quarter | groups with signal |")
        L.append("|---|---|---|---|---|---|---|")
        for arm, a in data["audits"].items():
            o, d = a["overall"], a["drift"]
            L.append(
                f"| {arm} | {o['n_rollouts']} | {_pct(o['match_single'][0])} | {_pct(o['match_suite'][0])} | "
                f"{_pct(o['false_positive_share'][0])} {_ci(o['false_positive_share'][1:])} | "
                f"{100 * d['diff']:+.1f} pts {_ci(d['ci95'])}{' ✱' if d.get('significant') else ''} | {_pct(o['groups_with_signal'])} |"
            )
        L.append("\n✱ interval excludes zero.\n")
        img("fidelity", "false-positive share over training")
        img("learning", "learning curves")
        img("hacks", "hack shapes")
        for arm, a in data["audits"].items():
            ex = a.get("false_positive_examples") or []
            if ex:
                L.append(f"<details><summary>{arm}: latest false-positive examples ({len(ex)})</summary>\n")
                L.append("| step | db | rewarded SQL | flags |\n|---|---|---|---|")
                for e in ex:
                    sql = (e["sql"] or "").replace("|", "\\|")
                    L.append(f"| {e['step']} | {e['db_id']} | `{sql}` | {', '.join(e['hacks']) or '-'} |")
                L.append("\n</details>\n")
        if not data["arms_comparable"]:
            L.append("> **Arms differ beyond the reward:** " + "; ".join(data["arm_problems"]) + "\n")

    if evals:
        L.append("## 3. Held-out accuracy\n")
        L.append(f"Evaluation set: **{st.get('eval_set', 'see study file')}**. Greedy decoding, identical prompt for every model.\n")
        cols = ["ex", "suite_ex", "pass@1", "pass@8", "suite_pass@8", "maj_ex", "maj_suite_ex"]
        present = [c for c in cols if any(c in e["overall"] for e in evals.values())]
        L.append("| model | n | " + " | ".join(present) + " | FP share |")
        L.append("|---|---|" + "---|" * len(present) + "---|")
        for name, e in evals.items():
            o = e["overall"]
            cells = [f"{_pct(o[c]['mean'])} {_ci(o[c]['ci95'])}" if c in o else "" for c in present]
            L.append(f"| {name} | {o['n']} | " + " | ".join(cells) + f" | {_pct(o['false_positive_share'])} |")
        L.append(
            "\n`ex`: matches gold on the original database. `suite_ex`: also on every perturbed variant. `pass@k` from "
            "sampled completions (unbiased estimator). `maj_*`: majority vote over samples by execution result.\n"
        )
        img("accuracy", "accuracy")
        img("hardness", "accuracy by difficulty")
        if data["comparisons"]:
            L.append("| comparison | metric | difference | 95% CI | gained / lost | McNemar p |\n|---|---|---|---|---|---|")
            for c in data["comparisons"]:
                L.append(
                    f"| {c['b']} − {c['a']} | {c['metric']} | {100 * c['diff']:+.1f} pts | {_ci(c['ci95'])} | "
                    f"{c['wins']} / {c['losses']} | {c['mcnemar_p']:.3f} |"
                )
            L.append("")
        L.append("### Why the wrong answers are wrong\n")
        kinds = [k for k in next(iter(evals.values()))["failures"]]
        L.append("| failure | " + " | ".join(evals) + " |\n|---|" + "---|" * len(evals))
        for k in kinds:
            if any(e["failures"].get(k) for e in evals.values()):
                L.append(f"| {k} | " + " | ".join(str(e["failures"].get(k, 0)) for e in evals.values()) + " |")
        L.append("")

    if data["cost"]:
        L.append("## 4. Cost per correct query\n")
        L.append(
            "Serving cost from a Vantage run (same prompt, base and adapters in one vLLM process); accuracy from section 3.\n"
        )
        L.append("| model | req/s | p95 total | $/1k requests | suite EX | $/1k suite-correct | correct/GPU-hour |")
        L.append("|---|---|---|---|---|---|---|")
        for r in data["cost"]:
            p95 = r.get("p95_total_s")
            L.append(
                f"| {r['name']} | {r['achieved_qps']:.2f} | {p95:.2f}s | ${(r.get('usd_per_1k_requests') or float('nan')):.3f} | "
                f"{_pct(r.get('suite_ex'))} | ${(r.get('usd_per_1k_correct_suite_ex') or float('nan')):.3f} | "
                f"{r.get('correct_per_gpu_hour_suite_ex', float('nan')):.0f} |"
                if p95 is not None
                else f"| {r['name']} | {r['achieved_qps']:.2f} | n/a | | | | |"
            )
        L.append("")
        img("cost", "cost per correct")
        if any(r.get("token_counts_estimated") for r in data["cost"]):
            L.append("> Some Vantage token counts were estimated, not reported by the server.\n")

    L.append("## What this run does not establish\n")
    caveats = [
        "**Suites are an approximation of test-suite accuracy.** They are built by a simpler procedure than the "
        "official distilled test suites (permutation, jitter, boundary planting and row edits, selected against "
        "mutants). For the headline dev number, also run the official evaluator (`mirage score --official`).",
        "**Mutants are not the policy's mistakes.** The blind-spot rate measures the check against a fixed family of "
        "near misses. How often a trained model produces such queries is what the ledger measures, separately.",
        "**Difficulty labels are recomputed** with a reimplementation of Spider's rule on a different parser and "
        "can differ from the official labels on edge cases.",
        "**One seed per arm unless stated.** RL runs vary between seeds; a difference between arms that is smaller "
        "than the seed-to-seed spread is not established by a single pair of runs.",
        "**Spider is old and small-schema.** Results may not transfer to large, messy production schemas, or to "
        "BIRD-style benchmarks with dirty values.",
        "**Cost figures are list-price arithmetic** from the Vantage run: they exclude engineering time, redundancy "
        "and peak-versus-mean provisioning, as Vantage's own report says.",
    ]
    if data["simulated"]:
        caveats.insert(
            0,
            "**This report is simulated.** The policy is a scripted softmax over candidate queries on synthetic "
            "databases. Its training dynamics follow from that construction. It establishes that the measurement "
            "pipeline works, nothing about language models.",
        )
    L.extend(f"- {c}" for c in caveats)
    L.append("")
    return "\n".join(L)


def _html(md: str, charts: dict, title: str) -> str:
    """A dependency-free Markdown-to-HTML pass sufficient for this report's
    own output (headings, lists, tables, images, details, quotes, code)."""
    import re

    def inline(t: str) -> str:
        t = html.escape(t, quote=False)
        t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
        t = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", t)
        return t

    imgs = {Path(p).name: base64.b64encode(Path(p).read_bytes()).decode() for p in charts.values() if p}
    out, in_table, in_list = [], False, False
    for line in md.splitlines():
        s = line.rstrip()
        if s.startswith("|"):
            cells = [c.strip() for c in s.strip("|").split("|")]
            if all(set(c) <= set("-: ") for c in cells):
                continue
            if not in_table:
                out.append("<table><thead><tr>" + "".join(f"<th>{inline(c)}</th>" for c in cells) + "</tr></thead><tbody>")
                in_table = True
            else:
                out.append("<tr>" + "".join(f"<td>{inline(c.replace(chr(92) + '|', '|'))}</td>" for c in cells) + "</tr>")
            continue
        if in_table:
            out.append("</tbody></table>")
            in_table = False
        if s.startswith("- "):
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{inline(s[2:])}</li>")
            continue
        if in_list:
            out.append("</ul>")
            in_list = False
        m = re.match(r"!\[([^\]]*)\]\(charts/([^)]+)\)", s)
        if m:
            b64 = imgs.get(m.group(2))
            if b64:
                out.append(f'<figure><img alt="{html.escape(m.group(1))}" src="data:image/png;base64,{b64}"></figure>')
            continue
        if s.startswith("#"):
            lvl = len(s) - len(s.lstrip("#"))
            out.append(f"<h{lvl}>{inline(s.lstrip('#').strip())}</h{lvl}>")
        elif s.startswith("> "):
            out.append(f"<blockquote>{inline(s[2:])}</blockquote>")
        elif s.startswith("<details") or s.startswith("</details"):
            out.append(s.replace("<summary>", "<summary>").replace("</summary>", "</summary>"))
        elif s:
            out.append(f"<p>{inline(s)}</p>")
    if in_table:
        out.append("</tbody></table>")
    if in_list:
        out.append("</ul>")
    css = f"""
    :root {{ color-scheme: light; }}
    body {{ background:{SURFACE}; color:{INK}; font: 15px/1.55 system-ui, -apple-system, 'Segoe UI', sans-serif;
           max-width: 920px; margin: 0 auto; padding: 32px 16px 64px; }}
    h1 {{ font-size: 28px; margin: 0 0 12px; }} h2 {{ margin-top: 40px; border-top: 1px solid {GRID}; padding-top: 20px; }}
    blockquote {{ margin: 12px 0; padding: 10px 14px; background: #fff4e0; border-left: 4px solid #eda100; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; margin: 12px 0; display: block; overflow-x: auto; }}
    th, td {{ border-bottom: 1px solid {GRID}; padding: 6px 8px; text-align: left; vertical-align: top; }}
    th {{ color: {INK2}; font-weight: 600; }}
    code {{ font: 12px ui-monospace, SFMono-Regular, Menlo, monospace; background: #f1f0ec; padding: 1px 4px; border-radius: 3px; }}
    figure {{ margin: 16px 0; }} img {{ max-width: 100%; height: auto; }}
    details {{ margin: 8px 0; }} summary {{ cursor: pointer; color: {INK2}; }}
    """
    return f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>{html.escape(title)}</title><style>{css}</style></head><body>{''.join(out)}</body></html>"
