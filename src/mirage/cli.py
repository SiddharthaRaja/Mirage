"""The `mirage` command.

    mirage demo                                  # the whole study, simulated, ~1 minute
    mirage data check  <spider_dir>              # counts, splits, overlap (must be empty)
    mirage suites build --spider <dir>           # perturbed suites + the pre-training blind spot
    mirage spike -c configs/spike.yaml           # baseline several models on a val slice
    mirage filter -c configs/sft.yaml            # pass-rate filter for GRPO
    mirage sft -c configs/sft.yaml               # optional warm-up
    mirage grpo -c configs/grpo_single.yaml      # one arm
    mirage grpo -c configs/grpo_suite.yaml       # the other arm
    mirage generate -c <cfg> --split dev         # completions -> predictions.jsonl (GPU)
    mirage score <predictions.jsonl> ...         # predictions -> scores (CPU)
    mirage audit <ledger.jsonl>                  # how often the reward lied, and drift
    mirage vantage export / cost                 # part D
    mirage report study.yaml                     # charts + REPORT.md + report.html
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from mirage.errors import MirageError

app = typer.Typer(add_completion=False, no_args_is_help=True, help="Reward-fidelity study for GRPO text-to-SQL.")
data_app = typer.Typer(no_args_is_help=True, help="Dataset utilities.")
suites_app = typer.Typer(no_args_is_help=True, help="Perturbed database suites.")
vantage_app = typer.Typer(no_args_is_help=True, help="Part D: serving cost through Vantage.")
app.add_typer(data_app, name="data")
app.add_typer(suites_app, name="suites")
app.add_typer(vantage_app, name="vantage")
console = Console()


def _run(fn):
    try:
        return fn()
    except MirageError as e:
        console.print(f"[red]error:[/red] {e}")
        raise typer.Exit(1) from None


def _pct(x):
    import math

    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.1f}%"


# --------------------------------------------------------------------- demo


@app.command()
def demo(
    out: Path = typer.Option(Path("runs/demo"), help="where to write everything"),
    steps: int = typer.Option(150, help="simulated GRPO steps per arm"),
    seed: int = 0,
    vantage_run: str = typer.Option(None, help="a real Vantage run directory to use for cost"),
):
    """Run the whole study end to end on the synthetic fixture with a simulated policy."""
    from mirage.demo import run_demo

    t0 = time.time()
    rep = _run(lambda: run_demo(out, steps=steps, seed=seed, vantage_run=vantage_run, log=console.print))
    console.print(f"\n[green]done[/green] in {time.time() - t0:.0f}s")
    console.print(f"report: {rep / 'REPORT.md'}\n        {rep / 'report.html'}")
    console.print("[yellow]Everything in it is SIMULATED; it demonstrates the measurements, not a result.[/yellow]")


# --------------------------------------------------------------------- data


@data_app.command("check")
def data_check(spider: Path):
    """Counts per split, difficulty mix, and train/dev database overlap (must be empty)."""
    from mirage.data.spider import Spider

    rep = _run(lambda: Spider(spider).check())
    console.print_json(json.dumps(rep))
    if rep.get("train_dev_db_overlap"):
        console.print("[red]train and dev share databases: dev is not a held-out schema test[/red]")
        raise typer.Exit(1)


@data_app.command("mini")
def data_mini(out: Path = typer.Argument(Path("data/mini_spider"))):
    """Write the bundled synthetic Spider-shaped fixture."""
    from mirage.data.fixtures import build_mini_spider

    console.print(f"wrote {build_mini_spider(out)}")


# --------------------------------------------------------------------- suites


@suites_app.command("build")
def suites_build(
    spider: Path = typer.Option(..., help="Spider-shaped folder"),
    out: Path = typer.Option(Path("runs/suites")),
    splits: str = typer.Option("train_all,dev", help="which question files supply gold queries"),
    n_candidates: int = 12,
    n_keep: int = 4,
    seed: int = 1,
    only: str = typer.Option(None, help="comma-separated db_ids to build"),
):
    """Build perturbed suites for every database and report the blind spot."""
    from mirage.data.spider import Spider
    from mirage.errors import DataError
    from mirage.reward.suite import build_suite

    sp = _run(lambda: Spider(spider))
    by_db: dict[str, list[str]] = defaultdict(list)
    for split in splits.split(","):
        try:
            for e in sp.load(split.strip(), hardness=False):
                by_db[e.db_id].append(e.gold_sql)
        except DataError:
            console.print(f"[yellow]skipping split {split}: not found[/yellow]")
    if only:
        keep = set(only.split(","))
        by_db = {k: v for k, v in by_db.items() if k in keep}
    tot = [0, 0, 0]
    t0 = time.time()
    for i, (db, qs) in enumerate(sorted(by_db.items()), 1):
        st = build_suite(db, sp.db_path(db), qs, out, n_candidates=n_candidates, n_keep=n_keep, seed=seed)
        tot[0] += st.n_mutants
        tot[1] += st.survivors
        tot[2] += st.survivors_killed_by_suite
        console.print(
            f"[{i}/{len(by_db)}] {db}: {st.n_gold} gold, {st.n_mutants} mutants, "
            f"{st.survivors} pass the original, {st.survivors_killed_by_suite} caught by {len(st.variants)} variants"
        )
    if tot[0]:
        console.print(
            f"\n[bold]blind spot[/bold]: original DB accepts {_pct(tot[1] / tot[0])} of mutants; "
            f"suites catch {_pct(tot[2] / max(1, tot[1]))} of those. ({time.time() - t0:.0f}s)"
        )


# --------------------------------------------------------------------- spike


@app.command()
def spike(
    config: Path = typer.Option(..., "-c", "--config"),
    n: int = typer.Option(100, help="val questions to use"),
    k: int = typer.Option(8, help="samples per question for the mixed-group estimate"),
    set_: list[str] = typer.Option([], "--set", help="override, e.g. --set eval.backend=vllm"),
):
    """Baseline each candidate model on a val slice: accuracy, speed, and how many questions give GRPO a signal."""
    from mirage.config import load_config, parse_overrides
    from mirage.eval.evaluate import generate_predictions, save_scores, score_predictions
    from mirage.eval.generate import GenConfig, build_generator
    from mirage.reward.suite import SuiteIndex
    from mirage.reward.verifier import Verifier
    from mirage.store import write_manifest
    from mirage.train.data import load_sets, prompt_spec

    cfg = _run(lambda: load_config(config, parse_overrides(set_)))
    sets = _run(lambda: load_sets(cfg))
    val = sets.val[:n]
    paths = sets.db_paths()
    verifier = Verifier(paths, SuiteIndex(cfg.suites))
    models = cfg.candidates or [cfg.model.name]
    tbl = Table(title=f"baseline spike on {len(val)} val questions ({len({e.db_id for e in val})} databases)")
    for c in ("model", "EX", "suite EX", f"pass@{k}", "mixed groups", "s/question"):
        tbl.add_column(c)
    for m in models:
        out = Path(cfg.out) / m.replace("/", "__")
        write_manifest(out, "spike", cfg.model_dump(), model=m)
        gen = build_generator(
            GenConfig(
                backend=cfg.eval.backend, model=m, dtype=cfg.model.dtype, max_new_tokens=cfg.eval.max_new_tokens,
                batch_size=cfg.eval.batch_size, chat_template_kwargs=cfg.model.chat_template_kwargs, base_url=cfg.eval.base_url,
            )
        )
        t0 = time.time()
        pred = generate_predictions(val, gen, paths, out / "predictions.jsonl", prompt_spec(cfg), n_samples=k, temperature=cfg.eval.temperature)
        dt = (time.time() - t0) / max(1, len(val))
        sc = score_predictions(pred, verifier, label=m)
        save_scores(sc, out)
        o = sc["overall"]
        mixed = sum(1 for i in sc["items"] if 0 < i.get(f"pass@{1}", 0) < 1) / max(1, len(sc["items"]))
        tbl.add_row(
            m, _pct(o["ex"]["mean"]), _pct(o.get("suite_ex", {}).get("mean")),
            _pct(o.get(f"pass@{k}", {}).get("mean")), _pct(mixed), f"{dt:.2f}",
        )
        del gen
    console.print(tbl)
    console.print(
        "Pick a model with EX roughly 15-50% and many mixed groups. Near 0% → SFT warm-up first; "
        "near the ceiling → nothing for GRPO to learn."
    )


# --------------------------------------------------------------------- training


@app.command("filter")
def filter_cmd(
    config: Path = typer.Option(..., "-c", "--config"),
    k: int = 8,
    out: Path = typer.Option(None, help="defaults to <cfg.out>/../filter/keep.jsonl"),
    set_: list[str] = typer.Option([], "--set"),
):
    """Sample k attempts per training question from the starting model; keep questions whose attempts disagree."""
    from mirage.config import load_config, parse_overrides
    from mirage.eval.generate import GenConfig, build_generator
    from mirage.reward.suite import SuiteIndex
    from mirage.reward.verifier import Verifier
    from mirage.train.common import resolve_start
    from mirage.train.data import load_sets, prompt_spec
    from mirage.train.filter import pass_rate_filter

    cfg = _run(lambda: load_config(config, parse_overrides(set_)))
    cfg.data.filtered_file = None
    sets = _run(lambda: load_sets(cfg))
    start = resolve_start(cfg)
    gen = build_generator(GenConfig(backend=cfg.eval.backend, model=start, dtype=cfg.model.dtype,
                                    batch_size=cfg.eval.batch_size, chat_template_kwargs=cfg.model.chat_template_kwargs,
                                    base_url=cfg.eval.base_url))
    out = out or Path(cfg.out).parent / "filter" / "keep.jsonl"
    res = pass_rate_filter(sets.train, gen, Verifier(sets.db_paths(), SuiteIndex(cfg.suites)), prompt_spec(cfg), out, k=k)
    console.print_json(json.dumps(res))
    console.print(f"wrote {out}; set data.filtered_file to it in the GRPO base config")


@app.command()
def sft(config: Path = typer.Option(..., "-c", "--config"), set_: list[str] = typer.Option([], "--set")):
    """Supervised warm-up (LoRA). Only if the spike says the baseline is too weak."""
    from mirage.config import load_config, parse_overrides
    from mirage.train.sft import run_sft

    cfg = _run(lambda: load_config(config, parse_overrides(set_)))
    console.print(f"adapter: {_run(lambda: run_sft(cfg))}")


@app.command()
def grpo(config: Path = typer.Option(..., "-c", "--config"), set_: list[str] = typer.Option([], "--set")):
    """Train one GRPO arm. Resumes automatically from the newest checkpoint."""
    from mirage.config import load_config, parse_overrides
    from mirage.train.grpo import run_grpo

    cfg = _run(lambda: load_config(config, parse_overrides(set_)))
    console.print(f"adapter: {_run(lambda: run_grpo(cfg))}")


@app.command()
def merge(base: str, adapter: Path, out: Path, dtype: str = "float16"):
    """Fold a LoRA adapter into its base model."""
    from mirage.train.common import merge_adapter

    console.print(f"merged model: {_run(lambda: merge_adapter(base, str(adapter), out, dtype))}")


# --------------------------------------------------------------------- evaluation


@app.command()
def generate(
    config: Path = typer.Option(..., "-c", "--config"),
    split: str = typer.Option("val", help="val for tuning; dev ONLY for the final number"),
    model: str = typer.Option(None, help="override the model to load"),
    adapter: str = typer.Option(None, help="a LoRA adapter to apply"),
    label: str = typer.Option(None),
    out: Path = typer.Option(None),
    set_: list[str] = typer.Option([], "--set"),
):
    """Generate completions (greedy + samples) into predictions.jsonl. Resumable."""
    from mirage.config import load_config, parse_overrides
    from mirage.eval.evaluate import generate_predictions
    from mirage.eval.generate import GenConfig, build_generator
    from mirage.store import write_manifest
    from mirage.train.common import resolve_start
    from mirage.train.data import load_sets, prompt_spec

    cfg = _run(lambda: load_config(config, parse_overrides(set_)))
    sets = _run(lambda: load_sets(cfg, include_dev=split == "dev"))
    exs = sets.dev if split == "dev" else sets.val
    if cfg.eval.max_examples:
        exs = exs[: cfg.eval.max_examples]
    m = model or resolve_start(cfg)
    label = label or Path(adapter).parent.name if adapter else (label or Path(m).name)
    out = out or Path(cfg.out) / "eval" / f"{label}-{split}"
    write_manifest(out, "generate", cfg.model_dump(), model=m, adapter=adapter, split=split, n=len(exs))
    gen = build_generator(GenConfig(backend=cfg.eval.backend, model=m, adapter=adapter, dtype=cfg.model.dtype,
                                    max_new_tokens=cfg.eval.max_new_tokens, batch_size=cfg.eval.batch_size,
                                    chat_template_kwargs=cfg.model.chat_template_kwargs, base_url=cfg.eval.base_url))
    p = generate_predictions(exs, gen, sets.db_paths(), out / "predictions.jsonl", prompt_spec(cfg),
                             n_samples=cfg.eval.n_samples, temperature=cfg.eval.temperature,
                             progress=lambda a, b: console.print(f"  {a}/{b}"))
    console.print(f"wrote {p}")


@app.command()
def score(
    predictions: Path,
    spider: Path = typer.Option(..., help="Spider-shaped folder"),
    suites: Path = typer.Option(None),
    out: Path = typer.Option(None, help="defaults to the predictions' folder"),
    label: str = typer.Option(None),
    official: Path = typer.Option(None, help="path to a test-suite-sql-eval checkout"),
    official_db: Path = typer.Option(None, help="the test-suite database folder"),
):
    """Score predictions on CPU: EX, suite EX, pass@k, majority vote, failures."""
    from mirage.data.spider import Spider, read_jsonl
    from mirage.eval.evaluate import save_scores, score_predictions
    from mirage.reward.suite import SuiteIndex
    from mirage.reward.verifier import Verifier

    sp = _run(lambda: Spider(spider))
    rows = _run(lambda: read_jsonl(predictions))
    verifier = Verifier({d: str(sp.db_path(d)) for d in {r["db_id"] for r in rows}}, SuiteIndex(suites))
    out = out or predictions.parent
    sc = score_predictions(predictions, verifier, label=label or out.name)
    save_scores(sc, out)
    o = sc["overall"]
    tbl = Table(title=f"{sc['label']} ({o['n']} questions)")
    for c in ("metric", "value", "95% CI"):
        tbl.add_column(c)
    for k in ("ex", "suite_ex", "pass@1", "pass@8", "suite_pass@8", "maj_ex", "maj_suite_ex"):
        if k in o:
            tbl.add_row(k, _pct(o[k]["mean"]), f"[{_pct(o[k]['ci95'][0])}, {_pct(o[k]['ci95'][1])}]")
    tbl.add_row("FP share", _pct(o["false_positive_share"]), "")
    console.print(tbl)
    console.print("by difficulty: " + ", ".join(f"{h} {_pct(b['ex']['mean'])} (n={b['n']})" for h, b in sc["by_hardness"].items()))
    console.print("failures: " + ", ".join(f"{k}={v}" for k, v in sc["failures"].items() if v))
    if official:
        from mirage.eval.official import run_official

        res = run_official(predictions, official, official_db or Path(spider) / "database", Path(spider) / "tables.json", out)
        console.print_json(json.dumps(res, default=str))


@app.command()
def compare(a: Path, b: Path, metric: str = "suite_ex"):
    """Paired bootstrap + McNemar between two scored runs (folders with scores.json)."""
    from mirage.eval.evaluate import compare_runs, load_scores

    res = compare_runs(load_scores(a), load_scores(b), metric)
    console.print_json(json.dumps(res, default=list))
    console.print("difference is " + ("[green]significant[/green]" if res["significant"] else "not distinguishable from zero"))


@app.command()
def audit(ledger: Path, out: Path = typer.Option(None)):
    """How often the single-database reward was wrong during training, and whether that grew."""
    from mirage.analysis.audit import audit_ledger

    res = _run(lambda: audit_ledger(ledger))
    if out:
        out.write_text(json.dumps(res, indent=1, default=list), encoding="utf-8")
    for arm, a in res["arms"].items():
        o, d = a["overall"], a["drift"]
        drift = f"{100 * d['diff']:+.1f} pts" if d["diff"] == d["diff"] else "n/a"
        console.print(
            f"[bold]{arm}[/bold]: {o['n_rollouts']} rollouts, match single {_pct(o['match_single'][0])}, "
            f"suite {_pct(o['match_suite'][0])}, false-positive share {_pct(o['false_positive_share'][0])} "
            f"[{_pct(o['false_positive_share'][1])}, {_pct(o['false_positive_share'][2])}]; drift "
            f"{drift} ({'significant' if d.get('significant') else 'not significant'}); "
            f"groups with signal {_pct(o['groups_with_signal'])}"
        )


@app.command()
def report(study: Path):
    """Build REPORT.md, report.html and report.json from a study file."""
    from mirage.analysis.report import build_report

    out = _run(lambda: build_report(study))
    console.print(f"wrote {out / 'REPORT.md'} and {out / 'report.html'}")


# --------------------------------------------------------------------- vantage


@vantage_app.command("export")
def vantage_export(
    config: Path = typer.Option(..., "-c", "--config"),
    split: str = typer.Option("dev"),
    out: Path = typer.Option(Path("runs/vantage/workload.mirage-dev.jsonl")),
):
    """Write a Vantage workload with the exact mirage prompt."""
    from mirage.config import load_config
    from mirage.integrations.vantage import export_workload
    from mirage.train.data import load_sets, prompt_spec

    cfg = _run(lambda: load_config(config))
    sets = _run(lambda: load_sets(cfg, include_dev=split == "dev"))
    exs = sets.dev if split == "dev" else sets.val
    console.print(f"wrote {export_workload(exs, sets.db_paths(), out, prompt_spec(cfg))} ({len(exs)} rows)")
    console.print("Apply integrations/vantage/vantage-raw-prompt.patch to Vantage, then use sweep.mirage-cost.yaml.")


@vantage_app.command("cost")
def vantage_cost(
    vantage_run: Path,
    pair: list[str] = typer.Option(..., "--pair", help="name=<vantage config id>:<scores folder>"),
    out: Path = typer.Option(Path("runs/vantage/cost.json")),
):
    """Join a Vantage run with scored evaluations: cost per 1,000 correct answers."""
    from mirage.eval.evaluate import load_scores
    from mirage.integrations.vantage import cost_per_correct

    pairs = {}
    for p in pair:
        name, rest = p.split("=", 1)
        vid, sdir = rest.rsplit(":", 1)
        pairs[name] = {"vantage": vid, "scores": load_scores(sdir)}
    rows = _run(lambda: cost_per_correct(vantage_run, pairs))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1, default=list), encoding="utf-8")
    for r in rows:
        console.print(
            f"{r['name']}: {r['achieved_qps']:.2f} req/s, ${r.get('usd_per_1k_requests') or float('nan'):.4f}/1k req, "
            f"suite EX {_pct(r.get('suite_ex'))}, ${r.get('usd_per_1k_correct_suite_ex', float('nan')):.4f}/1k correct"
        )
    console.print(f"wrote {out}")


if __name__ == "__main__":
    app()
