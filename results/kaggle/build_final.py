"""Builds kaggle/12_final.ipynb: the last Mirage run (seed-1 suite arm, seed-1 dev eval,
official test-suite evaluator for all five models, FINAL.md, packaged results).

    python kaggle/build_final.py
"""
import json
from pathlib import Path

CELLS = []


def md(s):
    CELLS.append(("markdown", s.strip("\n")))


def code(s):
    CELLS.append(("code", s.strip("\n")))


md(r"""
# Mirage · final run (seed 1 + official evaluator + final report)

The last Mirage notebook. One committed run, unattended, **about 5–6 hours** (up to ~10 if the
earlier outputs are not attached). It does everything still open and then packages the results:

| lane | what | time |
|---|---|---|
| GPU 0 | train the **seed-1 suite arm** (300 steps) → generate + score on Spider dev | ~4.5 h |
| GPU 1 | seed-1 single arm: reuse the trained adapter (or retrain it) → generate + score on dev | ~0.7 h (4.5 h if retrained) |
| CPU | **official test-suite evaluator** (taoyds) on all five models: base, seed-0 ×2, seed-1 ×2 | ~2–3 h, in parallel |
| end | training audit of all 4 arms, paired comparisons, seed-1 report, **FINAL.md**, `final_results.zip` | minutes |

**Inputs (Add Input):**
1. `spider` (Your Datasets) — required.
2. `notebook6e864222cc` (Your Work, latest version) — the seed-1 single run: its output holds the trained seed-1 single arm **and** the suites.
   Optional: without it the suites are rebuilt (+50 min, same seed, identical) and the single arm is retrained on GPU 1 in parallel.
3. `notebook376c8ebfe2` (Your Work) — seed 0's run. Optional, only a second source for the suites.

**Settings:** Accelerator **GPU T4 x2**, Internet **On**. Then **Save Version → Save & Run All (Commit)** and close the tab.

Nothing in here stops the run on a failure except the setup checks: every stage records `ok`/`failed`/`skipped`,
and the final cells always write `FINAL.md` with what did finish. A watchdog stops everything at 11.2 h so the
output is saved before Kaggle's 12 h limit.

**Result:** Output tab → `final_results.zip` (drop into the repo as `results/final/`) and `FINAL.md`.

**Re-running (version 2+):** add this notebook's own previous version as an input (Your Work → this notebook).
Everything already trained or scored is picked up from it and skipped, so a rerun with nothing left to train
needs **no GPU**: set Accelerator **None**. Only the official evaluator, audit, report and packaging run (~1–3 h CPU).
""")

code(r"""
# --- settings (identical to the seed-1 single run, so the two seed-1 manifests match) ---
SEED, STEPS, BUDGET_H = 1, 300, 5.5
HARD_STOP_H = 11.2          # Kaggle kills committed runs at 12 h; stop everything before that
RUN_OFFICIAL = True         # the official test-suite evaluator (CPU, Google Drive download)
""")

code(r"""
# --- setup ---------------------------------------------------------------
import os, subprocess, sys, time, json, shutil, glob, re, threading, gzip, traceback
T0 = time.time()
INPUT = os.environ.get("MIRAGE_KAGGLE_INPUT", "/kaggle/input")
W = os.environ.get("MIRAGE_KAGGLE_WORK", "/kaggle/working")
RUNS = f"{W}/runs"; REPO = f"{W}/mirage"
os.makedirs(f"{RUNS}/logs", exist_ok=True)
def hours(): return (time.time() - T0) / 3600

def find_dirs(pred, max_depth=9):
    hits = []
    for root, dirs, files in os.walk(INPUT, followlinks=True):
        if pred(root, dirs, files): hits.append(root)
        if root.count("/") - INPUT.count("/") >= max_depth: dirs[:] = []
    return hits

sp = find_dirs(lambda r, d, f: "train_spider.json" in f and "dev.json" in f and "database" in d)
assert sp, "Spider not found: Add Input -> Your Datasets -> spider"
SPIDER_DIR = sp[0]
os.environ.update(SPIDER_DIR=SPIDER_DIR, MIRAGE_RUNS=RUNS, PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false",
                  PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", HF_HUB_DISABLE_PROGRESS_BARS="1")
os.chdir(W)
if not os.path.exists(REPO):
    subprocess.run(["git", "clone", "-q", "--depth", "1", "https://github.com/SiddharthaRaja/Mirage", REPO], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", f"{REPO}[train]"], check=True)
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"])     # PEFT rejects Kaggle's torchao
os.chdir(REPO)
# The kernel started in /kaggle/working, which is on sys.path, so "import mirage" would find the repo FOLDER
# /kaggle/working/mirage (a namespace package with no code) instead of the package in src/. Put src first.
sys.path.insert(0, f"{REPO}/src")
for k in [k for k in sys.modules if k == "mirage" or k.startswith("mirage.")]: del sys.modules[k]
import importlib; importlib.invalidate_caches()
import mirage.analysis.audit                                     # fail here, in setup, not after hours of training
COMMIT = subprocess.run(["git", "log", "--oneline", "-1"], capture_output=True, text=True).stdout.strip()
assert os.path.exists("results/run2/eval/start-dev/scores.json"), "repo is missing the seed-0 dev results"
import torch
NGPU = torch.cuda.device_count() if torch.cuda.is_available() else 0
print("spider:", SPIDER_DIR); print("repo:", COMMIT); print("GPUs:", NGPU, torch.cuda.get_device_name(0) if NGPU else "(none)")
if NGPU == 1: print("only one GPU: the two GPU lanes will run one after the other")
if NGPU == 0: print("no GPU: only steps already done in an attached earlier version can be reused; anything needing a GPU is marked failed")
if NGPU:
    from huggingface_hub import snapshot_download      # download once, before two processes race for it
    snapshot_download("Qwen/Qwen2.5-Coder-0.5B-Instruct")
""")

code(r"""
# --- bring in earlier outputs: suites, filter, seed-1 arms, seed-1 evals -------
STATUS = {}
def note(stage, state, **kw):
    STATUS[stage] = {"state": state, "at_h": round(hours(), 2), **kw}
    print(f"[{hours():.2f} h] {stage}: {state} {kw if kw else ''}", flush=True)

runs_dirs = find_dirs(lambda r, d, f: r.endswith("/runs"))
print("runs folders in inputs:", runs_dirs)
skip_ck = shutil.ignore_patterns("optimizer.pt", "rng_state*.pth")
for src in runs_dirs:
    if os.path.isdir(f"{src}/suites") and len(os.listdir(f"{src}/suites")) >= 160 and not os.path.isdir(f"{RUNS}/suites"):
        shutil.copytree(f"{src}/suites", f"{RUNS}/suites")
    for d in glob.glob(f"{src}/arms/*-s1"):
        shutil.copytree(d, f"{RUNS}/arms/{os.path.basename(d)}", dirs_exist_ok=True)
    for f in glob.glob(f"{src}/logs/*.log"):                 # keep earlier versions' logs (e.g. the training log)
        if not os.path.exists(f"{RUNS}/logs/{os.path.basename(f)}"): shutil.copy2(f, f"{RUNS}/logs/")
    for d in glob.glob(f"{src}/eval/*-s1-dev"):
        if os.path.exists(f"{d}/scores.json"):
            shutil.copytree(d, f"{RUNS}/eval/{os.path.basename(d)}", dirs_exist_ok=True)

# the pass-rate filter is in the repo (identical to seed 0's: hash-selected), so it is always available
if not os.path.exists(f"{RUNS}/filter/keep.jsonl"):
    shutil.copytree("results/run2/filter", f"{RUNS}/filter", dirs_exist_ok=True)
note("filter", "ok", kept=sum(json.loads(l)["keep"] for l in open(f"{RUNS}/filter/keep.jsonl")))

def arm_done(a):
    t = f"{RUNS}/arms/{a}-s1/timing.json"
    return (os.path.exists(f"{RUNS}/arms/{a}-s1/adapter/adapter_config.json") and os.path.exists(t)
            and json.load(open(t)).get("global_step", 0) >= STEPS)
def eval_done(a): return os.path.exists(f"{RUNS}/eval/grpo-{a}-s1-dev/scores.json")

print("suites present:", len(os.listdir(f"{RUNS}/suites")) if os.path.isdir(f"{RUNS}/suites") else 0)
for a in ("single", "suite"):
    print(f"seed-1 {a}: trained={arm_done(a)} evaluated={eval_done(a)}")
""")

code(r"""
# --- helpers: logged subprocesses, lanes, watchdog -----------------------------
PROCS = set(); STOP = threading.Event(); LOCK = threading.Lock()

def run(stage, cmd, gpu=None, nice=False, cwd=None, force=False):
    # Run one command, stream it to logs/<stage>.log, record ok/failed. Never raises.
    env = dict(os.environ)
    if gpu is not None: env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    if nice: cmd = f"nice -n 15 {cmd}"
    if STOP.is_set() and not force:
        note(stage, "skipped", why="time limit"); return False
    note(stage, "running", gpu=gpu, cmd=cmd[:160])
    t = time.time()
    with open(f"{RUNS}/logs/{stage}.log", "w") as log:
        p = subprocess.Popen(cmd, shell=True, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=cwd or REPO,
                             start_new_session=True)
        with LOCK: PROCS.add(p)
        rc = p.wait()
        with LOCK: PROCS.discard(p)
    note(stage, "ok" if rc == 0 else "failed", rc=rc, minutes=round((time.time() - t) / 60, 1))
    return rc == 0

def last_line(stage):
    try:
        lines = [l.strip() for l in open(f"{RUNS}/logs/{stage}.log", errors="replace").read().replace("\r", "\n").splitlines() if l.strip()]
        return lines[-1][:150] if lines else ""
    except FileNotFoundError:
        return ""

def gpu_lane(arm, gpu):
    out = f"{RUNS}/arms/{arm}-s1"
    try:
        if not os.path.isdir(f"{RUNS}/suites") or len(os.listdir(f"{RUNS}/suites")) < 160:
            note(f"train_{arm}", "skipped", why="no suites"); return
        if arm_done(arm):
            note(f"train_{arm}", "reused")
        elif NGPU == 0:
            note(f"train_{arm}", "failed", why="needs a GPU (not trained in any attached version)"); return
        else:
            run(f"train_{arm}", f"mirage grpo -c configs/grpo_{arm}.yaml --set out={out} --set grpo.seed={SEED} "
                                f"--set grpo.max_steps={STEPS} --set budget.max_hours={BUDGET_H}", gpu=gpu)
        if not os.path.exists(f"{out}/adapter/adapter_config.json"):
            note(f"dev_{arm}", "skipped", why="no adapter"); return
        if eval_done(arm):
            note(f"dev_{arm}", "reused"); return
        ev = f"{RUNS}/eval/grpo-{arm}-s1-dev"
        if NGPU == 0:
            note(f"dev_{arm}", "failed", why="needs a GPU (not generated in any attached version)"); return
        if run(f"gen_{arm}", f"mirage generate -c configs/grpo_base.yaml --split dev --adapter {out}/adapter "
                             f"--label grpo-{arm}-s1 --out {ev}", gpu=gpu):
            run(f"score_{arm}", f"mirage score {ev}/predictions.jsonl --spider {SPIDER_DIR} --suites {RUNS}/suites "
                                f"--label grpo-{arm}-s1")
    except Exception:
        note(f"lane_{arm}", "failed", error=traceback.format_exc()[-400:])
""")

code(r"""
# --- suites (only if not attached) -------------------------------------------
if not os.path.isdir(f"{RUNS}/suites") or len(os.listdir(f"{RUNS}/suites")) < 160:
    run("suites", f"mirage suites build --spider {SPIDER_DIR} --out {RUNS}/suites")
else:
    note("suites", "reused", databases=len(os.listdir(f"{RUNS}/suites")))
""")

code(r"""
# --- the official test-suite evaluator (CPU lane) -------------------------------
OFF = f"{W}/official"; os.makedirs(OFF, exist_ok=True)
TSE = f"{W}/tse"; OFFICIAL = {}
MODELS = {  # name -> folder holding scored_items.jsonl
    "base": f"{REPO}/results/run2/eval/start-dev",
    "GRPO single s0": f"{REPO}/results/run2/eval/grpo-single-dev",
    "GRPO suite s0": f"{REPO}/results/run2/eval/grpo-suite-dev",
    "GRPO single s1": f"{RUNS}/eval/grpo-single-s1-dev",
    "GRPO suite s1": f"{RUNS}/eval/grpo-suite-s1-dev",
}

def ts_ready(d):
    # Official test-suite layout: <db>/<db>.sqlite plus extra variants of it. Mirage's own suites (<db>/<db>__sN.sqlite,
    # no <db>.sqlite) and plain Spider (one .sqlite per db) must NOT match: version 1 picked up Mirage's suites here.
    files = glob.glob(f"{d}/concert_singer/*.sqlite")
    return (os.path.exists(f"{d}/concert_singer/concert_singer.sqlite") and len(files) > 1
            and not any("__s" in os.path.basename(x) for x in files))

def prepare_test_suites():
    # Returns the folder of official test-suite databases, or None.
    if not os.path.exists(TSE):                                 # the evaluator code
        subprocess.run(["git", "clone", "-q", "--depth", "1", "https://github.com/taoyds/test-suite-sql-eval", TSE], check=True)
    hits = find_dirs(lambda r, d, f: "concert_singer" in d and "/runs/" not in r + "/" and ts_ready(r), max_depth=7)
    if hits: return hits[0]                                     # databases attached as a dataset
    if ts_ready(f"{TSE}/database"): return f"{TSE}/database"
    # NB: the repo ships database/ with only a readme.txt, so "folder exists" is not "databases exist".
    z = f"{W}/ts.zip"
    if not os.path.exists(z):
        run("ts_download", f"gdown --fuzzy 'https://drive.google.com/file/d/1mkCx2GOFIqNesD4y8TDAO1yX1QZORP5w/view?usp=sharing' -O {z}", cwd=W)
    if not os.path.exists(z) or os.path.getsize(z) < 10_000_000: return None
    shutil.rmtree(f"{W}/ts_raw", ignore_errors=True)
    subprocess.run(["unzip", "-q", z, "-d", f"{W}/ts_raw"], check=True)
    cands = sorted({os.path.dirname(os.path.dirname(p)) for p in glob.glob(f"{W}/ts_raw/**/concert_singer/*.sqlite", recursive=True)})
    os.remove(z)                                                # keep the output small
    return cands[0] if cands else None

def official_one(name, d, ts_dir):
    f = f"{d}/scored_items.jsonl"
    if not os.path.exists(f):
        note(f"official[{name}]", "skipped", why="no predictions"); return
    items = [json.loads(l) for l in open(f)]
    key = re.sub(r"\W+", "_", name)
    g, p = f"{OFF}/{key}_gold.txt", f"{OFF}/{key}_pred.txt"
    open(g, "w").write("\n".join(f"{' '.join(i['gold_sql'].split())}\t{i['db_id']}" for i in items) + "\n")
    open(p, "w").write("\n".join(" ".join((i.get("pred_sql") or "SELECT 1").split()) for i in items) + "\n")
    ok = run(f"official_{key}", f"{sys.executable} evaluation.py --gold {g} --pred {p} --db {ts_dir} "
                                f"--table {SPIDER_DIR}/tables.json --etype exec", nice=True, cwd=TSE)
    acc = None
    try:
        for line in open(f"{RUNS}/logs/official_{key}.log", errors="replace"):
            if line.strip().lower().startswith("execution"):
                nums = re.findall(r"\d+\.\d+", line)
                if nums: acc = float(nums[-1])                   # last column = "all"
    except FileNotFoundError:
        pass
    OFFICIAL[name] = dict(n=len(items), ex=sum(i["ex"] for i in items) / len(items),
                          suite_ex=sum(i["suite_ex"] for i in items) / len(items), official=acc)
    if ok and acc is None: note(f"official_{key}", "failed", why="could not parse evaluator output")

def official_lane(wait_for):
    try:
        if not RUN_OFFICIAL:
            note("official", "skipped", why="RUN_OFFICIAL=False"); return
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gdown", "sqlparse", "nltk", "tqdm"], check=True)
        subprocess.run([sys.executable, "-c", "import nltk\nfor x in ('punkt', 'punkt_tab'):\n    nltk.download(x, quiet=True)"])
        ts_dir = prepare_test_suites()
        ok = subprocess.run([sys.executable, "-c", "from nltk import word_tokenize; word_tokenize('select a from b')"],
                            capture_output=True).returncode == 0
        if not ok:   # no tokenizer data (download blocked): an equivalent tokenizer for SQL, recorded in the status
            src = open(f"{TSE}/process_sql.py").read()
            src = src.replace("from nltk import word_tokenize",
                              "import re as _re\ndef word_tokenize(s): return _re.findall(r'[\\w.]+|[^\\w\\s]', s)")
            open(f"{TSE}/process_sql.py", "w").write(src)
            note("official_tokenizer", "fallback", why="nltk punkt unavailable; regex tokenizer used")
        if not ts_dir:
            note("official", "failed", why="test-suite databases unavailable (Google Drive download failed)"); return
        note("official_dbs", "ok", path=ts_dir, databases=len(os.listdir(ts_dir)))
        from concurrent.futures import ThreadPoolExecutor
        workers = max(1, min(5, (os.cpu_count() or 2) - 1))
        with ThreadPoolExecutor(workers) as ex:
            futs = [ex.submit(official_one, n, MODELS[n], ts_dir) for n in ["base", "GRPO single s0", "GRPO suite s0"]]
            for t in wait_for: t.join()                         # seed-1 predictions come from the GPU lanes
            futs += [ex.submit(official_one, n, MODELS[n], ts_dir) for n in ["GRPO single s1", "GRPO suite s1"]]
            for f in futs: f.result()
    except Exception:
        note("official", "failed", error=traceback.format_exc()[-400:])
""")

code(r"""
# --- run all lanes; heartbeat every 10 minutes; watchdog before Kaggle's 12 h --------
if NGPU >= 2:
    lanes = [threading.Thread(target=gpu_lane, args=("suite", 0), name="gpu0-suite"),
             threading.Thread(target=gpu_lane, args=("single", 1), name="gpu1-single")]
else:
    g = 0 if NGPU else None
    lanes = [threading.Thread(target=lambda: (gpu_lane("suite", g), gpu_lane("single", g)), name="gpu-lane")]
cpu = threading.Thread(target=official_lane, args=(lanes,), name="cpu-official")
for t in lanes + [cpu]: t.start()

last_beat = 0
while any(t.is_alive() for t in lanes + [cpu]):
    time.sleep(20)
    if hours() > HARD_STOP_H and not STOP.is_set():
        STOP.set(); print(f"[{hours():.2f} h] WATCHDOG: stopping all running steps so the output gets saved", flush=True)
        with LOCK:
            for p in list(PROCS):
                try: os.killpg(p.pid, 15)
                except Exception: pass
    if time.time() - last_beat > 600:
        last_beat = time.time()
        running = [k for k, v in STATUS.items() if v["state"] == "running"]
        print(f"[{hours():.2f} h] running: " + (" | ".join(f"{k}: {last_line(k)}" for k in running) or "-"), flush=True)
note("lanes", "done")
print(json.dumps(STATUS, indent=1))
""")

code(r"""
# --- training audit of all four arms + the seed-1 report -------------------------
from mirage.analysis.audit import audit_ledger
os.makedirs(f"{RUNS}/audit", exist_ok=True)
AUDIT = {}
def ledger_path(arm, seed):
    if seed == 1: return f"{RUNS}/arms/{arm}-s1/ledger.jsonl"
    dst = f"{RUNS}/audit/{arm}-s0.ledger.jsonl"                 # seed 0's ledgers are in the repo, gzipped
    if not os.path.exists(dst):
        with gzip.open(f"{REPO}/results/run2/arms/{arm}/ledger.jsonl.gz", "rt") as fi, open(dst, "w") as fo:
            shutil.copyfileobj(fi, fo)
    return dst
for seed in (0, 1):
    for arm in ("single", "suite"):
        try:
            p = ledger_path(arm, seed)
            if not os.path.exists(p): continue
            res = audit_ledger(p)
            a = next(iter(res["arms"].values()))
            tf = f"{RUNS}/arms/{arm}-s1/timing.json" if seed == 1 else f"{REPO}/results/run2/arms/{arm}/timing.json"
            a["steps"] = json.load(open(tf)).get("global_step") if os.path.exists(tf) else None
            AUDIT[(arm, seed)] = a
            json.dump(res, open(f"{RUNS}/audit/{arm}-s{seed}.json", "w"), indent=1, default=list)
        except Exception:
            note(f"audit_{arm}_s{seed}", "failed", error=traceback.format_exc()[-300:])
note("audit", "ok", arms=[f"{a}-s{s}" for a, s in AUDIT])

study = f"{RUNS}/study_seed1.yaml"
open(study, "w").write(f'''title: "Mirage seed 1: reward fidelity for GRPO text-to-SQL (Qwen2.5-Coder-0.5B, Spider)"
simulated: false
eval_set: "Spider dev (1,034 questions, 20 unseen databases)"
suites: {RUNS}/suites
arms:
  single: {RUNS}/arms/single-s1
  suite: {RUNS}/arms/suite-s1
evals:
  start: {REPO}/results/run2/eval/start-dev
  grpo-single: {RUNS}/eval/grpo-single-s1-dev
  grpo-suite: {RUNS}/eval/grpo-suite-s1-dev
out: {RUNS}/report_seed1
''')
run("report_seed1", f"mirage report {study}", force=True)
""")

code(r"""
# --- FINAL.md: everything in one place ------------------------------------------
from mirage.eval.evaluate import compare_runs, load_scores
P = lambda x: f"{100*x:.1f}%"
EVALS = {}
for name, d in MODELS.items():
    if os.path.exists(f"{d}/scores.json"): EVALS[name] = load_scores(d)

L = ["# Mirage · final results (seeds 0 and 1)", "",
     f"Kaggle T4 · repo {COMMIT} · this run {hours():.1f} h · Qwen2.5-Coder-0.5B-Instruct + LoRA r16, GRPO 300 steps per arm.",
     "Spider dev: 1,034 questions on 20 databases never seen in training, touched once per model. 95% bootstrap CIs.",
     "If an interval for a difference includes zero, it is not called a difference.", ""]

L += ["## 1 · Training audit: how often the standard reward paid for wrong SQL", "",
      "| arm | seed | steps | rollouts | match single | match suite | false-positive share [95% CI] | drift last−first quarter [95% CI] | groups with signal |",
      "|---|---|---|---|---|---|---|---|---|"]
for (arm, seed), a in sorted(AUDIT.items(), key=lambda kv: (kv[0][1], kv[0][0])):
    o, d = a["overall"], a["drift"]; fp = o["false_positive_share"]
    drift = (f"{100*d['diff']:+.1f} pts [{100*d['ci95'][0]:+.1f}, {100*d['ci95'][1]:+.1f}] "
             f"{'(significant)' if d.get('significant') else '(n.s.)'}") if d["diff"] == d["diff"] else "n/a"
    L.append(f"| {arm} | {seed} | {a.get('steps')} | {o['n_rollouts']:,} | {P(o['match_single'][0])} | {P(o['match_suite'][0])} | "
             f"{P(fp[0])} [{P(fp[1])}, {P(fp[2])}] | {drift} | {P(o['groups_with_signal'])} |")
if len(AUDIT) < 4: L.append(f"\nOnly {len(AUDIT)} of 4 arms audited: see the status table.")

L += ["", "## 2 · Dev accuracy", "",
      "| model | EX [95% CI] | suite EX | official test-suite EX | pass@1 | pass@8 | majority EX |", "|---|---|---|---|---|---|---|"]
for name, s in EVALS.items():
    o = s["overall"]; ci = o["ex"]["ci95"]
    off = OFFICIAL.get(name, {}).get("official")
    L.append(f"| {name} | {P(o['ex']['mean'])} [{100*ci[0]:.1f}, {100*ci[1]:.1f}] | {P(o['suite_ex']['mean'])} | "
             f"{P(off) if off is not None else '—'} | {P(o['pass@1']['mean'])} | {P(o['pass@8']['mean'])} | {P(o['maj_ex']['mean'])} |")
L.append("\nOfficial = taoyds/test-suite-sql-eval on its released distilled test suites, same greedy predictions, `--etype exec`.")

L += ["", "## 3 · Paired comparisons (same 1,034 questions; bootstrap 95% CI + exact McNemar)", "",
      "| comparison | metric | difference [95% CI] | gained / lost | McNemar p | significant |", "|---|---|---|---|---|---|"]
PAIRS = [("GRPO single s0", "base"), ("GRPO suite s0", "base"), ("GRPO single s1", "base"), ("GRPO suite s1", "base"),
         ("GRPO suite s0", "GRPO single s0"), ("GRPO suite s1", "GRPO single s1"),
         ("GRPO single s1", "GRPO single s0"), ("GRPO suite s1", "GRPO suite s0")]
CMP = {}
for x, y in PAIRS:
    if x not in EVALS or y not in EVALS: continue
    for m in ("ex", "suite_ex"):
        r = compare_runs(EVALS[y], EVALS[x], m); CMP[(x, y, m)] = r
        lo, hi = r["ci95"]
        L.append(f"| {x} − {y} | {m} | {100*r['diff']:+.1f} [{100*lo:+.1f}, {100*hi:+.1f}] | {r['wins']} / {r['losses']} | "
                 f"{r['mcnemar_p']:.3g} | {'yes' if r['significant'] else 'no'} |")

# plain-language reading, generated from the numbers above (no hand-written claims)
R = []
fps = [a["overall"]["false_positive_share"][0] for a in AUDIT.values()]
if fps: R.append(f"- The single-database reward paid for provably wrong SQL in {P(min(fps))}–{P(max(fps))} of rewarded rollouts across {len(fps)} arms.")
sig_drift = [f"{a}-s{s}" for (a, s), v in AUDIT.items() if v["drift"].get("significant")]
R.append("- No arm shows a significant drift in that share over training: no evidence the policy learned to exploit it."
         if AUDIT and not sig_drift else f"- Significant drift in: {', '.join(sig_drift)}." if sig_drift else "")
gains = [(x, CMP[(x, 'base', 'ex')]) for x in EVALS if (x, "base", "ex") in CMP]
if gains: R.append("- GRPO vs base (EX): " + "; ".join(f"{x} {100*r['diff']:+.1f} pts ({'significant' if r['significant'] else 'n.s.'})" for x, r in gains) + ".")
ss = [(s, CMP[(f"GRPO suite s{s}", f"GRPO single s{s}", "ex")]) for s in (0, 1) if (f"GRPO suite s{s}", f"GRPO single s{s}", "ex") in CMP]
if ss:
    R.append("- Suite reward − single reward (EX): " + "; ".join(f"seed {s} {100*r['diff']:+.1f} pts ({'significant' if r['significant'] else 'not distinguishable'})" for s, r in ss)
             + (f"; mean over seeds {100*sum(r['diff'] for _, r in ss)/len(ss):+.1f} pts." if len(ss) == 2 else "."))
rep = [(a, CMP[(f"GRPO {a} s1", f"GRPO {a} s0", "ex")]) for a in ("single", "suite") if (f"GRPO {a} s1", f"GRPO {a} s0", "ex") in CMP]
if rep: R.append("- Seed 1 vs seed 0, same arm (EX): " + "; ".join(f"{a} {100*r['diff']:+.1f} pts ({'differs' if r['significant'] else 'replicates'})" for a, r in rep) + ".")
offs = {k: v["official"] for k, v in OFFICIAL.items() if v.get("official") is not None}
if offs: R.append("- Official test-suite EX: " + "; ".join(f"{k} {P(v)}" for k, v in offs.items()) + ".")
L += ["", "## 4 · Reading (generated from the tables)", ""] + [r for r in R if r]

L += ["", "## 5 · What this does not establish", "",
      "- Two seeds per arm is the minimum for a replication, not a distribution; seed-to-seed spread is only roughly known.",
      "- One model size (0.5B) and 300 steps: exploitation may appear with a larger model or longer training.",
      "- Mirage's suites approximate the official test suites; about half of suite-flagged false positives on dev were defensible answers (results/GALLERY.md).",
      "- 20.3% of seed-0 training rewards went to questions with empty gold results (7 empty-table databases); the suites cannot catch that leak.",
      "", "## 6 · Run status", "", "| stage | state | at (h) | detail |", "|---|---|---|---|"]
for k, v in STATUS.items():
    det = {kk: vv for kk, vv in v.items() if kk not in ("state", "at_h", "cmd")}
    L.append(f"| {k} | {v['state']} | {v['at_h']} | {json.dumps(det)[:120] if det else ''} |")
FINAL = "\n".join(L)
open(f"{RUNS}/FINAL.md", "w").write(FINAL); open(f"{W}/FINAL.md", "w").write(FINAL)
json.dump({"official": OFFICIAL, "status": STATUS}, open(f"{RUNS}/final.json", "w"), indent=1, default=list)
print(FINAL)
""")

code(r"""
# --- package: final_results.zip (repo-ready: unzip into the repo as results/final/) + seed-1 adapters ----
PK = f"{W}/pkg/final"; shutil.rmtree(f"{W}/pkg", ignore_errors=True); os.makedirs(PK)
def cp(src, dst):
    if os.path.isdir(src): shutil.copytree(src, dst, dirs_exist_ok=True)
    elif os.path.exists(src): os.makedirs(os.path.dirname(dst), exist_ok=True); shutil.copy2(src, dst)
cp(f"{RUNS}/FINAL.md", f"{PK}/FINAL.md"); cp(f"{RUNS}/final.json", f"{PK}/final.json")
for arm in ("single", "suite"):
    a = f"{RUNS}/arms/{arm}-s1"
    for f in ("manifest.json", "timing.json", "train_log.json"): cp(f"{a}/{f}", f"{PK}/arms/{arm}-s1/{f}")
    if os.path.exists(f"{a}/ledger.jsonl"):
        os.makedirs(f"{PK}/arms/{arm}-s1", exist_ok=True)
        with open(f"{a}/ledger.jsonl", "rb") as fi, gzip.open(f"{PK}/arms/{arm}-s1/ledger.jsonl.gz", "wb") as fo: shutil.copyfileobj(fi, fo)
    e = f"{RUNS}/eval/grpo-{arm}-s1-dev"
    for f in ("scores.json", "scored_items.jsonl", "manifest.json"): cp(f"{e}/{f}", f"{PK}/eval/grpo-{arm}-s1-dev/{f}")
for f in glob.glob(f"{RUNS}/audit/*.json"): cp(f, f"{PK}/audit/{os.path.basename(f)}")
for f in glob.glob(f"{OFF}/*_pred.txt"): cp(f, f"{PK}/official/{os.path.basename(f)}")
cp(f"{RUNS}/report_seed1", f"{PK}/report_seed1")
for f in glob.glob(f"{RUNS}/logs/*.log"): cp(f, f"{PK}/logs/{os.path.basename(f)}")
shutil.make_archive(f"{W}/final_results", "zip", f"{W}/pkg")
# adapters separately (for a Hugging Face upload later); drop training checkpoints from the output
AD = f"{W}/pkg_adapters"; shutil.rmtree(AD, ignore_errors=True)
for arm in ("single", "suite"): cp(f"{RUNS}/arms/{arm}-s1/adapter", f"{AD}/{arm}-s1")
if os.path.isdir(AD): shutil.make_archive(f"{W}/adapters_seed1", "zip", AD); shutil.rmtree(AD)
for ck in glob.glob(f"{RUNS}/arms/*/checkpoint-*"): shutil.rmtree(ck, ignore_errors=True)
for f in glob.glob(f"{RUNS}/audit/*.ledger.jsonl"): os.remove(f)
shutil.rmtree(f"{W}/pkg", ignore_errors=True); shutil.rmtree(f"{W}/ts_raw", ignore_errors=True)
print("final_results.zip:", round(os.path.getsize(f"{W}/final_results.zip") / 1e6, 1), "MB")
if os.path.exists(f"{W}/adapters_seed1.zip"): print("adapters_seed1.zip:", round(os.path.getsize(f"{W}/adapters_seed1.zip") / 1e6, 1), "MB")
print(f"done in {hours():.2f} h")
""")


def build(path):
    nb = {"cells": [], "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                    "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    for i, (kind, src) in enumerate(CELLS):
        cell = {"cell_type": kind, "id": f"c{i:02d}", "metadata": {}, "source": src.splitlines(keepends=True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        nb["cells"].append(cell)
    Path(path).write_text(json.dumps(nb, indent=1), encoding="utf-8")


if __name__ == "__main__":
    out = Path(__file__).with_name("12_final.ipynb")
    build(out)
    print("wrote", out)
