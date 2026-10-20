"""Pass-rate filter on a fixed, hashed subset of training questions.

Same logic as `mirage filter`, but on N questions instead of all ~7,800:
GRPO only sees 300 steps x 8 questions = 2,400 question draws, so filtering
the whole training set would spend ~6 GPU hours on questions it never uses.
The subset is chosen by hashing each question id, so it is spread across all
training databases, identical on every re-run, and the same for both arms.

usage: python filter_subset.py N [k]
"""

import hashlib
import json
import sys
from pathlib import Path

from mirage.config import load_config, parse_overrides
from mirage.eval.generate import GenConfig, build_generator
from mirage.reward.suite import SuiteIndex
from mirage.reward.verifier import Verifier
from mirage.train.common import resolve_start
from mirage.train.data import load_sets, prompt_spec
from mirage.train.filter import pass_rate_filter

N = int(sys.argv[1])
K = int(sys.argv[2]) if len(sys.argv) > 2 else 8
CONFIG = sys.argv[3] if len(sys.argv) > 3 else "configs/grpo_base.yaml"

cfg = load_config(CONFIG, parse_overrides([]))
cfg.data.filtered_file = None
sets = load_sets(cfg)


def h(qid: str) -> str:
    return hashlib.sha256(f"mirage-filter-v1:{qid}".encode()).hexdigest()


subset = sorted(sets.train, key=lambda e: h(e.qid))[:N]
print(f"[filter] {len(subset)} of {len(sets.train)} training questions, "
      f"{len({e.db_id for e in subset})} of {len({e.db_id for e in sets.train})} databases, k={K}", flush=True)

gen = build_generator(GenConfig(backend=cfg.eval.backend, model=resolve_start(cfg), dtype=cfg.model.dtype,
                                batch_size=cfg.eval.batch_size, chat_template_kwargs=cfg.model.chat_template_kwargs,
                                base_url=cfg.eval.base_url))
_gen, _done, _t0 = gen.generate, [0], __import__("time").time()
def _progress(prompts, n, temperature):
    r = _gen(prompts, n, temperature); _done[0] += len(prompts)
    el = __import__("time").time() - _t0
    print(f"[filter] {_done[0]}/{len(subset)} questions, {el/60:.1f} min, ~{el/_done[0]*(len(subset)-_done[0])/60:.0f} min left", flush=True)
    return r
gen.generate = _progress
out = Path(cfg.out).parent / "filter" / "keep.jsonl"
res = pass_rate_filter(subset, gen, Verifier(sets.db_paths(), SuiteIndex(cfg.suites)), prompt_spec(cfg), out, k=K)
res.update({"subset_of": len(sets.train), "k": K, "subset_rule": "sha256('mirage-filter-v1:'+qid), first N"})
(out.parent / "summary.json").write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))
print(f"wrote {out}")
