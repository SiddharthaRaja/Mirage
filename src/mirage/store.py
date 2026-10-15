"""Run directories and manifests.

Every command that produces a result writes `manifest.json` next to it:
the resolved config, prompt version, git commit (and whether the tree was
dirty), library versions, hardware, and timestamps. `mirage report` refuses
to put two GRPO arms side by side as a comparison if their manifests differ
in anything except the reward — that is the guarantee behind "one variable
changed".
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path
from typing import Any

from mirage import __version__
from mirage.data.prompts import PROMPT_VERSION

_LIBS = ("torch", "transformers", "trl", "peft", "accelerate", "datasets", "vllm", "sqlglot")


def _git() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=5).stdout.strip()
        dirty = bool(
            subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, timeout=5).stdout.strip()
        )
        return {"sha": sha or None, "dirty": dirty}
    except Exception:
        return {"sha": None, "dirty": None}


def _versions() -> dict[str, str | None]:
    out = {}
    for lib in _LIBS:
        try:
            out[lib] = metadata.version(lib)
        except metadata.PackageNotFoundError:
            out[lib] = None
    return out


def _hardware() -> dict[str, Any]:
    hw: dict[str, Any] = {"python": sys.version.split()[0], "platform": platform.platform()}
    try:
        import torch

        if torch.cuda.is_available():
            hw["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception:
        pass
    return hw


def config_hash(cfg: dict, ignore: tuple[str, ...] = ()) -> str:
    def strip(d):
        if isinstance(d, dict):
            return {k: strip(v) for k, v in sorted(d.items()) if k not in ignore}
        return d

    return hashlib.sha256(json.dumps(strip(cfg), sort_keys=True, default=str).encode()).hexdigest()[:12]


def write_manifest(out_dir: str | Path, kind: str, config: dict | None = None, **extra: Any) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    m = {
        "kind": kind,
        "mirage_version": __version__,
        "prompt_version": PROMPT_VERSION,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git": _git(),
        "versions": _versions(),
        "hardware": _hardware(),
        "config": config,
        **extra,
    }
    path = out / "manifest.json"
    path.write_text(json.dumps(m, indent=1, default=str), encoding="utf-8")
    return path


def read_manifest(out_dir: str | Path) -> dict:
    p = Path(out_dir) / "manifest.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def arms_comparable(manifests: list[dict]) -> tuple[bool, list[str]]:
    """True if GRPO arms differ only in reward mode (and their own names/paths)."""
    ignore = ("name", "out", "mode", "notes")
    hashes = {}
    for m in manifests:
        cfg = dict(m.get("config") or {})
        hashes[m.get("config", {}).get("name", "?")] = config_hash(cfg, ignore)
    problems = []
    if len(set(hashes.values())) > 1:
        problems.append(f"arm configs differ beyond the reward mode: {hashes}")
    inits = {m.get("init_fingerprint") for m in manifests}
    if len(inits) > 1:
        problems.append(f"arms start from different initial weights: {inits}")
    return (not problems), problems
