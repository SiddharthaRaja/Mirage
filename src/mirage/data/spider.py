"""Loading Spider (or anything laid out like it).

Spider ships as a folder:

    spider/
      train_spider.json   ~7,000 questions   (the HF "train" split)
      train_others.json   ~1,700 questions   (extra training DBs, same format)
      dev.json            ~1,030 questions   (20 databases, none shared with train)
      tables.json         schema + foreign keys for every database
      database/<db_id>/<db_id>.sqlite

Each question row has at least `db_id`, `question` and `query` (the gold SQL).
That is all this loader relies on, so the bundled mini fixture and any other
Spider-shaped dataset (your own databases, for instance) load the same way.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mirage.errors import DataError

SPLIT_FILES: dict[str, list[str]] = {
    "train": ["train_spider.json"],
    "train_all": ["train_spider.json", "train_others.json"],
    "dev": ["dev.json"],
}


@dataclass
class Example:
    qid: str
    db_id: str
    question: str
    gold_sql: str
    split: str
    hardness: str = "unknown"
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "qid": self.qid,
            "db_id": self.db_id,
            "question": self.question,
            "gold_sql": self.gold_sql,
            "split": self.split,
            "hardness": self.hardness,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Example:
        return cls(
            qid=d["qid"],
            db_id=d["db_id"],
            question=d["question"],
            gold_sql=d["gold_sql"],
            split=d.get("split", "unknown"),
            hardness=d.get("hardness", "unknown"),
            meta=d.get("meta", {}),
        )


def make_qid(db_id: str, question: str, sql: str) -> str:
    """A content hash, not a row index.

    Row indices shift when a file is re-exported or filtered; a hash of the
    content does not. Every ledger row, prediction and split decision is keyed
    on this, so a result from last week still joins to today's data.
    """
    h = hashlib.sha1(f"{db_id}\x1f{question.strip()}\x1f{sql.strip()}".encode()).hexdigest()
    return f"{db_id}:{h[:10]}"


class Spider:
    """A handle on a Spider-shaped folder."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        if not self.root.exists():
            raise DataError(
                f"Spider folder not found: {self.root}\n"
                "Download Spider 1.0 (CC BY-SA 4.0) from its project page and unzip it, "
                "or run `mirage data mini <dir>` for the bundled synthetic fixture."
            )
        self.db_root = self._find_db_root()
        self._tables: dict[str, dict[str, Any]] | None = None

    def _find_db_root(self) -> Path:
        for cand in ("database", "databases"):
            p = self.root / cand
            if p.is_dir():
                return p
        raise DataError(f"no database/ folder under {self.root}")

    # -- questions --------------------------------------------------------- #

    def load(self, split: str, hardness: bool = True) -> list[Example]:
        files = SPLIT_FILES.get(split, [f"{split}.json"])
        rows: list[dict[str, Any]] = []
        found = False
        for name in files:
            p = self.root / name
            if p.exists():
                found = True
                rows.extend(json.loads(p.read_text(encoding="utf-8")))
        if not found:
            raise DataError(f"none of {files} exist under {self.root}")

        if hardness:
            from mirage.data.hardness import classify_hardness
        out: list[Example] = []
        seen: set[str] = set()
        for r in rows:
            sql = r.get("query") or r.get("gold_sql") or r.get("sql")
            if not isinstance(sql, str):
                raise DataError(f"row without a gold SQL string in {split}: {r.get('question')!r}")
            qid = make_qid(r["db_id"], r["question"], sql)
            if qid in seen:  # Spider has a handful of exact duplicates
                continue
            seen.add(qid)
            out.append(
                Example(
                    qid=qid,
                    db_id=r["db_id"],
                    question=r["question"].strip(),
                    gold_sql=sql.strip().rstrip(";"),
                    split=split,
                    hardness=classify_hardness(sql) if hardness else "unknown",
                )
            )
        return out

    # -- databases --------------------------------------------------------- #

    def db_path(self, db_id: str) -> Path:
        p = self.db_root / db_id / f"{db_id}.sqlite"
        if not p.exists():
            raise DataError(f"database file missing: {p}")
        return p

    def db_ids(self) -> list[str]:
        return sorted(
            d.name for d in self.db_root.iterdir() if d.is_dir() and (d / f"{d.name}.sqlite").exists()
        )

    def tables(self) -> dict[str, dict[str, Any]]:
        if self._tables is None:
            p = self.root / "tables.json"
            self._tables = (
                {t["db_id"]: t for t in json.loads(p.read_text(encoding="utf-8"))} if p.exists() else {}
            )
        return self._tables

    def check(self) -> dict[str, Any]:
        """What `mirage data check` prints: counts, coverage and the overlap
        between splits that must be zero for dev to mean anything."""
        report: dict[str, Any] = {"root": str(self.root), "n_databases": len(self.db_ids())}
        db_sets: dict[str, set[str]] = {}
        for split in ("train", "train_all", "dev"):
            try:
                ex = self.load(split)
            except DataError:
                continue
            missing = sorted({e.db_id for e in ex} - set(self.db_ids()))
            hard: dict[str, int] = {}
            for e in ex:
                hard[e.hardness] = hard.get(e.hardness, 0) + 1
            db_sets[split] = {e.db_id for e in ex}
            report[split] = {
                "n_questions": len(ex),
                "n_databases": len(db_sets[split]),
                "by_hardness": dict(sorted(hard.items())),
                "missing_db_files": missing,
            }
        if "dev" in db_sets:
            train_key = "train_all" if "train_all" in db_sets else "train"
            if train_key in db_sets:
                report["train_dev_db_overlap"] = sorted(db_sets[train_key] & db_sets["dev"])
        return report


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise DataError(f"file not found: {p}")
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
