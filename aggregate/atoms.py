"""Atoms — the table the chain reads: one row per (worker, task text), the handoff a source produces.

Columns: worker, task, occupation (any code the source uses), title (the occupation's name, rendered into
prompts), status, original (the text before normalization), plus the optional FJA fields duty, function,
level, verb, object (defined by prompts/shared/fja_hta.txt; `blocks` reads them). An ATOM is one distinct task
text plus the set of workers that reported it; a worker is one interview, one employer — and every prevalence
number downstream counts that set. `read` also accepts the legacy column names (session_id, task_name,
onet_code, onet_title, original_task_name).

    atoms = Atoms.read("atoms.csv")
    atoms = Atoms.from_rows([{"worker": "s1", "task": "Draft motions.", "occupation": "23-1011.00", "title": "Lawyers"}])
    atoms.tasks, atoms.workers["Draft motions."], atoms.title, atoms.code, atoms.records()
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

COLUMNS = ["worker", "task", "occupation", "title", "status", "original"]
FJA = ["duty", "function", "level", "verb", "object"]
FUNCTIONS = ("data", "people", "things")
LEGACY = {"session_id": "worker", "task_name": "task", "onet_code": "occupation", "onet_title": "title",
          "original_task_name": "original"}


def _canon(r: dict) -> dict:
    out = {}
    for k, v in r.items():
        out[LEGACY.get(k, k)] = v
    for k in COLUMNS:
        out.setdefault(k, "")
    return out


class Atoms:
    def __init__(self, rows):
        self.rows = [_canon(dict(r)) for r in rows]
        self._workers = None
        self._meta = None

    @classmethod
    def read(cls, path) -> "Atoms":
        with Path(path).open(newline="") as fh:
            return cls(list(csv.DictReader(fh)))

    @classmethod
    def from_rows(cls, rows) -> "Atoms":
        return cls(rows)

    def write(self, path, legacy=False) -> None:
        """Write the CSV; `legacy=True` uses the old column names (session_id, task_name, onet_code, ...)."""
        cols = list(COLUMNS) + [k for k in FJA if any(k in r for r in self.rows)]
        for r in self.rows:
            for k in r:
                if k not in cols:
                    cols.append(k)
        back = {v: k for k, v in LEGACY.items()}
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=[back.get(c, c) if legacy else c for c in cols])
            w.writeheader()
            for r in self.rows:
                w.writerow({(back.get(k, k) if legacy else k): v for k, v in r.items()})

    @property
    def workers(self) -> dict[str, frozenset]:
        """{task text: frozenset(worker ids)}, tasks in order of first appearance."""
        if self._workers is None:
            sessions = defaultdict(set)
            for r in self.rows:
                t = (r.get("task") or "").strip()
                if t:
                    sessions[t].add(r["worker"])
            self._workers = {t: frozenset(s) for t, s in sessions.items()}
        return self._workers

    @property
    def tasks(self) -> list[str]:
        return list(self.workers)

    def records(self) -> list[dict]:
        return [{"task": t, "workers": sorted(s), "prevalence": len(s)} for t, s in self.workers.items()]

    @property
    def meta(self) -> dict[str, dict]:
        """{task: {"function", "duty"}} from the FJA columns; {} when the table has no `function` column."""
        if self._meta is None:
            self._meta = {}
            if self.rows and "function" in self.rows[0]:
                for r in self.rows:
                    t = (r.get("task") or "").strip()
                    if t and t not in self._meta:
                        self._meta[t] = {"function": (r.get("function") or "").strip().lower(), "duty": (r.get("duty") or "").strip()}
        return self._meta

    @property
    def title(self) -> str:
        return (self.rows[0].get("title") or "").strip() if self.rows else ""

    @property
    def code(self) -> str:
        return (self.rows[0].get("occupation") or "").strip() if self.rows else ""

    def __len__(self):
        return len(self.workers)

    def __iter__(self):
        return iter(self.workers)
