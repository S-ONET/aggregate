"""Atoms — the table the chain reads: one row per (source, task text), the handoff a source produces.

Columns: source, task, occupation (any code the source uses), title (the occupation's name, rendered into
prompts), status, original (the text before normalization), plus the optional FJA fields duty, function,
level, verb, object (defined by prompts/shared/fja_hta.txt; `blocks` reads them). An ATOM is one distinct task
text plus the set of sources that reported it; a source is one interview, one employer — and every prevalence
number downstream counts that set. `read` also accepts the legacy column names (session_id, task_name,
onet_code, onet_title, original_task_name).

INSTRUMENTS. An optional `instrument` column says which kind of data a source came from (interview, posting,
...). `Atoms.combine({"interview": a, "posting": b})` stacks per-instrument tables into one, stamping the column
and namespacing every source id as "<instrument>:<source>" so two instruments' ids can never collide. The same
task text reported under two instruments is ONE atom whose sources span both.

    atoms = Atoms.read("atoms.csv")
    atoms = Atoms.from_rows([{"source": "s1", "task": "Draft motions.", "occupation": "23-1011.00", "title": "Lawyers"}])
    atoms.tasks, atoms.sources["Draft motions."], atoms.title, atoms.code, atoms.records()
    atoms = Atoms.combine({"interview": Atoms.read("i.csv"), "posting": Atoms.read("p.csv")})
    atoms.instrument_of["posting:e12"], atoms.instrument_sizes   # {"interview": 40, "posting": 900}
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

COLUMNS = ["source", "task", "occupation", "title", "status", "original"]
FJA = ["duty", "function", "level", "verb", "object"]
FUNCTIONS = ("data", "people", "things")
LEGACY = {"session_id": "source", "worker": "source", "task_name": "task", "onet_code": "occupation", "onet_title": "title",
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
        self._sources = None
        self._meta = None
        self._instrument_of = None

    @classmethod
    def read(cls, path) -> "Atoms":
        with Path(path).open(newline="") as fh:
            return cls(list(csv.DictReader(fh)))

    @classmethod
    def from_rows(cls, rows) -> "Atoms":
        return cls(rows)

    @classmethod
    def coerce(cls, atoms) -> "Atoms":
        """An Atoms, a path to its CSV, a list of rows, or {instrument: any of those} -> Atoms."""
        if isinstance(atoms, Atoms):
            return atoms
        if isinstance(atoms, dict):
            return cls.combine(atoms)
        if isinstance(atoms, (str, Path)):
            return cls.read(atoms)
        return cls(atoms)

    @classmethod
    def combine(cls, by_instrument: dict) -> "Atoms":
        """{instrument: Atoms | CSV path | rows} -> one table with an `instrument` column and source ids
        namespaced "<instrument>:<source>"."""
        rows = []
        for inst, part in by_instrument.items():
            inst = str(inst).strip()
            if not inst or ":" in inst:
                raise ValueError(f"instrument names must be non-empty and contain no ':' (got {inst!r})")
            for r in cls.coerce(part).rows:
                r = dict(r)
                if (r.get("instrument") or inst) != inst:
                    raise ValueError(f"a row under instrument {inst!r} says instrument={r['instrument']!r}")
                r["instrument"] = inst
                r["source"] = f"{inst}:{r['source']}"
                rows.append(r)
        return cls(rows)

    def write(self, path, legacy=False) -> None:
        """Write the CSV; `legacy=True` uses the old column names (session_id, task_name, onet_code, ...)."""
        cols = list(COLUMNS) + [k for k in FJA if any(k in r for r in self.rows)]
        for r in self.rows:
            for k in r:
                if k not in cols:
                    cols.append(k)
        back = {"source": "session_id", "task": "task_name", "occupation": "onet_code", "title": "onet_title", "original": "original_task_name"}
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=[back.get(c, c) if legacy else c for c in cols])
            w.writeheader()
            for r in self.rows:
                w.writerow({(back.get(k, k) if legacy else k): v for k, v in r.items()})

    @property
    def sources(self) -> dict[str, frozenset]:
        """{task text: frozenset(source ids)}, tasks in order of first appearance."""
        if self._sources is None:
            sessions = defaultdict(set)
            for r in self.rows:
                t = (r.get("task") or "").strip()
                if t:
                    sessions[t].add(r["source"])
            self._sources = {t: frozenset(s) for t, s in sessions.items()}
        return self._sources

    @property
    def tasks(self) -> list[str]:
        return list(self.sources)

    @property
    def instrument_of(self) -> dict[str, str]:
        """{source id: instrument}; {} when the table has no `instrument` column (a single-instrument corpus).
        A source id under two instruments is an error: a source is one interview, one posting."""
        if self._instrument_of is None:
            out = {}
            if any(r.get("instrument") for r in self.rows):
                for r in self.rows:
                    src, inst = r["source"], (r.get("instrument") or "").strip()
                    if not inst:
                        raise ValueError(f"source {src!r} has no instrument, but other rows do")
                    if out.setdefault(src, inst) != inst:
                        raise ValueError(f"source {src!r} appears under instruments {out[src]!r} and {inst!r}")
            self._instrument_of = out
        return self._instrument_of

    @property
    def instrument_sizes(self) -> dict[str, int]:
        """{instrument: distinct sources} — the denominator of a share floor."""
        n = defaultdict(int)
        for inst in self.instrument_of.values():
            n[inst] += 1
        return dict(sorted(n.items()))

    def records(self) -> list[dict]:
        return [{"task": t, "sources": sorted(s), "prevalence": len(s)} for t, s in self.sources.items()]

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
        return len(self.sources)

    def __iter__(self):
        return iter(self.sources)
