"""dedup — drop exact-duplicate (worker, task) rows. No cross-worker merging, no fuzzy matching."""
from __future__ import annotations

from ..atoms import Atoms


def dedup(atoms) -> Atoms:
    rows = atoms.rows if isinstance(atoms, Atoms) else list(atoms)
    seen, kept = set(), []
    for r in rows:
        key = (r["worker"], (r.get("task") or "").strip())
        if key in seen:
            continue
        seen.add(key)
        kept.append(r)
    return Atoms(kept)
