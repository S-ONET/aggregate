"""subsample — b random subsets of a corpus's SOURCES, each a fraction of them.

    subs = subsample.draw(atoms, b=5, frac=0.8, seed=0)        # [(kept source ids, Atoms)]

A source is one interview, one posting: the unit a build's evidence is counted in, so it is the unit resampled.
Sources are drawn without replacement, `round(frac * n)` of them (at least one). With instruments
(`Atoms.combine`), the fraction is taken WITHIN each instrument, so every subsample keeps the corpus's mix and a
share floor (`instruments.<i>.min_share`) rescales with the instrument's size. Every row of a kept source is kept,
with all its columns.
"""
from __future__ import annotations

import numpy as np

from ..atoms import Atoms


def draw(atoms, b, frac, seed=0):
    atoms = Atoms.coerce(atoms)
    if not 0 < frac <= 1:
        raise ValueError(f"frac must be in (0, 1], got {frac}")
    inst = atoms.instrument_of
    groups = {}
    for src in sorted({r["source"] for r in atoms.rows if (r.get("task") or "").strip()}):
        groups.setdefault(inst.get(src, ""), []).append(src)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(int(b)):
        keep = set()
        for name in sorted(groups):
            srcs = groups[name]
            k = min(len(srcs), max(1, int(round(frac * len(srcs)))))
            keep.update(srcs[i] for i in rng.choice(len(srcs), size=k, replace=False))
        out.append((sorted(keep), Atoms([r for r in atoms.rows if r["source"] in keep])))
    return out
