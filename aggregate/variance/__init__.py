"""variance — how much a build moves when the sources and the clustering draw change.

    res = agg.variance(atoms, r=3, b=5, frac=0.8)      # b subsamples of 80% of the sources x r repetitions

The design is crossed: b subsamples of a fraction of the SOURCES (subsample.py), each built r times with an
independent clustering draw (replicate.py), b * r full builds in all. Each run is an ordinary `chain.build`
under `<workdir>/variance/s<i>_r<j>/`, sharing the verdict cache; a finished run is read back on a rerun.

    res["design"]        {"r", "b", "frac", "seed", "n_sources", "n_texts"}
    res["runs"]          [{"run", "subsample", "rep", "sources", "workdir", "metrics"}], the runs that finished
    res["failed"]        [{"run", "subsample", "rep", "error"}], runs whose build raised (left out below)
    res["variance"]      {metric: {"mean", "var", "sd", "between", "within", "values"}} (metrics.py): number
                         of clusters, share of texts filed, exposure, ...; between = which sources, within = the draw
    res["texts"]         the corpus's task texts, the order of the two matrices
    res["cooccurrence"]  texts x texts: the share of runs holding both texts that filed them in one family
    res["present"]       texts x texts: the runs whose subsample held both
    res["stability"]     {text: {"stability", "filed", "n_runs"}} (partitions.stability)
    res["pairwise"]      {"ami", "ari", "vi", "vi_norm"}: runs x runs matrices (in the order of res["runs"]) and
                         their means over all pairs, within a subsample and between subsamples (partitions.py)

Matrices are nested lists; the same dict is written to `<workdir>/variance/variance.json`. Cost: b * r builds,
of which repetition 1 of a subsample already built is free, plus two exposure calls per new statement.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from ..atoms import Atoms
from ..lib import atomic_write, load_json
from . import metrics, partitions, replicate, subsample


def _plain(x):
    """numpy -> lists and floats, so the result is JSON-serialisable."""
    if isinstance(x, dict):
        return {k: _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    return x


def _one(agg, atoms, kept, rep, workdir, design, exposure, build_kw):
    """One run -> (build, metrics). A run that finished under the same design is read back, not rebuilt."""
    stamp = {"sources": kept, "rep": rep, **design}
    if (workdir / "build.done").exists() and load_json(workdir / "run.json") == stamp:
        ctx, build = replicate.context(agg, atoms, rep, workdir), load_json(workdir / "build.json")
    else:
        ctx, build = replicate.build(agg, atoms, rep, workdir, **build_kw)
        atomic_write(workdir / "run.json", stamp)
    return build, metrics.of(ctx, build, ctx.occupation, exposure=exposure)


def run(agg, atoms, *, r=None, b=None, frac=None, seed=None, exposure=None, workdir=None, **build_kw) -> dict:
    """`agg` is the Aggregator; `atoms` anything `Atoms.coerce` takes. r, b, frac, seed and exposure default to
    the config's `variance` block. `build_kw` goes to every chain.build (max_cycles, cluster_draws, ...)."""
    cfg = agg.config
    r = int(cfg.get("variance.r", 3) if r is None else r)
    b = int(cfg.get("variance.b", 5) if b is None else b)
    frac = float(cfg.get("variance.frac", 0.8) if frac is None else frac)
    seed = int(cfg.get("variance.seed", 0) if seed is None else seed)
    exposure = bool(cfg.get("variance.exposure", True) if exposure is None else exposure)
    if r < 1 or b < 1:
        raise ValueError(f"r and b must be at least 1 (got r={r}, b={b})")
    root = Path(workdir) if workdir else agg.workdir
    if root is None:
        raise ValueError("variance needs a workdir: every run keeps its build and ledgers under <workdir>/variance/")
    root = root / "variance"
    atoms = Atoms.coerce(atoms)
    texts = sorted(atoms.sources)
    n_sources = len({s for ws in atoms.sources.values() for s in ws})
    design = {"frac": frac, "seed": seed}
    runs, failed, L = [], [], []
    values = {name: np.full((b, r), np.nan) for name in metrics.NAMES if exposure or name not in ("beta", "rl")}
    for i, (kept, sub) in enumerate(subsample.draw(atoms, b, frac, seed), 1):
        for j in range(1, r + 1):
            name = f"s{i:02d}_r{j:02d}"
            print(f"variance: run {name} ({len(kept)}/{n_sources} sources, {len(sub)} texts)", file=sys.stderr)
            try:
                build, m = _one(agg, sub, kept, j, root / name, design, exposure, build_kw)
            except Exception as e:                        # noqa: BLE001 — one failed build must not lose the rest
                print(f"variance: run {name} FAILED: {str(e)[:200]}", file=sys.stderr)
                failed.append({"run": name, "subsample": i, "rep": j, "error": str(e)})
                continue
            for k in values:
                values[k][i - 1, j - 1] = m.get(k, np.nan)
            runs.append({"run": name, "subsample": i, "rep": j, "sources": kept, "workdir": str(root / name), "metrics": m})
            L.append(partitions.labels(build, texts, sub.sources))
    L = np.array(L, dtype=np.int64).reshape(len(runs), len(texts))
    co = partitions.cooccurrence(L)
    res = _plain({"design": {"r": r, "b": b, **design, "n_sources": n_sources, "n_texts": len(texts)},
                  "runs": runs, "failed": failed, "variance": metrics.variance(values, b, r), "texts": texts,
                  "cooccurrence": co["share"], "present": co["present"], "stability": partitions.stability(co, texts),
                  "pairwise": partitions.pairwise(L, [x["subsample"] for x in runs])})
    atomic_write(root / "variance.json", res)
    return res
