"""replicate — one build of the design, with repetitions that are independent draws rather than cache replays.

    ctx, build = replicate.build(agg, sub_atoms, rep=2, workdir=root / "s01_r02")

Every judge is memoised on its inputs, so building the same atoms twice replays the first build. Repetition
j > 1 therefore runs on a context whose cache and engine are wrapped:

  RepCache    the CLUSTERING section's keys are salted with the repetition, so the families are drawn again;
              every other section is shared, so the writer, the hearings, refile and the exposure rubrics stay
              functions of their inputs and two repetitions differ exactly where their clusterings differ
  RepEngine   the request seed is offset by 1000 (j - 1), so a provider that honours `seed` does not replay
              another repetition

Repetition 1 is the plain context: an ordinary build, with the ordinary cache keys. Nothing in the chain is
changed for this — the wrappers sit on the two calls the clustering makes (`ctx.cache.memo` on its section,
`ctx.engine.chat_kwargs`), so repeated draws within one clustering (`cluster_draws`, where the chain offers
them) keep their own slots inside each repetition.
"""
from __future__ import annotations

from ..lib import sha256

CLUSTER_SECTION = "rollup_decompose_groups"
SEED_STRIDE = 1000           # far apart, so a repetition's seeds never meet another's repeated draws (seed + d - 1)


class RepCache:
    """The shared cache, with the clustering section's keys salted by the repetition."""

    def __init__(self, cache, rep):
        self._cache, self.rep = cache, int(rep)

    def __getattr__(self, name):
        return getattr(self._cache, name)

    def _key(self, section, key):
        return sha256(f"{key}␟REP{self.rep}") if section == CLUSTER_SECTION and self.rep > 1 else key

    def get(self, section, key):
        return self._cache.get(section, self._key(section, key))

    def put(self, section, key, value):
        return self._cache.put(section, self._key(section, key), value)

    def memo(self, section, key, compute, **kw):
        return self._cache.memo(section, self._key(section, key), compute, **kw)


class RepEngine:
    """The shared engine (one usage account), with the request seed offset per repetition."""

    def __init__(self, engine, rep):
        self._engine, self.rep = engine, int(rep)

    def __getattr__(self, name):
        return getattr(self._engine, name)

    def chat_kwargs(self, *a, **k):
        kw = self._engine.chat_kwargs(*a, **k)
        if "seed" in kw:
            kw["seed"] = int(kw["seed"]) + SEED_STRIDE * (self.rep - 1)
        return kw


def context(agg, atoms, rep=1, workdir=None):
    """The context one run builds on: its own workdir (so no ledger verdict carries between runs), the shared
    cache and engine, wrapped when `rep` > 1."""
    ctx = agg.context(atoms, workdir=workdir)
    if rep > 1:
        ctx.cache = RepCache(ctx.cache, rep)
        ctx.engine = RepEngine(ctx.engine, rep)
    return ctx


def build(agg, atoms, rep=1, workdir=None, **build_kw):
    """-> (ctx, build). `build_kw` goes to chain.build (max_cycles, cluster_draws, ...)."""
    from ..aggregation import chain
    ctx = context(agg, atoms, rep, workdir)
    return ctx, chain.build(ctx, **build_kw)
