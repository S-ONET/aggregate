"""metrics — the list-level numbers of one run, and their variance over the design.

    m = metrics.of(ctx, build, title)                  # {"n_clusters", "filed_share", "beta", "rl", ...}
    v = metrics.variance(values, b, r)                 # {metric: {"mean", "var", "sd", "between", "within", "values"}}

  n_clusters    families the build published (len(core))
  n_atoms       distinct task texts in the run's subsample
  filed         texts filed under a family; `pile` the texts left over; `filed_share` = filed / n_atoms
  prevalence    mean number of sources behind a family
  beta, rl      the two AI-exposure rubrics (evaluate/exposure.py), averaged over the run's statements

`values[metric]` is a b x r table (NaN for a run that failed). `var` is the sample variance over every run;
`between` is the variance of the b subsample means (which sources were drawn); `within` is the mean, over
subsamples, of the variance across the r repetitions (the draw alone). Each needs two values and is NaN without.
"""
from __future__ import annotations

import numpy as np

from .. import families
from ..evaluate import statements_of

NAMES = ("n_clusters", "n_atoms", "filed", "pile", "filed_share", "prevalence", "beta", "rl")


def of(ctx, build, title, exposure=True) -> dict:
    core = build.get("core") or []
    filed = sum(len(f.get("members") or []) for f in core)
    pile = len(families.pile(build))
    n = int((build.get("meta") or {}).get("n_atoms") or filed + pile)
    out = {"n_clusters": len(core), "n_atoms": n, "filed": filed, "pile": pile,
           "filed_share": filed / n if n else float("nan"),
           "prevalence": float(np.mean([f.get("prevalence") or 0 for f in core])) if core else float("nan")}
    if exposure:
        from ..evaluate import exposure as rubric
        got = rubric.run(ctx, {"run": statements_of(build)}, title)["lists"].get("run") or {}
        out["beta"], out["rl"] = got.get("beta", float("nan")), got.get("rl", float("nan"))
    return out


def _var(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    return float(x.var(ddof=1)) if len(x) > 1 else float("nan")


def _mean(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    return float(x.mean()) if len(x) else float("nan")


def variance(values, b, r) -> dict:
    out = {}
    for name, table in values.items():
        T = np.asarray(table, float).reshape(b, r)
        var = _var(T)
        out[name] = {"mean": _mean(T), "var": var, "sd": float(np.sqrt(var)),
                     "between": _var([_mean(row) for row in T]), "within": _mean([_var(row) for row in T]),
                     "values": T}
    return out
