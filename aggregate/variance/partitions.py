"""partitions — how alike the runs' families are: co-occurrence, per-text stability, pairwise agreement.

    L = np.array([partitions.labels(build, texts, present) for ...])    # runs x texts
    co = partitions.cooccurrence(L)                # {"together", "present", "share"}: texts x texts
    st = partitions.stability(co, texts)           # {text: {"stability", "filed", "n_runs"}}
    pw = partitions.pairwise(L, subsample_of)      # {"ami", "ari", "vi", "vi_norm"}: runs x runs + means

A run's partition is a label per text: its family's index, UNFILED (-1) for a text the run left in the pile,
ABSENT (-2) for a text whose sources were not in the run's subsample. Two texts co-occur in a run when it filed
them in the same family; an unfiled text co-occurs with nothing. Two runs are compared on the texts both
subsamples hold, an unfiled text being a family of its own.

  ami       adjusted mutual information: 1 identical, ~0 chance. Information-based; the steadier index when
            the families are many, small and unequal.
  ari       adjusted Rand index: 1 identical, ~0 chance. Pair-counting — the chance-corrected share of text
            pairs both runs treat alike — so large families weigh more.
  vi        variation of information, H(U) + H(V) - 2 I(U; V) in nats: 0 identical, a metric on partitions.
  vi_norm   vi / log(n) over the n shared texts, in [0, 1]: comparable between pairs of different overlap.
"""
from __future__ import annotations

import numpy as np

UNFILED, ABSENT = -1, -2
INDICES = ("ami", "ari", "vi", "vi_norm")


def labels(build, texts, present):
    """One run's label per text of `texts` (the full corpus, in a fixed order); `present` is the run's texts."""
    at = {t: i for i, t in enumerate(texts)}
    out = np.full(len(texts), ABSENT, dtype=np.int64)
    for t in present:
        if t in at:
            out[at[t]] = UNFILED
    for gi, f in enumerate(build.get("core") or []):
        for m in f.get("members") or []:
            if m in at:
                out[at[m]] = gi
    return out


def cooccurrence(L):
    """-> {"together", "present", "share"}, each texts x texts. together[i, j] = runs that filed i and j in the
    same family; present[i, j] = runs whose subsample held both; share = together / present (NaN where the two
    were never in one subsample). The diagonal of share is the share of a text's runs that filed it."""
    L = np.atleast_2d(np.asarray(L, dtype=np.int64))
    n = L.shape[1]
    together, present = np.zeros((n, n), dtype=np.int64), np.zeros((n, n), dtype=np.int64)
    for row in L:
        here, filed = row > ABSENT, row > UNFILED
        present += np.outer(here, here)
        together += (row[:, None] == row[None, :]) & np.outer(filed, filed)
    with np.errstate(divide="ignore", invalid="ignore"):
        share = np.where(present > 0, together / np.maximum(present, 1), np.nan)
    return {"together": together, "present": present, "share": share}


def stability(co, texts):
    """-> {text: {"stability", "filed", "n_runs"}}. `stability` is the text's mean co-occurrence share with the
    texts it was EVER filed with, weighted by the runs that held both: 1 when its companions are the same in
    every run, lower as they change; NaN for a text never filed with another. `filed` is the share of its runs
    that filed it; `n_runs` the runs whose subsample held it."""
    together, present = np.asarray(co["together"], float), np.asarray(co["present"], float)
    runs = np.diag(present).copy()
    filed_runs = np.diag(together).copy()
    T = together.copy()
    np.fill_diagonal(T, 0)
    den = (present * (T > 0)).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        stab = np.where(den > 0, T.sum(axis=1) / np.maximum(den, 1), np.nan)
        filed = np.where(runs > 0, filed_runs / np.maximum(runs, 1), np.nan)
    return {t: {"stability": float(stab[i]), "filed": float(filed[i]), "n_runs": int(runs[i])} for i, t in enumerate(texts)}


def _entropy(x):
    p = np.unique(x, return_counts=True)[1] / len(x)
    return float(-(p * np.log(p)).sum())


def compare(a, b):
    """Two runs' label vectors -> {"ami", "ari", "vi", "vi_norm", "n"} on the texts both hold (NaN under 2)."""
    from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, mutual_info_score
    a, b = np.asarray(a), np.asarray(b)
    both = (a > ABSENT) & (b > ABSENT)
    n = int(both.sum())
    if n < 2:
        return {**{k: float("nan") for k in INDICES}, "n": n}

    def own(x):                                          # an unfiled text is a family of its own
        x = x[both].copy()
        loose = x == UNFILED
        x[loose] = x.max() + 1 + np.arange(int(loose.sum()))
        return x
    x, y = own(a), own(b)
    vi = max(0.0, _entropy(x) + _entropy(y) - 2 * float(mutual_info_score(x, y)))
    return {"ami": float(adjusted_mutual_info_score(x, y)), "ari": float(adjusted_rand_score(x, y)),
            "vi": vi, "vi_norm": vi / float(np.log(n)), "n": n}


def pairwise(L, subsample_of):
    """Every pair of runs -> {index: {"matrix", "mean", "within_subsample", "between_subsamples"}} for ami, ari,
    vi, vi_norm, plus "n_shared" (the texts each pair was compared on). `subsample_of[k]` is run k's subsample:
    a pair WITHIN a subsample differs by the draw alone, a pair BETWEEN subsamples by the sources as well."""
    L = np.asarray(L, dtype=np.int64)
    m = len(L)
    mats = {k: np.full((m, m), np.nan) for k in INDICES}
    shared = np.zeros((m, m), dtype=np.int64)
    for i in range(m):
        for j in range(i, m):
            c = compare(L[i], L[j])
            shared[i, j] = shared[j, i] = c["n"]
            for k in INDICES:
                mats[k][i, j] = mats[k][j, i] = c[k]
    sub = np.asarray(subsample_of)
    off = ~np.eye(m, dtype=bool)
    same = (sub[:, None] == sub[None, :]) & off if m else off

    def mean(M, mask):
        v = M[mask]
        v = v[~np.isnan(v)]
        return float(v.mean()) if len(v) else float("nan")
    out = {k: {"matrix": M, "mean": mean(M, off), "within_subsample": mean(M, same),
               "between_subsamples": mean(M, off & ~same)} for k, M in mats.items()}
    out["n_shared"] = shared
    return out
