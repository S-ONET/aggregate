"""variance: source subsamples x independent repetitions, the metrics' variance, co-occurrence, stability and
the pairwise indices. No model is reached: the chain's judges are stubbed on the Judges class, because every
run makes its own context."""
import json
import math

import numpy as np
import pytest

from aggregate import Aggregator, Atoms, Config
from aggregate.aggregation import chain, cluster, synthesize
from aggregate.evaluate import exposure
from aggregate.judges import Judges
from aggregate.variance import metrics, partitions, replicate, run, subsample
from conftest import FakeClient, atoms_from_sources, orth_embed

U, A = partitions.UNFILED, partitions.ABSENT
W = {f"{fam}{i}": {f"w{k}", f"w{k + 1}"} for k, (fam, i) in enumerate([(f, i) for f in "abc" for i in (1, 2, 3)], 1)}


# ── subsample ─────────────────────────────────────────────────────────────────────────────────────

def test_subsample_keeps_a_fraction_of_the_sources_and_is_seeded():
    atoms = atoms_from_sources(W)                           # 10 sources
    subs = subsample.draw(atoms, b=4, frac=0.8, seed=3)
    assert [len(kept) for kept, _ in subs] == [8] * 4
    for kept, sub in subs:
        assert {r["source"] for r in sub.rows} == set(kept)
        assert all(ws <= set(kept) for ws in sub.sources.values())
    assert len({tuple(kept) for kept, _ in subs}) > 1       # the subsamples differ
    assert [k for k, _ in subsample.draw(atoms, 4, 0.8, seed=3)] == [k for k, _ in subs]
    assert [k for k, _ in subsample.draw(atoms, 4, 0.8, seed=4)] != [k for k, _ in subs]
    with pytest.raises(ValueError):
        subsample.draw(atoms, 2, 0.0)


def test_subsample_is_taken_within_each_instrument():
    atoms = Atoms.combine({"interview": atoms_from_sources({"x": {f"i{k}" for k in range(10)}}).rows,
                           "posting": atoms_from_sources({"x": {f"p{k}" for k in range(4)}, "y": {"p0"}}).rows})
    for kept, sub in subsample.draw(atoms, b=5, frac=0.5, seed=0):
        assert sub.instrument_sizes == {"interview": 5, "posting": 2}
        assert all(r["instrument"] for r in sub.rows)


# ── replicate ─────────────────────────────────────────────────────────────────────────────────────

def test_a_repetition_redraws_the_clustering_and_nothing_else(agg, monkeypatch):
    atoms = atoms_from_sources(W)
    calls = []

    def chat(ctx, prompt, cap, tag, **kw):
        calls.append(tag)
        return "Alpha :: 1, 2", ctx.effort("cluster")
    monkeypatch.setattr(cluster, "chat", chat)
    labels = sorted(W)
    for rep, expected in [(1, 1), (1, 1), (2, 2), (2, 2), (3, 3), (1, 3)]:
        ctx = replicate.context(agg, atoms, rep)
        assert cluster.group(ctx, labels) == [{"head": "Alpha", "members": labels[:2]}]
        assert len(calls) == expected                       # one call per repetition, then the cache
    plain, rep2 = replicate.context(agg, atoms, 1), replicate.context(agg, atoms, 2)
    assert plain.cache is agg.cache and plain.engine is agg.engine      # repetition 1 is an ordinary context
    plain.cache.memo("some_judge", "k", lambda: "v")
    assert rep2.cache.memo("some_judge", "k", lambda: pytest.fail("other sections are shared")) == "v"
    assert rep2.cache.dir == agg.cache.dir


def test_a_repetition_offsets_the_seed(agg):
    base = agg.engine.chat_kwargs("gpt-5.4", 100)
    kw = replicate.RepEngine(agg.engine, 3).chat_kwargs("gpt-5.4", 100)
    assert kw["seed"] == base["seed"] + 2 * replicate.SEED_STRIDE and {k: v for k, v in kw.items() if k != "seed"} == {k: v for k, v in base.items() if k != "seed"}


# ── partitions ────────────────────────────────────────────────────────────────────────────────────

def test_labels_mark_unfiled_and_absent_texts():
    build = {"core": [{"members": ["a", "b"]}, {"members": ["c"]}]}
    assert partitions.labels(build, ["a", "b", "c", "d", "e"], {"a", "b", "c", "d"}).tolist() == [0, 0, 1, U, A]


def test_cooccurrence_counts_runs_together_over_runs_present():
    L = [[0, 0, 1, U],                                      # a+b together, c alone, d unfiled
         [0, 1, 1, A],                                      # b+c together, d not in the subsample
         [0, 0, 0, U]]
    co = partitions.cooccurrence(L)
    assert co["together"][0, 1] == 2 and co["together"][1, 2] == 2 and co["together"][0, 2] == 1
    assert co["present"][0, 3] == 2 and co["present"][0, 1] == 3
    assert co["together"][3].sum() == 0                     # unfiled co-occurs with nothing, itself included
    assert co["share"][0, 1] == pytest.approx(2 / 3) and co["share"][0, 3] == 0 and co["share"][3, 3] == 0
    assert np.isnan(partitions.cooccurrence([[0, A], [A, 0]])["share"][0, 1])    # never in one subsample


def test_stability_is_one_for_fixed_companions_and_lower_for_a_text_that_moves():
    L = [[0, 0, 1, 1, 0], [0, 0, 1, 1, 1], [0, 0, 1, 1, U], [0, 0, 1, 1, A]]     # e moves: a/b, c/d, unfiled, absent
    st = partitions.stability(partitions.cooccurrence(L), list("abcde"))
    assert st["e"] == {"stability": pytest.approx(1 / 3), "filed": pytest.approx(2 / 3), "n_runs": 3}
    assert st["a"]["stability"] == pytest.approx((4 + 1) / (4 + 3)) and st["a"]["filed"] == 1 and st["a"]["n_runs"] == 4
    lone = partitions.stability(partitions.cooccurrence([[0, 1], [0, 1]]), ["x", "y"])
    assert math.isnan(lone["x"]["stability"]) and lone["x"]["filed"] == 1


def test_compare_identical_unrelated_and_hand_computed():
    same = partitions.compare([0, 0, 1, 1, 2, 2], [5, 5, 3, 3, 9, 9])           # the labels' names do not matter
    assert same["ami"] == pytest.approx(1) and same["ari"] == pytest.approx(1) and same["vi"] == pytest.approx(0)
    halves = partitions.compare([0, 0, 0, 0], [0, 0, 1, 1])                      # one family against two halves
    assert halves["vi"] == pytest.approx(math.log(2)) and halves["vi_norm"] == pytest.approx(math.log(2) / math.log(4))
    assert halves["ari"] == pytest.approx(0) and halves["n"] == 4
    rng = np.random.default_rng(0)
    x, y = rng.integers(0, 5, 2000), rng.integers(0, 5, 2000)
    rand = partitions.compare(x, y)
    assert abs(rand["ami"]) < 0.02 and abs(rand["ari"]) < 0.02 and 0 < rand["vi_norm"] <= 1
    assert partitions.compare(y, x)["vi"] == pytest.approx(rand["vi"])


def test_compare_uses_shared_texts_and_unfiled_as_singletons():
    c = partitions.compare([0, 0, 1, 1, A, 0], [0, 0, 1, 1, 0, A])
    assert c["n"] == 4 and c["ami"] == pytest.approx(1)
    assert partitions.compare([U, U, U], [0, 1, 2])["vi"] == pytest.approx(0)    # three singletons both ways
    assert partitions.compare([U, U, U], [0, 0, 0])["vi"] == pytest.approx(math.log(3))
    assert math.isnan(partitions.compare([0, A, A], [A, 0, 0])["ami"])


def test_pairwise_separates_pairs_within_and_between_subsamples():
    L = [[0, 0, 1, 1], [0, 0, 1, 1], [0, 1, 0, 1], [0, 1, 0, 1]]
    pw = partitions.pairwise(L, [1, 1, 2, 2])
    assert pw["ari"]["within_subsample"] == pytest.approx(1) and pw["ari"]["between_subsamples"] < 0.01
    assert pw["vi"]["within_subsample"] == pytest.approx(0) and pw["vi"]["between_subsamples"] > 0
    M = pw["ami"]["matrix"]
    assert M.shape == (4, 4) and np.allclose(M, M.T) and np.allclose(np.diag(M), 1)
    assert pw["ami"]["mean"] == pytest.approx((2 * 1 + 4 * M[0, 2]) / 6) and pw["n_shared"][0, 1] == 4
    assert math.isnan(partitions.pairwise(L[:2], [1, 2])["ami"]["within_subsample"])    # r = 1: no such pair


# ── metrics ───────────────────────────────────────────────────────────────────────────────────────

def test_variance_splits_between_subsamples_and_within():
    v = metrics.variance({"n": [[10, 12], [20, 22]], "flat": [[1, 1], [1, 1]], "gap": [[1, float("nan")], [3, 5]]}, 2, 2)
    assert v["n"]["mean"] == 16 and v["n"]["within"] == pytest.approx(2) and v["n"]["between"] == pytest.approx(50)
    assert v["n"]["var"] == pytest.approx(np.var([10, 12, 20, 22], ddof=1)) and v["n"]["sd"] == pytest.approx(math.sqrt(v["n"]["var"]))
    assert v["flat"]["var"] == 0 and v["flat"]["between"] == 0
    assert v["gap"]["mean"] == 3 and v["gap"]["within"] == pytest.approx(2)      # the failed run is left out
    one = metrics.variance({"n": [[4]]}, 1, 1)["n"]
    assert one["mean"] == 4 and math.isnan(one["var"]) and math.isnan(one["between"])


# ── the whole design ──────────────────────────────────────────────────────────────────────────────

def _stub(monkeypatch):
    """The chain's judges, for every context. Repetition 1 files a text by its first letter; a later one
    files the a's and b's together, so repetitions of one subsample differ and subsamples of one repetition
    agree. -> the exposure calls made."""
    def group(ctx, pile, **kw):
        key = (lambda t: t[0]) if getattr(ctx.cache, "rep", 1) == 1 else (lambda t: "ab" if t[0] in "ab" else t[0])
        fams = {}
        for t in sorted(pile):
            fams.setdefault(key(t), []).append(t)
        return [{"head": f"Family {k}", "members": ms} for k, ms in sorted(fams.items()) if len(ms) > 1]
    monkeypatch.setattr(cluster, "group", group)
    monkeypatch.setattr(synthesize, "statement", lambda ctx, head, atoms, siblings=(), standing=None: standing or f"Do {head.lower()} work.")
    monkeypatch.setattr(Judges, "embed", lambda self, xs: orth_embed(xs))
    monkeypatch.setattr(Judges, "covers", lambda self, atom, stmt, model=None: True)
    monkeypatch.setattr(Judges, "duties", lambda self, stmt, members, draw=1: {"one_duty": True, "outcomes": [], "reason": ""})
    monkeypatch.setattr(Judges, "overlaps", lambda self, a, b, draw=1: False)
    monkeypatch.setattr(Judges, "dispose", lambda self, atom, tasks, home=None, siblings=(), draw=1: {"fate": "LEAVE", "task": None})
    monkeypatch.setattr(Judges, "writing_defects", lambda self, stmt, atoms, exclude=(): {"defect": False, "principle": "none", "trigger": "", "confidence": 0.0})
    rated = []

    def rate(ctx, metric, task, title):
        rated.append((metric, task))
        return {"label": "E1", "beta": 1.0} if metric == "eloundou" else {"gate": True, "score": 40.0 if "ab" in task else 20.0}
    monkeypatch.setattr(exposure, "rate", rate)
    return rated


def test_run_builds_every_cell_and_reports_the_design(agg, monkeypatch, tmp_path):
    _stub(monkeypatch)
    atoms = atoms_from_sources(W)
    res = agg.variance(atoms, r=2, b=3, frac=0.8, seed=1)
    root = tmp_path / "work" / "variance"
    assert res["design"] == {"r": 2, "b": 3, "frac": 0.8, "seed": 1, "n_sources": 10, "n_texts": 9}
    assert [x["run"] for x in res["runs"]] == [f"s{i:02d}_r{j:02d}" for i in (1, 2, 3) for j in (1, 2)] and res["failed"] == []
    assert all((root / x["run"] / "build.done").exists() and len(x["sources"]) == 8 for x in res["runs"])
    assert json.loads((root / "variance.json").read_text())["design"] == res["design"]
    json.dumps(res)                                         # plain lists and floats throughout
    v = res["variance"]["n_clusters"]
    assert np.shape(v["values"]) == (3, 2) and v["within"] > 0          # the repetitions disagree
    assert set(res["variance"]) == set(metrics.NAMES) and res["variance"]["beta"]["mean"] == 1.0
    assert res["texts"] == sorted(W) and np.shape(res["cooccurrence"]) == (9, 9) == np.shape(res["present"])
    pw = res["pairwise"]
    assert set(pw) == {"ami", "ari", "vi", "vi_norm", "n_shared"} and np.shape(pw["ari"]["matrix"]) == (6, 6)
    assert pw["ami"]["between_subsamples"] > pw["ami"]["within_subsample"]      # here only the repetition moves a text
    assert pw["vi"]["within_subsample"] > 0
    st = res["stability"]
    assert set(st) == set(W) and all(0 <= s["filed"] <= 1 for s in st.values())
    kept = {t for t, s in st.items() if s["n_runs"] and s["filed"] == 1}
    assert kept and all(st[t]["stability"] == 1.0 for t in kept if t[0] == "c")      # the c's never change company
    assert any(st[t]["stability"] < 1 for t in kept if t[0] in "ab")


def test_a_rerun_reads_finished_runs_back_and_a_failed_run_is_set_aside(agg, monkeypatch):
    rated = _stub(monkeypatch)
    atoms = atoms_from_sources(W)
    first = agg.variance(atoms, r=2, b=2, frac=0.8)
    built, real = [], chain.build
    monkeypatch.setattr(chain, "build", lambda ctx, **kw: built.append(ctx.workdir.name) or real(ctx, **kw))
    again = agg.variance(atoms, r=2, b=2, frac=0.8)
    assert built == [] and again["variance"]["n_clusters"]["values"] == first["variance"]["n_clusters"]["values"]
    assert again["pairwise"]["ami"]["matrix"] == first["pairwise"]["ami"]["matrix"]
    agg.variance(atoms, r=2, b=2, frac=0.8, seed=5)        # another design: other subsamples, so every run is rebuilt
    assert sorted(built) == ["s01_r01", "s01_r02", "s02_r01", "s02_r02"]

    def flaky(ctx, **kw):
        if ctx.workdir.name == "s02_r01":
            raise RuntimeError("invariants failed after cycle 1: boom")
        return real(ctx, **kw)
    monkeypatch.setattr(chain, "build", flaky)
    res = agg.variance(atoms, r=2, b=2, frac=0.8, seed=9, exposure=False)
    assert [f["run"] for f in res["failed"]] == ["s02_r01"] and "boom" in res["failed"][0]["error"]
    assert len(res["runs"]) == 3 and np.shape(res["pairwise"]["ami"]["matrix"]) == (3, 3)
    assert math.isnan(res["variance"]["n_clusters"]["values"][1][0]) and not math.isnan(res["variance"]["n_clusters"]["mean"])
    assert "beta" not in res["variance"] and "beta" not in res["runs"][0]["metrics"]
    assert {m for m, _ in rated} == {"eloundou", "rl"}             # the earlier runs rated both rubrics


def test_run_needs_a_workdir_and_takes_its_defaults_from_the_config(monkeypatch, tmp_path):
    _stub(monkeypatch)
    atoms = atoms_from_sources(W)
    bare = Aggregator(Config.load(prewarm={"on": False}), client=FakeClient(), cache_dir=tmp_path / "cache")
    with pytest.raises(ValueError, match="workdir"):
        run(bare, atoms)
    cfg = Config.load(prewarm={"on": False}, variance={"r": 1, "b": 2, "frac": 0.5, "seed": 2, "exposure": False})
    res = run(Aggregator(cfg, client=FakeClient(), cache_dir=tmp_path / "cache"), atoms, workdir=tmp_path / "v")
    assert res["design"]["r"] == 1 and res["design"]["b"] == 2 and res["design"]["frac"] == 0.5 and len(res["runs"]) == 2
    assert (tmp_path / "v" / "variance" / "s02_r01" / "build.json").exists()
    assert math.isnan(res["variance"]["n_clusters"]["within"])          # r = 1: no repetition to compare


def test_run_on_a_multi_instrument_corpus(monkeypatch, tmp_path):
    _stub(monkeypatch)
    monkeypatch.setattr(Judges, "in_scope", lambda self, statement, instrument, description, only_cached=False: True)
    monkeypatch.setattr(Judges, "in_scope_many", lambda self, items: 0)
    inst = {"interview": {"scope": "What a worker says they do.", "min_share": 0.2},
            "posting": {"scope": "What an employer hires for.", "min_sources": 1}}
    agg = Aggregator(Config.load(prewarm={"on": False}, instruments=inst, min_sources=2), client=FakeClient(),
                     workdir=tmp_path / "work", cache_dir=tmp_path / "cache")
    atoms = Atoms.combine({"interview": atoms_from_sources(W).rows,
                           "posting": atoms_from_sources({"a1": {"p1", "p2"}, "b1": {"p2", "p3"}, "c1": {"p3", "p4"}}).rows})
    res = agg.variance(atoms, r=2, b=2, frac=0.75, exposure=False)
    assert res["failed"] == [] and len(res["runs"]) == 4 and res["design"]["n_sources"] == 14
    for x in res["runs"]:
        assert sum(s.startswith("posting:") for s in x["sources"]) == 3 and sum(s.startswith("interview:") for s in x["sources"]) == 8
        assert x["metrics"]["n_clusters"] >= 1
