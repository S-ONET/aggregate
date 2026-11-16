"""cluster.propose reconciles the artifact with the corpus before it draws, applies the floor, and the chunked
draw unions families split across chunks."""
from aggregate import families
from aggregate.aggregation import blocks, cluster
from conftest import atoms_from_sources


def _art():
    return {"meta": {"n_atoms": 4},
            "core": [{"sid": "s_a", "head": "A", "members": ["a1", "a2"], "sources": ["w1", "w2"], "prevalence": 2},
                     {"sid": "s_b", "head": "B", "members": ["b1"], "sources": ["w3"], "prevalence": 1}],
            "released": ["p1"]}


def _accounted(art):
    filed = {m for f in art["core"] for m in f["members"]}
    pile = set(families.pile(art))
    return len(filed) + len(pile) == art["meta"]["n_atoms"] and not (filed & pile)


def _ctx(agg, sources):
    return agg.context(atoms_from_sources(sources))


def test_new_corpus_atoms_enter_the_pile(agg, monkeypatch):
    ctx = _ctx(agg, {"a1": {"w1"}, "a2": {"w2"}, "b1": {"w3"}, "p1": {"w4"}, "n1": {"w5"}, "n2": {"w6"}})
    monkeypatch.setattr(cluster, "group", lambda ctx, pile: [])
    art = cluster.propose(ctx, _art())
    assert set(art["released"]) == {"p1", "n1", "n2"}
    assert art["meta"]["n_atoms"] == 6 and art["draw_pile_log"]["corpus_new"] == 2 and _accounted(art)


def test_vanished_members_leave_their_family(agg, monkeypatch):
    ctx = _ctx(agg, {"a1": {"w1"}, "a2": {"w2"}, "p1": {"w4"}})
    monkeypatch.setattr(cluster, "group", lambda ctx, pile: [])
    art = cluster.propose(ctx, _art())
    assert [f["head"] for f in art["core"]] == ["A"] and art["draw_pile_log"]["corpus_gone"] == ["b1"]
    assert art["meta"]["n_atoms"] == 3 and _accounted(art)


def test_unchanged_corpus_is_a_no_op(agg, monkeypatch):
    ctx = _ctx(agg, {"a1": {"w1"}, "a2": {"w2"}, "b1": {"w3"}, "p1": {"w4"}})
    monkeypatch.setattr(cluster, "group", lambda ctx, pile: [])
    art = cluster.propose(ctx, _art())
    assert art["released"] == ["p1"] and art["meta"]["n_atoms"] == 4
    assert art["draw_pile_log"]["corpus_new"] == 0 and art["draw_pile_log"]["corpus_gone"] == []


_CORPUS = {"a1": {"w1"}, "a2": {"w2"}, "b1": {"w3"}, "b2": {"w4"}, "p1": {"w5"}, "p2": {"w6"}, "p3": {"w7"}, "p4": {"w8"}}


def _art8():
    return {"meta": {"n_atoms": 8},
            "core": [{"sid": "s1", "statement": "Do alpha work.", "head": "Do alpha work.", "members": ["a1", "a2"], "sources": ["w1", "w2"], "prevalence": 2},
                     {"sid": "s2", "statement": "Do beta work.", "head": "Do beta work.", "members": ["b1", "b2"], "sources": ["w3", "w4"], "prevalence": 2}],
            "released": ["p1", "p2", "p3", "p4"]}


def test_cluster_over_pile_founds_an_ordinary_family(agg, monkeypatch):
    ctx = _ctx(agg, _CORPUS)
    monkeypatch.setattr(cluster, "group", lambda ctx, pile: [{"head": "Run gamma drills", "members": ["p1", "p2"]}])
    art = cluster.propose(ctx, _art8())
    cand = [f for f in art["core"] if f["head"] == "Run gamma drills"]
    assert len(cand) == 1 and sorted(cand[0]["members"]) == ["p1", "p2"] and not cand[0].get("written_for") and cand[0]["sid"]
    assert _accounted(art)


def test_cluster_over_pile_applies_the_floor(agg, monkeypatch):
    ctx = _ctx(agg, {**_CORPUS, "p1": {"w5"}, "p2": {"w5"}})
    monkeypatch.setattr(cluster, "group", lambda ctx, pile: [{"head": "Run gamma drills", "members": ["p1", "p2"]}])
    art = cluster.propose(ctx, _art8())
    assert len(art["core"]) == 2 and {"p1", "p2"} <= set(art["released"])


def test_cluster_over_an_empty_core_is_a_fresh_build(agg, monkeypatch):
    ctx = _ctx(agg, {"p1": {"w5"}, "p2": {"w6"}, "p3": {"w7"}})
    monkeypatch.setattr(cluster, "group", lambda ctx, pile: [{"head": "Run gamma drills", "members": ["p1", "p2"]}])
    art = cluster.propose(ctx, {"meta": {"n_atoms": 3}, "core": [], "released": ["p1", "p2", "p3"]})
    assert [f["head"] for f in art["core"]] == ["Run gamma drills"] and art["released"] == ["p3"]


def test_seed_holds_every_atom_in_the_pile(ctx):
    art = cluster.seed(ctx)
    assert art["core"] == [] and set(art["released"]) == set(ctx.atoms.tasks) and art["meta"]["n_atoms"] == 3
    assert art["meta"]["title"] == "Keytest Occupation" and art["meta"]["code"] == "99-9999.00"


def test_group_chunked_unions_across_chunks(agg, monkeypatch):
    ctx = _ctx(agg, {k: {"w"} for k in ["a1", "a2", "b1", "a3", "b2", "c1"]})
    ctx.config["blocks"]["cluster_chunk"] = 3
    monkeypatch.setattr(blocks, "kmeans_chunks", lambda ctx, labels, chunk, seed=0: [labels[:3], labels[3:]])
    calls = []

    def fake_group(ctx, labels):
        calls.append(sorted(labels))
        labels = sorted(labels)
        if labels == ["a1", "a2", "b1"]:
            return [{"head": "wash dishes", "members": ["a1", "a2"]}, {"head": "mop floors", "members": ["b1"]}]
        if labels == ["a3", "b2", "c1"]:
            return [{"head": "clean dishware", "members": ["a3"]}, {"head": "mop the floor", "members": ["b2"]}]
        return [{"head": "Wash dishes", "members": ["clean dishware", "wash dishes"]},
                {"head": "Mop floors", "members": ["mop floors", "mop the floor"]}]
    monkeypatch.setattr(cluster, "group", fake_group)
    got = cluster.group_chunked(ctx, ["a1", "a2", "b1", "a3", "b2", "c1"])
    assert len(calls) == 3
    assert {g["head"]: g["members"] for g in got} == {"Wash dishes": ["a1", "a2", "a3"], "Mop floors": ["b1", "b2"]}
    ctx.config["blocks"]["cluster_chunk"] = 0
    assert cluster.group_chunked(ctx, ["a1", "a2"]) == fake_group(ctx, ["a1", "a2"])


# ── consensus over repeated draws ─────────────────────────────────────────────────────────────────

def _fam(head, *members):
    return {"head": head, "members": sorted(members)}


def test_consensus_keeps_what_a_majority_of_draws_agree_on():
    labels = ["a", "b", "c", "d", "e"]
    draws = [[_fam("Wash", "a", "b", "c"), _fam("Mop", "d", "e")],
             [_fam("Wash up", "a", "b"), _fam("Mop floors", "c", "d", "e")],
             [_fam("Wash dishes", "a", "b", "c"), _fam("Mop the floor", "d", "e")]]
    got = cluster._consensus(draws, labels)
    assert got == [_fam("Wash", "a", "b", "c"), _fam("Mop", "d", "e")]   # c went with a, b in two draws of three


def test_consensus_leaves_a_minority_filing_unfiled():
    draws = [[_fam("Wash", "a", "b", "x")], [_fam("Wash", "a", "b")], [_fam("Wash", "a", "b")]]
    assert cluster._consensus(draws, ["a", "b", "x"]) == [_fam("Wash", "a", "b")]
    assert cluster._consensus([[_fam("Wash", "a", "b")], [], []], ["a", "b"]) == []


def test_consensus_of_two_draws_is_what_both_agree_on():
    draws = [[_fam("Wash", "a", "b", "c")], [_fam("Wash up", "a", "b"), _fam("Dry", "c", "d")]]
    # c was filed in both draws but never with the same company: a family of its own, which propose drops
    assert cluster._consensus(draws, ["a", "b", "c", "d"]) == [_fam("Wash up", "a", "b"), _fam("Dry", "c")]


def test_consensus_of_identical_draws_is_the_draw():
    one = [_fam("Wash", "a", "b"), _fam("Mop", "c"), _fam("Dry", "d", "e")]
    assert cluster._consensus([one, one, one], list("abcde")) == sorted(one, key=lambda g: g["members"])


def test_consensus_takes_the_best_overlapping_head_and_uses_it_once():
    draws = [[_fam("Clean", "a", "b", "c", "d")],
             [_fam("Wash", "a", "b"), _fam("Mop", "c", "d")],
             [_fam("Wash dishes", "a", "b"), _fam("Mop floors", "c", "d")]]
    assert cluster._consensus(draws, list("abcd")) == [_fam("Wash", "a", "b"), _fam("Mop", "c", "d")]
    same = [[_fam("Clean", "a", "b"), _fam("Tidy", "c", "d")], [_fam("Tidy", "a", "b"), _fam("Clean", "c", "d")],
            [_fam("Clean", "a", "b"), _fam("Tidy", "c", "d")]]
    assert [g["head"] for g in cluster._consensus(same, list("abcd"))] == ["Clean", "Tidy"]


def test_group_takes_one_draw_by_default_and_the_consensus_of_several(ctx, monkeypatch):
    seen = []
    by_draw = {1: [_fam("Wash", "a", "b", "c")], 2: [_fam("Wash up", "a", "b")], 3: [_fam("Rinse", "a", "b", "c")]}

    def fake_draw(ctx, labels, d=1):
        seen.append((list(labels), d))
        return by_draw[d]
    monkeypatch.setattr(cluster, "_draw", fake_draw)
    assert cluster.group(ctx, ["c", "a", "b"]) is by_draw[1] and seen == [(["a", "b", "c"], 1)]
    seen.clear()
    assert cluster.group(ctx, ["c", "a", "b"], draws=3) == [_fam("Wash", "a", "b", "c")]
    assert sorted(d for _, d in seen) == [1, 2, 3]
    seen.clear()
    ctx.config["models"]["cluster"]["draws"] = 2                        # the config key alone
    assert cluster.group(ctx, ["c", "a", "b"]) == [_fam("Wash up", "a", "b")]
    assert sorted(d for _, d in seen) == [1, 2]


def test_each_draw_has_its_own_cache_slot_and_the_first_keeps_its_key(ctx, monkeypatch):
    keys = []
    monkeypatch.setattr(ctx.cache, "memo", lambda section, key, compute, **k: keys.append(key) or [])
    labels = [f"task {i}" for i in range(10)]
    cluster.group(ctx, labels)
    first = keys[-1]
    keys.clear()
    cluster.group(ctx, labels, draws=3)
    assert len(set(keys)) == 3 and first in keys


def test_a_repeated_draw_offsets_the_seed(ctx, monkeypatch):
    seeds = []

    class _R:
        class _C:
            finish_reason = "stop"

            class message:
                content = "ok"
        choices = [_C]
    monkeypatch.setattr(ctx.engine, "create", lambda model, messages, **kw: seeds.append(kw["seed"]) or _R)
    cluster.chat(ctx, "p", 100, "t")
    cluster.chat(ctx, "p", 100, "t", draw=3)
    assert seeds == [0, 2]


def test_draws_reach_group_from_propose_and_from_the_aggregator(agg, monkeypatch):
    ctx = _ctx(agg, {"p1": {"w5"}, "p2": {"w6"}, "p3": {"w7"}})
    got = []

    def fake_group(ctx, pile, draws=None):
        got.append(draws)
        return [{"head": "Run gamma drills", "members": ["p1", "p2"]}]
    monkeypatch.setattr(cluster, "group", fake_group)
    art = cluster.propose(ctx, {"meta": {"n_atoms": 3}, "core": [], "released": ["p1", "p2", "p3"]}, draws=3)
    assert got == [3] and art["draw_pile_log"]["draws"] == 3
    art = cluster.propose(ctx, {"meta": {"n_atoms": 3}, "core": [], "released": ["p1", "p2", "p3"]})
    assert "draws" not in art["draw_pile_log"]

    class Reached(Exception):
        pass

    def fake_propose(ctx, build, draws=None):
        got.append(("propose", draws))
        raise Reached
    monkeypatch.setattr(cluster, "propose", fake_propose)
    try:
        agg.build(atoms_from_sources({"p1": {"w5"}, "p2": {"w6"}}), cluster_draws=5)
    except Reached:
        pass
    assert got[-1] == ("propose", 5)
