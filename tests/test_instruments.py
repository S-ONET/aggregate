"""instruments: several kinds of data in one build; the scope judge decides which instruments' floors apply."""
import json

import pytest

from aggregate import Aggregator, Atoms, Config
from aggregate.aggregation import cluster, fusion, refile, verify
from aggregate.aggregation.floor import Floor
from conftest import FakeClient, atoms_from_sources

INSTRUMENTS = {"interview": {"scope": "What a worker says they do.", "min_share": 0.5},
               "posting": {"scope": "What an employer hires for.", "min_sources": 2}}


def _rows(inst_sources):
    """{atom: {source}} -> rows (no instrument column)."""
    return atoms_from_sources(inst_sources).rows


def _corpus():
    # interviews i1..i4 (share floor 0.5 -> 2), postings p1..p3 (count floor 2)
    return Atoms.combine({
        "interview": _rows({"Draft motions.": {"i1", "i2"}, "Advise clients.": {"i3"}, "Chat at lunch.": {"i4"}}),
        "posting": _rows({"Draft motions.": {"p1"}, "Hold a bar license.": {"p1", "p2"}, "Advise clients.": {"p3"}}),
    })


def _agg(tmp_path, **cfg):
    return Aggregator(Config.load(prewarm={"on": False}, instruments=INSTRUMENTS, min_sources=1, **cfg),
                      client=FakeClient(), workdir=tmp_path / "work", cache_dir=tmp_path / "cache")


def _scoped(ctx, monkeypatch, table):
    """Stub the scope judge: table = {statement: {instrument: bool}} (missing = in scope)."""
    asked = []

    def fake(statement, instrument, description, only_cached=False):
        asked.append((statement, instrument))
        return table.get(statement, {}).get(instrument, True)
    monkeypatch.setattr(ctx.judges, "in_scope", fake)
    monkeypatch.setattr(ctx.judges, "in_scope_many", lambda items: 0)
    return asked


# ── atoms ──────────────────────────────────────────────────────────────────────────────────────────

def test_combine_namespaces_sources_and_merges_shared_texts():
    a = _corpus()
    assert a.sources["Draft motions."] == {"interview:i1", "interview:i2", "posting:p1"}
    assert a.instrument_of["posting:p2"] == "posting"
    assert a.instrument_sizes == {"interview": 4, "posting": 3}
    assert {r["instrument"] for r in a.rows} == {"interview", "posting"}


def test_coerce_takes_a_dict_and_rejects_ambiguous_sources():
    assert len(Atoms.coerce({"interview": _rows({"x": {"s1"}})})) == 1
    rows = _rows({"x": {"s1"}})
    bad = Atoms([{**rows[0], "instrument": "interview"}, {**rows[0], "task": "y", "instrument": "posting"}])
    with pytest.raises(ValueError, match="appears under instruments"):
        bad.instrument_of
    with pytest.raises(ValueError, match="no ':'"):
        Atoms.combine({"a:b": rows})


def test_single_instrument_atoms_have_no_instruments(atoms):
    assert atoms.instrument_of == {} and atoms.instrument_sizes == {}


# ── the floor ──────────────────────────────────────────────────────────────────────────────────────

def test_single_instrument_floor_is_the_plain_count_and_calls_no_judge(ctx, no_live_calls):
    fl = ctx.floor
    assert not fl.multi and fl.clears("x", {"s1", "s2"}) and not fl.clears("x", {"s1"})


def test_floor_per_instrument_applies_only_where_in_scope(tmp_path, monkeypatch):
    ctx = _agg(tmp_path).context(_corpus())
    fl = ctx.floor
    assert fl.need("interview") == 2 and fl.need("posting") == 2
    W = ctx.atoms.sources
    _scoped(ctx, monkeypatch, {"Hold a bar license.": {"interview": False},
                               "Draft motions.": {"posting": True},
                               "Chat at lunch.": {"interview": False, "posting": False}})
    # 2 interviews but only 1 posting, and postings are in scope: fails the posting floor
    assert not fl.clears("Draft motions.", W["Draft motions."])
    # posting-only task: out of scope for interviews, 2 postings clears
    assert fl.clears("Hold a bar license.", W["Hold a bar license."])
    # in scope for nothing: never publishes
    assert fl.shortfall({"interview:i1", "interview:i2"}, {"interview": False, "posting": False}) == ["in scope for no instrument"]


def test_floor_out_of_scope_instrument_asks_nothing(tmp_path, monkeypatch):
    ctx = _agg(tmp_path).context(_corpus())
    _scoped(ctx, monkeypatch, {"Draft motions.": {"posting": False}})
    assert ctx.floor.clears("Draft motions.", ctx.atoms.sources["Draft motions."])


def test_global_min_sources_still_applies(tmp_path, monkeypatch):
    ctx = _agg(tmp_path).context(_corpus())
    ctx.config["min_sources"] = 5
    ctx._floor = None
    asked = _scoped(ctx, monkeypatch, {})
    assert not ctx.floor.clears("Draft motions.", ctx.atoms.sources["Draft motions."]) and asked == []


def test_config_and_corpus_must_agree(tmp_path, atoms):
    with pytest.raises(ValueError, match="no `instrument` column"):
        _agg(tmp_path).context(atoms).floor.sizes
    only = Atoms.combine({"survey": _rows({"x": {"s1"}})})
    with pytest.raises(ValueError, match="does not describe"):
        _agg(tmp_path).context(only).floor.sizes
    with pytest.raises(ValueError, match="scope is empty"):
        Aggregator(Config.load(instruments={"interview": {"min_share": 0.1}}), client=FakeClient(),
                   cache_dir=tmp_path / "c").context(only).floor


# ── the judge ──────────────────────────────────────────────────────────────────────────────────────

def test_scope_judge_parses_caches_and_keys_on_the_description(tmp_path, monkeypatch):
    ctx = _agg(tmp_path).context(_corpus())
    calls = []

    def chat(prompt, model, **kw):
        calls.append(prompt)
        return json.dumps({"reason": "r", "in_scope": "posting" in prompt.split("THE INSTRUMENT:")[1][:20]})
    monkeypatch.setattr(ctx.engine, "chat", chat)
    j = ctx.judges
    assert j.in_scope("Draft motions.", "posting", "employers") is True
    assert j.in_scope("Draft motions.", "posting", "employers") is True and len(calls) == 1
    assert j.in_scope("Draft motions.", "interview", "workers") is False and len(calls) == 2
    j.in_scope("Draft motions.", "posting", "employers, revised")
    assert len(calls) == 3
    assert j.in_scope("New text.", "posting", "employers", only_cached=True) is None and len(calls) == 3
    assert "What an employer" not in calls[0] and "employers" in calls[0] and ctx.occupation in calls[0]


def test_scope_judge_refuses_to_cache_garbage(tmp_path, monkeypatch):
    ctx = _agg(tmp_path).context(_corpus())
    monkeypatch.setattr(ctx.engine, "chat", lambda *a, **k: "not json")
    with pytest.raises(RuntimeError, match="unusable"):
        ctx.judges.in_scope("x", "posting", "employers")


# ── the stages ─────────────────────────────────────────────────────────────────────────────────────

def test_admit_uses_the_floor_callable():
    groups = [{"head": "A", "members": ["a"]}, {"head": "B", "members": ["b"]}]
    kept, thin = cluster.admit(groups, {"a": {"s1"}, "b": {"s2", "s3"}}, 2, floor=lambda head, w: head == "A")
    assert [k["head"] for k in kept] == ["A"] and [t["head"] for t in thin] == ["B"]


def test_fusion_decide_holds_parts_to_the_instrument_floor(tmp_path, monkeypatch):
    ctx = _agg(tmp_path).context(_corpus())
    _scoped(ctx, monkeypatch, {"P-int": {"posting": False}, "P-post": {"interview": False}})
    W = ctx.atoms.sources
    row = {"members": ["Draft motions.", "Hold a bar license."], "n_members": 2, "prevalence": 5, "reason": "",
           "outcomes": [{"statement": "P-int", "members": ["Draft motions."]},
                        {"statement": "P-post", "members": ["Hold a bar license."]}]}
    v, parts, basis, _ = fusion.decide(row, W, 1, ctx.floor)
    pub = {p["statement"]: p["publishable"] for p in parts}
    assert pub == {"P-int": False, "P-post": False}       # one-member parts never publish
    assert v == "split" and basis == "no-supported-part"


def test_refile_demotes_by_the_floor_and_stamps_scope(tmp_path, monkeypatch):
    ctx = _agg(tmp_path).context(_corpus())
    _scoped(ctx, monkeypatch, {"Draft motions.": {"posting": True}, "Hold a bar license.": {"interview": False}})
    monkeypatch.setattr(refile, "excluded", lambda ctx, core, unfiled=(): ([], {}))
    monkeypatch.setattr(ctx.judges, "dispose", lambda *a, **k: {"fate": "LEAVE", "task": None})
    art = {"meta": {"n_atoms": 4},
           "core": [{"sid": "s_a", "statement": "Draft motions.", "members": ["Draft motions."]},
                    {"sid": "s_b", "statement": "Hold a bar license.", "members": ["Hold a bar license."]}],
           "released": ["Advise clients.", "Chat at lunch."]}
    art, _ = refile.place(ctx, art)
    assert [f["statement"] for f in art["core"]] == ["Hold a bar license."]
    f = art["core"][0]
    assert f["scope"] == {"interview": False, "posting": True} and f["by_instrument"] == {"interview": 0, "posting": 2}
    assert "Draft motions." in art["released"] and verify.invariants(ctx, art) == []


def test_verify_reads_the_stamp_and_catches_a_stale_one(tmp_path, monkeypatch):
    ctx = _agg(tmp_path).context(_corpus())
    _scoped(ctx, monkeypatch, {"Hold a bar license.": {"interview": False}})
    art = {"meta": {"n_atoms": 4}, "core": [{"sid": "s_1", "statement": "Hold a bar license.", "members": ["Hold a bar license."]}],
           "released": ["Advise clients.", "Chat at lunch.", "Draft motions."]}
    assert any(p.startswith("scope:") for p in verify.invariants(ctx, art))
    ctx.floor.stamp(art["core"])
    assert verify.invariants(ctx, art) == []
    art["core"][0]["statement"] = "Hold a license to practice law."
    assert any(p.startswith("scope:") for p in verify.invariants(ctx, art))
    art["core"][0]["statement"] = "Hold a bar license."
    art["core"][0]["scope"]["interview"] = True             # now held to the interviews' floor of 2: has 0
    assert any("in scope for interview" in p for p in verify.invariants(ctx, art))


def test_seed_records_the_instruments(tmp_path):
    ctx = _agg(tmp_path).context(_corpus())
    assert cluster.seed(ctx)["meta"]["instruments"] == {"interview": {"sources": 4, "floor": 2},
                                                       "posting": {"sources": 3, "floor": 2}}


def test_build_accepts_a_dict(tmp_path, monkeypatch):
    agg = _agg(tmp_path)
    seen = {}
    from aggregate.aggregation import chain
    monkeypatch.setattr(chain, "build", lambda ctx, **kw: seen.setdefault("atoms", ctx.atoms))
    agg.build({"interview": _rows({"x": {"s1"}}), "posting": _rows({"x": {"e1"}})})
    assert seen["atoms"].sources["x"] == {"interview:s1", "posting:e1"}


def test_fusion_scope_decides_whether_a_part_publishes(tmp_path, monkeypatch):
    corpus = Atoms.combine({"interview": _rows({"a1": {"i1"}, "a2": {"i2"}}), "posting": _rows({"b1": {"p1"}, "b2": {"p2"}})})
    ctx = _agg(tmp_path).context(corpus)
    row = {"members": ["a1", "a2", "b1", "b2"], "n_members": 4, "prevalence": 4, "reason": "",
           "outcomes": [{"statement": "P-int", "members": ["a1", "a2"]}, {"statement": "P-post", "members": ["b1", "b2"]}]}
    _scoped(ctx, monkeypatch, {"P-int": {"posting": False}, "P-post": {"interview": False}})
    v, parts, basis, _ = fusion.decide(row, ctx.atoms.sources, 1, ctx.floor)
    assert (v, basis) == ("split", "exclusive-sources") and all(p["publishable"] for p in parts)
    _scoped(ctx, monkeypatch, {"P-int": {"posting": False}})   # P-post now also owes the interviews a source
    v, parts, basis, fb = fusion.decide(row, ctx.atoms.sources, 1, ctx.floor)
    assert basis == "one-supported-part" and {p["statement"]: p["publishable"] for p in parts} == {"P-int": True, "P-post": False}
    assert "interview sources when in scope for interview" in fb
