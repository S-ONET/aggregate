"""Fixes found building task lists from job postings (ported from the TaskNet System's in-tree fork): the
resolver's reply is salvaged rather than thrown away, colliding groups are capped, the fusion hearing runs again
after overlap, a failed one_duty verdict is never ledgered, output caps stay under the model's limit, a text shared
by two families never loses one of them, and a rewrite never takes another family's statement."""
import json
from types import SimpleNamespace

from aggregate.aggregation import chain, cluster, fusion, overlap, synthesize
from conftest import atoms_from_sources, orth_embed
from test_chain import W, _stub

A, B, C = "Draft trial motions.", "Draft appellate briefs.", "File pleadings with the clerk."


# ── merge_or_separate: a flawed reply is salvaged ─────────────────────────────────────────────────

def _resolve(agg, monkeypatch, groups):
    ctx = agg.context(atoms_from_sources({"a1": {"w1"}, "b1": {"w2"}, "c1": {"w3"}}))
    monkeypatch.setattr(ctx.cache, "memo", lambda section, key, compute, **k: {"groups": groups})
    return ctx.judges.merge_or_separate([A, B, C], [(A, B), (B, C)], {A: ["a1"], B: ["b1"], C: ["c1"]})


def test_a_statement_listed_twice_keeps_its_first_decision(agg, monkeypatch):
    out = _resolve(agg, monkeypatch, [{"ids": [1], "decision": "SEPARATE"}, {"ids": [1, 2], "decision": "KEEP"},
                                      {"ids": [3], "decision": "KEEP"}])
    assert [(g["members"], g["decision"]) for g in out] == [([A], "SEPARATE"), ([B], "KEEP"), ([C], "KEEP")]


def test_a_statement_left_out_is_kept(agg, monkeypatch):
    out = _resolve(agg, monkeypatch, [{"ids": [1, 2], "decision": "MERGE", "relation": "EQUIVALENT", "proposed": "P"}])
    assert [(g["members"], g["decision"]) for g in out] == [([A, B], "MERGE"), ([C], "KEEP")]
    assert "left out" in out[1]["basis"]


def test_an_unknown_number_still_rejects_the_reply(agg, monkeypatch):
    assert _resolve(agg, monkeypatch, [{"ids": [9], "decision": "KEEP"}]) is None


# ── colliding groups are capped, closest pairs first ──────────────────────────────────────────────

def test_capped_components_join_closest_first():
    chain_pairs = [("a", "b", 0.9), ("b", "c", 0.8), ("c", "d", 0.7), ("d", "e", 0.6)]
    groups, dropped = overlap.capped_components(chain_pairs, 3)
    assert groups == [["a", "b", "c"], ["d", "e"]] and dropped == 1
    assert overlap.capped_components(chain_pairs, 10) == (overlap.components([(a, b) for a, b, _ in chain_pairs]), 0)
    assert overlap.capped_components(chain_pairs, 0)[0] == [["a", "b", "c", "d", "e"]]


def test_route_hears_each_capped_group_on_its_own(ctx, monkeypatch):
    ctx.config["chain"]["overlap_group_cap"] = 2
    heard = []
    monkeypatch.setattr(overlap, "resolve_component", lambda ctx, nodes, edges, m_of: heard.append(sorted(nodes)) or
                        [{"members": [n], "decision": "KEEP"} for n in nodes])
    rows = [{"a": "a", "b": "b", "cos": 0.9, "labels_collide": True, "why": ""},
            {"a": "b", "b": "c", "cos": 0.5, "labels_collide": True, "why": ""},
            {"a": "c", "b": "d", "cos": 0.8, "labels_collide": True, "why": ""}]
    overlap.route(ctx, rows)
    assert sorted(heard) == [["a", "b"], ["c", "d"]]


# ── one_duty ──────────────────────────────────────────────────────────────────────────────────────

def test_one_duty_output_cap_grows_with_the_family(ctx):
    assert ctx.judges.duties_cap(["m"] * 10) == 2000 + 30 * 10
    assert ctx.judges.duties_cap(["m"] * 200) > 2000 + 20 * 200       # room for every member under its part


def test_a_failed_verdict_is_kept_but_never_ledgered(agg, monkeypatch):
    ctx = agg.context(atoms_from_sources({"a1": {"w1"}, "a2": {"w2"}, "b1": {"w3"}, "b2": {"w4"}}))
    monkeypatch.setattr(ctx.judges, "duties_many", lambda rows: 0)
    monkeypatch.setattr(ctx.judges, "duties", lambda s, m, draw=1: (
        {"one_duty": True, "outcomes": [], "reason": "", "error": "truncated"} if s == A else
        {"one_duty": True, "outcomes": [], "reason": "fine"}))
    build = {"core": [{"statement": A, "members": ["a1", "a2"]}, {"statement": B, "members": ["b1", "b2"]}]}
    rows = fusion.hear(ctx, build)
    assert {r["statement"]: r["verdict"] for r in rows} == {A: "keep", B: "keep"}
    led = json.loads(ctx.ledger("defuse_seen.json").read_text())
    assert [v["statement"] for v in led.values()] == [B]


# ── the fusion hearing again after overlap ────────────────────────────────────────────────────────

def test_merged_families_are_heard_again_and_judged_ones_replay(agg, monkeypatch, tmp_path):
    ctx = agg.context(atoms_from_sources(W), workdir=tmp_path / "w")
    _stub(ctx, monkeypatch)
    heard = []
    monkeypatch.setattr(ctx.judges, "duties", lambda s, m, draw=1: heard.append(tuple(sorted(m))) or
                        {"one_duty": True, "outcomes": [], "reason": ""})
    real = overlap.resolve

    def merge_all(ctx, build, standing=None, tau=None):
        out, rep = real(ctx, build, standing, tau)
        mem = sorted(m for f in out["core"] for m in f["members"])
        out["core"] = [{"sid": "s_merged", "statement": "Do alpha and beta work.", "head": "Do alpha and beta work.",
                        "members": mem}]
        return out, rep
    monkeypatch.setattr(overlap, "resolve", merge_all)
    chain.build(ctx, max_cycles=1)
    assert ("a1", "a2", "b1", "b2") in heard                             # the merged family was heard
    assert heard.count(("a1", "a2")) == 1 and heard.count(("b1", "b2")) == 1
    assert (tmp_path / "w" / "verdicts" / "cycle1_fused_merged.json").exists()


def test_the_recheck_can_be_turned_off(agg, monkeypatch, tmp_path):
    ctx = agg.context(atoms_from_sources(W), workdir=tmp_path / "w")
    ctx.config["chain"]["recheck_after_overlap"] = False
    _stub(ctx, monkeypatch)
    chain.build(ctx, max_cycles=1)
    assert not (tmp_path / "w" / "verdicts" / "cycle1_fused_merged.json").exists()


# ── cluster caps ──────────────────────────────────────────────────────────────────────────────────

def test_the_truncation_ladder_never_exceeds_the_model_limit():
    assert cluster._ladder(60000, "medium", 128000) == [(60000, "medium"), (128000, "medium"), (128000, "low")]
    assert cluster._ladder(20000, "low", 128000) == [(20000, "low"), (60000, "low")]
    assert cluster._ladder(200000, "low", 128000) == [(200000, "low"), (200000, "low")]   # never below the ask
    assert cluster._ladder(60000, "low", 0) == [(60000, "low"), (180000, "low")]          # 0 = no limit


def test_the_draw_cap_is_clamped_only_above_the_limit(ctx, monkeypatch):
    keys = []
    monkeypatch.setattr(ctx.cache, "memo", lambda section, key, compute, **k: keys.append(key) or [])
    cluster.group(ctx, [f"task {i}" for i in range(10)])
    small = keys[-1]
    ctx.config["models"]["cluster"]["max_out_cap"] = 0
    cluster.group(ctx, [f"task {i}" for i in range(10)])
    assert keys[-1] == small                                             # under the cap: the key is unchanged


# ── a shared text, and a rewrite into a taken text ─────────────────────────────────────────────────

def test_a_merge_takes_every_family_carrying_a_named_text(agg, monkeypatch):
    ctx = agg.context(atoms_from_sources({"a1": {"w1"}, "a2": {"w2"}, "b1": {"w3"}}))
    monkeypatch.setattr(synthesize, "statement", lambda ctx, seed, mem, siblings=(), standing=None: "Merged.")
    monkeypatch.setattr(synthesize, "covered", lambda ctx, stmt, atoms: [True] * len(atoms))
    core = [{"statement": A, "members": ["a1"]}, {"statement": A, "members": ["a2"]}, {"statement": B, "members": ["b1"]}]
    fams, log, lost = overlap.apply_merges(ctx, core, [[A, B]])
    assert [(f["statement"], f["members"]) for f in fams] == [("Merged.", ["a1", "a2", "b1"])]


def test_identical_statements_merge_without_a_judge(agg):
    ctx = agg.context(atoms_from_sources({"a1": {"w1"}, "a2": {"w2"}, "b1": {"w3"}}))
    core = [{"sid": "s1", "statement": A, "members": ["a1"], "written_for": "x"}, {"sid": "s2", "statement": B, "members": ["b1"]},
            {"sid": "s3", "statement": A, "members": ["a2"]}]
    out, dups = overlap.merge_identical(ctx, core)
    assert dups == [A] and [f["statement"] for f in out] == [A, B]
    assert out[0]["members"] == ["a1", "a2"] and out[0]["prevalence"] == 2 and "written_for" not in out[0]
    assert out[0]["sid_merged_from"] == ["s1", "s3"]
    assert overlap.merge_identical(ctx, out[1:]) == (out[1:], [])


def test_sharpen_keeps_two_families_that_share_a_text(ctx, monkeypatch):
    monkeypatch.setattr(overlap, "sharpen_component", lambda c, n, e, m, frozen=(): {A: "Rewritten A."})
    core = [{"statement": A, "members": ["a1"]}, {"statement": A, "members": ["a2"]}, {"statement": B, "members": ["b1"]}]
    sg = [{"members": [A], "each": [{"statement": A, "siblings": [B], "complaints": [{"partner": B}]}]}]
    out, log = overlap.apply_sharpens(ctx, core, sg)
    assert [(f["statement"], f["members"]) for f in out] == [("Rewritten A.", ["a1"]), ("Rewritten A.", ["a2"]), (B, ["b1"])]


def test_a_rewrite_never_takes_another_familys_statement(ctx, monkeypatch):
    core = [{"statement": A, "members": ["a1"]}, {"statement": B, "members": ["b1"]}]
    sg = [{"members": [A], "each": [{"statement": A, "siblings": [B], "complaints": [{"partner": B}]}]}]
    monkeypatch.setattr(overlap, "sharpen_component", lambda c, n, e, m, frozen=(): {A: B})
    out, log = overlap.apply_sharpens(ctx, [dict(f) for f in core], sg)
    assert [f["statement"] for f in out] == [A, B] and log == []
    ctx.sharpen_alternatives[A] = [B, "Draft trial motions for the court."]
    out, log = overlap.apply_sharpens(ctx, [dict(f) for f in core], sg)
    assert [f["statement"] for f in out] == ["Draft trial motions for the court.", B]


# ── the per-call usage log ────────────────────────────────────────────────────────────────────────

def test_usage_log_writes_one_line_per_call(ctx, tmp_path):
    log = tmp_path / "usage.jsonl"
    ctx.engine.config["usage_log"] = str(log)
    u = SimpleNamespace(prompt_tokens=100, completion_tokens=7, prompt_tokens_details=SimpleNamespace(cached_tokens=40))
    ctx.engine.record_usage("gpt-5.4-mini", SimpleNamespace(usage=u))
    ctx.engine.record_usage("gpt-5.4-mini", SimpleNamespace(usage=u))
    lines = [json.loads(x) for x in log.read_text().splitlines()]
    assert len(lines) == 2 and lines[0]["in"] == 100 and lines[0]["out"] == 7 and lines[0]["cached"] == 40


def test_usage_log_is_off_by_default(ctx, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    u = SimpleNamespace(prompt_tokens=1, completion_tokens=1, prompt_tokens_details=None)
    ctx.engine.record_usage("gpt-5.4-mini", SimpleNamespace(usage=u))
    assert list(tmp_path.iterdir()) == []
