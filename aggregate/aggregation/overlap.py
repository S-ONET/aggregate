"""overlap — the merge / sharpen hearing over pairs of published statements.

    build, report = overlap.resolve(ctx, build, standing=None)

Pairs above the embedding prefilter (chain.overlap_tau) go to the overlaps judge (does an act appear on BOTH
statements). That tier ROUTES — `yes` edges form components heard by merge_or_separate (MERGE with a basis /
SEPARATE / KEEP), which answers the merge question on the MEMBERS. A MERGE then passes strain evidence
(ledgers/sharpen_seen: reworded N× and still colliding) and merge concurrence (a second resolver draw must
merge the same set; chain.concurrence.merge, on by default). `apply_merges` writes the union's statement with
the writer from the resolver's seed. SEPARATE verdicts go to sharpen: fresh statements take the rewrite;
standing statements are never machine-rewritten — the proposal is docketed with a concurrence triage flag.
"""
from __future__ import annotations

import collections
import itertools
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor

from .. import families
from ..lib import load_json, sha256
from . import blocks, synthesize

_STRAIN_STOP = {"with", "and", "for", "the", "from", "that", "their", "client", "clients", "case", "cases",
                "matter", "matters", "active", "legal", "work", "during", "other", "such", "including"}


def near_pairs(ctx, stmts, tau=None, funcs=None):
    """Index pairs above the cosine prefilter -> [(i, j, cos)]; `funcs` keeps a pair inside one block."""
    import numpy as np
    tau = float(ctx.config.get("chain.overlap_tau", 0.44)) if tau is None else tau
    V = ctx.judges.embed(stmts)
    E = np.array([V[s] for s in stmts])
    C = E @ E.T
    return [(i, j, float(C[i, j])) for i, j in itertools.combinations(range(len(stmts)), 2)
            if C[i, j] >= tau and (not funcs or blocks.same_block(ctx, funcs[i], funcs[j]))]


def components(pairs):
    """Connected components over [(a, b)] by union-find -> [[member, ...]]."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in pairs:
        parent[find(a)] = find(b)
    out = collections.defaultdict(list)
    for x in list(parent):
        out[find(x)].append(x)
    return [sorted(v) for v in out.values()]


def capped_components(pairs, cap):
    """Components over [(a, b, cos)] of at most `cap` items: pairs join closest-first, and a pair that would make a
    group bigger than `cap` is left out. -> ([[member, ...]], pairs left out). cap <= 0, or every component at
    or under the cap, gives exactly `components`. One resolver call over a 49-statement postings component (chained
    through one umbrella statement) listed a statement twice and proposed SEPARATE for 45 of them."""
    if cap <= 0:
        return components([(a, b) for a, b, _ in pairs]), 0
    parent, size, dropped = {}, {}, 0

    def find(x):
        parent.setdefault(x, x)
        size.setdefault(x, 1)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b, _ in sorted(pairs, key=lambda t: (-t[2], t[0], t[1])):
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        if size[ra] + size[rb] > cap:
            dropped += 1
            continue
        parent[ra] = rb
        size[rb] += size[ra]
    out = collections.defaultdict(list)
    for x in list(parent):
        out[find(x)].append(x)
    return [sorted(v) for v in out.values() if len(v) > 1], dropped


def merge_identical(ctx, core):
    """Families whose statements are word-for-word identical become one: the same text claims the same work, so no
    judge is needed. The merged family takes the first one's place, the union of the members and a new sid; its
    writer stamps go, so synthesize checks the text against the new members. -> (core, [merged texts])"""
    by = collections.defaultdict(list)
    for c in core:
        by[c["statement"]].append(c)
    dups = [t for t, cs in by.items() if len(cs) > 1]
    if not dups:
        return core, []
    out, done = [], set()
    for c in core:
        t = c["statement"]
        if t not in dups:
            out.append(c)
        elif t not in done:
            cs = by[t]
            keep = {k: v for k, v in cs[0].items() if k not in ("written_for", "written_with", "form_flag")}
            out.append({**keep, "sid": families.new_sid(), "sid_merged_from": [f.get("sid") for f in cs], "head": t,
                        "statement": t, "members": sorted({m for f in cs for m in (f.get("members") or [])}),
                        "merged_from": [t] * len(cs)})
            done.add(t)
    families.recount(ctx, out)
    for t in dups:
        print(f"  MERGED identical statements: {t[:90]}", file=sys.stderr)
    return out, dups


def judge_pairs(ctx, core, tau=None):
    """Run the overlaps judge over every prefiltered pair -> [{a, b, cos, label, labels_collide, why}]."""
    stmts = [c["statement"] for c in core]
    funcs = [blocks.function_of(ctx, c.get("members") or []) for c in core] if blocks.block_by(ctx) else None
    pairs = near_pairs(ctx, stmts, tau, funcs)
    print(f"  {len(stmts)} statements · {len(pairs)} pairs at cosine >= "
          f"{float(ctx.config.get('chain.overlap_tau', 0.44)) if tau is None else tau}", file=sys.stderr)

    def one(t):
        i, j, cos = t
        a, b = stmts[i], stmts[j]
        collide = ctx.judges.overlaps(a, b)
        return {"a": a, "b": b, "cos": round(cos, 3), "label": "yes" if collide else "no",
                "labels_collide": bool(collide), "why": f"an act appears in both labels at cosine {cos:.2f}" if collide else ""}
    ctx.judges.overlaps_many([(stmts[i], stmts[j]) for i, j, _ in pairs])
    with ThreadPoolExecutor(ctx.workers_for("overlap")) as ex:
        return list(ex.map(one, pairs))


# ── the strain ledger ──────────────────────────────────────────────────────────────────────────────

def _pairkey(members_a, members_b):
    return sha256("|".join(sorted((families.key(members_a), families.key(members_b)))))


def _ledger(ctx):
    return load_json(ctx.ledger("sharpen_seen.json"), {}) or {}


def record_sharpen_edges(ctx, edges, members_of):
    """Record that sharpen APPLIED a rewrite on each of these pairs."""
    p = ctx.ledger("sharpen_seen.json")
    led = _ledger(ctx)
    for a, b in edges:
        k = _pairkey(members_of[a], members_of[b])
        rec = led.get(k) or {"count": 0, "labels": []}
        rec["count"] += 1
        rec["labels"] = [a, b]
        led[k] = rec
    p.write_text(json.dumps(led, indent=1))


def _content_words(text):
    return {w for w in re.sub(r"[^a-z ]", " ", text.lower()).split() if len(w) > 4 and w not in _STRAIN_STOP}


def strain_count(ctx, members_a, members_b, label_a=None, label_b=None):
    """Ledgered rewording failures for this member-pair, discounted by one after a material rewrite."""
    rec = _ledger(ctx).get(_pairkey(members_a, members_b))
    if not rec:
        return 0
    n = rec.get("count", 0)
    if n and label_a and label_b and rec.get("labels"):
        def sim(x, y):
            if x == y:
                return 1.0
            wx, wy = _content_words(x), _content_words(y)
            if not wx or not wy:
                return 1.0
            return len(wx & wy) / min(len(wx), len(wy))
        la, lb = rec["labels"][0], rec["labels"][-1]
        best = max(min(sim(label_a, la), sim(label_b, lb)), min(sim(label_a, lb), sim(label_b, la)))
        if best < 0.5:
            n -= 1
    return n


def strain_evidence(ctx, edges, members_of, idx):
    """-> (evidence lines for the resolver's worker block, cache-key tokens)."""
    lines, tokens = [], []
    for a, b in edges:
        n = strain_count(ctx, members_of[a], members_of[b], a, b)
        if not n:
            continue
        lines.append(f"{idx[a]}-{idx[b]} STRAIN: this pair was already reworded to separate "
                     f"{n} time(s) and still reads as one duty — the occupation's vocabulary has so "
                     f"far refused to separate them. Weigh this, in proportion to the count, as "
                     f"evidence the two are ONE entry at the published level, unless a hard "
                     f"structural divider holds.")
        tokens.append(f"STRAIN:{idx[a]}-{idx[b]}:{n}")
    return lines, tokens


# ── resolution ─────────────────────────────────────────────────────────────────────────────────────

def merge_concurrence(groups, second):
    """A MERGE applies only when an independent second resolver draw merges the same member set."""
    if not any(g["decision"] == "MERGE" for g in groups):
        return groups
    merged2 = {frozenset(g2["members"]) for g2 in (second() or []) if g2.get("decision") == "MERGE"}
    for g in groups:
        if g["decision"] != "MERGE":
            continue
        if frozenset(g["members"]) in merged2:
            g["basis"] = ((g.get("basis") or "") + " [+concurrence: an independent second draw merges the same group]").strip()
            continue
        g["decision"], g["relation"] = "KEEP", ""
        g["contested"] = ((g.get("contested") or "") + " no-concurrence: MERGE on one resolver draw; the second draw "
                          "did not merge this group and no confirmed same-work edge covers it").strip()
    return groups


def resolve_component(ctx, nodes, edges, members_of):
    """One component through merge_or_separate, with strain evidence and merge concurrence -> groups or None."""
    idx = {t: i for i, t in enumerate(nodes, 1)}
    strain = strain_evidence(ctx, edges, members_of, idx)
    out = ctx.judges.merge_or_separate(nodes, edges, members_of, strain=strain)
    if not out:
        return None
    if ctx.config.get("chain.concurrence.merge", True):
        out = merge_concurrence(out, lambda: ctx.judges.merge_or_separate(nodes, edges, members_of, strain=strain, draw=2) or [])
    return out


def route(ctx, rows, by_mem=None):
    """The collisions, then ONE joint decision per component -> {"merge", "sharpen", "withheld", "mixed", "partitions"}."""
    by_mem = by_mem or {}
    merge_cap = int(ctx.config.get("chain.merge_cap", 4))
    group_cap = int(ctx.config.get("chain.overlap_group_cap", 10) or 0)
    colliding = [(r["a"], r["b"]) for r in rows if r["labels_collide"]]
    why = collections.defaultdict(dict)
    for r in rows:
        if r["labels_collide"]:
            why[r["a"]][r["b"]] = r["why"]
            why[r["b"]][r["a"]] = r["why"]
    mg, mixed, parts_log, specs = [], [], [], []
    comps, dropped = capped_components([(r["a"], r["b"], float(r.get("cos") or 0)) for r in rows if r["labels_collide"]],
                                       group_cap)
    if dropped:
        print(f"  {dropped} colliding pair(s) left out to keep each group to {group_cap} statements", file=sys.stderr)
    for c in comps:
        inner = [(a, b) for (a, b) in colliding if a in c and b in c]
        part = resolve_component(ctx, c, inner, {n: blocks.cap_members(ctx, by_mem.get(n, [])) for n in c})
        if not part:
            mixed.append({"members": sorted(c), "merge_edges": [], "sharpen_edges": [list(e) for e in inner],
                          "why": "merge_or_separate returned no usable partition"})
            continue
        parts_log.append({"component": sorted(c), "edges": [list(e) for e in inner], "same_work_edges": [], "groups": part})
        sep = [g["members"][0] for g in part if g["decision"] == "SEPARATE"]
        mg += [sorted(g["members"]) for g in part if g["decision"] == "MERGE"]
        if sep:
            keep = [g["members"][0] for g in part if g["decision"] != "SEPARATE"]
            specs.append((sorted(sep), sorted(keep)))
    withheld = [g for g in mg if len(g) > merge_cap]
    mg = [g for g in mg if len(g) <= merge_cap]
    sg = [{"members": ms,
           "each": [{"statement": m, "atoms": sorted(set(by_mem.get(m, []))),
                     "siblings": [x for x in ms if x != m] + list(peers),
                     "complaints": [{"partner": p, "why": w} for p, w in sorted(why.get(m, {}).items()) if p in set(ms) | set(peers)]}
                    for m in ms]}
          for ms, peers in specs]
    return {"merge": mg, "sharpen": sg, "withheld": withheld, "mixed": mixed, "partitions": parts_log}


def apply_merges(ctx, core, mg, partitions=()):
    """Combine each merge group into one family (its statement written from the resolver's seed) -> (fams, log, lost)."""
    # two families can carry one text (a merge or rewrite can reproduce a neighbour's), so a merge takes every
    # family carrying a named text rather than keeping one and losing the others' members
    by = collections.defaultdict(list)
    for c in core:
        by[c["statement"]].append(c)
    W = ctx.atoms.sources

    def cov(stmt, atoms):
        return sum(synthesize.covered(ctx, stmt, atoms))
    merged, log, lost, fams = set(), [], [], []
    for g in mg:
        mem = sorted({m for s in g for f in by[s] for m in (f.get("members") or [])})
        before = sum(cov(s, sorted(set(f.get("members") or []))) for s in g for f in by[s])
        seed = next((x["proposed"] for pt in partitions for x in pt.get("groups", [])
                     if x.get("decision") == "MERGE" and set(x.get("members") or []) == set(g) and x.get("proposed")), g[0])
        stmt = synthesize.statement(ctx, seed, mem, [s for s in by if s not in g]) or g[0]
        after = cov(stmt, mem)
        if after < before:
            lost.append({"from": g, "statement": stmt, "cov_before": before, "cov_after": after, "n_atoms": len(mem)})
        w = sorted({x for m in mem for x in W.get(m, ())})
        fams.append({"sid": families.new_sid(), "sid_merged_from": [f.get("sid") for s in g for f in by.get(s, [])],
                     "head": stmt, "statement": stmt, "members": mem, "sources": w, "prevalence": len(w),
                     "n_atoms": len(mem), "merged_from": list(g)})
        merged.update(g)
        log.append({"into": stmt, "from": g, "n_members": len(mem), "prevalence": len(w), "cov_before": before, "cov_after": after})
    for c in core:
        if c["statement"] in merged:
            continue
        fams.append({**c, "head": c["statement"], "statement": c["statement"], "members": sorted(set(c.get("members") or [])),
                     "sources": sorted(set(c.get("sources") or [])), "prevalence": len(set(c.get("sources") or []))})
    families.recount(ctx, fams)
    return fams, log, lost


def sharpen_concurrence(ctx, stmt, new, partners):
    """A sharpen of ADJUDICATED prose counts only when the rewrite RESOLVES a collision on a second draw."""
    for p in partners:
        if ctx.judges.overlaps(stmt, p, draw=2) and not ctx.judges.overlaps(new, p):
            return p
    return None


def sharpen_component(ctx, nodes, edges, members_of, frozen=()):
    """The judge's rewrite proposals for a component, gated by the writer's form judge -> {old: new} or None."""
    h = ctx.judges.sharpen(nodes, edges, members_of, frozen)
    if not h:
        return None
    out = {}
    for node, cands in h.items():
        passers = []
        for c in (synthesize.one_line(x) for x in cands):
            if c == node or not c:
                continue
            why, _w = synthesize.rejects(ctx, c, members_of.get(node) or [])
            if not why:
                passers.append(c)
        if passers:
            out[node] = passers[0]
            ctx.sharpen_alternatives[node] = passers
    return out or None


def apply_sharpens(ctx, core, sg, standing_reg=None):
    """Reword each sharpen group's statements in ONE call per group -> (core, log). A family is standing only
    when `standing_reg` holds its member set AND its statement is still the adopted text."""
    by = {c["statement"]: c for c in core}             # members for the prompt only; the output is built from core
    out, log, rewritten = [], [], {}
    for g in sg:
        movable = [m for m in g["members"] if m in by]
        frozen = [x for x in dict.fromkeys(sib for e in g["each"] for sib in (e.get("siblings") or []))
                  if x in by and x not in movable]
        nodes = movable + frozen
        edges = sorted({tuple(sorted((e["statement"], c["partner"]))) for e in g["each"] for c in e["complaints"] if c["partner"] in nodes})
        if not movable or not edges:
            continue
        m_of = {n: blocks.cap_members(ctx, by[n].get("members") or []) for n in nodes}
        h = sharpen_component(ctx, nodes, edges, m_of, frozen=frozen) or {}
        # a rewrite may not be a statement the list already has (it would fuse two families): fall to the next
        # candidate that passed the form gate, else leave the statement as it is
        taken = set(by) | set(rewritten.values())
        h = {old: next((c for c in [new, *ctx.sharpen_alternatives.get(old, [])] if c not in taken), None)
             for old, new in h.items()}
        h = {old: new for old, new in h.items() if new}
        docket, applied = [], set()
        for old, new in h.items():
            if old not in by or old == new or old in rewritten:
                continue
            f = by[old]
            adopted = families.match(f.get("members") or [], standing_reg) if standing_reg else None
            if bool(adopted) and adopted == f["statement"]:
                partners = sorted({b if a == old else a for a, b in edges if old in (a, b)})
                concur = sharpen_concurrence(ctx, old, new, partners)
                docket.append({"kind": "sharpen", "statement": old, "proposed": new, "group": movable, "concurrence": bool(concur),
                               "why": (f"resolves the collision with: {concur[:80]}" if concur else
                                       "no-concurrence: the rewrite does not resolve a confirmed collision")})
                continue
            log.append({"before": old, "after": new, "group": movable,
                        "alternatives": [c for c in ctx.sharpen_alternatives.get(old, []) if c != new]})
            rewritten[old] = new
            applied.add(old)
        if applied:
            record_sharpen_edges(ctx, [e for e in edges if e[0] in applied or e[1] in applied], m_of)
        if docket:
            ctx.docket.extend(docket)
            for e in docket:
                print(f"  DOCKET sharpen{' [concurrence]' if e.get('concurrence') else ''} {e['statement'][:64]}\n"
                      f"            -> {e['proposed'][:64]}", file=sys.stderr)
    # from core itself, so two families sharing a text each keep their own members
    for c in core:
        new = rewritten.get(c["statement"])
        out.append(synthesize.stamp_flag(ctx, {**c, "statement": new, "sharpened_from": c["statement"]}, new) if new else c)
    return out, log


def resolve(ctx, build, standing=None, tau=None):
    """The whole hearing: judge pairs, route, apply merges, apply sharpens. -> (build, report)."""
    core, _ = merge_identical(ctx, build["core"])      # identical statements, e.g. from synthesis
    rows = judge_pairs(ctx, core, tau)
    r = route(ctx, rows, {c["statement"]: (c.get("members") or []) for c in core})
    fams, mlog, lost = apply_merges(ctx, core, r["merge"], r["partitions"])
    fams, _ = merge_identical(ctx, fams)               # a merge that wrote a neighbour's statement
    if mlog:
        print(f"  APPLIED {len(mlog)} merges: {len(core)} families -> {len(fams)}")
    newcore, slog = apply_sharpens(ctx, fams, r["sharpen"], standing)
    newcore, _ = merge_identical(ctx, newcore)         # backstop: two rewrites in one pass can still meet
    if slog:
        print(f"  SHARPENED {len(slog)} statements across {len(r['sharpen'])} group(s)")
    out = {"meta": {**build.get("meta", {}), "label_model": ctx.model("overlaps"), "resolve_model": ctx.model("merge"),
                    "sharpen_model": ctx.model("sharpen"), "n_merged": len(mlog), "n_sharpened": len(slog)},
           "core": newcore, "sharpen_log": slog, "coverage_loss": lost, **families.leftovers(build)}
    report = {"merge": r["merge"], "sharpen": r["sharpen"], "withheld": r["withheld"], "mixed": r["mixed"],
              "partitions": r["partitions"], "pairs": rows, "merges_applied": mlog, "sharpens_applied": slog}
    return out, report
