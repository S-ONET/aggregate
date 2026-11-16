"""cluster — draw the occupation's task FAMILIES over a pile of atoms, and apply the source floor.

    build = cluster.seed(ctx)                 # fresh: empty core, every atom in the pile
    build = cluster.propose(ctx, build)       # draw over the pile, append the new families (head-only)
    n_new = len(build["draw_pile_log"]["founded"])

One call sees every atom in the pile at once (prompts/aggregation/cluster/cluster.txt) and returns
`[{"head", "members"}]`; the same partitioner serves the first round of a fresh build (the whole corpus) and
every later round (what refile left). It may leave atoms UNFILED by design — that pile is what the next round
reads. A family publishes only on min_sources distinct sessions; survivors are appended head-only for
synthesize to write and the hearings to judge. Draw variance is real (~15-20% per draw), which is why an
adopted list is amended rather than redrawn, and why `group` can take several draws and keep what they agree
on (models.cluster.draws, or `draws=` on group / group_chunked / propose; 1 = a single draw, the default).
"""
from __future__ import annotations

import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .. import families
from ..cache import NoCache
from ..engine import bare_model
from ..lib import sha256
from . import blocks

SEP = "␟"


def _ladder(max_out, effort, max_cap=128000):
    """The truncation rungs: raise the cap at the configured effort FIRST, degrade effort only last. No rung asks
    for more than `max_cap` (the model's completion limit, models.cluster.max_out_cap) unless max_out already does."""
    big = min(max_out * 3, max(max_cap, max_out)) if max_cap else max_out * 3
    return ([(max_out, effort), (big, effort), (big, "low")] if effort != "low"
            else [(max_out, "low"), (big, "low")])


def chat(ctx, prompt, max_out, tag, model=None, effort=None, cache_prefix=None, draw=1):
    """One call, retried up the ladder on truncation. -> (content, the effort that produced it). Shared with
    the writer — the package's only call path that escalates on `finish_reason == "length"`. `draw > 1` is a
    repeat of the same prompt: its seed is offset so a provider that honours `seed` does not replay draw 1."""
    model = model or ctx.model("cluster")
    effort = effort or ctx.effort("cluster")
    timeout = int(ctx.config.get("models.cluster.timeout_s", 300))
    if cache_prefix and bare_model(model).startswith("claude") and prompt.startswith(cache_prefix):
        content = [{"type": "text", "text": cache_prefix, "cache_control": {"type": "ephemeral"}},
                   {"type": "text", "text": prompt[len(cache_prefix):]}]
    else:
        content = prompt
    rungs = _ladder(max_out, effort, int(ctx.config.get("models.cluster.max_out_cap", 128000) or 0))
    for i, (cap, eff) in enumerate(rungs):
        kw = {**ctx.engine.chat_kwargs(model, cap, effort=eff), "timeout": timeout}
        if draw > 1 and "seed" in kw:
            kw["seed"] = int(kw["seed"]) + draw - 1
        r = ctx.engine.create(model, [{"role": "user", "content": content}], **kw)
        if r.choices[0].finish_reason != "length":
            return r.choices[0].message.content or "", eff
        nxt = rungs[i + 1] if i + 1 < len(rungs) else None
        print(f"  chat[{tag}]: truncated at cap={cap:,} effort={eff}"
              + (f" — retrying at cap={nxt[0]:,} effort={nxt[1]}" if nxt else " — no retry left"), file=sys.stderr)
    raise RuntimeError(f"{model} hit the output cap ({max_out}) at every effort")


def _items(labels):
    return "\n".join(f"{i}. {lab}" for i, lab in enumerate(labels, 1))


def _parse_groups(content, labels):
    """"<heading> :: 3, 7, 12" lines -> [{"head", "members"}]; one atom, one family."""
    out, seen_heads, claimed, junk = [], set(), set(), []
    for line in (content or "").splitlines():
        line = line.strip().lstrip("-•*").strip()
        if "::" not in line:
            continue
        parts = line.split("::")
        head = parts[0].strip().rstrip(".")
        pos = parts[-1]
        if not head or head in seen_heads:
            continue
        members = []
        for tok in re.split(r"[,;\s]+", pos.strip()):
            tok = tok.strip(".:;()[]").strip()
            if not tok:
                continue
            nums = []
            if tok.isdigit():
                nums = [int(tok)]
            elif (m := re.fullmatch(r"(\d+)\s*[-–—]\s*(\d+)", tok)):
                a, b = int(m.group(1)), int(m.group(2))
                nums = list(range(a, b + 1)) if a <= b <= a + 200 else []
            if not nums:
                junk.append(tok)
                continue
            for n in nums:
                if 1 <= n <= len(labels) and (lab := labels[n - 1]) not in claimed and lab not in members:
                    members.append(lab)
        if not members:
            continue
        seen_heads.add(head)
        claimed.update(members)
        out.append({"head": head, "members": sorted(members)})
    if junk:
        print(f"  _parse_groups: {len(junk)} unparsed member token(s) discarded: {', '.join(sorted(set(junk))[:8])}",
              file=sys.stderr)
    return out


def _draw(ctx, labels, d=1):
    """ONE clustering draw over sorted `labels`: families AND their members from a single call ->
    [{"head", "members"}]. The memo section and key prefix are historical ("rollup_decompose_groups",
    "DECOMPOSEG") and must stay; draw 1 keeps the pinned key and draw d > 1 takes its own slot ("D{d}")."""
    model, effort = ctx.model("cluster"), ctx.effort("cluster")
    template = ctx.prompt("cluster", occupation=ctx.occupation)
    items = _items(labels)
    cap = max(int(ctx.config.get("models.cluster.max_out_min", 32000)),
              int(ctx.config.get("models.cluster.max_out_per_atom", 120)) * len(labels))
    if int(ctx.config.get("models.cluster.max_out_cap", 128000) or 0):  # never above the model's completion limit
        cap = min(cap, int(ctx.config.get("models.cluster.max_out_cap", 128000)))
    key = sha256(SEP.join(["DECOMPOSEG", model, effort, str(cap), sha256(template), items]
                          + ([f"D{d}"] if d > 1 else [])))

    def compute():
        content, eff = chat(ctx, template.replace("$items", items), cap, "draw" if d == 1 else f"draw {d}", draw=d)
        got = _parse_groups(content, labels)
        if eff != effort:
            print(f"  group: DEGRADED to effort={eff}; using it but NOT caching it", file=sys.stderr)
            return NoCache(got)
        return got
    return ctx.cache.memo("rollup_decompose_groups", key, compute)


def _consensus(partitions, labels, threshold=0.5):
    """The partition several draws agree on (evidence accumulation) -> [{"head", "members"}].

    The co-association of two atoms is the share of draws that filed them in the same family; an atom a draw
    left unfiled co-occurs with nothing in that draw. Families are grown by average linkage and two join only
    while their mean co-association is strictly above `threshold` (0.5 = a strict majority of the draws), so
    an atom filed in no more than that share of the draws stays unfiled. Each family takes the heading of the
    draw family it overlaps most (Jaccard; ties to the earlier draw), a heading being used once where the
    draws offer enough of them. Deterministic: members sorted, families ordered by their first member."""
    n = len(partitions)
    if not n:
        return []
    known = set(labels)
    sig = {}                                            # atom -> its family index in each draw (-1 = unfiled)
    for d, groups in enumerate(partitions):
        for gi, g in enumerate(groups):
            for m in g["members"]:
                if m in known:
                    sig.setdefault(m, [-1] * n)[d] = gi
    by_sig = defaultdict(list)                          # atoms every draw treated alike start as one block
    for lab in sorted(sig):
        if sum(x >= 0 for x in sig[lab]) / n > threshold + 1e-9:   # can never clear the threshold otherwise
            by_sig[tuple(sig[lab])].append(lab)
    if not by_sig:
        return []
    clusters = [list(ms) for ms in by_sig.values()]
    S = np.array(list(by_sig), dtype=np.int64)
    b = len(clusters)
    C = np.zeros((b, b))
    for d in range(n):
        col = S[:, d]
        C += (col[:, None] == col[None, :]) & (col >= 0)[:, None]
    C /= n
    np.fill_diagonal(C, -1.0)
    size = np.array([len(ms) for ms in clusters], dtype=float)
    while b > 1:
        i, j = divmod(int(np.argmax(C)), b)
        if C[i, j] <= threshold + 1e-9:
            break
        merged = (size[i] * C[i] + size[j] * C[j]) / (size[i] + size[j])   # average linkage, exact by size
        C[i, :] = merged
        C[:, i] = merged
        C[j, :] = -1.0
        C[:, j] = -1.0
        C[i, i] = -1.0
        size[i] += size[j]
        clusters[i] += clusters[j]
        clusters[j] = None
    fams = sorted(sorted(ms) for ms in clusters if ms)
    where = defaultdict(set)                            # atom -> the draw families holding it
    for d, groups in enumerate(partitions):
        for gi, g in enumerate(groups):
            for m in g["members"]:
                where[m].add((d, gi))
    out, used = [], set()
    for ms in fams:
        mset, cands = set(ms), []
        for d, gi in {c for m in ms for c in where[m]}:
            g = partitions[d][gi]
            inter = len(mset & set(g["members"]))
            cands.append((-inter / (len(mset) + len(g["members"]) - inter), d, g["head"]))
        cands.sort()
        head = next((h for _, _, h in cands if h not in used), cands[0][2])
        used.add(head)
        out.append({"head": head, "members": ms})
    return out


def n_draws(ctx, draws=None):
    """How many draws a clustering takes: the argument, else models.cluster.draws, else 1."""
    return max(1, int(draws or ctx.config.get("models.cluster.draws", 1) or 1))


def group(ctx, task_labels, draws=None):
    """Cluster `task_labels` into families -> [{"head", "members"}]. `draws` (default models.cluster.draws,
    1) is how many times the clustering is repeated: one draw is returned as it came; several are drawn in
    parallel, each in its own cache slot, and reduced to their consensus (`_consensus`, at
    models.cluster.consensus)."""
    labels = sorted(task_labels)
    n = n_draws(ctx, draws)
    if n == 1:
        return _draw(ctx, labels)
    with ThreadPoolExecutor(max_workers=min(n, ctx.workers_for("cluster"))) as ex:
        parts = list(ex.map(lambda d: _draw(ctx, labels, d) or [], range(1, n + 1)))
    got = _consensus(parts, labels, float(ctx.config.get("models.cluster.consensus", 0.5)))
    print(f"  group: consensus of {n} draws ({', '.join(str(len(p)) for p in parts)} group(s)) -> {len(got)} group(s), "
          f"{sum(len(g['members']) for g in got)}/{len(labels)} atoms filed", file=sys.stderr)
    return got


def group_chunked(ctx, pile, draws=None):
    """The draw over a pile too large for one call (blocks.cluster_chunk). Chunks are k-means over the atoms'
    embeddings, inside one source-function block when blocking is on; the chunk HEADS are then drawn once more
    and heads grouped together union their members. With the knob unset this is exactly `group`. `draws` is
    handed to every `group` call (each chunk and the heads draw)."""
    kw = {} if draws is None else {"draws": draws}
    chunk = blocks.cluster_chunk(ctx)
    if not chunk or len(pile) <= chunk:
        return group(ctx, pile, **kw) or []
    meta = ctx.atoms.meta
    by_fn = defaultdict(list)
    for a in pile:
        by_fn[meta.get(a, {}).get("function", "") if blocks.block_by(ctx) == "function" else ""].append(a)
    chunks = [ch for fn in sorted(by_fn) for ch in blocks.kmeans_chunks(ctx, by_fn[fn], chunk)]
    print(f"  group_chunked: {len(pile)} atoms in {len(chunks)} chunk(s) of {min(map(len, chunks))}-{max(map(len, chunks))} "
          "atoms; drawing in parallel", file=sys.stderr, flush=True)
    done = [0]

    def one(ch):
        g = group(ctx, ch, **kw) or []
        done[0] += 1
        print(f"  group_chunked: chunk {done[0]}/{len(chunks)} ({len(ch)} atoms) -> {len(g)} group(s)", file=sys.stderr, flush=True)
        return g
    with ThreadPoolExecutor(max_workers=min(len(chunks), ctx.workers_for("cluster"))) as ex:
        first = [g for gs in ex.map(one, chunks) for g in gs]
    if len(first) < 2:
        return first
    by_head = defaultdict(list)
    for g in first:
        by_head[g["head"]].append(g)
    second = group(ctx, sorted(by_head), **kw) or []
    merged, used = [], set()
    for g2 in second:
        parts = [g for h in g2["members"] for g in by_head.get(h, [])]
        if not parts:
            continue
        members = sorted({m for p in parts for m in p["members"]})
        head = g2["head"] if len(parts) > 1 else parts[0]["head"]
        merged.append({"head": head, "members": members})
        used.update(id(p) for p in parts)
    merged += [g for g in first if id(g) not in used]
    print(f"  group_chunked: heads draw unioned {len(first)} -> {len(merged)} group(s)", file=sys.stderr)
    return merged


def admit(groups, sources, min_sources, floor=None):
    """Apply the source floor -> (kept, thin). `floor(head, sources) -> bool` is the rule (ctx.floor.clears,
    aggregation/floor.py); without it, min_sources distinct sources."""
    kept, thin = [], []
    for g in groups:
        w = sorted(set().union(*(set(sources[m]) for m in g["members"])))
        rec = {"sid": families.new_sid(), "head": g["head"], "members": g["members"], "sources": w, "prevalence": len(w)}
        ok = floor(g["head"], w) if floor is not None else len(w) >= min_sources
        (kept if ok else thin).append(rec)
    return kept, thin


def seed(ctx) -> dict:
    """A FRESH seed artifact: empty core, every corpus atom in the pile."""
    atoms = ctx.atoms
    return {"meta": {"occupation": ctx.occupation, "code": atoms.code, "title": atoms.title,
                     "model": ctx.model("cluster"), "effort": ctx.effort("cluster"), "min_sources": ctx.min_sources,
                     "n_atoms": len(atoms), "prompt": sha256(ctx.prompt("cluster", occupation="$OCC"))[:12],
                     **({"instruments": {i: {"sources": n, "floor": ctx.floor.need(i)} for i, n in ctx.floor.sizes.items()}}
                        if ctx.floor.multi else {})},
            "core": [], "released": sorted(atoms.tasks)}


def propose(ctx, build, draws=None) -> dict:
    """Draw families over the build's PILE and append the survivors as head-only families. Consumed atoms leave
    the pile; the rest stay; nothing is written for the new families here. Reconciles with the corpus first:
    atoms ingested since the seed enter the pile, members the corpus no longer has leave their family.
    Mutates and returns build, with `draw_pile_log` for this round and `draw_log` appended. `draws` is the
    number of clustering repetitions (see `group`); the log records it when it is more than one."""
    core = build["core"]
    W = ctx.atoms.sources
    corpus = set(W)
    known = {m for f in core for m in f["members"]} | {a for k in families.LEFTOVERS for a in build.get(k) or []}
    new = sorted(corpus - known)
    gone = sorted(known - corpus)
    if gone:
        gs = set(gone)
        for f in core:
            f["members"] = [m for m in f["members"] if m not in gs]
        for k in families.LEFTOVERS:
            if build.get(k):
                build[k] = sorted(set(build[k]) - gs)
        core[:] = [f for f in core if f["members"]]
    if new:
        build["released"] = sorted(set(build.get("released") or []) | set(new))
    build.setdefault("meta", {})["n_atoms"] = len(corpus)
    filed = {m for f in core for m in f["members"]}
    pile = [a for a in families.pile(build) if a not in filed]
    founded = []
    if len(pile) >= 2:
        groups = group_chunked(ctx, pile, draws)
        fl = ctx.floor
        if fl.multi:
            fl.prefill([(g["head"], {w for m in g["members"] for w in W.get(m) or ()}) for g in groups])
        fams, _thin = admit(groups, {m: set(W.get(m) or ()) for m in pile}, ctx.min_sources,
                            floor=fl.clears if fl.multi else None)
        consumed = set()
        for f in fams:
            ms = sorted(set(f["members"]) - consumed - filed)
            if len(ms) < 2:
                continue
            core.append({"sid": families.new_sid(), "head": f["head"], "members": ms, "sources": f["sources"],
                         "prevalence": f["prevalence"], "founded_by": "cluster --over-pile",
                         **({"function": blocks.function_of(ctx, ms), "duty": blocks.duty_of(ctx, ms)}
                            if blocks.block_by(ctx) else {})})
            consumed.update(ms)
            founded.append(f["head"])
        for k in families.LEFTOVERS:
            if build.get(k):
                build[k] = sorted(set(build[k]) - consumed)
    families.recount(ctx, core)
    entry = {"pile": len(pile), "founded": founded, "corpus_new": len(new), "corpus_gone": gone,
             **({"draws": n_draws(ctx, draws)} if n_draws(ctx, draws) > 1 else {})}
    build["draw_pile_log"] = entry
    build.setdefault("draw_log", []).append({"round": len(build.get("draw_log") or []) + 1, **entry})
    return build
