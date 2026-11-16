"""floor — THE publishing rule: does a family have the source evidence to publish? One instrument or several.

    fl = ctx.floor
    fl.clears(statement, sources) -> bool        the rule, for a statement and the source ids behind its members
    fl.prefill([(statement, sources)])           scope verdicts for many at once, skipping any that fail the total
    fl.stamp(families)                           record each family's scope on it, for verify (instruments only)
    fl.problems(family, sources) -> [str]        verify's check: from the stamp, no model calls

ONE INSTRUMENT (config `instruments` empty): a family publishes on `min_sources` distinct sources. No model call.

SEVERAL INSTRUMENTS: a family publishes when
  1. it has `min_sources` distinct sources in all,
  2. the scope judge puts it in scope for at least one instrument present in the corpus, and
  3. for EVERY instrument it is in scope for, it has that instrument's floor of sources:
     max(instruments.<i>.min_sources (default 1), ceil(instruments.<i>.min_share * the instrument's sources)).
An instrument the family is out of scope for asks nothing of it: absence from data that could not have reported
the task is no evidence against it. The scope verdict is per (statement text, instrument), cached; a rewrite of
the statement re-asks, and the stamp records which text was judged so verify can tell a stale one.
"""
from __future__ import annotations

import math
import sys
from collections import Counter

from ..lib import sha256


class Floor:
    def __init__(self, ctx):
        self.ctx = ctx
        self.min_sources = int(ctx.config.get("min_sources", 2))
        self.spec = {str(k): dict(v or {}) for k, v in (ctx.config.get("instruments") or {}).items()}
        self.multi = bool(self.spec)
        self._sizes = None
        if self.multi:
            for name, s in self.spec.items():
                if not str(s.get("scope") or "").strip():
                    raise ValueError(f"instruments.{name}.scope is empty: the scope judge needs a description")
                share = float(s.get("min_share", 0) or 0)
                if not 0 <= share <= 1:
                    raise ValueError(f"instruments.{name}.min_share must be in [0, 1], got {share}")

    # ── the corpus ────────────────────────────────────────────────────────────────────────────────
    @property
    def sizes(self) -> dict[str, int]:
        """{instrument: distinct sources} over the instruments present in BOTH the config and the corpus."""
        if self._sizes is None:
            atoms = self.ctx.atoms
            have = atoms.instrument_sizes if atoms is not None else {}
            if self.multi and atoms is not None:
                if not have:
                    raise ValueError("config.instruments is set but the atoms have no `instrument` column; pass "
                                     "{instrument: atoms} to build, or use Atoms.combine")
                unknown = sorted(set(have) - set(self.spec))
                if unknown:
                    raise ValueError(f"atoms carry instrument(s) {unknown} that config.instruments does not describe")
                absent = sorted(set(self.spec) - set(have))
                if absent:
                    print(f"  floor: configured instrument(s) {absent} have no atoms; they ask nothing of any family",
                          file=sys.stderr)
            self._sizes = {i: n for i, n in have.items() if i in self.spec}
        return self._sizes

    def need(self, instrument) -> int:
        """The sources a family in scope for `instrument` needs from it."""
        s = self.spec[instrument]
        return max(int(s.get("min_sources", 1) or 1),
                   math.ceil(float(s.get("min_share", 0) or 0) * self.sizes.get(instrument, 0) - 1e-9))

    def counts(self, sources) -> dict[str, int]:
        """{instrument: how many of `sources` came from it}, over every present instrument (zeros included)."""
        of = self.ctx.atoms.instrument_of
        c = Counter(of.get(s) for s in set(sources))
        return {i: c.get(i, 0) for i in self.sizes}

    # ── scope ─────────────────────────────────────────────────────────────────────────────────────
    def scope(self, statement) -> dict[str, bool]:
        """{instrument: in scope?} for every present instrument (the judge, cached)."""
        j = self.ctx.judges
        return {i: j.in_scope(statement, i, self.spec[i]["scope"]) for i in self.sizes}

    def prefill(self, items) -> None:
        """Scope verdicts for [(statement, sources)] in one fan-out; sources=None always asks. A statement whose
        sources miss the overall min_sources is skipped: `clears` never asks about it."""
        if not self.multi:
            return
        texts = {t for t, w in items if t and (w is None or len(set(w)) >= self.min_sources)}
        self.ctx.judges.in_scope_many([(t, i, self.spec[i]["scope"]) for t in texts for i in self.sizes])

    # ── the rule ──────────────────────────────────────────────────────────────────────────────────
    def shortfall(self, sources, scope) -> list[str]:
        """Why a family with these sources and this scope does not publish -> [reasons] ([] = it does)."""
        sources = set(sources)
        out = []
        if len(sources) < self.min_sources:
            out.append(f"{len(sources)} source(s) in all, floor is {self.min_sources}")
        if not self.multi:
            return out
        inside = [i for i in self.sizes if scope.get(i)]
        if not inside:
            out.append("in scope for no instrument")
        n = self.counts(sources)
        for i in inside:
            if n[i] < self.need(i):
                out.append(f"in scope for {i} but {n[i]} of its {self.sizes[i]} sources, floor is {self.need(i)}")
        return out

    def clears(self, statement, sources) -> bool:
        if not self.multi:
            return len(set(sources)) >= self.min_sources
        if len(set(sources)) < self.min_sources:
            return False                                  # no evidence to judge: skip the scope calls
        return not self.shortfall(sources, self.scope(statement))

    def describe(self) -> str:
        """The floor in words, for a writer's or reviewer's feedback."""
        if not self.multi:
            return f"{self.min_sources} sources"
        per = ", ".join(f"{self.need(i)} {i} sources when in scope for {i}" for i in self.sizes)
        return f"{self.min_sources} sources in all and {per}"

    # ── the stamp verify reads ────────────────────────────────────────────────────────────────────
    @staticmethod
    def _text(f) -> str:
        return f.get("statement") or f.get("head") or ""

    def stamp(self, fams) -> list:
        """Record {instrument: in scope} on each family, with the hash of the text judged, in place. -> fams."""
        if not self.multi:
            return fams
        self.prefill([(self._text(f), None) for f in fams])
        for f in fams:
            t = self._text(f)
            f["scope"] = self.scope(t)
            f["scope_for"] = sha256(t)[:12]
        return fams

    def problems(self, f, sources) -> list[str]:
        """The floor, checked from the family's stamp: no model calls. -> [failure strings]."""
        t = self._text(f)
        if not self.multi:
            return [f"floor: {t[:50]!r} rests on {len(set(sources))} source(s), floor is {self.min_sources}"] \
                if len(set(sources)) < self.min_sources else []
        if not isinstance(f.get("scope"), dict) or f.get("scope_for") != sha256(t)[:12]:
            return [f"scope: {t[:50]!r} carries no scope verdict for its current text"]
        return [f"floor: {t[:50]!r} {r}" for r in self.shortfall(sources, f["scope"])]
