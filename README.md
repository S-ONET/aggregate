# aggregate

Task atoms in, an occupational task list out, and the evaluation of that list. `aggregate` is the Python
library behind TaskNet / Terkel: it takes a table of task statements reported by workers (from interviews, job
postings, anything a source turns into atoms), groups them into families, writes one statement per family,
hears every structural change twice, and re-files every atom until the corpus is exhausted. It knows nothing
about where the atoms came from and nothing about O*NET: comparison lists, held-out documents and covariates
are inputs the caller passes in.

Built for *Emerging Tasks From Worker Interviews* (Wang, Haupt, Robinson, Lewis, Brynjolfsson, Yang, 2026).
MIT.

## Install

```bash
pip install tasknet-aggregate                                        # the distribution; the import name is `aggregate`
pip install 'tasknet-aggregate @ git+https://github.com/S-ONET/aggregate.git@v0.1.0'
```

Developing: `pip install -e '.[test]'` from a checkout, then `pytest -q` (no test calls a model).

## Use

```python
import openai
from aggregate import Aggregator, Atoms, Config

agg = Aggregator(Config.load("configs.yml"), client=openai.OpenAI(), workdir="results/lawyers")

atoms = agg.process(rows)                 # raw task rows -> atoms: split, filter, rewrite, debrand, dedup
atoms = Atoms.read("atoms.csv")           # or a table a source already produced
build = agg.build(atoms)                  # the list; every invariant is checked before it returns
```

`build` is a dict, saved as `workdir/build.json`: `core` holds one entry per published family — its
`statement`, the member atoms, the workers behind them and their count (`prevalence`) — and the rest of the
atoms sit in the pile (`released`, `declined`, ...). `aggregate.statements(build)` is the list of statements.

The client is anything with the OpenAI SDK's `chat.completions.create` and `embeddings.create`; pass
`router=lambda model: client` to route by model, or nothing to read `OPENAI_API_KEY` (and
`ONET_CLAUDE_API_KEY` + `ONET_CLAUDE_BASE_URL` for `claude-*` models through an OpenAI-compatible gateway).
Every model, threshold and batching knob is a key in `aggregate/configs.yml`; `Config.load("my.yml")` overlays
a file on those defaults.

### Evaluating a list

Every evaluation takes the lists to compare and the data they are measured on, nothing is read from disk:

```python
import aggregate
from aggregate import DocSet

lists = {"ours": aggregate.statements(build), "reference": reference_statements}   # the first key is "ours"
heldout = DocSet.read("heldout.json")     # documents outside the build, each with its atoms and covariates

agg.evaluate.coverage(lists, heldout)                  # share of held-out atoms a list describes, worker bootstrap
agg.evaluate.silhouette(lists, heldout, pool=atoms.tasks)
agg.evaluate.incidence(lists, heldout)                 # documents x statements, PCA / CA, covariate correlations
agg.evaluate.crosswalk(lists["ours"], lists["reference"])   # new / refined / related per statement
agg.evaluate.exclusivity(lists); agg.evaluate.exposure(lists, title="Lawyers")
agg.judges.covers(atom, statement)                     # the judges themselves, as an instrument
print(agg.usage())
```

A `DocSet` file is `{"docs": [{"id", "split": "heldout"|"construction", "atoms": [...], "meta": {covariates}}]}`;
`split` says whether the document was held out of the build or was part of it.

### Rebuilding

An adopted list is amended, not redrawn: `agg.build(atoms, standing=final)` seeds the chain from `final`
(the dict of an adopted `build.json`) and keeps its statements wherever the member sets still match, so a
rebuild is an amendment of what was reviewed rather than a fresh opinion. Adopting a build is an explicit
copy by the caller; the chain never writes `final.json`.

## How the chain works

cluster → write → split → write → merge / sharpen → refile, repeated until cluster proposes no new family
that clears the worker floor (cycle 1 always runs in full), capped at `chain.max_cycles`. A family publishes
only with `min_workers` distinct workers behind it. Every judge verdict is memoised on (model, prompt hash,
inputs) under `cache_dir`, so a rebuild is nearly free, and the ledgers under `workdir/ledgers/` replay
settled hearings. Cache keys are an interface: `tests/test_cache_keys.py` pins every recipe to what earlier
builds paid for.

## What is where

| | |
|---|---|
| `aggregate/configs.yml` | every knob and its default |
| `config.py`, `prompts.py`, `engine.py`, `cache.py`, `context.py` | the infrastructure every stage receives as one `Context` |
| `atoms.py` | the atoms table: `worker, task, occupation, title, status, original` + optional FJA fields (legacy column names accepted on read) |
| `judges.py` | `covers`, `overlaps`, `duties`, `dispose`, `merge_or_separate`, `sharpen`, `writing_defects`, `relation`, `embed` |
| `aggregation/` | `cluster`, `synthesize`, `fusion`, `overlap`, `refile`, `verify`, `blocks`, and `chain.build` |
| `processing/` | `normalize`, `dedup`, `condense`, `lines` |
| `evaluate/` | `coverage`, `silhouette`, `incidence`, `crosswalk`, `exclusivity`, `occupational`, `exposure`, `equal_length`, `discovery` |
| `prompts/` | every prompt, by bare name; `prompt_set: fja` picks `<name>_fja.txt` variants |
