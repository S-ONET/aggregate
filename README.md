# aggregate

Task atoms in, an occupational task list out, and the evaluation of that list. `aggregate` is the Python
library behind TaskNet / Terkel: it takes a table of task statements reported by workers (interviews, job
postings, anything a source turns into atoms), groups them into families, writes one statement per family,
hears every structural change twice, and re-files every atom until the corpus is exhausted. It knows nothing
about where the atoms came from and nothing about O*NET: reference lists, held-out documents and covariates
are inputs the caller passes in.

Built for *Emerging Tasks From Worker Interviews* (Wang, Haupt, Robinson, Lewis, Brynjolfsson, Yang, 2026).
MIT.

## Install

```bash
pip install tasknet-aggregate     # the distribution name (PyPI / git); the import name is `aggregate`
pip install -e '.[test]'          # from a checkout: openai, numpy, pyyaml; pytest for the offline suite
pytest -q                     # no test calls a model
```

## Use

```python
import openai
from aggregate import Aggregator, Config, Atoms, DocSet

cfg = Config.load("configs.yml")                                    # overlays the packaged defaults by key
agg = Aggregator(cfg, client=openai.OpenAI(), workdir="results/lawyers")

atoms = agg.process(rows)                 # raw task rows -> atoms: split, filter, rewrite, debrand, dedup
atoms = Atoms.read("atoms.csv")           # or a table a source already produced
build = agg.build(atoms)                  # the chain; writes workdir/build.json, stages/, verdicts/, ledgers/
build = agg.build(atoms, standing=final)  # amend an adopted list instead of redrawing it
problems = agg.verify(build)              # [] when filed + pile == n_atoms, no duplicates, floor holds

lists = {"ours": [c["statement"] for c in build["core"]], "reference": reference_statements}
docs = DocSet.read("docs.json")           # held-out documents with their atoms and covariates
agg.evaluate.coverage(lists, docs)
agg.evaluate.silhouette(lists, docs, pool=atoms.tasks)
agg.evaluate.crosswalk(lists["ours"], lists["reference"])
agg.judges.covers(atom, statement)        # the judges as an instrument
print(agg.usage())
```

The client is anything with the OpenAI SDK's `chat.completions.create` and `embeddings.create`; pass
`router=lambda model: client` to route by model, or nothing to read `OPENAI_API_KEY` (and
`ONET_CLAUDE_API_KEY` + `ONET_CLAUDE_BASE_URL` for `claude-*` models through an OpenAI-compatible gateway).

## What is where

| | |
|---|---|
| `aggregate/configs.yml` | every knob and its default: models and efforts per judge, the floor, batching, the scale knobs, prices |
| `config.py`, `prompts.py`, `engine.py`, `cache.py`, `context.py` | the infrastructure every stage receives as one `Context` |
| `atoms.py` | the atoms table: `worker, task, occupation, title, status, original` + optional FJA fields |
| `judges.py` | `covers`, `overlaps`, `duties`, `dispose`, `merge_or_separate`, `sharpen`, `writing_defects`, `relation`, `embed` |
| `aggregation/` | `cluster`, `synthesize`, `fusion`, `overlap`, `refile`, `verify`, `blocks`, and `chain.build` |
| `processing/` | `normalize`, `dedup`, `condense`, `lines` |
| `evaluate/` | `coverage`, `silhouette`, `incidence`, `crosswalk`, `exclusivity`, `occupational`, `exposure`, `equal_length`, `discovery` |
| `prompts/` | every prompt, by bare name; `prompt_set: fja` picks `<name>_fja.txt` variants |

## The chain

cluster → statements → split → statements → merge / sharpen → refile, repeated until cluster proposes no new
family that clears the worker floor (cycle 1 always runs in full), capped at `chain.max_cycles`. Every judge
verdict is memoised on (model, prompt hash, inputs) under `cache_dir`, so a rebuild is nearly free and the
ledgers under `workdir/ledgers/` replay settled hearings. Cache keys are an interface:
`tests/test_cache_keys.py` pins every recipe to what earlier builds paid for.

## Contracts

- **Atoms** (`Atoms.read`): one row per (worker, task); legacy column names (`session_id`, `task_name`,
  `onet_code`, `onet_title`) are accepted on read.
- **Build**: `{"meta", "core": [{sid, statement, members, workers, prevalence, ...}], "released", ...}`;
  `final.json` is an adopted build, copied by the caller.
- **DocSet** (`docs.json`): `{"docs": [{"id", "split": "heldout"|"construction", "atoms": [...], "meta": {covariates}}]}`.
- **Lists** for the evaluations: `{"ours": [...], "<other>": [...]}` — the first key is ours.
