# TaskNet-aggregate

Aggregate takes a table of task statements reported by workers (interviews, job postings, anything a source
turns into atoms), groups them into families, and writes one statement per family, hearing every structural
change twice and re-filing every atom until the corpus is exhausted. It is agnostic to the source of the
atoms, and it knows nothing about O*NET: comparison lists, held-out documents and covariates are inputs the
caller passes in. Repositories that produce atoms include this package to write the list.

The code builds on the work for interviews in *Building Occupational Task Inventories from Worker Interviews
at Scale* (Wang, Haupt, Robinson, Lewis, Brynjolfsson, Yang, 2026). MIT.

## Install

```bash
pip install 'tasknet-aggregate @ git+https://github.com/S-ONET/aggregate.git@v0.1.0'   # the import name is `aggregate`
```

## Building a task list

A build starts from the task statements workers reported, as rows, and ends with a list of statements, each
backed by the atoms and workers behind it. Three calls:

```python
import openai
from aggregate import Aggregator, Atoms, Config

cfg = Config.load("configs.yml")          # to overwrite defaults, e.g. the models or the worker floor
agg = Aggregator(cfg, client=openai.OpenAI(), workdir="results/lawyers")

atoms = agg.process(rows)                 # generate atoms from "raw" task rows: split, filter, rewrite, debrand, deduplicate
atoms = Atoms.read("atoms.csv")           # or read them in from a file
build = agg.build(atoms)                  # the list; every invariant is checked before it returns
```

### Rows

A row is one thing one worker said they do. A worker is a session, an employer, a respondent — whatever the
source counts people by; `occupation` is any code the source uses and `title` is the occupation's name, which
the prompts are rendered with.

```python
rows = [
    {"worker": "s17", "task": "I draft motions and file them with the court", "occupation": "23-1011.00", "title": "Lawyers", "status": "added"},
    {"worker": "s17", "task": "Advising clients on whether to settle",         "occupation": "23-1011.00", "title": "Lawyers", "status": "edited"},
    {"worker": "s42", "task": "Draft motions for filing.",                      "occupation": "23-1011.00", "title": "Lawyers", "status": "confirmed"},
]
```

### Atoms

`agg.process` rewrites each row into a verb-led statement, drops non-tasks, and removes exact duplicates
within a worker. An atom is one distinct task text plus the set of workers who reported it; the table has one
row per (worker, task). It is a CSV you can keep and hand around:

```
worker,task,occupation,title,status,original
s17,Draft motions and file them with the court.,23-1011.00,Lawyers,added,I draft motions and file them with the court
s17,Advise clients on whether to settle.,23-1011.00,Lawyers,edited,Advising clients on whether to settle
s42,Draft motions and file them with the court.,23-1011.00,Lawyers,confirmed,Draft motions for filing.
```

Optional columns `duty, function, level, verb, object` carry the structured task fields an extractor can
fill (`prompts/shared/fja_hta.txt` defines them); the chain uses `function` to keep comparisons within one
worker function when `blocks.by: function` is set. `Atoms.read` also accepts the older column names
(`session_id`, `task_name`, `onet_code`, `onet_title`).

### The build

`agg.build` returns a dict and saves it as `workdir/build.json`. `core` holds one entry per published family:
its statement, the atoms it covers (`members`), the workers behind them and their count (`prevalence`), plus
the stamps the chain uses to skip work it has already done. Atoms that reached no family sit in the pile
(`released`, `declined`, `dropped_atoms`, `thin_members`); `meta.n_atoms` is what the two sides must add up to.

```json
{"meta": {"occupation": "Lawyers", "code": "23-1011.00", "n_atoms": 412, "min_workers": 2, "mode": "FRESH"},
 "core": [
   {"sid": "s_3f9a21c0",
    "statement": "Draft motions, briefs, and other pleadings for filing with the court.",
    "members": ["Draft motions and file them with the court.", "Write appellate briefs.", "..."],
    "workers": ["s17", "s42", "s63"], "prevalence": 3,
    "written_for": "…", "written_with": "…"},
   {"sid": "s_b07d4e11", "statement": "Advise clients on the risks and likely outcomes of settling a dispute.", "...": "..."}
 ],
 "released": ["Order office supplies for the practice.", "..."]}
```

`aggregate.statements(build)` is the plain list of statements. A family publishes only with `min_workers`
distinct workers behind it (2 by default; the postings runs use 3).

### The client and the configuration

The client is anything with the OpenAI SDK's `chat.completions.create` and `embeddings.create`; pass
`router=lambda model: client` to route by model, or nothing to read `OPENAI_API_KEY` (and
`ONET_CLAUDE_API_KEY` + `ONET_CLAUDE_BASE_URL` for `claude-*` models through an OpenAI-compatible gateway).
Every model, threshold and batching knob is a key in `aggregate/configs.yml`; a file passed to `Config.load`
overlays those defaults by key, so a run that only changes the writer is a one-line file:

```yaml
models:
  write: {model: gpt-5.4}
```

## Evaluating a list

An evaluation compares lists of statements on data that was not used to build them. Each step takes the
lists to compare and the documents they are measured on; nothing is read from disk, and the first key of
`lists` is treated as ours:

```python
import aggregate
from aggregate import DocSet

lists = {"ours": aggregate.statements(build), "reference": reference_statements}
heldout = DocSet.read("heldout.json")

agg.evaluate.coverage(lists, heldout)                       # share of held-out atoms a list describes, worker bootstrap
agg.evaluate.silhouette(lists, heldout, pool=atoms.tasks)   # do held-out tasks group more tightly under one list?
agg.evaluate.incidence(lists, heldout)                      # documents x statements, PCA / CA, covariate correlations
agg.evaluate.crosswalk(lists["ours"], lists["reference"])   # each of our statements: new / refined / related
agg.evaluate.exclusivity(lists)                             # related pairs within a list
agg.evaluate.exposure(lists, title="Lawyers")               # AI-exposure rubrics per statement
agg.judges.covers(atom, statement)                          # the judges themselves, as an instrument
print(agg.usage())
```

### Held-out documents

A held-out document is a worker's interview or an employer's posting that was not among those the list was
built from, with the atoms extracted from it the same way the build's atoms were, and any covariates the
source knows. `split` says whether the document was held out or was part of the build (`construction`);
the incidence step uses both.

```json
{"slug": "lawyers", "source": "postings", "docs": [
  {"id": "job_8812", "split": "heldout",
   "atoms": ["Draft discovery requests.", "Negotiate settlements with opposing counsel.", "Bill hours to client matters."],
   "meta": {"employer": "rcid:4471", "salary": 118000, "seniority": 4, "year": 2025}},
  {"id": "job_0193", "split": "construction", "atoms": ["Draft motions and file them with the court."], "meta": {"salary": 95000}}
]}
```

### What an evaluation returns

Every step returns a dict the caller saves or renders. Coverage, for instance, gives the share of held-out
atoms each list describes, the per-list gap with a bootstrap over documents, and the per-atom verdicts:

```python
{"available": True, "n_workers": 40, "n_tasks": 212, "k": {"ours": 47, "reference": 32},
 "cov": {"ours": 0.78, "reference": 0.61},
 "gaps": {"ours-reference": {"est": 0.17, "se": 0.034, "ci": [0.10, 0.24], "p": 0.0001, "n": 40}},
 "per_task": [{"worker": "job_8812", "text": "Draft discovery requests.", "ours": True, "reference": False, "n_cover": {"ours": 2, "reference": 0}}, "..."]}
```

## Amending an adopted list

A build you have reviewed and adopted is `final.json`: a copy of a `build.json` made by you, never by the
chain. When new atoms arrive, pass that dict as `standing` and the chain seeds itself from it instead of
redrawing: standing families stay byte-identical, their statements are kept wherever the member sets still
match, and only the new atoms are clustered, heard and filed. A rebuild is then an amendment of what was
reviewed rather than a fresh opinion.

```python
final = json.loads(Path("results/lawyers/final.json").read_text())
build = agg.build(atoms, standing=final)
```

## How the chain works

cluster → write → split → write → merge / sharpen → refile, repeated until cluster proposes no new family
that clears the worker floor (cycle 1 always runs in full), capped at `chain.max_cycles`. Every judge
verdict is memoised on (model, prompt hash, inputs) under `cache_dir`, so a rebuild is nearly free, and the
ledgers under `workdir/ledgers/` replay settled hearings. Cache keys are an interface:
`tests/test_cache_keys.py` pins every recipe to what earlier builds paid for.

## Files

| | |
|---|---|
| `aggregate/configs.yml` | every knob and its default |
| `config.py`, `prompts.py`, `engine.py`, `cache.py`, `context.py` | the infrastructure every stage receives as one `Context` |
| `atoms.py` | the atoms table |
| `judges.py` | `covers`, `overlaps`, `duties`, `dispose`, `merge_or_separate`, `sharpen`, `writing_defects`, `relation`, `embed` |
| `aggregation/` | `cluster`, `synthesize`, `fusion`, `overlap`, `refile`, `verify`, `blocks`, and `chain.build` |
| `processing/` | `normalize`, `dedup`, `condense`, `lines` |
| `evaluate/` | `coverage`, `silhouette`, `incidence`, `crosswalk`, `exclusivity`, `occupational`, `exposure`, `equal_length`, `discovery` |
| `prompts/` | every prompt, by bare name; `prompt_set: fja` picks `<name>_fja.txt` variants |
| `tests/` | the offline suite; `test_cache_keys.py` is the key guard |
