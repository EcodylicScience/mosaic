# Pipeline graphs

A pipeline as a **file**: a JSON recipe of steps and the references between
them, checked before it runs, resolved against a dataset into a plan, and
executed. Portable across datasets, diffable in a review, and identified by
content digest.

```bash
mosaic pipeline validate --recipe @recipe.json                     # no dataset
mosaic pipeline show     --recipe @recipe.json                     # no dataset
mosaic pipeline plan     --recipe @recipe.json -m dataset.yaml
mosaic pipeline run      --recipe @recipe.json -m dataset.yaml
mosaic pipeline submit   --recipe @recipe.json -m dataset.yaml     # record it, run nothing
mosaic pipeline status   --request <id>       -m dataset.yaml
```

```python
from mosaic.core.pipeline.graph import Recipe, load_recipe, plan_pipeline, run_pipeline

recipe = load_recipe("recipe.json")
plan = plan_pipeline(ds, recipe)            # resolves; submits nothing
for step in plan.steps:
    print(step.step_id, step.run_id, step.status, step.reason)

run_pipeline(ds, recipe)                    # executes here, in this process
```

## The recipe

```jsonc
{
  "schema_version": 1,
  "name": "trex to global tsne",
  "steps": [
    { "id": "transcode", "type": "op", "kind": "transcode",
      "params": { "target": "analysis" } },

    { "id": "trex", "type": "op", "kind": "trex",
      "params": { "track_max_individuals": 4 },
      "after": ["transcode"] },            // ordering only: no data reference exists

    { "id": "speed", "type": "feature", "feature": "speed-angvel",
      "inputs": ["tracks"],
      "tracks": { "step": "trex" } },      // -> --tracks-run-id <variant>

    { "id": "templates", "type": "feature", "feature": "extract-templates",
      "inputs": [ { "step": "speed" } ], "params": { "n_templates": 8 } },

    { "id": "tsne", "type": "feature", "feature": "global-tsne",
      "inputs": [ { "step": "speed" } ],
      "params": { "templates": { "step": "templates",
                                 "pattern": "templates.parquet" },
                  "perplexity": 50 } }
  ]
}
```

Three things the format leaves out, each on purpose:

- **No resolved `run_id`.** Those are dataset state, and a recipe travels.
- **No entry list.** Which sequences to run over is a submission's choice, not a
  property of the analysis, so it is passed at plan time (`--entry`,
  `intended_entries=`). An op step's scope reaches it the same way.
- **No `overwrite`.** It mutates content under a stable address, so a concurrent
  downstream reader gets a mixed read that its own `run_id` records nothing
  about. Validation refuses the key on presence.

`pattern` is spelled out on the `templates` reference because an artifact
reference defaults its glob to `*.parquet`, which silently resolves the wrong
file out of a producer that writes more than one.

**One place per fact.** A cross-step reference sits at the exact site it will be
substituted, so there is no `edges` array to drift from the bodies; `edges()`
reads them back out for anything that draws the graph. The one explicit list is
`after`, which is ordering-only and corresponds to nothing in any payload.

## Validation happens before the dataset is opened

Most of what can be wrong with a recipe is wrong whatever dataset it is pointed
at, and discovering it at run time means discovering it one step at a time,
after the expensive steps above it have already run.

```python
from mosaic.core.pipeline.graph import check_recipe, reject_unless_valid

for problem in check_recipe(recipe):
    print(problem)
```

Every problem is reported rather than the first, and a step below a broken one
restates nothing — fix the upstream and re-check.

::: mosaic.core.pipeline.graph.validate
    options:
      show_source: false
      members_order: source

## Which steps may be wired to which

Dataset-independent, so a canvas can refuse a wire as it is drawn with nothing
selected. The sharpest refusal is `can_join`'s: two inputs at different entity
levels share no identity column, so merging them on `frame` alone pairs every row
of one with every row of the other — a per-frame cartesian product that raises
nothing and produces a plausible table.

::: mosaic.core.pipeline.graph.compatibility
    options:
      show_source: false
      members_order: source

## The plan

Every step resolves in one topological walk: step A's identity is a function of
its params, B's of its params plus A's identity, C's of B's. Nothing waits on
execution, because every term of an identifier is either in the recipe or on disk
beforehand — a feature-to-feature edge reads nothing at all, a tracks variant is
minted from the recipe's settings rather than read back from tables an op has not
written, and a `scope_dependent` step's entry set comes from the scope the graph
is planned over.

**A resolved `run_id` is never load-bearing at execution.** It drives the
preview, the estimate, validation and the decision to enqueue. It never skips a
step and never enters a downstream job's payload: every step resolves its own
identity at its own start. That is what makes prediction safe rather than
authoritative — a wrong resolution makes the preview wrong, and the next call
corrects it.

::: mosaic.core.pipeline.graph.plan
    options:
      show_source: false
      members_order: source

## Running it here

The no-queue path, and not a lesser one: it is what a notebook has and what a
bare compute node has. Each step is planned again at its own start, which is the
rule a queued job follows and what keeps the loop correct — a step's coverage,
and so what it should be asked to compute, changes the moment its parent
finishes.

A coverage shortfall is refused rather than proceeded through, and only for a
`scope_dependent` step. The asymmetry is the point: a scope-free step over 89 of
90 entries writes 89 correct outputs under the identifier they belong to, and the
ninetieth arrives later under the same one. A scope-dependent step over 89 writes
*one* artifact that is not the one anyone asked for, under a name saying it is.
`allow_partial` is where the decision to proceed is recorded.

::: mosaic.core.pipeline.graph.run
    options:
      show_source: false
      members_order: source

## Running one step at a time

The same graph, driven by anything that can start processes in order — a shell
loop, a job array, a scheduler. `submit` records the pipeline against the dataset
and assigns every step its attempt id **before anything runs**; each step is then
run by naming itself:

```bash
mosaic pipeline submit --recipe @recipe.json -m dataset.yaml --json > request.json
mosaic run -m dataset.yaml --graph-request <request-id> --step speed --execution-id <eid>
```

The command is **step-addressed** rather than spelled out, and that is strictly
more expressive: several of the arguments reaching a feature's identity have no
flag on `mosaic run` at all, and a step re-planning itself reads all of them out
of the recipe. The request is found from the manifest's parent, never from a path
flag of its own — a path a queue does not know about is one it cannot translate
for a substrate that mounts the dataset somewhere else.

At its own start a step reads the request, checks that the recipe still digests
to the name it was submitted under, pins each feature ancestor's identity from
**that ancestor's run-log**, re-plans itself against what is now on disk, and
runs its preflight. Pinning is what stops two requests on one dataset from
cross-binding: resolving an input by feature *name* falls through to the
latest-run rule, which is wall clock, so the second step of one request would
pick up the other's output because its index row landed a second later.

::: mosaic.core.pipeline.graph.request
    options:
      show_source: false
      members_order: source

::: mosaic.core.pipeline.graph.step
    options:
      show_source: false
      members_order: source

## Refusing before doing the work

A step about to compute something is the last party in a position to notice that
what it is about to read is not what was intended. Coverage counts cannot see
wrongness — 120 parquets of NaN is 120 of 120 — so the preflight runs the
predicates that can, where a refusal is still free.

A refusal is an ordinary failure carrying a reason: a reserved exit code, the
run-log status left at `failed`, and the reason in `error_json`. There is no new
terminal status, because that set is read across three repositories and adding to
it would make a live run reap as finished.

::: mosaic.core.pipeline.graph.preflight
    options:
      show_source: false
      members_order: source

## What has been tried

The one piece of state that cannot be re-derived from the artifacts:
absent-because-quarantined and absent-because-never-run are the same observation
on disk. It is also the only bound on retrying a sequence that cannot succeed.

Attempts are global and survive a resubmit — a counter reset by the cheap
recovery would bound nothing. The decision to proceed *without* an entry is per
request, because it is a scientific choice: a model fitted on 89 sequences is a
different model from one fitted on 90.

::: mosaic.core.pipeline.graph.failures
    options:
      show_source: false
      members_order: source

::: mosaic.core.pipeline.graph.claims
    options:
      show_source: false
      members_order: source

::: mosaic.core.pipeline.graph.rollup
    options:
      show_source: false
      members_order: source

## The recipe and request files

A recipe is copied into the dataset on first use, addressed by digest, so the
dataset records which pipelines were applied to it and can be handed to someone
else intact.

::: mosaic.core.pipeline.graph.model
    options:
      show_source: false
      members_order: source

::: mosaic.core.pipeline.graph.store
    options:
      show_source: false
      members_order: source

## Ordering, scope and lanes

::: mosaic.core.pipeline.graph.topo
    options:
      show_source: false
      members_order: source

::: mosaic.core.pipeline.graph.scope
    options:
      show_source: false
      members_order: source

::: mosaic.core.pipeline.graph.lanes
    options:
      show_source: false
      members_order: source
