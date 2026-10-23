# Pipelines as documents

A pipeline in mosaic is a **document**: a JSON recipe of steps and the references
between them, validated against the real registries, resolved against a dataset into
a plan, and then run.

That is a deliberate choice over the more obvious one — a pipeline as a live object —
and the two coexist. `Pipeline` holds feature *classes* and a `CallbackStep` wrapping
a live callable, so it has no wire form at all. A recipe has nothing but a wire form.

## What the document form buys

**It is portable.** A recipe never carries a resolved `run_id`, because those are
dataset state, and never carries an entry list, because those are about one dataset.
So the same file runs over several. What is specific to one submission — which
entries, which pinned artifact, which resolved feature versions — lives in a
`Request` beside the recipe rather than inside it.

**It can be checked before anything happens.** Validation runs before the dataset is
touched and reports all of its problems at once. It decides from declarations alone,
with no dataset at all. An editor can therefore refuse a wire as it is drawn. Each
feature declares the entity level of its output. Each feature mosaic registers also
declares the columns it reads from its merged input (`requires`) and the columns of its
output (`provides`). A feature registered outside mosaic may declare neither. Validation
walks the recipe and works out, for each step, the columns its input is guaranteed to
contain and the columns it may contain. A required column absent from every input is a
problem, and the problem names the inputs providing it. A required column is a caution
when an input may contain it. The tracks may contain each column allowed by a schema.
They may also contain columns outside every schema and feature declaration, such as
SLEAP's `poseP0`. A column that an undeclared feature may produce is a caution too. A
caution does not block a run. A step whose feature does not declare its requirements is
listed as unchecked, and the columns that it reads are not checked. The same declaration
is checked against each entry's merged frame before the feature runs. Validation and
execution therefore agree on the columns a feature reads. The sharpest refusal is a
multi-input join of mismatched entity granularity. Merging such inputs on the frame
alone pairs every row of one input with every row of the other in the same frame.

**It can hold an op.** `Pipeline` cannot express a tracker, a conversion or a
transcode; a recipe can, which is why a whole workflow from raw files or video to a
scored classifier is one document rather than a shell script wrapping a notebook.

**The dataset records what was applied to it.** On first use the recipe is copied to
`<dataset>/.mosaic/pipelines/<digest>.json`.
The digest is taken over the recipe's canonical form: keys sorted, defaults filled,
and a whole number in a step's `params` written as an integer, so `5` and `5.0`
name one recipe.

## One place per fact

Cross-step references sit at the exact site they substitute — inside `inputs`, inside
`tracks`, inside a `params` field. There is no `edges` array, so there is nothing that
can drift from the step bodies; the edge list is a derived view. The one explicit list
is `after`, and it corresponds to nothing in any payload because it is ordering only.

## Planning resolves everything, and commits to nothing

`plan_pipeline(ds, recipe)` resolves every step in one topological walk and submits
nothing. Step A's identity is a function of its parameters, B's of its parameters plus
A's identity, C's of B's. The walk terminates because every term is either in the
recipe or already on disk: a feature-to-feature edge reads nothing, and a tracks
variant is *minted* from the recipe's settings rather than read back from tables no op
has written yet.

**A resolved identifier is never load-bearing at execution.** It drives the preview,
the estimate, the validation and the decision to enqueue. It never skips a step and
never enters a downstream job's payload — every step resolves its own identity at its
own start. A submitted identifier is a prediction, not an instruction.

## A graph that converts its own data

A `convert-tracks` or `convert-labels` step names the variant it writes from its
settings alone, so a step reading it through a `tracks` or `labels` reference is named
before the conversion runs. Which entries the conversion writes is not known until it
runs. On a dataset with no tracks yet, a step whose identity covers its entries, and
an op that is asked for entries, are left unresolved with the conversion as the
reason, rather than named over no entries. Each is named at its own start, when the
tracks exist. A conversion takes no scope, and on a dataset that already holds tracks
the plan covers those.

## A step scope narrows one step

A request names the entries a graph covers. `Request.step_scopes` narrows single steps
within them, so one request can fit on the training groups and score on the test
groups. A step's target is the request's entries narrowed by its scope, and its
identity, its coverage and what it is asked for are measured over that target.

The data a step reads must be there for every entry of its target, so a step reading
another through `inputs` or `tracks` is refused when the other is narrowed to fewer of
them. A step reading another through a params reference reads one artifact, fitted on
whatever that step covered, and no such rule applies. A request that narrows no step is
written without the field, and planned and fingerprinted as before.

## A fit is asked for all of its scope

A step whose result depends on which entries were in scope is asked for its *whole*
scope, never for the remainder. Its identity **is** its scope, so a fit over what is
left, recorded under the name of a fit over everything, is exactly the silent wrong
answer the scheme exists to prevent. A step with no scope dependence gets the
remainder, as it should. `pair-posedistance-pca` is the first when it fits and the
second when it is given a `model`, which applies a basis fitted by another step.

Which entries a fit learns from can be named elsewhere than in its scope. The
classifiers `xgboost`, `sequence-classifier` and `lightning-action` train on the entries
of their `templates` run, and a `probability-ensemble` with `weights: fit` chooses its
weights on the entries of its members' fold table. Their identity holds no scope, so
widening one to the test groups is the same run: the stored fit is applied to the new
entries and nothing fits again. For the two networks and the blend, the first request
has to cover the training entries, and is refused otherwise. The same holds for `arhmm`, `kpms`, `vame`, the identity
models and `pair-posedistance-pca` when given a `model`. The template steps,
`fit-label-map` and any of these without a `model` are still asked for the whole of
their scope, so scope them to the training groups once and apply what they fitted from
a second step.

That asymmetry is also why a coverage shortfall is **refused** rather than proceeded
through, and refused only for a scope-dependent step. A scope-free step over 89 of 90
entries writes 89 correct outputs under the identifier they belong to, and the
ninetieth arrives later under the same one. A scope-dependent step over 89 writes
*one* artifact that is not the one anyone asked for, under a name saying it is.
`allow_partial` (`--allow-partial` at the CLI) is where a decision to proceed
regardless is recorded, and it answers
that question alone — a moved version or a disagreeing tracks variant is not a
question about how much, and no flag unlocks them.

## A held-out group is not fitted on

A score on data that a fit has seen is not a held-out score, and a scope-dependent fit
makes it easy to arrive there without noticing: widen a step from the training groups
to the whole dataset and its identity changes, so it fits again, this time on the test
groups too. The dataset can name its held-out groups ([`splits`](../dataset.md#hold-groups-out-of-training)),
and the plan then refuses a step that is both scope-dependent and a fit (`fits` in the
step catalog) over a `test` or `sealed` group. A step that is scope-dependent and learns
nothing, such as `evaluate-labels`, may cover them. So may a step that applies a fitted
model, because the model is named by the step that fitted it and not by the entries it
is applied to.

Data that arrives after the fits is the same question asked later. Adding a group
changes the entries of every scope-dependent step that covers it, so each one's
identity moves and the recipe is not the one that was fitted. `assume_added` on a plan
asks that before the data exists, and `mosaic pipeline score` checks it by comparing
the run id of every step of the recipe before and after the data joins. Both name the
steps that would fit again. A step whose scope is narrowed to the training groups does
not move, which is why the protocol scopes every fit.

## A refusal names its reason

A refusal is a failure that names its reason, which tells it apart from a crash.
`mosaic pipeline run`, `mosaic run` and `mosaic track` exit with code 65 and print the
reason. Under `--json` they print one object with `"status": "refused"` and the
`reason`. From Python a refusal is raised as a `Refusal`
(`mosaic.core.pipeline.refusal`), with `reason` and a `detail` with the numbers
behind it. A pipeline step's refusal is its subclass `StepRefused`. The run log
records the attempt as `failed`, with the reason beside it.

The reasons form a closed set, such as `coverage_shortfall`, `version_moved`,
`variant_mismatch` and `undecodable_codec`. Each reason except `coverage_shortfall` is
fixed at its cause, and the step then runs again.

## Failed entries: waiting and quarantined

A pipeline step that fails on one entry records the failure under
`<dataset>/.mosaic/claims/`, and the count carries across resubmissions. The count
decides what happens to that entry next:

- **Waiting.** After a failure the entry is not retried for a minute, and for two
  minutes after a second failure. A step that meets a waiting entry runs without it
  this time and records no decision about it.
- **Quarantined.** After three failures the entry is no longer attempted. A step whose
  scope holds it is refused with `coverage_shortfall`, naming the entry. With
  `allow_partial` the step proceeds without it, and that exclusion is recorded for that
  request only. For a fit this is a fit over fewer entries, named for the entries it
  covered.

A success clears an entry's record. To retry a quarantined entry once its input is
fixed, delete its record,
`.mosaic/claims/entries/<name>/<run_id>/<group>__<sequence>.json` (just
`<sequence>.json` when the group is empty), which also holds the entry's last error.
`<name>` is the step's directory under `features/`. Keep the rest of
`.mosaic/claims/`, since it is the only record of which entries are known to fail.

---

To build one, see [Chain steps into a recipe](../guides/pipelines/chain-steps.md).
