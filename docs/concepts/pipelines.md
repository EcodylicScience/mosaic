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
touched and reports every problem rather than the first. `can_connect` and `can_join`
answer from declarations alone, with no dataset at all, which is what lets an editor
refuse a wire as it is drawn. The sharpest of those refusals is a multi-input join of
mismatched entity granularity: silently, that is a per-frame cartesian product.

**It can hold an op.** `Pipeline` cannot express a tracker or a transcode; a recipe
can, which is why a whole workflow from video to embedding is one document rather
than a shell script wrapping a notebook.

**The dataset records what was applied to it.** On first use the recipe is copied to
`<dataset>/.mosaic/pipelines/<digest>.json`.

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

## A fit is asked for all of its scope

A step whose result depends on which entries were in scope is asked for its *whole*
scope, never for the remainder. Its identity **is** its scope, so a fit over what is
left, recorded under the name of a fit over everything, is exactly the silent wrong
answer the scheme exists to prevent. A step with no scope dependence gets the
remainder, as it should.

That asymmetry is also why a coverage shortfall is **refused** rather than proceeded
through, and refused only for a scope-dependent step. A scope-free step over 89 of 90
entries writes 89 correct outputs under the identifier they belong to, and the
ninetieth arrives later under the same one. A scope-dependent step over 89 writes
*one* artifact that is not the one anyone asked for, under a name saying it is.
`allow_partial` is where a decision to proceed regardless is recorded, and it answers
that question alone — a moved version or a disagreeing tracks variant is not a
question about how much, and no flag unlocks them.

---

To build one, see [Chain steps into a recipe](../guides/pipelines/chain-steps.md).
