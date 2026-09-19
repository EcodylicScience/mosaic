# Reproducibility, run_id and caching

Every feature run is tagged with a run identifier:

```
run_id = "<version>-<hash>"
```

The hash covers the feature's parameters, its inputs, and the frame range. Identical
inputs and parameters give an identical `run_id`. That single fact produces most of
mosaic's useful behavior.

## What it buys you

- **Re-running costs nothing.** The run is already there, so a script you run twice
  does the work once.
- **Parameter sweeps organize themselves.** Each setting lands in
  `features/<name>/<run_id>/`, with no naming scheme to invent and no chance of two
  settings overwriting each other.
- **A result cannot be silently applied to the wrong input.** Change an upstream
  feature and everything downstream re-addresses, because the upstream identity is one
  of the terms.

Throughput knobs — worker counts, batch sizes — are deliberately excluded from the
hash. Retuning them for a bigger machine does not invalidate a cache, because they
change how fast a result is produced and not what it is.

## Version is a decision, not an accident

The `<version>` segment is declared by the feature, and moving it re-addresses every
run of that feature. That is what a version bump is *for*: it says the meaning of the
output changed, so previous results should not be reused as if they were the same
thing.

Adding a parameter with a default does not need a bump — the hash covers it. Changing
what an existing parameter means does.

## Status is derived, never stored

`mosaic inventory` reports what a dataset holds, and its answers are computed at read
time from what is on disk: `absent`, `partial`, `complete`, `complete-but-drifted`,
`inconsistent`.

There is no inventory database and no filesystem watcher. Every view is a cache thrown
away rather than reconciled, which is why a stale answer is impossible rather than
merely unlikely — and why you can delete a run directory by hand and have the next
inventory simply tell you the truth.

Coverage is which entries exist, never a boolean. "Complete" means every entry in scope
has an artifact, and `complete-but-drifted` means they all exist but the code that
produced them has since moved.

## A trained model can say what it saw

A model's identifier covers its training data by content, so it is reproducible in the
usual sense. It is also *traceable*, because each step records the one before it:

```
the model              records the prepared dataset it was trained on
the prepared dataset   records each annotation set, revision and content digest it merged
the revision           records what was noted when it was saved -- a database commit, say
```

`mosaic models provenance <run_id>` walks that chain and prints it. Nothing stores the
chain as a whole, for the same reason status is not stored: a summary is a second thing
to keep agreeing with the records it summarizes. When a link is missing the answer
says where it stopped, for instance a model trained from a folder of labels that no
preparation run wrote, or a revision in a dataset that has since been archived.

Only the revision's *content* enters an identifier. Its number is a label, so asking
for "the latest" and asking for the number that resolves to are the same run, and
annotations restored to an earlier state reuse what was computed for that state.

## What this does not promise

Determinism is a property of the *code*, and the identifier only records inputs. A
feature that used unseeded randomness or depended on filesystem ordering would produce
a different answer under the same `run_id`, and nothing would notice. mosaic's own
features avoid that; a feature you write must too.
