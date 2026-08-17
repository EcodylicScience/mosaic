# Pipeline

The composable pipeline infrastructure that orchestrates feature extraction,
frame sampling, and model training. See
[Chain steps into a recipe](../../guides/pipelines/chain-steps.md) for practical
usage.

## Pipeline Orchestrator

Declarative multi-step feature pipeline with automatic caching, staleness
detection, and dependency chaining.

::: mosaic.core.pipeline.pipeline.Pipeline
    options:
      show_source: true
      members_order: source

::: mosaic.core.pipeline.pipeline.FeatureStep
    options:
      show_source: true

::: mosaic.core.pipeline.pipeline.CallbackStep
    options:
      show_source: true

## Run-logs (attempt status & progress)

Append-only JSONL run-logs, one per attempt under
`<dataset_root>/.mosaic/runs/<execution_id>.jsonl` — the job-kind-agnostic,
NFS-safe status/progress store that replaced the per-dataset SQLite `.mosaic.db`.
The reader helpers reduce a log back to an attempt-status snapshot (what the
`mosaic status` / `runs` / `cancel` commands consume).

::: mosaic.core.pipeline.run_log.JsonlRunLog
    options:
      show_source: true
      members_order: source

::: mosaic.core.pipeline.run_log.read_run
    options:
      show_source: true

::: mosaic.core.pipeline.run_log.read_runs
    options:
      show_source: true

::: mosaic.core.pipeline.run_log.read_run_progress
    options:
      show_source: true

## Feature Protocol

The `Feature` protocol is rendered from its own source under Pipeline Types below,
which is the one place its attributes are written down. For what each one decides
when you are implementing it, see
[Write your own feature](../../guides/analysis/write-your-own-feature.md).

## Pipeline Types

::: mosaic.core.pipeline.types
    options:
      show_source: true
      members_order: source

## Pipeline Runner

::: mosaic.core.pipeline.run
    options:
      show_source: true
      members:
        - run_feature

## Training Progress

Callback protocol and storage-free backends for monitoring training progress in
real time. The Job Contract's default backend is the JSONL run-log
(`JsonlRunLog`, above), which implements this protocol.

::: mosaic.core.pipeline.progress.TrainingProgressCallback
    options:
      show_source: true

::: mosaic.core.pipeline.progress.CSVProgressCallback
    options:
      show_source: true
      members_order: source

::: mosaic.core.pipeline.progress.CompositeProgressCallback
    options:
      show_source: true
