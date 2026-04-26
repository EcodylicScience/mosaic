# Mosaic MVP Feature List

Based on Jacob's outline. Each bullet is annotated with context from the Apr 23 discussion and references to existing repo components where relevant.

## In Minimal V1

- **User management (create acct): ecodylic admin, admin, user.**
  - Three-tier model: *ecodylic super-admin* (read all data across all admins, for debugging), *admin* (can create other admins and create users), *user* (standard).
  - Admins and users are strictly data-separated — Ecodylic does not interfere with or see their data beyond seeding credentials.
  - Same auth logic must work for both deployment modes: SaaS-style web instance hosted by us, and self-installed local server.
  - Standard authentication approach (API-level, not especially complex — "Loopy also has this").

- **User permissions**
  - Enforced at the API level: every API call is checked against the caller's DB entry, rejected if not permitted.
  - File-system-level separation on the backend (per-user roots) since data is organized as datasets on disk.

- **Video import**
  - Support user-provided video files end-to-end.
  - **Convert (exact format?)** — open question. Use ffmpeg with GPU-accelerated transcode (HEVC/H.265 or AV1). Tradeoffs discussed:
    - Always-re-encode (Max's preference): guarantees smooth scrubbing and codec consistency, but slow and destructive on large datasets, and can drop frames causing track/overlay drift.
    - Never-re-encode: respects raw frames (critical when tracks are computed on raw video), but scrubbing may be poor with insufficient keyframes. H.264 worked fine out of the box; AV1 required denser keyframing for responsive scrubbing.
    - MVP direction: don't force re-encoding; document which input formats/codecs are "guaranteed smooth" and which are "user's own risk." Revisit with Max.
  - **imgstore support** — Loopy-style hierarchical folder spec (e.g. Basler 10-minute chunks). Medium complexity, but needed for realistic recording setups. Low-to-mid integration effort compared with the "single video per sequence" case, which is already handled.
  - Configurable data directory — API server points at a mountable location (external drive, shared mount), not hardcoded.
  - Repo touch points: `media/video_io.py`, `media/extraction.py` (already has ffprobe fallback, hw-accel detection, raw H.264 handling).

- **Data (tracks) import**
  - Index raw tracking files (`index_tracks_raw()`) and convert to the standardized parquet schema (`convert_all_tracks()`).
  - Standard schema columns: `frame, time, id, group, sequence, X, Y, ANGLE, SPEED, poseX0..N, poseY0..N` — validated via `TrackSchema` (default `trex_v1`).
  - **Add other format support** — existing converters: TREx, CalMS21, MABe22. Template exists (`track_converter_template.py`) and new formats are registered via `register_track_converter()` decorator.

- **Dataset create and mgmt**
  - Core class `Dataset` (dataset.py, ~6.5k lines) manages roots: `media`, `tracks_raw`, `tracks`, `labels`, `features`, `models`, `inputsets`.
  - **Add metadata tags ability for groups, seqs, individuals (UIDs)**:
    - *Sequence tags* — flat user-defined list (train/test, day 1, condition, etc.).
    - *Individual tags* — flat user-defined list (groups, cohorts, etc.).
    - *Unique IDs (UIDs)* — treated as a tag with the constraint that each UID can only be used once per sequence. Enables cross-sequence identity matching.
    - *Free-text metadata* — general searchable text fields attached to sequences.
  - All tags are user-configurable flat lists. Keep modifiers (behavior-level attributes like "near wall") separate from these metadata tags — modifiers live on scoring events, tags live on dataset entities.

- **Annotate pose**
  - **Exists but needs integration** — workflow is set up end-to-end but not smoothly wired into the product.
  - **Get frames from videos** — use k-means frame extraction (better than uniform). Already in `media/extraction.py` via `extract_frames(method="kmeans")`. Do not use CVAT native video annotation (not a good idea per Jacob).
  - **Annotate in CVAT** — ship CVAT alongside Mosaic (Docker). Pre-seed skeleton / class configs so users can "start project → load configuration from shared location."
    - Mount a shared directory into CVAT.
    - Write skeleton JSON files from the frontend into that shared directory.
    - On new project creation, user can select the pre-seeded config and begin immediately.
  - Deferred: in-house fully integrated annotation tool (see "Not in Minimal V1").

- **Train pose model**
  - **Ultralytics pose** — main supported pose framework. Works well, general-purpose CV standard, regularly updated. Position Mosaic's story around Ultralytics pose + T-Rex tracking rather than wrapping DeepLabCut/SLEAP.
  - **POLO** — point-detection training supported (`train_point_model()` via Ultralytics/POLO fork).
  - Repo: `tracking/pose_training/` includes YOLO pose, POLO, and localizer heatmap training; converters for CVAT XML, Lightning Pose, COCO; augmentation presets; split-by-group dataset prep.

- **Track video (TRex)**
  - T-Rex as the main supported tracker for the MVP, invoked via **CLI**. Link already added from Mosaic side but not yet tested.

- **Running / resource mgmt (basic)**
  - Flagged as complicated — needs more thinking, but some minimum is required or users will trip over each other.
  - Minimum behavior for MVP:
    - Sorted FIFO queue on a single server, jobs run sequentially.
    - Entry point for execution is a **CLI** that takes a pipeline config file and runs as a subprocess (API server should never import mosaic directly — always shell out).
    - Log files written per job so the server can surface progress to the frontend.
    - Frontend job controls: start, stop/interrupt, "run up to this node," "run all," see what's running and progress.
    - Heavy-lifting backend runs in a Docker container receiving dataset + feature dependencies + video if needed.
  - Pipelines can be saved with associated run status (analogous to Jupyter-notebook kernel state — open a new pipeline while another runs).

- **Scoring: Ethogram creation and mgmt**
  - Ethogram = behavior list, synced directly with the SQLite database via API.
  - **Current features**: per-behavior fields `name`, `type` (interval / point), symmetry, directed/undirected — mostly implemented.
  - **Add text comments** — free-text field per scoring event (separate from modifiers).
  - **Modifiers** — project-level attributes applied per scoring event (e.g. "near wall," "in cage"). Defined per project; separate from the ethogram but can be tied to specific entries.
  - Import/export ethogram as a text file.

- **Scoring tool: score sequence and save**
  - Tool is ~80% done, needs finishing.
  - Manual interval/point scoring on the timeline.
  - **New UI capability (this week)**: direct selection on the video canvas (click animal → select → assign behavior), with multi-individual / directed-interaction support (actor → target arrow).
  - Scrub the video and place start/stop markers on the timeline; label as directed interaction (e.g. "chasing").
  - Per-row DB fields: `name, type, scope, actor, target, modifier, comment, start, stop`.
  - Database persistence across sessions (needs DB wiring).
  - Export from DB to the feature-pipeline label format — **database is the source of truth**.

- **Database (api-server), used for user mgmt, ethograms, scoring**
  - SQLite-backed via the API server.
  - Stores: users/auth, projects, ethograms, modifiers, scoring events, tags, metadata, pipeline specs and run status.
  - Acts as the canonical source for everything the frontend reads; features/label storage on disk is derived by export.

- **Analysis (mosaic-behavior)**
  - **Basic-indiv features** — needs more. E.g. smooth, speed, ROI, angular velocity, body scale, orientation.
    - Already in repo (`behavior/feature_library/`): speed-angvel, body-scale, orientation-rel, trajectory-smooth, movement-smooth, movement-filter-interpolate, etc.
    - Individual-ID-based features are the primary supported type for MVP.
  - **Feature pipeline (composable)** — pipeline that exists, **map to UI using React Flow**.
    - Node types to define: input nodes (tracks / features / results), feature nodes (each wraps a registered feature with typed Inputs/Params/OutputType), parameter nodes (reusable param sets), model-artifact nodes (trained weights consumed downstream).
    - Each node has a side panel for parameter editing.
    - Graph serializes to a config file → handed to the CLI runner.
    - Repo backing: `core/pipeline/` (typed protocol with `Params`, `Inputs`, `Result`, `ArtifactSpec`, `OutputType`); features registered via `@register_feature`; `run_feature()` orchestrates with dependency resolution.
  - **Models (global features)** — trainable features that fit-then-apply per sequence. Supervised (individual and social) and unsupervised.
    - Available in repo: `xgboost`, `arhmm`, `feral`, `kpms`, `lightning-action`, `global-scaler`, `global-tsne`, `global-kmeans`, `global-ward`, `global-identity-model`.
    - **Feral** called out specifically as expected to be heavy and a prime candidate for future cloud scale-out.
  - **Summary features** — per-sequence, per-group, per-UID aggregates (e.g. time in ROI, mean speed).

- **Visualization (front-end) / interacting with data**
  - **Verification purpose** — did the model/feature do something sensible? Are parameters reasonable?
    - Timeline of predictions for supervised models, synced with video playback.
    - For unsupervised: egocentric crop segments per cluster, partially supported already via mixed-feature workflows (`egocentric-crop`, `viz-global-colored`, `viz-timeline`).
  - **Summarizing** — per sequence, per group, per UID.
    - Basic common-denominator metrics: region-of-interest time, speed, proportion-based summaries.
  - **Export** — summary, chosen results.
    - CSV export as the primary "anything you want" output path (open in Excel, pandas, R, etc.).
  - **Overall, needs more thought** — huge variance expected between scientific users (prefer raw files + their own notebooks) and applied-research users (want in-app dashboards).
    - Max's concern: built-in stats invites misinterpretation; keep it minimal.
    - Plan to gather user feedback before investing heavily here.

## Not in Minimal V1

- **User roles (e.g. who can edit ethogram)** — fine-grained permissions. In MVP everyone can do everything within their own scope. Post-MVP: project-level admin roles, invite-normal-user flows, role-gated actions like ethogram editing.

- **Custom-made, integrated pose annotation** — building our own in-app annotation tool. MVP uses CVAT shipped alongside.

- **"Convert" pose model workflow** — converting models/annotations across ecosystems (e.g. DeepLabCut → Ultralytics, or Lightning Pose single-animal → Ultralytics multi-animal via pseudo-annotations from tracks). Partially exists in repo but won't be extensively supported in V1. Message to users who come from DLC/SLEAP: retrain in Ultralytics rather than porting.

- **Non-TRex tracker built-in (e.g. Ultralytics)** — a lightweight single-animal / single-object tracker using Ultralytics' built-in tracking. Easy to add post-MVP (basically a single call; scripted path already exists in the tracking package) but not needed for V1.

- **Custom identity model for individuals** — a Mosaic-native identity network. `TRexIdentityNetwork` exists in `behavior/model_library/` as transitional; its long-term placement is undecided. T-Rex handles identity for MVP.

- **Model predictions shown and can be corrected in scoring tool (human-in-loop)** — load model predictions onto the scoring timeline, correct intervals, save, retrain. Discussed as the "most used feature" once it exists, but out of V1 scope. Related deferred items:
  - Confidence visualization on the timeline.
  - Threshold dragging on scalar features.
  - Cluster-index visualization (k-means, keypoint-MoSeq) as timeline categories (bypasses ethogram).

- **Other dataset types: anonymous (tracklets), continuous**
  - MVP supports sequence + individual-ID-based only.
  - Deferred: tracklet-based (no stable individual IDs), continuous (segment-defined, uses `segment_duration`/`time_column`), tracker-based continuous, anonymous / detections-only aggregate analyses. Repo already has `dataset_type="discrete"|"continuous"` plumbing for eventual support.

- **Lambda-AWS-type scaling (distributing jobs)** — cloud scale-out for heavy features (Feral especially). Transfer dataset in, compute on ephemeral node, transfer results back. Would enable 10-users-simultaneously and parallel execution across nodes. Flagged as "likely the most important thing to add right after MVP" if queueing becomes a bottleneck.
