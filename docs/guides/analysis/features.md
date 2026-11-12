# Features

Everything mosaic computes is a **feature**: a registered plugin that reads
standardized tracks, or another feature's output, and writes parquet under
`features/<name>/<run_id>/`.

**The library is meant to be extended.** The features that ship with the
distribution are the ones that have been needed so far, not a closed set — they are
plugins registered through a decorator, and yours registers the same way. When the
quantity you want is not here, [write it](write-your-own-feature.md); it will chain,
cache and appear in the reference exactly like the shipped ones.

## Running one

```bash
mosaic run -m dataset.yaml --feature speed-angvel
```

or from Python, which is what a notebook does:

```python
from mosaic.core.dataset import open_dataset
from mosaic.behavior.feature_library import SpeedAngvel
from mosaic.core.scope import Scope

ds = open_dataset("dataset.yaml")
result = ds.run_feature(SpeedAngvel(), scope=Scope(entries=[("", "trial01")]))
```

If you run the exact same call twice, the second call computes nothing: the `run_id` is a hash of the
parameters, inputs and frame range, so identical work is already on disk. That is
also what makes a parameter sweep organize itself — see [Reproducibility, run_id and
caching](../../concepts/reproducibility.md).


## What the library holds

[The features reference](../../reference/features.md) is the complete list with every
parameter, generated from the registry. Grouped by the question they answer:

**Individual.** `speed-angvel` (speed and angular velocity),
`heading` (body orientation from keypoints, under a method you name),
`keypoint-kinematics` (smoothed speed and acceleration of every keypoint, distances
between keypoints, and with `heading` each keypoint's velocity and position in the
animal's own frame), `trajectory-smooth`, `body-scale`, `body-length` (length along a
chain of keypoints, or one a tracker measured), `scale-to-cm`, `scale-to-bl` (lengths in
body lengths, for comparing animals of different sizes), `individual-wavelet` (the log
power of chosen columns at a band of frequencies), `track-subsample`, and, for 3D tracks
in the world frame, `pose-egocentric` (each keypoint in the animal's own body frame,
from a left-right pair or a front-rear pair of keypoints).

**Whatever came in.** `temporal-stack` (gives a frame its context window) and the
global fitters below key their output the way their input was keyed, so they sit at
whichever level you feed them.

**Pair.** One row per ordered pair per frame: the `pair-*` family — `pair-position`,
`pair-egocentric`, `pair-keypoints` (each partner keypoint's position and velocity in
the focal's body frame, and chosen distances between the two animals' keypoints),
`pair-facing`, `pair-wavelet`, `pair-posedistance-pca`, `pair-interaction-filter` — plus
`orientation-rel` and `approach-avoidance`. For 3D tracks in the world frame,
`pair-world` measures distance, bearing, elevation, relative heading and approach speed
in meters and radians. `visual-network` says which individuals each one can see, by
casting rays from its eyes against every body outline, and writes only the visible
pairs. Set its `include_hidden` before `temporal-stack` or `frame-aggregate`, which
expect every pair in every frame.

**Neighbors and social forces.** `nearest-neighbor` and its `nn-delta-response` /
`nn-delta-bins` chain, and `attention-target`. These describe an individual's
relationship to the others around it, so their rows are keyed per individual, not per
pair. `neighbor-delta-response` writes the same response against every neighbor, one
row per focal and neighbor, and `nn-delta-bins` maps either: where neighbors sit
(`occupancy`), and how the focal turns and speeds against them, within a
`start_time`/`end_time` window. Its `time_shuffle` gives the chance-level map.

**The group as a whole.** `collective-motion-metrics` (polarization and rotation
order parameters, and the discrete states they imply), `local-order-metrics`,
`ffgroups` and `ffgroups-metrics`, `frame-aggregate`, `social-motion-summary`.
`frame-aggregate`, `sequence-aggregate` and `individual-aggregate` reduce columns with a
shared set of parameters: to one row per frame, per sequence or per individual. Each of
`columns` is reduced by each of `aggs` into `<column>_<agg>`. An input with an `arena`
column gives one row per arena, and `subgroup_col` splits the rows by a column such as a
tag. Over pair rows, `individual-aggregate` reduces every row of a focal `id1` to that
individual's row.

**Responses to events.** `event-response` measures a value, by default the speed of
`speed-angvel`, around each onset, one row per individual and event: the baseline mean
over `baseline_s` before it, the minimum and mean over `response_s` after it, and the
latency from the onset to the first value at or above `recover_fraction` times its
baseline after the response window's minimum (NaN, with `recovered` false, when it does
not; `recovered` is null when the baseline or the response has no value). A single low
sample sets the minimum, so filter a noisy value first: `speed-angvel` writes
`speed_smooth` when `smooth_window` is set (use it as `value_column`), and
`trajectory-smooth` smooths the positions before `speed-angvel`. `event` counts an
individual's own onsets under `onset_column` and `labels`; `onset_time` joins one
stimulus across individuals. Give the onsets as `onset_times`, as the rising edge of
`onset_column` (`onset_value` names the active value of a text column, such as `Y` for
Tracktor's `stimuli`), or as the runs of `onset_classes` in a `labels` kind. The windows
and the latency follow the `time` column in seconds, so a table that dropped frames is
measured the same way. `individual-aggregate` then summarizes the rows.

**A label sequence as a fingerprint.** `label-statistics` reduces labels to one row per
individual: how much of the time each state occupies, how often it starts, how long its
bouts last and what tends to follow it. It reads ground-truth labels or a categorical
column such as AR-HMM's `syllable`, so the same row compares annotated and discovered
behavior.

**Models fitted over a collection.** t-SNE, k-means, Ward, AR-HMM, keypoint-MoSeq,
VAME and the supervised classifiers. See [Find behaviors without
labels](discover-behaviors.md), [Train a behavior classifier](train-a-classifier.md)
and [Identify individuals by appearance](identify-individuals.md); each feature is
listed with its parameters under *Global* in the
[features reference](../../reference/features.md).

**Image and video artifacts.** `egocentric-crop` and `interaction-crop-pipeline` cut
animal-centered clips; `overlay` renders an annotated video. See
[Render an annotated video](../media/render-a-video.md).

## Features and track spaces

A feature declares the spaces it computes in. Tracks are in pixels, centimeters or, for
3D data, world meters (see [what a tracker
reports](../../concepts/tracks.md#spatial-columns-are-in-one-of-three-spaces)). Most
features are planar and refuse a world-frame table; `pose-egocentric` and `pair-world`
refuse a planar one. The refusal is `schema_family_mismatch`. It comes before anything
runs, from a run and from a recipe when it is planned or, for tracks the recipe itself
writes, when the step starts. Features whose computation does not depend on the space
(`keypoint-kinematics` measures in the table's own units, meters on a world table) and
features that read no spatial column (the aggregates) accept every space.

## Chaining

A feature reads tracks by default, or another feature's output when you pass the
`Result` its run returned:

```python
from mosaic.behavior.feature_library import TrajectorySmooth
from mosaic.core.pipeline.types import Inputs

smoothed = ds.run_feature(TrajectorySmooth())
ds.run_feature(SpeedAngvel(inputs=Inputs((smoothed,))))
```

A hand-written `Result` names the upstream's storage directory under `features/`,
`<slug>__from__<inputs>` for a feature that reads inputs, such as
`Result(feature="trajectory-smooth__from__tracks")`. A bare slug names no stored run
and is refused with `UnrunFeatureInput`.

The upstream identity enters the downstream `run_id`, so a result computed from
smoothed tracks can never be mistaken for the same feature computed from raw ones.

For chains of more than two steps, and for chains that also contain tracking or media
work, use a pipeline: [Chain steps into a recipe](../pipelines/chain-steps.md).

## Summaries across entries

To compare groups of recordings or individuals, such as strains or sexes, reduce each
recording or individual to one row and then summarize the rows by group. A feature run
summarizes by its rows, so a run of one row per frame is not a candidate.

**Keep the labels.** `frame-aggregate`, `sequence-aggregate` and `individual-aggregate`
write only their reduction keys and their aggregates. `keep_columns` carries columns of
the input to each output row, after the reduction keys, so a label such as `strain`
reaches the rows that are compared. A kept column must take one value over the rows
reduced together, null counting as a value. A group that disagrees, or an input
without the column, fails the entry and the run lists it in `failed_entries`. A label
of the tracks table, such as Tracktor's `strain`, comes through by listing `"tracks"`
after the feature's output among the aggregate's inputs:

```json
{"id": "by_recording", "type": "feature", "feature": "sequence-aggregate",
 "inputs": [{"step": "speed"}, "tracks"],
 "params": {"columns": ["speed"], "aggs": ["mean"],
            "keep_columns": ["strain", "sex"]}}
```

The tracks table comes second, so a column that both hold is read from the feature. A
plan cautions that it cannot check a label of the tracks table, and refuses a kept
column that no input can provide, when the tracks table is not among the inputs. The
feature's output cannot take `"tracks"` beside it when it is pair-level, so a tracks
label does not reach a pair feature's aggregate: key it by a dataset group or by a
column that the pair feature's own input carries.

**Summarize.** In Python:

```python
from mosaic.core.pipeline import summarize_run

summary = summarize_run(
    ds,
    result,
    by=["strain", "sex"],
    columns="speed_mean",
    tests=["kruskal"],
    where={"sex": ["female", "male"]},
)
summary.groups, summary.statistics, summary.kruskal
```

On the command line, with the run's ref as `mosaic inventory` prints it:

```bash
mosaic summarize feature/<storage>/<run_id> -m dataset.yaml \
    --by strain --by sex --column speed_mean --test kruskal \
    --where sex=female --where sex=male
```

`by` takes `group`, `sequence`, an identity column the run writes or any other column it
writes, such as a kept label. Several keys cross into one grouping. `where` keeps the
rows whose key takes one of the values listed, and a key of `where` need not be in
`by`. `--json` prints the same summary as `RunSummaryWire`, and `--allow-partial` reads
a run that lacks entries or is not finished. Rows with a null key are counted in
`rows_without_key`, and the entries that have no row in `entries_without_rows`.

A `where` value from the command line, or a string in a `where` of the MCP tool, is read
as text and as a number or a boolean where the key holds one. `--where n_inds=1` keeps
the integer 1, `--where code=1` keeps the text `"1"` of a text key, and `true` and
`false` match a boolean key. A JSON number in an MCP call, or a Python number, matches an
integer or float key alike, and a boolean matches a boolean key. `where` in Python does
not read a string as a number or a boolean.

**The statistics.** Each group and column has `n`, `n_missing`, `mean`, `sd` (`ddof=1`),
`sem` and `median`. A NaN or infinite cell is missing, so one infinite value does not
turn a mean infinite. Each column is handled alone: a row missing one column still
counts in another. `mean` and `median` are NaN when `n` is 0, and `sd` and `sem` when
`n` is below 2.

**The tests.** `kruskal` is Kruskal-Wallis across all groups of a column. `mannwhitney`
is a two-sided Mann-Whitney test for every pair of groups, with `p_holm`, Holm's
step-down adjustment over the pairs of that column. Columns are not adjusted against
each other: a table of many columns holds an uncorrected family, and the correction
across them is yours. A test row without a statistic names its `reason`:
`too_few_groups`, `no_values` or `no_variation` (every value the test reads is equal).
`mannwhitney` over more than 12 groups is refused before it runs, since 66 pairs per
column leave little power. Narrow with `where`, group by fewer keys, or run `kruskal`
alone.

**The unit of analysis.** A row is one observation: a recording for `sequence-aggregate`
and an individual for `individual-aggregate`. A run with a `frame` or a `time` column is
refused, because frames of one recording are not independent and a test over them
returns p values near zero. Reduce it first. Rows of one recording are not independent
either. When a recording holds several rows (`max_rows_per_entry` above 1) and a key is
constant within it, as a strain is, the tests count each individual of a recording as a
replicate of its strain. `n_entries` beside `n_rows` shows how many recordings stand
behind a group, so read it before reading a p value.

**Through MCP.** `preview_output` with `mode: "summary"` takes `by`, `tests`, `where`,
`columns`, `groups` and `allow_partial` and returns the same summary. A summary has no
default bound: about 265 characters per group and column, so about 150 groups of one
column fit under the 40,000 characters a result may hold. The tests add to that.
`mann_whitney` takes about 190 characters per pair and column with one short key, and
about 230 with three, more for longer key text. Its limit of 12 groups makes 66 pairs,
so 12,700 to 15,000 characters per column, and three columns of 12 groups exceed a
result. `kruskal` adds a small row per column. A larger design is refused with
`result_too_large`, which names `where`, `columns` and `by`. The command line and
`summarize_run` return it whole.

**What this does not do.** Variance components, mixed models, heritability, interaction
terms and other multi-factor designs are not part of mosaic. A summary crosses its keys
into one grouping, and its tests compare those groups as the levels of one factor. For
a factorial analysis, or a test per frame, read the rows in your own code with
`load_run_matrix`. To read a label beside the values, call `plan_run_read(..., keys=[...])`
and `read_run`, which `load_run_matrix` calls in turn, and the keys come back in
`.index`.

## Filled positions

`trajectory-smooth` fills every missing or speed-flagged centroid position by
interpolation, so features that need continuous positions (groups, distances,
fission-fusion events) can run, and it marks each row it filled:

- `interpolated` and `interp_run_frames`, the length of the filled run;
- `unbridgeable`, a run whose straight line between the positions on either side is
  faster than `speed_threshold`, usually a jump between two animals;
- `long_interp`, a run longer than `long_run_frames`.

Exclude the marks where per-frame values become per-animal numbers. Put the
`trajectory-smooth` step among the summary's inputs and name the marks in its
`exclude_cols`, as `social-motion-summary` and `ffgroups-metrics` take them.
`social-motion-summary` refuses a recipe whose inputs lack a column it names. With
`exclude_scope="frame"`, one flagged animal excludes its whole frame in its arena.

## The rule that catches most mistakes

**Inputs must align at one entity level.** A per-individual result has one row per
individual per frame; a pair result has one row per *ordered* pair per frame. They
share only `frame`, so merging them would pair every individual with every pair and fit
downstream on rows that never existed.

mosaic refuses that merge and names both levels rather than performing it. A
classifier, a global model or a pair feature reads per-individual results beside a
pair result put first: each pair row gets the focal's columns as `<column>__id1` and
the other's as `<column>__id2`. [Individual results on pair
rows](../../concepts/features.md#individual-results-on-pair-rows) gives the rules.

Two pair results join one-to-one only when both carry `perspective`, which is what
separates the two orderings of a pair. [What identifies a pair
row](../../concepts/features.md#what-identifies-a-pair-row) has the key and why every
column of it matters; [Features and composition](../../concepts/features.md) has the
tracks-variant rule.
