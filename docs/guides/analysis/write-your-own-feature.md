# Write your own feature

The feature library is a registry, not a fixed menu. A feature you write registers
through the same decorator the shipped ones use, and from that moment it chains,
caches, gets a `run_id` and appears in the generated reference exactly like them.

Two templates in the library are the starting point. Copy one, rename the class and
its `name`, and fill in the logic:

- [`feature_template__per_sequence.py`](https://github.com/EcodylicScience/mosaic/blob/main/src/mosaic/behavior/feature_library/feature_template__per_sequence.py)
  — a stateless transform: reads one sequence, writes one sequence. `speed-angvel` is
  the real feature to read beside it.
- [`feature_template__global.py`](https://github.com/EcodylicScience/mosaic/blob/main/src/mosaic/behavior/feature_library/feature_template__global.py)
  — fit once over a collection, then apply per sequence. `global-tsne` and
  `global-ward` are the real ones.

Neither is registered: the decorator is commented out and neither is imported by the
package, so copying one does not add a half-finished feature to the registry.

## The protocol

Nine attributes and four methods.

| Attribute | Answers |
| --- | --- |
| `name` | The registry slug, as typed at `--feature` |
| `version` | Bump it when the meaning of the output changes; it re-addresses every run |
| `category` | Which group it appears under in the reference and in a diagram |
| `parallelizable` | Whether sequences may be processed concurrently |
| `scope_dependent` | Whether the output depends on which sequences were in scope |
| `accepts_overlap` | Whether `apply` may be handed rows from neighboring sequences |
| `emits` | At what entity level the output rows are keyed |
| `follows_ids` | Whether `apply` follows `id` across frames; `True` unless it reads each row alone. A feature that declares nothing reads `True` |
| `consumed_roots` | Source roots the feature opens directly, past its inputs, and `arenas`, the one non-source root, which carries an entry's layout digest and is never blind |

```python
def load_state(self, run_root, artifact_paths, dependency_lookups) -> bool: ...
def fit(self, inputs: InputStream) -> None: ...
def save_state(self, run_root: Path) -> None: ...
def apply(self, df: pd.DataFrame) -> pd.DataFrame: ...
```

A stateless feature returns `True` from `load_state`, leaves `fit` and `save_state`
empty, and does all its work in `apply`. A global one restores a fitted model in
`load_state`, builds it in `fit`, persists it in `save_state`, and maps with it in
`apply`.

A feature that needs a package outside the base install declares it in a
`prerequisites` tuple, and imports it through the same declaration:

```python
from mosaic.optional_dependency import OptionalModule

MOVEMENT = OptionalModule("movement", "movement", "the movement-library features")

class MovementSmooth:
    prerequisites = (MOVEMENT,)

    def apply(self, df):
        _ = MOVEMENT.load()
        ...
```

The arguments are the import name, the mosaic extra that installs it, and what needs
it. Where the package is not installed, the pipeline catalog lists the feature as
unavailable, and `mosaic pipeline validate` reports a step running it as a problem
whose message says what to install.

## `emits` is the one to get right

`emits` has no default, and it is the one where a wrong answer is silent.

It declares what your output rows are keyed by — `"individual"`, `"pair"`,
`"unidentified"` (a per-frame aggregate over everyone present), or `"as-input"`
(whatever came in goes out). It is what lets a chain be refused *before it runs*: a
pair-level output joined to an individual-level one shares only `frame`, so the merge
is a cartesian product rather than an alignment. A pair-producing feature left
declared as individual would have exactly that join permitted.

Declare `"as-input"` only where the level genuinely follows the input. A feature that
always produces the same level declares that level, even when its only legal input
happens to share it.

### What a pair row looks like

Declaring `"pair"` commits you to one row per **ordered** pair per frame: `id1` is the
focal, `id2` the other, and `perspective` says which of the two orderings the row is.
The key is `(frame, id1, id2, perspective)`. A symmetric quantity — a distance, a
mutual-interaction flag — is written the same on both rows rather than collapsed into
one.

Two habits to avoid, because neither raises:

- Writing the same `id1`/`id2` on both perspectives. `id1` then means the focal on
  some features and the lower id on yours, and a merge between them binds every row
  to the wrong perspective while silently dropping half of one side.
- Rebuilding your output frame and copying the identity across by name. Use
  `meta_columns(df)` from `feature_library/helpers.py`; a hand-written list is how
  `perspective` gets left behind, and without it the rows downstream cannot be told
  apart at all.

## Parameters are a Pydantic model

Never a bare dict. Add typed fields with defaults on the nested `Params` class; the
`run_id` hash covers them, so every field affects reproducibility — except one tagged
`Annotated[T, HASH_EXCLUDE]`, which is for throughput knobs like a batch size that
should not bust the cache when retuned.

Do not reuse one field for two meanings across versions. Bump `version` instead.

## Registration

A decorator only runs when its module is imported, and **mosaic discovers nothing on
its own** — no entry-point scan, no import hook. Where the module lives decides what
can reach it:

- **In the tree.** Put it in `feature_library/`, uncomment `@register_feature`, and
  add the import line to `feature_library/__init__.py`. It is then reachable from
  `mosaic run --feature`, appears in `mosaic features list`, and lands in the
  generated reference.
- **In a notebook.** Import your module before you build the pipeline and it
  registers on import. This is Python-API only. A fresh `mosaic` process does not
  import it, and the CLI cannot see it.
- **In a dataset.** [Define the feature](#define-a-feature-in-a-dataset) from one
  file, and every process that opens the dataset can run it.

## Define a feature in a dataset

A definition stores a feature's source in the dataset, as `def:<name>`. The CLI, a
recipe, the MCP server and every worker that opens the dataset then find it, with no
change to mosaic and no registration. Write one Python file that declares the feature
class, with the definition's name as the class's `name`, and store it:

```bash
mosaic features define my-feature my_feature.py --manifest study/dataset.yaml
```

From Python, `define_feature(ds, "my-feature", source_bytes)` in
`mosaic.core.pipeline.definitions.define` does the same. A name is lowercase kebab-case
of at most 64 characters and may not equal a built-in feature's. The file is imported
once in a child process and checked against the rules below as far as a check can test
them. The version bump, determinism in `apply` and the environments that other machines
run in stay the author's to keep. A file that fails is refused with every finding,
listed under `detail.findings` with `--json`.

Defining the same source again reuses its version. Other source adds a version and
makes it current. A new version gives new run ids, and the runs of earlier versions
stay readable. A feature's slug names which version runs:

- `def:my-feature` is the current version when a recipe is validated or planned. A
  plan pins the version's digest, and a request runs that version after a later
  definition moves `current`.
- `def:my-feature@<digest>` names one version, by 12 to 64 hex characters of its
  SHA-256, as `mosaic features list --manifest study/dataset.yaml` shows them.
  Pin a recipe this way to keep it on a version.

Use the slug as the feature of a step, or run it alone:

```bash
mosaic run -m study/dataset.yaml --feature def:my-feature
```

`mosaic features list`, `mosaic features describe def:my-feature` and
`mosaic pipeline validate` list, describe and check definitions when given
`--manifest`, and without it know only the built-ins. A run writes under
`features/def.my-feature__from__tracks/`, since `:` is not allowed in a folder name on
every platform. Definitions are unavailable to a `Dataset` built in code with roots
that omit `feature_definitions`.

### Rules for the code

Start from the per-sequence template above, with these differences:

- **One file, with absolute imports.** Write
  `from mosaic.behavior.feature_library.helpers import meta_columns`. A relative
  import such as `from .helpers import ...` fails. Leave `register_feature` out, as the
  template does. The file is UTF-8 of at most 1 MiB. The class whose `name` is the
  definition's name is the only one that counts, and other classes and helpers in the
  file are allowed. Do not declare `__mosaic_definition__`.
- **Import only what the environment has.** A definition runs wherever a step runs:
  your machine, the MCP server's, a queue worker's. A module missing there fails the
  step.
- **Be deterministic**, at import as well as in `apply`. A file that reads its
  environment while it is imported can declare different params in two imports, and
  its load then fails with `definition_mismatch`.
- **Bump `version` when the output's meaning changes.** An edit gives new run ids
  whether or not you bump it. The version is the run id's prefix and what a request
  checks to find that a producer moved.
- **Do not call `register_feature`.** An import that changes a registry (features,
  ops, converters, track schemas) is refused. A definition stays out of the registries.
- **Declare the flags as class constants**, outside `__init__`. Planning reads
  `parallelizable`, `scope_dependent`, `accepts_overlap`, `emits`, `consumed_roots`,
  `fits` and `costly_fit` from the class, and an instance that sets one runs under a
  different value than it was planned with. `parallelizable`, `scope_dependent`,
  `accepts_overlap`, `emits` and `consumed_roots` are required.
- **Give `temporal_reach` as an int or `None`.** A function of the params and a
  `temporal_reach_by_option` are refused, because planning reads the reach without
  the class.
- **Type the params from a closed set.** Every field has a default and is one of:
  `bool`, `int`, `float`, `str` or `None`; a `Literal` of those; a `list`, a `tuple`
  (`tuple[T, ...]` or a fixed length) or a `dict` with `str` keys of these types; a
  union of them; a toolkit config model such as `PoseConfig`; or a model defined in the
  file from the same types, subclassing `DeclaredModel` or `NestedParams`. A reference
  to another run (`Result`, `ArtifactSpec`, a labels source) is refused. Constraints
  (`gt`, `ge`, `lt`, `le`, `multiple_of`, `min_length`, `max_length`, `pattern`),
  `Declared` and `HASH_EXCLUDE` go at the top level of a field: `Annotated[list[str] |
  None, Declared(...)]`. `Optional[Annotated[list[str], Declared(...)]]` is refused.
  Describe a field with `Declared`, because `Field(description=...)` is refused. `Field`
  takes a default, a default factory and the constraints above, and any other argument
  is refused.
- **Leave the params models plain.** No validator, serializer, computed field, alias,
  `model_config`, `model_post_init` or `__init__`, on `Params` or on a model it
  contains. Planning rebuilds the model from its fields alone, and each of these
  changes validation or the dump outside them.
- **Subclass the toolkit's `Inputs`** with items of `TrackInput`, `Result` or both,
  and declare nothing on it but `_require` and `_track_input`. The per-sequence
  template declares `Inputs[TrackInput | Result]`, which reads raw tracks, another
  feature's output or both. `TrackInputs` is for a feature that needs the track frame
  (`X` and `Y`). A recipe refuses it after a feature that writes neither, such as
  `keypoint-kinematics`, so a definition that reads such a feature's columns declares
  `Inputs[TrackInput | Result]` or `Inputs[Result]`.

The class must also build as `MyFeature(MyFeature.Inputs(...), None)`, as the template
does.

### Trust

A definition is Python, run with every privilege of the process that runs a step naming
it: your account at a workstation, the server's under the MCP server, the worker's
identity on a queue. It runs outside any sandbox. The check at definition runs in a
child process, which keeps your session clean and isolates nothing else.

- Whoever can write a dataset's `feature_definitions/` folder runs code as whoever
  later runs a step naming `def:` on it. A recipe pinned to `def:<name>@<digest>`
  runs that version only, and one naming `def:<name>` runs whichever version is
  current when it is planned.
- Opening, listing, describing, validating and planning a dataset you received runs none
  of its code. Running or verifying (`mosaic pipeline verify`) a recipe that names a
  `def:` step does, and loads only that step's definition at its pinned version. Read
  `feature_definitions/<name>/<date>_<digest>.py` first. `mosaic features list
  --manifest` shows the digest.
- Your CLI and Python calls run without a flag, because you already run code as
  yourself. The MCP server runs definitions only when its operator
  [allows them](../../agents.md#feature-definitions).

## Determinism

Identical parameters and inputs must produce an identical `run_id` and identical
output. No unseeded random state, no iteration order that depends on a dict built
from a filesystem walk, no wall-clock in the output. The cache is only trustworthy
because this holds.

## Write it out with `run_feature`

Never write into `features/` by hand. `run_feature` owns the output layout, the index
row and the `run_id` registration, and a side-loaded file desynchronizes all three.
