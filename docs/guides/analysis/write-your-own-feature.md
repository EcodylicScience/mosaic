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

Seven attributes and four methods.

| Attribute | Answers |
| --- | --- |
| `name` | The registry slug, as typed at `--feature` |
| `version` | Bump it when the meaning of the output changes; it re-addresses every run |
| `category` | Which group it appears under in the reference and in a diagram |
| `parallelizable` | Whether sequences may be processed concurrently |
| `scope_dependent` | Whether the output depends on which sequences were in scope |
| `accepts_overlap` | Whether `apply` may be handed rows from neighboring sequences |
| `emits` | At what entity level the output rows are keyed |
| `consumed_roots` | Source roots the feature opens directly, past its inputs |

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
- **Out of the tree, or in a notebook.** Import your module before you build the
  pipeline and it registers on import. This is Python-API only: a fresh `mosaic`
  process never imports it, so the CLI cannot see it.

## Determinism

Identical parameters and inputs must produce an identical `run_id` and identical
output. No unseeded random state, no iteration order that depends on a dict built
from a filesystem walk, no wall-clock in the output. The cache is only trustworthy
because this holds.

## Write it out with `run_feature`

Never write into `features/` by hand. `run_feature` owns the output layout, the index
row and the `run_id` registration, and a side-loaded file desynchronizes all three.
