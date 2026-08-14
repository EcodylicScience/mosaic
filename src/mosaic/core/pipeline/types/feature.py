from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Literal, Protocol, final

import pandas as pd

from mosaic.core.pipeline.types.inputs import InputsLike
from mosaic.core.pipeline.types.params import Params

DependencyLookup = dict[tuple[str, str], Path]

type EmitsLevel = Literal["individual", "pair", "unidentified", "as-input"]
"""At what entity level a feature's output is keyed. See ``Feature.emits``."""


@final
class InputStream:
    """Factory for fit() input iterators, with entry count metadata.

    Wraps a callable that produces ``(entry_key, DataFrame)`` iterators.
    Each call creates a fresh iterator over the manifest entries.
    ``n_entries`` exposes the total number of entries so features can
    make exact allocation decisions (e.g. train/test split counts)
    without an extra data pass.
    """

    def __init__(
        self,
        factory: Callable[[], Iterator[tuple[str, pd.DataFrame]]],
        n_entries: int,
    ) -> None:
        self._factory = factory
        self._n_entries = n_entries

    @property
    def n_entries(self) -> int:
        return self._n_entries

    def __call__(self) -> Iterator[tuple[str, pd.DataFrame]]:
        return self._factory()


class Feature(Protocol):
    """Feature protocol -- 6 attributes, 4 methods."""

    name: str
    version: str
    parallelizable: bool
    scope_dependent: bool
    accepts_overlap: bool
    """Whether ``apply`` may be handed rows from the neighbouring sequences.

    Only asked when a run sets ``overlap_frames > 0``, which is how a feature
    reads across the boundary between the time divisions of one continuous
    recording -- a rolling window, a backward difference, a wavelet, all of which
    otherwise return an edge artifact at every division.

    The cost of saying yes is that ``group`` and ``sequence`` stop being constant
    down the frame. The loader documents them as constant (it is why they are not
    ``ALIGN_COLS``), and a good deal of the library relies on that without saying
    so: a feature that reads its identity from row 0 stamps its neighbour's name
    onto every output row, and one that opens media for that identity reads the
    wrong video. So this is ``True`` only where ``apply`` reads nothing from the
    frame that is true of one entry alone.

    It has no default, for the reason ``scope_dependent`` has none: a default
    would let the next feature ship without answering, and the wrong answer is
    silent in one direction.
    """

    emits: EmitsLevel
    """At what entity level this feature's output is keyed.

    What lets a chain be checked **before it runs**. ``alignment_verdict``
    decides whether two inputs can be joined, and it decides from the identity
    columns each one carries -- so on a produced parquet it needs no declaration
    at all. Before anything has run there is no parquet, and the commonest real
    mistake is exactly the one that check exists to catch: joining an
    individual-level output to a pair-level one shares no identity column, so
    the merge pairs every row of one with every row of the other. Declaring the
    level is what moves that from a runtime surprise to a refused connection.

    The four values, and which identity columns each one means:

    - ``"individual"`` -- one row per ``(frame, id)``, or per ``id`` for a
      per-sequence summary. ``speed-angvel``, ``nearest-neighbor``, ``arhmm``.
    - ``"pair"`` -- one row per ordered or unordered pair, whatever the pair
      columns are spelled. The ``pair-*`` family, ``orientation-rel``,
      ``interaction-crop-pipeline``.
    - ``"unidentified"`` -- no per-animal identity at all: a per-frame or
      per-chunk aggregate over everyone present. ``collective-motion-metrics``,
      ``frame-aggregate``.
    - ``"as-input"`` -- whatever came in goes out. The global fitters augment
      the frame they were given rather than re-keying it, and ``temporal-stack``
      and ``feral`` branch on whether their input carries a pair.

    **No default**, for the reason ``scope_dependent`` and ``accepts_overlap``
    have none: a default would let the next feature ship without answering, and
    the wrong answer here is silent in the dangerous direction -- a
    pair-producing feature that read as passthrough would have its cartesian
    join *permitted* rather than refused.

    Declare ``"as-input"`` only where the level genuinely follows the input.
    A feature that always produces the same level declares that level, even
    where its only legal input happens to share it.
    """

    consumed_roots: tuple[str, ...]
    """The **source roots this feature opens directly**, outside its inputs.

    The obvious reading is wrong, so read this one. It is not "the roots my data
    came from": a feature consuming ``"tracks"`` declares ``()``, because
    ``tracks/`` is a *derived* root and the manifest already hands it the tables.
    It is the roots the feature reaches past its inputs to read -- today, the two
    that open video through ``resolve_media`` / ``MultiVideoReader``.

    **Why a tracks-consuming feature declares nothing.** If it carried
    ``tracks_raw``'s composition, a change under ``tracks_raw`` would move its
    identifier without the tracks parquet having changed a byte -- a false
    invalidation, and precisely the "couple every per-sequence feature to the
    whole dataset" hazard the media-storage decision note warns about.

    **Not transitive through the variant identity, which this used to claim.** A
    change under ``tracks_raw`` re-produces the tracks table but does *not* move the
    tracks variant identity: that identity is params-only, and ``tracks_identity``
    says so -- "this names the recipe, not the input". So the ``_tracks`` hash term
    is byte-identical across a re-conversion of changed bytes, and nothing about the
    identifier notices.

    What closes it is a recorded comparison rather than an identifier: the tracks row
    already carries the source composition its table was converted from, and a
    feature row carries that value forward in ``consumed_tracks_composition``, so the
    per-entry cache check compares the two. Declaring ``("tracks_raw",)`` would have
    closed it by false invalidation instead, which is the wrong trade -- an honest
    miss beats a confident wrong value, but a *spurious* miss is neither.
    """

    @property
    def inputs(self) -> InputsLike: ...

    @property
    def params(self) -> Params: ...

    def load_state(
        self,
        run_root: Path,
        artifact_paths: dict[str, Path],
        dependency_lookups: dict[str, DependencyLookup],
    ) -> bool: ...

    def fit(self, inputs: InputStream) -> None: ...

    def save_state(self, run_root: Path) -> None: ...

    def apply(self, df: pd.DataFrame) -> pd.DataFrame: ...
