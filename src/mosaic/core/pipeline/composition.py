"""What a sequence is made of, as one comparable value per source root.

A recipe name cannot carry source content -- that is rule P2, and it is why
``run_id`` names how an artifact was produced and never what it was produced
from. Source content is tracked separately, and this is where it is turned into
something comparable: a composition hash, per sequence, per source root.

``media_raw`` composes the ordered video uids of a sequence; ``tracks_raw`` the
checksums of its raw track files. They are deliberately separate values rather
than one hash over everything a sequence has, because a single value cannot
answer the question that matters. A track-only sequence gaining its first video
changes its media composition and nothing else, and everything built from its
tracks is untouched -- where one combined hash would have thrown away a feature
chain months deep to no purpose.

**No I/O here.** No pandas, no ``Dataset``, no ``mosaic_media``: this module
turns already-read rows into a digest and nothing else. That is what lets the
golden corpus pin these values as literals with no fixture, and what keeps the
storage half (:mod:`mosaic.core.pipeline.sequence_index`) free of the hashing
half -- the same split ``tracks_index`` and ``tracks_identity`` already have.

**The ordering rule is mixed, and a naive reading of item 0.5 gets it wrong.**
The outer list of ``(camera, uids)`` pairs is sorted, because two callers may
hand the cameras over in either order and mean the same thing. The inner uid list
is **never** sorted: it is ``video_order``, it is semantic, and sorting it would
make two different arrangements of one sequence hash alike -- which is the exact
change a composition exists to detect. ``identity_ready`` already preserves list
order and sorts only sets, so the rule survives by construction; the comment
there naming this item is the other half of the contract.

**Nothing here is ever a ``set``.** A set deduplicates at construction, before
``identity_ready`` ever sees it, so a set of uids would silently drop the second
of two byte-identical videos and a set of checksums the second of two identical
files. Both are real states -- two byte-identical videos in one sequence share
one ``video_uuid`` by design -- so cardinality has to survive, and only a list
carries it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from ._utils import hash_params

__all__ = [
    "MEDIA_COMPOSITION_SCHEME",
    "TRACKS_RAW_COMPOSITION_SCHEME",
    "MediaMember",
    "SequenceComposition",
    "SourceMember",
    "media_composition",
    "media_composition_payload",
    "source_composition_payload",
    "tracks_raw_composition",
]

MEDIA_COMPOSITION_SCHEME: Final = "1"
"""The contract :func:`media_composition` implements today.

Per family, like ``FEATURE_IDENTITY_SCHEME`` and ``TRACKS_IDENTITY_SCHEME``: one
number cannot honestly cover two independent hash functions over two different
shapes, and this one will gain terms the checksum composition will not. Bumped
for a change to what the digest covers, never for anything else, and recorded on
the row rather than folded into the payload -- a marker that changed the value it
marks would cause the event it exists to detect.
"""

TRACKS_RAW_COMPOSITION_SCHEME: Final = "1"
"""The contract :func:`tracks_raw_composition` implements today. See above."""


@dataclass(frozen=True, slots=True)
class MediaMember:
    """One video's place in a sequence, and what names it.

    ``uid`` is the row's ``video_uuid``: content plus exact timing, and the only
    value that identifies a single file. ``content_digest`` deliberately cannot
    serve -- it is timing-blind and not unique, so a composition over it could
    not tell two sequences apart that differ only in timing. ``""`` means the
    member carries no identity, which makes the whole composition unestablishable
    rather than partially guessed.
    """

    camera: str
    video_order: int
    uid: str


@dataclass(frozen=True, slots=True)
class SourceMember:
    """One raw source file: a within-sequence label and a checksum of its bytes.

    ``name`` is the basename rather than the stored path, so a sequence's
    composition survives the dataset moving between machines -- the same
    portability the index's root-relative ``abs_path`` buys, applied to a value
    that is hashed and therefore cannot be rewritten later.

    ``algo`` rides *inside* the hashed payload rather than beside it, following
    ``mosaic-media``'s rule for its own content digest: hashed whole, algorithm
    included, so changing the algorithm changes every composition instead of
    producing an incomparable one that looks comparable.
    """

    name: str
    digest: str
    algo: str


@dataclass(frozen=True, slots=True)
class SequenceComposition:
    """A sequence's composition under one root: the digest, and what it covered.

    ``member_count`` is load-bearing rather than diagnostic. It is the only thing
    that tells an empty ``digest`` meaning "three files are here and one carries
    no checksum" apart from one meaning "there is nothing here", and it is the
    single cell a human can read without recomputing anything.
    """

    digest: str
    member_count: int


def media_composition_payload(
    members: Sequence[MediaMember],
) -> dict[str, object]:
    """What determines a sequence's media composition.

    A named builder rather than a dict literal at the mint site, for the reason
    ``tracks_identity``'s three payload builders give: it lets the golden corpus
    pin the **wrapper**, not just the digest. Renaming ``"cameras"`` would move
    every stored composition in existence, and a corpus that called
    ``hash_params`` with a hand-built dict would stay green straight through it.

    The shape is a sorted list of ``[camera, [uid, ...]]`` pairs, ``camera=""``
    for single-camera media. Sorted outside, ordered inside -- see the module
    docstring. The outer sort is total because camera keys are unique by
    construction, so the comparison never descends into a uid list whose order is
    semantic.
    """
    by_camera: dict[str, list[MediaMember]] = {}
    for member in members:
        by_camera.setdefault(member.camera, []).append(member)
    return {
        "kind": "media_raw",
        "cameras": [
            [
                camera,
                [
                    member.uid
                    for member in sorted(group, key=lambda m: (m.video_order, m.uid))
                ],
            ]
            for camera, group in sorted(by_camera.items())
        ],
    }


def source_composition_payload(
    kind: str, members: Sequence[SourceMember]
) -> dict[str, object]:
    """What determines a sequence's composition over a root of raw files.

    ``kind`` is domain separation: the same set of members under two roots must
    not mint one digest, or a ``tracks_raw`` change would read as a ``labels_raw``
    one to anything comparing values without knowing which root it asked about.

    Raw sources have no within-sequence ordering -- no ``video_order``, no camera
    axis -- so the file list is sorted here, by the builder. It stays a list
    rather than a set so a legitimately duplicated file survives as two members.
    """
    return {
        "kind": kind,
        "files": sorted(
            [member.name, member.algo, member.digest] for member in members
        ),
    }


def _mint(
    payload: Mapping[str, object], members: int, complete: bool
) -> SequenceComposition:
    """Digest *payload*, or state honestly that it cannot be established."""
    if not complete:
        return SequenceComposition("", members)
    return SequenceComposition(hash_params(dict(payload)), members)


def media_composition(members: Sequence[MediaMember]) -> SequenceComposition:
    """Mint a sequence's ``media_raw`` composition.

    One member without a uid makes the whole composition unestablishable, not
    partially established: a digest over "the members that happened to have an
    identity" would compare equal to a genuinely different sequence whose missing
    member differed, which is a confident wrong answer where an empty is an
    honest one.
    """
    return _mint(
        media_composition_payload(members),
        len(members),
        all(member.uid for member in members),
    )


def tracks_raw_composition(members: Sequence[SourceMember]) -> SequenceComposition:
    """Mint a sequence's ``tracks_raw`` composition. Same completeness rule."""
    return _mint(
        source_composition_payload("tracks_raw", members),
        len(members),
        all(member.digest for member in members),
    )
