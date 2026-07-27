"""The plain-video probe entry point, in the shape the media index stores.

:func:`mosaic.core.media.imgstore_io.imgstore_probe` is this module's
counterpart for a store directory: both return a
:class:`~mosaic.core.media.facts_columns.ProbeMetadata`, so a caller assembles a
media-index row the same way whichever media type it holds. It lives here rather
than in ``facts_columns`` (which maps facts to cells and probes nothing) or in
``dataset``: its two consumers are ``dataset`` itself and
:mod:`mosaic.core.media.reprobe`, and the latter sits below the dataset layer, so
leaving it there would have made a re-probe import ``Dataset`` to reach a probe.
"""

from __future__ import annotations

from pathlib import Path

from mosaic_media import CHROME_149, MediaFacts, derive, probe_media

from mosaic.media_probe_config import media_thresholds

from .facts_columns import ProbeMetadata, facts_to_row


def row_from_facts(facts: MediaFacts) -> ProbeMetadata:
    """Build the media-index ProbeMetadata from an already-measured MediaFacts.

    Stores CODED width/height (un-oriented); get_video_metadata returns display
    dims, and row_to_facts injects coded dims + rotation so the reader orients
    once. Shared by the probe path and the injection path so the row shape is
    constructed in exactly one place.
    """
    verdict = derive(facts, CHROME_149, media_thresholds())
    return {
        "width": facts.width,
        "height": facts.height,
        "fps": facts.fps,
        "codec": facts.codec_name,
        **facts_to_row(facts, verdict),
    }


def probe_video_metadata(path: Path) -> ProbeMetadata:
    """Probe *path* and build its media-index row. Raises MediaProbeError on an
    unreadable file."""
    return row_from_facts(probe_media(path))
