"""Labelled frames, as one representation every reader and writer agrees on.

A *reader* turns some external format into an :class:`AnnotationSet`; an
*emitter* turns one into a training dataset. Neither knows about the other, so
adding a source costs one reader and adding a target costs one emitter, rather
than one function per pair -- which is what the converter package became.

This lives in ``core`` because it is a representation rather than a pipeline
stage, and it imports nothing from ``tracking`` or ``behavior``: the emitters
that write YOLO and POLO layouts are a tracking concern and stay there. Reading
labels out of the control plane will land here too, which is why the levels are
named for what that side already calls them.
"""

from mosaic.core.annotations.model import (
    AnnotationFrame,
    AnnotationObject,
    AnnotationSet,
    Bbox,
    Keypoint,
    KeypointSchema,
    Split,
    Visibility,
)

__all__ = [
    "AnnotationFrame",
    "AnnotationObject",
    "AnnotationSet",
    "Bbox",
    "Keypoint",
    "KeypointSchema",
    "Split",
    "Visibility",
]
