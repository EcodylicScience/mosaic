"""The environments Ultralytics-backed tracking runs in, and their programs.

Ultralytics and the POLO fork are AGPL-3.0, so mosaic drives them the way it
drives TRex, SLEAP, Lightning Pose and keypoint-MoSeq: as a separate program in
an environment the user builds, reached by exchanging files and command-line
arguments.

The separation is not yet complete.
:mod:`mosaic.tracking.ultralytics_track.run` still imports Ultralytics in four
function bodies and tracks in mosaic's own process; ``runner/`` is what it is
rewired to spawn instead, and this paragraph goes when that lands.

``runner/`` is a sibling of the environment directories rather than inside one,
because ``ultralytics-env/`` and the POLO environment beside it install the same
two files. See ``README.md`` here.
"""
