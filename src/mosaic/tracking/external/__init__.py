"""The environments Ultralytics-backed tracking runs in, and their programs.

Ultralytics and the POLO fork are AGPL-3.0, so mosaic drives them the way it
drives TRex, SLEAP, Lightning Pose and keypoint-MoSeq: as a separate program in
an environment the user builds, reached by exchanging files and command-line
arguments.

:mod:`mosaic.tracking.ultralytics_track.run` locates the environment and
launches ``runner/ultralytics_runner.py`` inside it, exchanging a JSON request
file, a JSON response file and progress lines on standard output.

Tracking is what runs out of process. :mod:`mosaic.tracking.pose_training` --
YOLO and POLO training, and single-model inference -- still imports Ultralytics
in mosaic's own process.

``runner/`` is a sibling of the environment directories rather than inside one,
because ``ultralytics-env/`` and the POLO environment beside it install the same
two files. See ``README.md`` here.
"""
