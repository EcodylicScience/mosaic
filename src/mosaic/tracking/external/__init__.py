"""The environments Ultralytics-backed work runs in, and their program.

Ultralytics and the POLO fork are AGPL-3.0, so mosaic drives them the way it
drives TRex, SLEAP, Lightning Pose and keypoint-MoSeq: as a separate program in
an environment the user builds, reached by exchanging files and command-line
arguments.

:mod:`mosaic.tracking.common.ultralytics_env` locates either environment and
launches ``runner/ultralytics_runner.py`` inside it, exchanging a JSON request
file, a JSON response file and progress lines on standard output.

**Two environments.** POLO ships under the distribution name ``ultralytics``, so
it and upstream cannot occupy one: ``ultralytics-env/`` runs the tracker and
everything upstream-backed, ``polo-env/`` runs the fork's point detection.
``runner/`` is a sibling of both rather than inside either, because they run the
same program and which one runs is chosen by the interpreter mosaic spawns.

**Every path that reaches Ultralytics reaches it here** -- tracking, single-model
inference and model training alike -- so no mosaic install carries it and nothing
under ``src/mosaic/`` imports it outside ``runner/``. See ``README.md`` here.
"""
