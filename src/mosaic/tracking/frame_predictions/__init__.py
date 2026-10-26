"""Stores pose-model predictions on extracted frames and their scores.

A predict-frames run stores one model's predictions on the frames of an extraction
run, one ``mosaic-pose-predictions`` file per entry camera, under the dataset's
``frame_predictions`` root. A score-predictions run compares predictions with
annotated keypoint sets. :mod:`.index` defines both indexes and :mod:`.read` the
readers. This file imports no submodule.
"""
