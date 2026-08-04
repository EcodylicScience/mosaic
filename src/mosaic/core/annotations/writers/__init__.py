"""Serializing the canonical representation to interchange formats.

Interchange, not training. A writer here produces something another tool reads
as annotations -- COCO being the one most tools speak. The directory tree a
particular trainer walks is a different artifact, owned by the integration that
knows that trainer, and lives under ``tracking``.
"""

from mosaic.core.annotations.writers.coco import write_coco_keypoints

__all__ = ["write_coco_keypoints"]
