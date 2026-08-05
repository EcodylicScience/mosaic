"""Model library for behavior datasets.

Networks for visual individual identification, each reached by importing its own
module rather than from here. That is deliberate: ``mosaic.behavior`` imports
this package eagerly, and every network needs PyTorch, so exporting them would
make the optional ``identity`` extra a hard requirement for importing mosaic at
all. Each module imports torch lazily inside its methods, and
``tests/test_behavior_import_is_torch_free.py`` holds that line.

Available modules:

* :mod:`~mosaic.behavior.model_library.identity_classifier` --
  ``ClassifierIdentityNetwork``, a pretrained image backbone with a trained
  linear head over a closed set of animals.
* :mod:`~mosaic.behavior.model_library.identity_embedding` --
  ``EmbeddingIdentityNetwork``, the same family of backbones frozen, with
  identity decided by k-NN against per-identity prototypes. Trains nothing.
* :mod:`~mosaic.behavior.model_library.dinov2_temporal_identity` --
  ``DinoV2TemporalNetwork``, frozen DINOv2 per frame plus a trained temporal
  head over clips. The only one that sees time.
* :mod:`~mosaic.behavior.model_library.timm_backbone` -- backbone resolution,
  preprocessing, and device selection shared by the three.
* :mod:`~mosaic.behavior.model_library.identity_common` -- label mapping, crop
  loading, and the prototype/k-NN helpers shared by the two embedding models.

Usage
-----
>>> from mosaic.behavior.model_library.identity_classifier import (
...     ClassifierIdentityNetwork,
... )
"""
