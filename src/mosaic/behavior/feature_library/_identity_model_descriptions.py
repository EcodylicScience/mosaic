"""Field prose shared across the crop-based identity models.

:class:`~mosaic.behavior.feature_library.dinov2_temporal_identity_model.GlobalIdentityDinoV2Temporal`
and
:class:`~mosaic.behavior.feature_library.identity_embedding_model.GlobalIdentityEmbedding`
both train from
:class:`~mosaic.behavior.visualization_library.egocentric_crop.EgocentricCrop`
output. They share the identity-selection, crop-reading and
checkpoint-naming fields below, the text living here once rather than
once per class.

LEARNING_RATE_DESCRIPTION additionally serves
:class:`~mosaic.behavior.feature_library.identity_model.GlobalIdentityModel`,
the third crop-based identity model: all three train with Adam and describe
its learning rate the same way. GlobalIdentityModel's identity-selection,
crop-reading and checkpoint-naming fields describe a different fallback
order and keep separate text under ``_CLASSIFIER_*`` names in that module --
they must not share a name with the ones below.
"""

from __future__ import annotations

IDENTITIES_DESCRIPTION = (
    "An explicit mapping of identity name to the sequences containing "
    "that individual alone. Takes precedence over group_as_identity."
)

GROUP_AS_IDENTITY_DESCRIPTION = (
    "Treat each sequence's group name as its identity, instead of "
    "listing sequences explicitly under identities. Ignored when "
    "identities is set."
)

CHANNELS_DESCRIPTION = (
    "How many channels the crop image is read from disk with. 1 reads "
    "grayscale and is replicated to 3 channels before the backbone "
    "reads it. Any other value reads 3-channel RGB."
)

WEIGHTS_NAME_DESCRIPTION = (
    "The filename stem for the exported checkpoint, written as <weights_name>.pth."
)

CROP_ROOT_DESCRIPTION = (
    "Override for the directory EgocentricCrop output is read from. "
    "When it names no readable directory, the directory the loaded "
    "input came from is used instead."
)

LEARNING_RATE_DESCRIPTION = "The Adam learning rate."
