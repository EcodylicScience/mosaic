"""Field prose shared across the crop-based identity models.

:class:`~mosaic.behavior.feature_library.dinov2_temporal_identity_model.GlobalIdentityDinoV2Temporal`
and
:class:`~mosaic.behavior.feature_library.identity_embedding_model.GlobalIdentityEmbedding`
both fit from
:class:`~mosaic.behavior.visualization_library.egocentric_crop.EgocentricCrop`
output. They share the identity-selection, crop-reading and
checkpoint-naming fields below, the text living here once rather than
once per class.

CROP_ROOT_DESCRIPTION and CROP_IMAGES_CAUSES also serve
:class:`~mosaic.behavior.feature_library.identity_model.GlobalIdentityModel`,
the third crop-based identity model, because all three read their crops from
``crop_root`` alone. LEARNING_RATE_DESCRIPTION serves the two that train with
Adam, GlobalIdentityModel and GlobalIdentityDinoV2Temporal; the embedding model
trains nothing. GlobalIdentityModel's identity-selection, channels and
checkpoint-naming fields keep their own wording under ``_CLASSIFIER_*`` names
in that module -- they must not share a name with the ones below.
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

# A placeholder in a field's prose is written {name}: the generated reference
# renders the prose as Markdown, which reads <name> as an HTML tag and * as
# emphasis.

WEIGHTS_NAME_DESCRIPTION = (
    "The filename stem for the exported checkpoint, written as {weights_name}.pth."
)

CROP_ROOT_DESCRIPTION = (
    "The directory of the EgocentricCrop run to read, "
    "features/{storage}/{run_id}/, or the output_root that run was given. "
    "Fitting reads each sequence's crops from "
    "{group}__{sequence}/frames_id{id}/frame_{frame}.png under it, so the "
    "crops must be made with output_mode frames or both and frame_format "
    "png. Nothing else locates them: unset, fitting finds no crops and "
    "raises."
)

# The causes named when a fit collects nothing. Each identity model's error
# opens with its own prefix and what it collected, then reads this.
CROP_IMAGES_CAUSES = (
    "The identity models read "
    "<crop_root>/<group>__<sequence>/frames_id*/frame_*.png, and the usual "
    "causes of finding none are: crop_root unset or not the directory of the "
    "egocentric-crop run, features/<storage>/<run_id>/ (Result.feature is "
    "the storage); crops made with output_mode='video', which writes no "
    "images, where 'frames' or 'both' writes them; and a frame_format other "
    "than 'png'. Check also that identities names sequences in this run's "
    "scope."
)

LEARNING_RATE_DESCRIPTION = "The Adam learning rate."
