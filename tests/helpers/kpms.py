"""The params of a kpms feature on three named keypoints."""

from __future__ import annotations


def kpms_params(**overrides: object) -> dict[str, object]:
    """Return kpms params for three keypoints, ``nose``, ``left`` and ``right``.

    The nose is the front of the body, and the other two are its back.
    ``latent_dim`` is 4, the most that three keypoints allow. Each of *overrides*
    replaces or adds a param.
    """
    return {
        "pose": {"pose_n": 3, "keypoint_names": ["nose", "left", "right"]},
        "anterior_bodyparts": ["nose"],
        "posterior_bodyparts": ["left", "right"],
        "latent_dim": 4,
        **overrides,
    }
