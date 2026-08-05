"""Backbone selection and preprocessing resolution for the embedding identity model.

Everything here runs without ``torch`` or ``timm``, which is why the model-id
rule and the data-config parsing live at module scope as pure functions rather
than inside the network's ``__init__``. Constructing a real backbone is one
``slow`` test at the end, skipped unless the extra is installed.
"""

from __future__ import annotations

import pytest

from mosaic.behavior.feature_library.identity_embedding_model import (
    GlobalIdentityEmbedding,
)
from mosaic.behavior.model_library.timm_backbone import (
    DEFAULT_MODEL_NAME,
    FALLBACK_DATA_CONFIG,
    IMAGENET_MEAN,
    IMAGENET_STD,
    BackboneDataConfig,
    data_config_from_mapping,
    resolve_backbone_data_config,
    resolve_timm_model_id,
)

# What ``timm.data.resolve_model_data_config`` returns for
# ``BVRA/MegaDescriptor-L-384``, transcribed from that repository's config.json.
MEGADESCRIPTOR_CFG: dict[str, object] = {
    "input_size": [3, 384, 384],
    "interpolation": "bicubic",
    "crop_pct": 0.9,
    "crop_mode": "center",
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
}


# --- The licensing guard --------------------------------------------------


def test_default_backbone_is_permissively_licensed() -> None:
    """The default weights must stay commercially usable.

    ``BVRA/MegaDescriptor-*`` is CC-BY-NC-4.0. It is a documented and
    recommended option for wildlife re-identification, and it must never become
    the value a user gets by not choosing -- a default that cannot be shipped
    commercially is a licensing decision made silently. Changing this literal is
    a licensing decision, not a tuning one, and moving it belongs in the same
    commit as the corresponding row in ``docs/licensing.md``.
    """
    assert DEFAULT_MODEL_NAME == "timm/swin_large_patch4_window12_384.ms_in22k_ft_in1k"
    assert not DEFAULT_MODEL_NAME.startswith("BVRA/")
    # The feature reads the literal from one home rather than repeating it.
    assert GlobalIdentityEmbedding.Params().model_name == DEFAULT_MODEL_NAME


# --- Model id resolution --------------------------------------------------


@pytest.mark.parametrize(
    ("model_name", "expected"),
    [
        # A bare timm architecture tag resolves through timm's own registry.
        (
            "swin_large_patch4_window12_384.ms_in22k_ft_in1k",
            "swin_large_patch4_window12_384.ms_in22k_ft_in1k",
        ),
        # A hub id is a name with an owner, and timm needs the prefix to see it.
        ("BVRA/MegaDescriptor-L-384", "hf-hub:BVRA/MegaDescriptor-L-384"),
        # ``timm/<tag>`` carries a slash and is a real hub repository, so it
        # takes the hub path and reaches the same weights the bare tag does.
        (
            "timm/swin_large_patch4_window12_384.ms_in22k_ft_in1k",
            "hf-hub:timm/swin_large_patch4_window12_384.ms_in22k_ft_in1k",
        ),
        # An explicit source prefix is already an answer.
        ("hf-hub:BVRA/MegaDescriptor-T-224", "hf-hub:BVRA/MegaDescriptor-T-224"),
        ("local-dir:/models/mine", "local-dir:/models/mine"),
    ],
)
def test_model_id_resolution(model_name: str, expected: str) -> None:
    """All three spellings a user might copy must reach a backbone."""
    assert resolve_timm_model_id(model_name) == expected


# --- Data config resolution -----------------------------------------------


def test_fallback_data_config_is_the_pre_resolution_behavior() -> None:
    """The no-information path must be the previous behavior, not a third one.

    Before resolution existed this class hardcoded ImageNet statistics and
    384x384. A backbone that declares nothing gets exactly that, so adopting
    resolution changed no result that resolution cannot explain.
    """
    assert FALLBACK_DATA_CONFIG == BackboneDataConfig(
        image_size=(384, 384), mean=IMAGENET_MEAN, std=IMAGENET_STD
    )


def test_a_full_config_is_honored_in_every_key() -> None:
    """MegaDescriptor declares the statistics this class used to assert."""
    resolved = data_config_from_mapping(MEGADESCRIPTOR_CFG)
    assert resolved.image_size == (384, 384)
    assert resolved.mean == IMAGENET_MEAN
    assert resolved.std == IMAGENET_STD


@pytest.mark.parametrize(
    ("raw", "why"),
    [
        ({}, "declares nothing"),
        ({"input_size": [3, 384]}, "input_size is not (channels, height, width)"),
        ({"input_size": "384x384"}, "input_size is not a sequence"),
        ({"mean": [0.5, 0.5]}, "mean is not a triple"),
        ({"std": None}, "std is absent in all but name"),
        ({"input_size": [3, 224, "224"]}, "a size is not an integer"),
    ],
)
def test_data_config_falls_back_per_key(raw: dict[str, object], why: str) -> None:
    """A missing or malformed key takes the fallback alone, and never raises.

    A partial ``pretrained_cfg`` is a real shape, so falling back wholesale
    would discard the keys the repository did answer.
    """
    resolved = data_config_from_mapping(raw)
    assert isinstance(resolved, BackboneDataConfig), why


def test_a_partial_config_keeps_the_keys_it_answered() -> None:
    """Statistics without a size resolve the statistics and fall back on size."""
    resolved = data_config_from_mapping(
        {"mean": [0.5, 0.5, 0.5], "std": [0.2, 0.2, 0.2]}
    )
    assert resolved.mean == (0.5, 0.5, 0.5)
    assert resolved.std == (0.2, 0.2, 0.2)
    assert resolved.image_size == FALLBACK_DATA_CONFIG.image_size


class _UndeclaredBackbone:
    """A model carrying no ``pretrained_cfg`` -- a locally defined architecture."""

    pretrained_cfg: None = None


def test_resolve_falls_back_without_a_pretrained_cfg() -> None:
    """No declaration means the fallback, reached without importing timm.

    The check precedes the import deliberately: a backbone with nothing to read
    should not pay for loading the library that would read it. timm is not
    installed in this environment, so an import here would raise rather than
    return.
    """
    assert resolve_backbone_data_config(_UndeclaredBackbone()) == FALLBACK_DATA_CONFIG


# --- The feature's defaults -----------------------------------------------


def test_image_size_default_follows_the_backbone() -> None:
    """``None`` is the only input size correct for every backbone.

    A literal here silently mis-sizes anything that is not 384x384, which is
    what naming an arbitrary backbone makes possible.
    """
    assert GlobalIdentityEmbedding.Params().image_size is None


# --- Against a real backbone ----------------------------------------------


@pytest.mark.slow
def test_constructs_the_real_backbone() -> None:
    """The resolution path against timm itself, not a transcribed config."""
    _ = pytest.importorskip("timm")
    _ = pytest.importorskip("torch")
    from mosaic.behavior.model_library.identity_embedding import (
        EmbeddingIdentityNetwork,
    )

    net = EmbeddingIdentityNetwork(model_name=DEFAULT_MODEL_NAME)
    assert net.image_size == (384, 384)
    assert net.embedding_dim == 1536
