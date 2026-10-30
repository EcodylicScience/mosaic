"""Training a SLEAP model from a mosaic-owned labels file.

``sleap-nn train`` is configuration-driven, not flag-driven: it takes a directory
and the name of a YAML inside it, plus Hydra-style ``key=value`` overrides. So
training from mosaic means *writing* that YAML, which is what this module does --
mosaic owns the config, and the config is what makes a run reproducible.

**A generated config, not a template the user maintains.** The alternative is
asking for a config path, and then a run identifier means nothing: two runs with
the same mosaic parameters and different YAML on disk would share a name. What is
declared here reaches the identifier; what is not, cannot be varied.

**Everything sleap-nn can express is still reachable.** The typed fields are the
few decisions mosaic has an opinion about -- which head, which backbone, how long.
The rest is ``overrides``, passed through as Hydra assignments and folded into
identity, because a run trained with a different learning rate is a different
model whether or not mosaic has a field for it.

Requires:
    ``sleap-nn``, from a SLEAP install, run as ``sleap-nn train`` with the
    ``--config-dir`` / ``--config-name`` flags. sleap-nn has shipped only the
    ``sleap-nn`` script since 0.0.2. The ``sleap-nn-train`` script that mosaic
    used to run belonged to the ``sleap`` distribution (1.5.1 to 1.6.4), and
    sleap 1.6.5 removed it. The subcommand takes the same flags and works from
    sleap-nn 0.3.1, so running it drops no older environment. Tracking runs
    the same ``sleap-nn``, so the environment variables that point mosaic at
    SLEAP serve training too -- see :data:`mosaic.tracking.sleap.run.SLEAP_ENV`.
"""

from __future__ import annotations

import copy
import logging
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, NotRequired, TypedDict, get_args

import yaml

from mosaic.core.narrowing import is_str_mapping
from mosaic.core.pipeline.subprocess_util import command_summary, run_supervised
from mosaic.core.pipeline.types import JsonValue
from mosaic.core.pipeline.toolenv import subprocess_env, tool_invocation
from mosaic.tracking.sleap.run import SLEAP_ENV, SleapError

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_EMBEDDING_OBJECTIVE",
    "PREPARED_EMBEDDING_OBJECTIVE",
    "SLEAP_LOADER_WORKER_KEYS",
    "EmbeddingObjective",
    "EmbeddingPositiveScope",
    "EmbeddingSettings",
    "SleapBackbone",
    "SleapHead",
    "SleapTrainConfig",
    "effective_train_config",
    "embedding_settings",
    "loader_worker_keys_in",
    "sleap_device_overrides",
    "sleap_loader_overrides",
    "sleap_train_config",
    "train_sleap",
]

_SLEAP_NN: Final = "sleap-nn"
_TRAIN_SUBCOMMAND: Final = "train"

_ACCELERATOR_FAMILIES: Final = frozenset({"cpu", "gpu", "mps"})
"""The accelerator families sleap-nn's ``trainer_accelerator`` names.

``auto`` is deliberately absent: it is sleap-nn's own default, so selecting it
means injecting nothing rather than injecting the word.
"""

SleapHead = Literal[
    "single_instance",
    "centroid",
    "centered_instance",
    "bottomup",
    "multi_class_bottomup",
    "multi_class_topdown",
    "embedding",
]
"""Which task the network is trained for.

``centroid`` and ``centered_instance`` are the two halves of a top-down model and
are trained separately; the pair is what inference is later handed, in that
order. The ``multi_class_`` heads add identity classification. ``embedding`` maps
each detection's crop to an appearance vector and predicts no pose.
"""

SleapBackbone = Literal["unet", "convnext", "swint"]
"""The feature extractor. Orthogonal to the head, which is why it is its own
field rather than folded into a single model name."""

_HEAD_SECTIONS: Final[Mapping[SleapHead, tuple[str, ...]]] = {
    "single_instance": ("confmaps",),
    "centroid": ("confmaps",),
    "centered_instance": ("confmaps",),
    "bottomup": ("confmaps", "pafs"),
    "multi_class_bottomup": ("confmaps", "class_maps"),
    "multi_class_topdown": ("confmaps", "class_vectors"),
    "embedding": ("embedding",),
}
"""The output sections each head owns, which must be present for sleap-nn to fill.

An empty head block is **not** the same as a defaulted one. sleap-nn merges what
is written here over its own structured config, where every section defaults to
``None``; it then walks the head's sections looking for ``part_names`` and
``edges`` to fill in from the labels file, and a section left at ``None`` has no
keys to walk. Writing ``{head: {}}`` therefore reaches
``AttributeError: 'NoneType' object has no attribute 'keys'`` inside
``model_trainer._setup_head_config`` before training starts. Naming each section
as an empty mapping instantiates it at its own defaults, which is what leaves
``part_names`` and ``edges`` present-and-``None`` for sleap-nn to complete.

Listed here rather than derived from sleap-nn's classes because mosaic never
imports it: the tool lives in its own environment, and this file is the record of
what mosaic asks of it.
"""


_EMBEDDING_DIM: Final = 128
"""The length of an appearance vector, sleap-nn's default."""

_EMBEDDING_OUTPUT_STRIDE: Final[Mapping[SleapBackbone, int]] = {
    "unet": 16,
    "convnext": 32,
    "swint": 32,
}
"""The stride of the embedding head's input, each backbone's default ``max_stride``."""

EmbeddingPositiveScope = Literal["global_id", "tracklet", "aug_view"]
"""What makes two crops one identity: one track name across the labels, one
track within a video, or two augmented views of one crop."""

EmbeddingNegativeSource = Literal["same_frame", "in_batch"]
"""Where the crops of other identities come from."""

EmbeddingLoss = Literal["supcon"]
"""The contrastive loss."""

EmbeddingSampler = Literal["pk", "within_video"]
"""How a batch is drawn: groups of identities across the labels, or within one
video."""


@dataclass(frozen=True, slots=True)
class EmbeddingObjective:
    """The contrastive objective of an embedding head, every term of it.

    The document states every term rather than leaving any to sleap-nn. A
    run id then names the objective that trained even when a later sleap-nn release
    changes a default.
    """

    positive_scope: EmbeddingPositiveScope
    aug_views: int
    negative_sources: tuple[EmbeddingNegativeSource, ...]
    exclude_same_track: bool
    restrict_same_video: bool
    loss: EmbeddingLoss
    temperature: float
    sampler: EmbeddingSampler
    groups_per_batch: int
    samples_per_group: int
    use_projection: bool
    projection_dim: int

    def document(self) -> dict[str, JsonValue]:
        """The ``objective`` block of the head leaf, in sleap-nn's layout."""
        return {
            "positives": {"scope": self.positive_scope, "aug_views": self.aug_views},
            "negatives": {
                "sources": list(self.negative_sources),
                "exclude_same_track": self.exclude_same_track,
                "restrict_same_video": self.restrict_same_video,
            },
            "loss": {"name": self.loss, "temperature": self.temperature},
            "sampler": {
                "kind": self.sampler,
                "groups_per_batch": self.groups_per_batch,
                "samples_per_group": self.samples_per_group,
            },
            "use_projection": self.use_projection,
            "projection_dim": self.projection_dim,
        }


DEFAULT_EMBEDDING_OBJECTIVE: Final = EmbeddingObjective(
    positive_scope="global_id",
    aug_views=2,
    negative_sources=("same_frame", "in_batch"),
    exclude_same_track=True,
    restrict_same_video=False,
    loss="supcon",
    temperature=0.1,
    sampler="pk",
    groups_per_batch=8,
    samples_per_group=16,
    use_projection=True,
    projection_dim=128,
)
"""sleap-nn 0.3.5's default objective, for every labels reference other than a
``prepare-embedding-data`` run id."""

PREPARED_EMBEDDING_OBJECTIVE: Final = EmbeddingObjective(
    positive_scope="tracklet",
    aug_views=2,
    negative_sources=("same_frame", "in_batch"),
    exclude_same_track=True,
    restrict_same_video=True,
    loss="supcon",
    temperature=0.1,
    sampler="within_video",
    groups_per_batch=8,
    samples_per_group=8,
    use_projection=True,
    projection_dim=128,
)
"""The objective for labels that a ``prepare-embedding-data`` run cut.

Their tracks are a tracker's tracks, one video each. Positives are the crops of
one track, and negatives come from the same video.
"""

_EMBEDDING_AUGMENTATION: Final[dict[str, JsonValue]] = {
    "geometric": {
        "rotation_min": -10.0,
        "rotation_max": 10.0,
        "scale_min": 1.0,
        "scale_max": 1.0,
        "translate_width": 0.05,
        "translate_height": 0.05,
        "affine_p": 1.0,
        "erase_p": 0.0,
        "flip_p": 0.0,
    },
    "intensity": {"contrast_p": 0.0, "brightness_p": 0.0, "gaussian_noise_p": 0.0},
}
"""Augmentation that keeps appearance cues: small rotations and shifts only.

A flip, an erased patch or an intensity change can remove the mark that tells
two animals apart. The values are a starting point that no comparison chose.
"""


class _PreprocessingConfig(TypedDict):
    ensure_rgb: bool
    ensure_grayscale: bool


class _IdentityConfig(TypedDict):
    track_names_are_global: bool


class _DataConfig(TypedDict):
    train_labels_path: list[str]
    # sleap_train_config writes exactly one of these two.
    val_labels_path: NotRequired[list[str]]
    validation_fraction: NotRequired[float]
    provider: str
    preprocessing: _PreprocessingConfig
    # Written for the embedding head only.
    identity: NotRequired[_IdentityConfig]
    use_augmentations_train: NotRequired[bool]
    augmentation_config: NotRequired[dict[str, JsonValue]]


class _ModelConfig(TypedDict):
    backbone_config: dict[str, dict[str, object]]
    head_configs: dict[str, dict[str, object]]
    # sleap_train_config writes both or neither, from one base checkpoint.
    pretrained_backbone_weights: NotRequired[str]
    pretrained_head_weights: NotRequired[str]


class _TrainerConfig(TypedDict):
    max_epochs: int
    seed: int
    save_ckpt: bool
    ckpt_dir: str
    run_name: str
    # Written for the embedding head only.
    trainer_devices: NotRequired[int]


class SleapTrainConfig(TypedDict):
    """The sleap-nn configuration document mosaic writes.

    Typed rather than a bare mapping so the sections can be read back and
    asserted on -- a config is the whole of what a run was, and a test that
    cannot index it can only check that a file exists.
    """

    data_config: _DataConfig
    model_config: _ModelConfig
    trainer_config: _TrainerConfig


def sleap_train_config(
    labels_path: Path,
    run_root: Path,
    *,
    head: SleapHead,
    backbone: SleapBackbone,
    max_epochs: int,
    seed: int,
    validation_fraction: float,
    validation_labels: Path | None,
    base_weights: Path | None,
    run_name: str,
    objective: EmbeddingObjective = DEFAULT_EMBEDDING_OBJECTIVE,
) -> SleapTrainConfig:
    """The training configuration, as the mapping that is written to YAML.

    Built rather than templated so it can be asserted on without running
    anything, which is what the tests do. A *validation_labels* file is
    written as ``val_labels_path``, and *validation_fraction* only without one.
    A *base_weights* checkpoint is written as both
    ``model_config.pretrained_backbone_weights`` and
    ``model_config.pretrained_head_weights``, sleap-nn's fine-tune mapping. The
    run starts from the base's weights at epoch zero.

    The ``embedding`` head's leaf holds the vector length, the backbone's
    output stride and every term of *objective*. No other head reads
    *objective*, and its default suits only labels that no
    ``prepare-embedding-data`` run cut. A caller building an embedding
    document chooses it with
    :func:`mosaic.tracking.ops.train_sleap.embedding_objective`. That head's
    document also declares track names per video, augments, and trains on one
    device.
    """
    data_config: _DataConfig = {
        "train_labels_path": [str(labels_path)],
        "provider": "LabelsReader",
        # Written at sleap-nn's own defaults, and only because sleap-nn reads
        # these two off the *unmerged* document. ``run_training`` completes the
        # config through ``verify_training_cfg`` and keeps the result on
        # ``trainer.config``, but its post-training evaluation reaches back to
        # the raw ``config`` for ``ensure_rgb`` / ``ensure_grayscale`` -- so a
        # config that omits them trains to completion, writes its checkpoint,
        # and then dies on the evaluation pass with
        # ``ConfigAttributeError: Key 'preprocessing' is not in struct``.
        # Stating them changes no behavior and keeps the run from ending on an
        # error after the model is already on disk.
        "preprocessing": {"ensure_rgb": False, "ensure_grayscale": False},
    }
    if validation_labels is None:
        data_config["validation_fraction"] = validation_fraction
    else:
        data_config["val_labels_path"] = [str(validation_labels)]
    model_config: _ModelConfig = {
        # The backbone takes an empty block: its fields carry real defaults, so
        # merging one over sleap-nn's structured config yields those. A head
        # does not -- see ``_HEAD_SECTIONS``.
        "backbone_config": {backbone: {}},
        "head_configs": {head: {section: {} for section in _HEAD_SECTIONS[head]}},
    }
    if base_weights is not None:
        model_config["pretrained_backbone_weights"] = str(base_weights)
        model_config["pretrained_head_weights"] = str(base_weights)
    trainer_config: _TrainerConfig = {
        "max_epochs": max_epochs,
        "seed": seed,
        "save_ckpt": True,
        "ckpt_dir": str(run_root),
        "run_name": run_name,
    }
    if head == "embedding":
        model_config["head_configs"][head]["embedding"] = {
            "embedding_dim": _EMBEDDING_DIM,
            "output_stride": _EMBEDDING_OUTPUT_STRIDE[backbone],
            "objective": objective.document(),
        }
        identity: _IdentityConfig = {"track_names_are_global": False}
        data_config["identity"] = identity
        data_config["use_augmentations_train"] = True
        data_config["augmentation_config"] = copy.deepcopy(_EMBEDDING_AUGMENTATION)
        trainer_config["trainer_devices"] = 1
    return {
        "data_config": data_config,
        "model_config": model_config,
        "trainer_config": trainer_config,
    }


type _HydraForm = Literal["", "+", "++", "~"]


def _hydra_key(key: str) -> tuple[_HydraForm, str]:
    """Split an override key into its Hydra form and the dotted path it names.

    Hydra spells appending ``+key=value``, forcing an append ``++key=value`` and
    deleting ``~key``. A bare key has the empty form.
    """
    if key.startswith("++"):
        return "++", key[2:]
    if key.startswith("+"):
        return "+", key[1:]
    if key.startswith("~"):
        return "~", key[1:]
    return "", key


def _declared_keys(node: Mapping[str, object], prefix: str = "") -> frozenset[str]:
    """Every dotted key *node* carries, at every depth.

    Derived from the document rather than listed, so a key added to
    :func:`sleap_train_config` changes how an override naming it is rendered
    without anyone remembering to say so here.
    """
    keys: set[str] = set()
    for name, value in node.items():
        dotted = f"{prefix}{name}"
        keys.add(dotted)
        if is_str_mapping(value):
            keys |= _declared_keys(value, f"{dotted}.")
    return frozenset(keys)


def _rendered_key(declared: frozenset[str], key: str) -> tuple[_HydraForm, str]:
    """The Hydra form in which *key* reaches sleap-nn, and the dotted path it names.

    ``sleap_train_config`` writes a deliberately minimal document, so most of
    what sleap-nn exposes has no key in it -- and Hydra refuses a bare
    assignment to a key the config it composed does not already hold, with
    ``Could not override '<key>'``. Every override therefore has to know whether
    mosaic happened to write its key, which is not a thing a caller can be
    expected to know. So a bare key that *declared* lacks is spelled with ``+``.
    A caller who wrote a Hydra form has answered the question, and the form is
    kept.
    """
    form, path = _hydra_key(key)
    if not form and key not in declared:
        return "+", path
    return form, path


def _hydra_assignment(declared: frozenset[str], key: str, value: JsonValue) -> str:
    """One Hydra argument, with its key in :func:`_rendered_key`'s form.

    A deletion whose value is ``None`` carries no value.
    """
    form, path = _rendered_key(declared, key)
    if form == "~" and value is None:
        return f"~{path}"
    return f"{form}{path}={_hydra_value(value)}"


_BARE_STRING: Final = re.compile(r"[A-Za-z_/][A-Za-z0-9_\-./]*")
"""A string that Hydra's override grammar reads unquoted as that same string."""

_BARE_MAPPING_KEY: Final = re.compile(r"[A-Za-z_][A-Za-z0-9_\-]*")
"""A mapping key that Hydra's override grammar reads as that same string."""

_HYDRA_WORDS: Final = frozenset({"true", "false", "null", "inf", "nan"})
"""Words that Hydra reads unquoted, in any case, as a boolean, null or float."""

_QUOTED_ESCAPES: Final = re.compile(r"(\\*)(\$\{|'|$)")
"""What a single-quoted Hydra string escapes, with the backslashes before it.

Each takes a backslash, because a quote ends the string and ``${`` starts an
interpolation. A run of backslashes before either, or at the end, is doubled.
"""


def _hydra_value(value: JsonValue) -> str:
    """*value* in Hydra's override grammar, which parses back to the same value.

    Raises:
        ValueError: A mapping key that the grammar cannot spell as a string.
    """
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return repr(value)
    if isinstance(value, str):
        return _hydra_string(value)
    if isinstance(value, list):
        return "[" + ",".join(_hydra_value(item) for item in value) + "]"
    items = [
        f"{_hydra_mapping_key(name)}:{_hydra_value(item)}"
        for name, item in value.items()
    ]
    return "{" + ",".join(items) + "}"


def _hydra_string(text: str) -> str:
    if _BARE_STRING.fullmatch(text) and text.lower() not in _HYDRA_WORDS:
        return text
    escaped = _QUOTED_ESCAPES.sub(
        lambda match: match[1] * 2 + (f"\\{match[2]}" if match[2] else ""), text
    )
    return f"'{escaped}'"


def _hydra_mapping_key(name: str) -> str:
    if _BARE_MAPPING_KEY.fullmatch(name) and name.lower() not in _HYDRA_WORDS:
        return name
    msg = (
        f"Hydra's override grammar cannot spell the mapping key {name!r}. Set "
        "the keys below it one at a time instead."
    )
    raise ValueError(msg)


_DEVICE_COUNT_PATH: Final = ("trainer_config", "trainer_devices")
"""Where sleap-nn reads how many devices to train on. Absent, it trains on all."""


def effective_train_config(
    document: Mapping[str, JsonValue], overrides: Mapping[str, JsonValue]
) -> dict[str, JsonValue]:
    """Apply *overrides* to *document* the way Hydra 1.3 composes them for sleap-nn.

    Each override applies in order, in the form :func:`_rendered_key` gives it,
    as Hydra's ``_apply_overrides_to_config`` calls ``OmegaConf.update(cfg,
    key, value, merge=True)`` on a config in struct mode. A mapping value
    merges into a mapping at its dotted path and keeps the sibling keys there.
    Any other value replaces the value at the path. A ``+`` or ``++`` key
    creates the mappings above its path, replacing a value that is not a
    mapping. A ``~`` key deletes the path.

    Every override that Hydra rejects is refused here, before a run is
    claimed. Struct mode refuses a bare key whose path, or any
    key its mapping value would add, the running document does not hold. A
    path through a list is refused, although Hydra reads a numeric name there
    as a list index. The device-count refusal covers deletion only: the device
    check reads the returned document, where a null or a replaced
    ``trainer_config`` also leaves the count unset.

    Args:
        document: The generated training document. It is not modified.
        overrides: Hydra assignments by key, in the order sleap-nn receives
            them. A ``~`` key's value is ``None`` for a plain deletion, or the
            value that the deletion requires the path to hold.

    Returns:
        A new document, sharing no mutable value with *document* or
        *overrides*.

    Raises:
        ValueError: A ``~`` key deletes ``trainer_config.trainer_devices`` or a
            mapping that contains it, which leaves sleap-nn to train on every
            visible device. A ``~`` key names a path that is absent or null, or
            carries a value other than the one at the path. A ``+`` key names a
            path that holds a value other than null, and its own value is
            neither a mapping nor a list. A bare key names a path, or its
            mapping value a key, that the document no longer holds. A mapping
            meets a list, in either order. A path runs through a list. A
            mapping key is one that Hydra's override grammar cannot spell.
    """
    declared = _declared_keys(document)
    effective = copy.deepcopy(dict(document))
    for key, value in overrides.items():
        form, path = _rendered_key(declared, key)
        _ = _hydra_value(value)  # refuses a mapping key Hydra cannot spell
        names = tuple(path.split("."))
        present = _select(effective, names)
        if form == "~":
            _refuse_deletion(key, names, value, present)
            _delete_path(effective, names)
            continue
        if form == "+" and present is not None and not isinstance(value, dict | list):
            msg = (
                f"Override {key!r} reaches Hydra as '+{path}', which appends "
                f"{path!r}. The training document already sets it to "
                f"{present!r}, and Hydra refuses the append. Write '++{path}' "
                "to replace the value."
            )
            raise ValueError(msg)
        _merge_at(effective, names, value, key, struct=not form)
    return effective


def _refuse_deletion(
    key: str, names: tuple[str, ...], value: JsonValue, present: JsonValue | None
) -> None:
    """Raise when Hydra would refuse the deletion *key* of the value *present*."""
    if _DEVICE_COUNT_PATH[: len(names)] == names:
        msg = (
            f"Override {key!r} deletes trainer_config.trainer_devices, "
            "and sleap-nn then trains on every visible device. "
            "Set trainer_config.trainer_devices to 1 instead."
        )
        raise ValueError(msg)
    if present is None:
        msg = (
            f"Override {key!r} deletes a path that the training document does "
            "not set, and Hydra refuses it."
        )
        raise ValueError(msg)
    if value is not None and value != present:
        msg = (
            f"Override {key!r} deletes a path that holds {present!r} rather "
            f"than {value!r}, and Hydra refuses it."
        )
        raise ValueError(msg)


def _select(document: dict[str, JsonValue], names: tuple[str, ...]) -> JsonValue | None:
    """The value at *names*, or ``None`` when the path is absent, as ``OmegaConf.select``."""
    node: JsonValue = document
    for name in names:
        if not isinstance(node, dict):
            return None
        node = node.get(name)
    return node


def _delete_path(document: dict[str, JsonValue], names: tuple[str, ...]) -> None:
    """Delete the key at *names*, which :func:`_refuse_deletion` found present."""
    parent = _select(document, names[:-1])
    assert isinstance(parent, dict)
    del parent[names[-1]]


def _merge_at(
    document: dict[str, JsonValue],
    names: tuple[str, ...],
    value: JsonValue,
    key: str,
    *,
    struct: bool,
) -> None:
    """Merge or set *value* at *names* in *document*.

    With *struct*, as for Hydra's bare form, every name on the path is already
    there. Without it, as ``OmegaConf.update`` does with ``force_add`` for
    ``+`` and ``++``, an empty mapping replaces each value above the path that
    is absent or not a mapping.
    """
    parent = document
    for depth, name in enumerate(names[:-1]):
        child = parent.get(name)
        if not isinstance(child, dict):
            above = ".".join(names[: depth + 1])
            if isinstance(child, list):
                msg = (
                    f"Override {key!r} names a path through the list at "
                    f"{above!r}, which mosaic does not follow. Override the "
                    "whole list instead."
                )
                raise ValueError(msg)
            if struct:
                raise _struct_refusal(key, above)
            child = {}
            parent[name] = child
        parent = child
    if struct and names[-1] not in parent:
        raise _struct_refusal(key, ".".join(names))
    parent[names[-1]] = _merged(
        parent.get(names[-1]), value, key, ".".join(names), struct=struct
    )


def _merged(
    present: JsonValue, value: JsonValue, key: str, path: str, *, struct: bool
) -> JsonValue:
    """*value* merged over *present*, as ``OmegaConf.merge`` merges a node.

    A mapping merges into a mapping, key by key. Any other value replaces
    *present*, except that Hydra refuses to merge a mapping into a list or a
    list into a mapping. With *struct*, a mapping adds no key.
    """
    if isinstance(value, dict) and isinstance(present, dict):
        for name, item in value.items():
            where = f"{path}.{name}"
            if struct and name not in present:
                raise _struct_refusal(key, where)
            present[name] = _merged(present.get(name), item, key, where, struct=struct)
        return present
    if (isinstance(value, dict) and isinstance(present, list)) or (
        isinstance(value, list) and isinstance(present, dict)
    ):
        msg = (
            f"Override {key!r} merges {value!r} into {present!r} at {path!r}, "
            "and Hydra refuses to merge a mapping into a list or a list into a "
            f"mapping. Delete it first with '~{path}'."
        )
        raise ValueError(msg)
    return copy.deepcopy(value)


def _struct_refusal(key: str, path: str) -> ValueError:
    """Hydra's refusal of a bare key that would add *path* to its config."""
    msg = (
        f"Override {key!r} sets {path!r}, which the training document does not "
        "hold once the overrides before it apply, and Hydra refuses a bare key "
        f"that adds one. Write '++{key}' to add it."
    )
    return ValueError(msg)


_EMBEDDING_OBJECTIVE_PATH: Final = (
    "model_config",
    "head_configs",
    "embedding",
    "embedding",
    "objective",
)


@dataclass(frozen=True, slots=True)
class EmbeddingSettings:
    """What an effective embedding document asks of its labels and devices.

    A key that the document deletes, leaves null or never sets takes sleap-nn's
    default.

    Attributes:
        scope: ``objective.positives.scope``, ``global_id`` by default.
        track_names_are_global: Whether
            ``data_config.identity.track_names_are_global`` is true. False by
            default.
        restrict_same_video: Whether ``objective.negatives.restrict_same_video``
            is true. False by default.
        trainer_devices: ``trainer_config.trainer_devices`` as the document
            holds it, ``None`` when unset, which trains on every device.
        trainer_device_indices: ``trainer_config.trainer_device_indices`` as
            the document holds it, ``None`` when unset.
    """

    scope: EmbeddingPositiveScope
    track_names_are_global: bool
    restrict_same_video: bool
    trainer_devices: JsonValue
    trainer_device_indices: JsonValue


def embedding_settings(document: dict[str, JsonValue]) -> EmbeddingSettings:
    """Read the settings that an embedding run checks from an effective document.

    Args:
        document: A document that :func:`effective_train_config` returned.

    Raises:
        ValueError: The positive scope is not one that sleap-nn names.
    """
    objective = _EMBEDDING_OBJECTIVE_PATH
    match _select(document, (*objective, "positives", "scope")):
        case None:
            scope = DEFAULT_EMBEDDING_OBJECTIVE.positive_scope
        case "global_id" | "tracklet" | "aug_view" as named:
            scope = named
        case other:
            accepted = ", ".join(get_args(EmbeddingPositiveScope))
            msg = (
                f"The embedding objective's positive scope is {other!r}, and "
                f"sleap-nn accepts one of {accepted}."
            )
            raise ValueError(msg)
    restrict = _select(document, (*objective, "negatives", "restrict_same_video"))
    declared = _select(document, ("data_config", "identity", "track_names_are_global"))
    return EmbeddingSettings(
        scope=scope,
        track_names_are_global=declared is True,
        restrict_same_video=(
            DEFAULT_EMBEDDING_OBJECTIVE.restrict_same_video
            if restrict is None
            else restrict is True
        ),
        trainer_devices=_select(document, _DEVICE_COUNT_PATH),
        trainer_device_indices=_select(
            document, ("trainer_config", "trainer_device_indices")
        ),
    )


def sleap_device_overrides(device: str) -> dict[str, JsonValue]:
    """The ``trainer_config`` assignments that put training on *device*.

    sleap-nn splits the question in two, and neither half takes the spelling
    the rest of mosaic uses: ``trainer_accelerator`` is an accelerator *family*
    (``cpu`` / ``gpu`` / ``mps`` / ``auto``) and ``trainer_device_indices`` is
    the list of devices within it. Passing ``"0"`` straight through would set
    the family to the string ``"0"``. So *device* is spelled the way
    ``train-pose`` and ``train-localizer`` spell it -- a family name or a CUDA
    index -- and translated here.

    Args:
        device: ``auto`` or empty to leave the choice to sleap-nn; ``cpu``,
            ``gpu`` or ``mps`` to name a family; or a comma-separated list of
            CUDA indices such as ``0`` or ``0,1``.

    Returns:
        The Hydra assignments, empty when *device* leaves the choice open.

    Raises:
        ValueError: *device* is none of those. Raised from a field validator
            too, so an unusable value is refused when the run is submitted
            rather than on a GPU node once it is scheduled.
    """
    if device in ("", "auto"):
        return {}
    if device in _ACCELERATOR_FAMILIES:
        return {"trainer_config.trainer_accelerator": device}
    parts = [part.strip() for part in device.split(",")]
    if parts and all(part.isdigit() for part in parts):
        return {
            "trainer_config.trainer_accelerator": "gpu",
            "trainer_config.trainer_device_indices": [int(part) for part in parts],
        }
    families = ", ".join(sorted(_ACCELERATOR_FAMILIES))
    msg = (
        f"unusable device {device!r}: give 'auto', one of {families}, or a "
        f"comma-separated list of CUDA indices such as '0' or '0,1'."
    )
    raise ValueError(msg)


SLEAP_LOADER_WORKER_KEYS: Final = (
    "trainer_config.train_data_loader.num_workers",
    "trainer_config.val_data_loader.num_workers",
)
"""Where sleap-nn reads its data-loader worker count: once per loader.

Both are set from one value, because the question a caller is answering -- how
many processes may load data -- does not differ between them. Neither key is in
the document :func:`sleap_train_config` writes, so :func:`_hydra_assignment`
appends them.
"""


def sleap_loader_overrides(num_workers: int | None) -> dict[str, JsonValue]:
    """The assignments that give both of sleap-nn's data loaders *num_workers*.

    Args:
        num_workers: Worker processes per loader, or ``None`` to leave sleap-nn's
            own default, which loads in the main process.

    Returns:
        The Hydra assignments, empty for ``None``.
    """
    if num_workers is None:
        return {}
    return {key: num_workers for key in SLEAP_LOADER_WORKER_KEYS}


def loader_worker_keys_in(overrides: Mapping[str, JsonValue]) -> list[str]:
    """The keys of *overrides* that set a loader worker count, however spelled.

    Each key is compared with its Hydra prefix removed. :func:`_hydra_assignment`
    passes a ``+``, ``++`` or ``~`` spelling through as written, so without this
    ``+trainer_config.train_data_loader.num_workers`` would still reach the
    worker count through a hashed parameter.
    """
    return sorted(
        key for key in overrides if _hydra_key(key)[1] in SLEAP_LOADER_WORKER_KEYS
    )


def train_sleap(
    labels_path: str | Path,
    run_root: str | Path,
    *,
    config: SleapTrainConfig,
    overrides: Mapping[str, JsonValue] | None = None,
    sleap_conda_env: str | None = None,
    sleap_bin: str | Path | None = None,
    idle_timeout: float = 1800,
    max_runtime: float | None = None,
    cancel_check: Callable[[], bool] | None = None,
    on_output: Callable[[str], None] | None = None,
    on_activity: Callable[[str], None] | None = None,
) -> Path:
    """Train a SLEAP model and return the directory it produced.

    Args:
        labels_path: The ``.slp`` that *config* trains on.
        run_root: Where the model directory is written. The configuration is
            written here too, so a finished run carries the exact inputs it ran
            with beside its weights.
        config: The document from :func:`sleap_train_config`, written as
            ``config.yaml``. Its ``trainer_config.run_name`` names the model
            directory under *run_root*.
        overrides: Hydra assignments applied on top, for anything sleap-nn
            exposes that the document does not set. A key the document holds
            is assigned bare, and any other key is appended.
        sleap_conda_env: Run in this conda env, overriding the environment.
        sleap_bin: A SLEAP console script naming the install, overriding the
            environment.
        idle_timeout: Kill the subprocess after this long with no output.
            Generous by default -- an epoch on a large set is slow, and the
            watchdog must not mistake slow for dead.
        max_runtime: Optional absolute ceiling.
        cancel_check: Polled while the subprocess runs.
        on_output: Called with each line the trainer writes to standard output,
            for a caller parsing epochs out of them.
        on_activity: Called with each line on **either** stream, for a caller
            keeping the run root's claim alive. Both are passed straight to
            ``run_supervised``; see its two parameters of the same names.

    Returns:
        The model directory, which is what
        :data:`mosaic.tracking.model_refs.MODEL_KINDS` describes as a ``sleap``
        artifact and what inference is later handed.

    Raises:
        FileNotFoundError: *labels_path* does not exist, or training exited
            zero without producing a model directory.
        SleapNotFoundError: No SLEAP install could be located.
        SleapError: Training exited non-zero.
    """
    labels_path = Path(labels_path)
    if not labels_path.exists():
        raise FileNotFoundError(f"SLEAP labels file does not exist: {labels_path}")
    run_root = Path(run_root)
    run_root.mkdir(parents=True, exist_ok=True)

    config_path = run_root / "config.yaml"
    _ = config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    declared = _declared_keys(config)

    args = [
        _TRAIN_SUBCOMMAND,
        "--config-dir",
        str(run_root),
        "--config-name",
        config_path.stem,
        *(
            _hydra_assignment(declared, key, value)
            for key, value in (overrides or {}).items()
        ),
    ]
    invocation = tool_invocation(
        SLEAP_ENV.placed(conda_env=sleap_conda_env, bin_path=sleap_bin),
        executable=_SLEAP_NN,
    )
    cmd = [*invocation, *args]
    logger.info("Running: %s", command_summary(cmd))

    stdout, stderr, returncode = run_supervised(
        cmd,
        env=subprocess_env(),
        cancel_check=cancel_check,
        timeout=max_runtime,
        idle_timeout=idle_timeout,
        on_output=on_output,
        on_activity=on_activity,
    )
    if returncode != 0:
        raise SleapError(cmd, returncode, stdout, stderr)

    produced = run_root / config["trainer_config"]["run_name"]
    if not produced.is_dir():
        raise FileNotFoundError(
            f"sleap-nn train exited cleanly but wrote no model directory at {produced}"
        )
    return produced
