"""GlobalIdentityModel feature.

Trains a visual identification model from egocentric crop images of individual
animals: a pretrained image backbone with a linear classification head on top,
fitted with cross-entropy against a closed set of known individuals.

Each identity is named by the sequences that contain that individual alone, so
the training labels come from the dataset's own structure rather than from
per-frame annotation.

Choosing among the three identity features: this one *trains* a head and wants
enough crops per animal to fit one. ``global-identity-embedding`` trains nothing
and answers in a single pass over the same backbone family, which makes it the
right first attempt. ``global-identity-dinov2-temporal`` reads clips rather than
single frames, so it can use cues that only exist over time.

Mosaic distributes no model weights. Whatever ``model_name`` names is fetched at
run time under its own license -- see
:mod:`mosaic.behavior.model_library.timm_backbone`.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated, ClassVar, TypedDict, final

import joblib
import numpy as np
import pandas as pd
from pydantic import Field

from mosaic.behavior.model_library.timm_backbone import DEFAULT_MODEL_NAME
from mosaic.core.pipeline.types import (
    EmitsLevel,
    DependencyLookup,
    InputRequire,
    Inputs,
    InputStream,
    JoblibArtifact,
    JoblibLoadSpec,
    Result,
)
from mosaic.core.params import Declared, Params

from ._identity_model_descriptions import LEARNING_RATE_DESCRIPTION
from .registry import register_feature

# --- Model artifact ---

# The exported weights are a torch ``.pth``, and an ArtifactSpec can only load
# npz / parquet / joblib -- so the referencable artifact is this joblib sidecar,
# written beside the checkpoint and naming it. Same shape as
# ``identity_embedding_model``. The name is fixed rather than derived from
# ``weights_name`` because dependency resolution globs it, and the run root also
# holds ``identity_names.joblib`` and ``training_history.joblib``.
_BUNDLE_NAME = "identity_classifier_model.joblib"


class ClassifierIdentityBundle(TypedDict):
    """Sidecar naming the exported identity-classifier checkpoint.

    Attributes:
        weights: Checkpoint filename, relative to the bundle's directory.
        identity_names: Class order the checkpoint was exported with.
        version: Feature version that wrote the bundle.
    """

    weights: str
    identity_names: list[str]
    version: str


class ClassifierIdentityArtifact(JoblibArtifact[ClassifierIdentityBundle]):
    """Fitted identity-classifier bundle (identity_classifier_model.joblib)."""

    feature: str = "global-identity-model"
    pattern: str = _BUNDLE_NAME
    load: JoblibLoadSpec = Field(default_factory=JoblibLoadSpec)


_MODEL_DESCRIPTION = (
    "Unset, the fit runs over the input scope. Set to a pre-fitted "
    "ClassifierIdentityArtifact, the fit is skipped and the referenced "
    "training run enters this run's identity beside its scope."
)

_CLASSIFIER_IDENTITIES_DESCRIPTION = (
    "The mapping from an identity name to the group/sequence entries that "
    "contain that individual alone. Takes priority over group_as_identity "
    "when both are given."
)

_CLASSIFIER_GROUP_AS_IDENTITY_DESCRIPTION = (
    "Derive one identity per group name from the input entries, used only "
    "when identities is unset. Fitting raises when neither is given."
)

_MODEL_NAME_DESCRIPTION = (
    "A bare timm architecture tag or a Hugging Face hub id naming the "
    "backbone. Mosaic ships no weights, and whatever is named here is "
    "downloaded at run time under its own license."
)

_IMAGE_SIZE_DESCRIPTION = (
    "Unset, the backbone's declared input size is used. Set, crops are "
    "resized to (height, width) before the backbone reads them."
)

_CLASSIFIER_CHANNELS_DESCRIPTION = (
    "The number of channels read from disk: 1 for grayscale, 3 for RGB. "
    "A grayscale image is replicated to 3 channels for the backbone."
)

_FREEZE_BACKBONE_DESCRIPTION = (
    "Train the classification head alone, leaving the pretrained backbone "
    "untouched. False fine-tunes the whole network end to end, which needs "
    "considerably more data."
)

_EPOCHS_DESCRIPTION = "How long the classifier trains."

_BATCH_SIZE_DESCRIPTION = "How many crops the classifier reads per training batch."

_VAL_SPLIT_DESCRIPTION = (
    "The fraction of collected images held out for validation. 0 disables "
    "the split, training on every image."
)

_MAX_IMAGES_PER_IDENTITY_DESCRIPTION = (
    "The most images used per identity. Balances classes by capping how "
    "many crops from any one identity enter training."
)

_CLASSIFIER_WEIGHTS_NAME_DESCRIPTION = (
    "The stem of the exported .pth checkpoint filename."
)

_CLASSIFIER_CROP_ROOT_DESCRIPTION = (
    "Unset, the EgocentricCrop output root is resolved from the input "
    "Result. Set, crops are read from this path first, falling back to "
    "that resolution when the derived directory does not exist."
)

# --- Feature class ---


@final
@register_feature
class GlobalIdentityModel:
    """Train a visual identity model from individual animal sequences.

    Takes EgocentricCrop output as input. Each identity is specified as a
    mapping of identity names to lists of sequences containing that
    individual alone. Trains a classification head over a pretrained image
    backbone and exports the fitted weights.

    Example::

        ego_result = dataset.run_feature(ego_crop)

        identity_model = GlobalIdentityModel(
            Inputs((Result(feature="egocentric-crop"),)),
            params={
                "identities": {
                    "mouse_A": ["cage1/day1_mouseA_alone", "cage1/day3_mouseA_alone"],
                    "mouse_B": ["cage1/day1_mouseB_alone"],
                    "mouse_C": ["cage1/day2_mouseC_alone"],
                    "mouse_D": ["cage1/day1_mouseD_alone"],
                },
            },
        )
        result = dataset.run_feature(identity_model)

    Field documentation is on
    :class:`~mosaic.behavior.feature_library.identity_model.GlobalIdentityModel.Params`.
    """

    category = "global"
    name: str = "global-identity-model"
    # 0.3: the network changed outright -- a trained head over a pretrained
    # image backbone, where 0.2 was a CNN trained from scratch. Network numerics
    # are not part of the run_id payload, so only this version string can
    # express "the recipe changed" and stop `load_state` adopting a checkpoint
    # the previous code wrote, which this network cannot read at all.
    version: str = "0.3"
    parallelizable = False
    # fit() reads the ambient stream -- both to discover the label set under
    # group_as_identity and to collect the training crops -- so the scope IS the
    # training set and belongs in the identifier (P2f). An inference run pins
    # ``model`` instead and carries its training set by reference, so fit and
    # apply are two runs with two identifiers rather than one that silently
    # reuses a network fitted on a narrower scope.
    scope_dependent = True
    accepts_overlap = (
        False  # trains per entry; a neighbour's rows carry another entry's labels
    )
    consumed_roots: tuple[str, ...] = ()
    emits: EmitsLevel = "as-input"
    ModelArtifact = ClassifierIdentityArtifact

    class Inputs(Inputs[Result]):
        _require: ClassVar[InputRequire] = "any"

    class Params(Params):
        # Pre-fitted model reference: when set (and resolvable), fit is skipped.
        model: Annotated[
            ClassifierIdentityArtifact | None, Declared(_MODEL_DESCRIPTION)
        ] = None

        # Primary: explicit identity -> sequences mapping
        identities: Annotated[
            dict[str, list[str]] | None, Declared(_CLASSIFIER_IDENTITIES_DESCRIPTION)
        ] = None
        # Convenience shortcut: treat each group as one identity
        group_as_identity: Annotated[
            bool, Declared(_CLASSIFIER_GROUP_AS_IDENTITY_DESCRIPTION)
        ] = False

        # Backbone selection. Changing ``model_name`` is a licensing decision as
        # well as an accuracy one -- see the module docstring.
        model_name: Annotated[str, Declared(_MODEL_NAME_DESCRIPTION)] = (
            DEFAULT_MODEL_NAME
        )
        # None means follow the backbone's declared input size.
        image_size: Annotated[
            tuple[int, int] | None, Declared(_IMAGE_SIZE_DESCRIPTION, unit="px")
        ] = None
        channels: Annotated[
            int, Field(examples=[1, 3]), Declared(_CLASSIFIER_CHANNELS_DESCRIPTION)
        ] = 3
        freeze_backbone: Annotated[bool, Declared(_FREEZE_BACKBONE_DESCRIPTION)] = True

        # Training
        epochs: Annotated[int, Declared(_EPOCHS_DESCRIPTION, unit="epochs")] = 30
        learning_rate: Annotated[float, Declared(LEARNING_RATE_DESCRIPTION)] = 0.001
        batch_size: Annotated[int, Declared(_BATCH_SIZE_DESCRIPTION)] = 32
        val_split: Annotated[float, Declared(_VAL_SPLIT_DESCRIPTION)] = Field(
            default=0.2, ge=0.0, lt=1.0
        )

        # Sampling
        max_images_per_identity: Annotated[
            int, Declared(_MAX_IMAGES_PER_IDENTITY_DESCRIPTION)
        ] = Field(default=2000, ge=1)

        # Export
        weights_name: Annotated[str, Declared(_CLASSIFIER_WEIGHTS_NAME_DESCRIPTION)] = (
            "identity_classifier"
        )

        # Path to EgocentricCrop output root (contains group__sequence/ subdirs).
        # If None, the feature tries to resolve it from the input Result.
        crop_root: Annotated[
            str | None, Declared(_CLASSIFIER_CROP_ROOT_DESCRIPTION)
        ] = None

    def __init__(
        self,
        inputs: GlobalIdentityModel.Inputs,
        params: dict[str, object] | None = None,
    ) -> None:
        self.inputs = inputs
        self.params = self.Params.from_overrides(params)

        self._network: object | None = None
        self._history: dict[str, list[float]] | None = None
        self._identity_names: list[str] | None = None

    # --- Feature protocol ---

    def load_state(
        self,
        run_root: Path,
        artifact_paths: dict[str, Path],
        dependency_lookups: dict[str, DependencyLookup],
    ) -> bool:
        from mosaic.behavior.model_library.identity_classifier import (
            ClassifierIdentityNetwork,
        )

        self._network = None
        self._history = None
        self._identity_names = None

        # Branch 1: this run's own cached checkpoint.
        cached_path = run_root / f"{self.params.weights_name}.pth"
        if cached_path.exists():
            self._network = ClassifierIdentityNetwork.from_checkpoint(cached_path)
            history_path = run_root / "training_history.joblib"
            if history_path.exists():
                self._history = joblib.load(history_path)
            names_path = run_root / "identity_names.joblib"
            if names_path.exists():
                self._identity_names = joblib.load(names_path)
            return True

        # Branch 2: a pre-fitted model pinned in params. The checkpoint name
        # comes from the bundle, never from self.params -- an inference run's
        # weights_name need not match the training run's.
        if self.params.model is not None and "model" in artifact_paths:
            bundle_path = artifact_paths["model"]
            bundle = self.params.model.from_path(bundle_path)
            self._network = ClassifierIdentityNetwork.from_checkpoint(
                bundle_path.parent / bundle["weights"]
            )
            self._identity_names = list(bundle["identity_names"])
            return True

        return False

    def fit(self, inputs: InputStream) -> None:
        from mosaic.behavior.model_library.identity_classifier import (
            ClassifierIdentityNetwork,
        )
        from mosaic.behavior.model_library.identity_common import (
            build_label_mapping,
            load_crop_frames,
        )

        p = self.params

        seq_to_label, identity_names = build_label_mapping(p, inputs)
        self._identity_names = identity_names
        num_classes = len(identity_names)

        if num_classes < 2:
            msg = (
                f"[identity-model] Need at least 2 identities, "
                f"got {num_classes}: {identity_names}"
            )
            raise ValueError(msg)

        print(
            f"[identity-model] Training with {num_classes} identities: "
            f"{identity_names}",
            file=sys.stderr,
        )

        # Collect images and labels from input sequences
        all_images: dict[int, list[np.ndarray]] = {i: [] for i in range(num_classes)}
        for entry_key, df in inputs():
            label = seq_to_label.get(entry_key)
            if label is None:
                continue
            frames = load_crop_frames(
                entry_key,
                df,
                crop_root=p.crop_root,
                channels=p.channels,
                max_frames=p.max_images_per_identity,
            )
            if frames:
                all_images[label].extend(frames)

        # Cap per-identity and report counts
        images_list: list[np.ndarray] = []
        labels_list: list[int] = []
        for label_idx in range(num_classes):
            imgs = all_images[label_idx]
            if not imgs:
                print(
                    f"[identity-model] WARNING: no images for "
                    f"{identity_names[label_idx]}",
                    file=sys.stderr,
                )
                continue
            if len(imgs) > p.max_images_per_identity:
                rng = np.random.default_rng(42)
                indices = rng.choice(
                    len(imgs), p.max_images_per_identity, replace=False
                )
                imgs = [imgs[i] for i in indices]
            print(
                f"[identity-model]   {identity_names[label_idx]}: {len(imgs)} images",
                file=sys.stderr,
            )
            images_list.extend(imgs)
            labels_list.extend([label_idx] * len(imgs))

        if not images_list:
            msg = (
                "[identity-model] No images collected. Check sequence keys "
                "and crop output."
            )
            raise RuntimeError(msg)

        # Crops go to the network at their stored size. The network resizes
        # once, to whatever the backbone declares -- resizing here as well would
        # resample twice, and with ``image_size`` free to follow the backbone
        # this layer no longer knows the target.
        images_arr = np.stack(images_list, axis=0)
        labels_arr = np.array(labels_list, dtype=np.int64)

        # Train/val split
        val_images: np.ndarray | None = None
        val_labels: np.ndarray | None = None
        if p.val_split > 0:
            rng = np.random.default_rng(42)
            n = len(images_arr)
            n_val = max(1, int(n * p.val_split))
            perm = rng.permutation(n)
            val_idx = perm[:n_val]
            train_idx = perm[n_val:]

            val_images = images_arr[val_idx]
            val_labels = labels_arr[val_idx]
            images_arr = images_arr[train_idx]
            labels_arr = labels_arr[train_idx]

        self._network = ClassifierIdentityNetwork(
            num_classes=num_classes,
            model_name=p.model_name,
            image_size=p.image_size,
            freeze_backbone=p.freeze_backbone,
        )
        self._history = self._network.fit(
            images_arr,
            labels_arr,
            val_images=val_images,
            val_labels=val_labels,
            epochs=p.epochs,
            lr=p.learning_rate,
            batch_size=p.batch_size,
        )

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Passthrough -- identity predictions are consumed downstream."""
        return df

    def save_state(self, run_root: Path) -> None:
        from mosaic.behavior.model_library.identity_classifier import (
            ClassifierIdentityNetwork,
        )

        if self._network is None:
            return
        run_root.mkdir(parents=True, exist_ok=True)

        if isinstance(self._network, ClassifierIdentityNetwork):
            weights_name = f"{self.params.weights_name}.pth"
            # The head scores identities by index and nothing in the weights
            # records what animal an index means, so the class order is the
            # only link back to the animals.
            self._network.export_checkpoint(
                run_root / weights_name,
                class_labels=self._identity_names,
            )
            # The sidecar a later run references as ``model``. Written here so
            # this run's output is loadable as the next run's pre-fitted model.
            bundle: ClassifierIdentityBundle = {
                "weights": weights_name,
                "identity_names": list(self._identity_names or ()),
                "version": self.version,
            }
            joblib.dump(bundle, run_root / _BUNDLE_NAME)

        if self._history is not None:
            joblib.dump(self._history, run_root / "training_history.joblib")

        if self._identity_names is not None:
            joblib.dump(self._identity_names, run_root / "identity_names.joblib")
