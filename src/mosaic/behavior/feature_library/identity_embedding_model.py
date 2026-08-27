"""GlobalIdentityEmbedding feature.

Sibling of
:class:`~mosaic.behavior.feature_library.identity_model.GlobalIdentityModel`
that uses any timm-resolvable image backbone as a frozen embedding
extractor and predicts identities by cosine-similarity k-NN against
per-identity prototypes computed at fit time. Nothing is trained, so there is
no per-animal training cycle -- ``fit()`` only computes prototypes.

Choosing between this and ``global-identity-dinov2-temporal``: this one trains
nothing and embeds **single frames**, so it is a zero-shot baseline that
returns an answer in one pass. That one trains a small head on **clips**, so it
can learn identity cues that only appear over time (gait, posture dynamics) and
needs a training cycle. Run the cheap one first.

Mosaic distributes no model weights. Whatever ``model_name`` names is fetched
at run time under its own license -- see
:mod:`mosaic.behavior.model_library.identity_embedding`.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import ClassVar, TypedDict, final

import joblib
import numpy as np
import pandas as pd
from pydantic import Field

from mosaic.behavior.model_library.identity_embedding import DEFAULT_MODEL_NAME
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
from mosaic.core.params import Params

from .registry import register_feature

# --- Model artifact ---

# The exported weights are a torch ``.pth``, and an ArtifactSpec can only load
# npz / parquet / joblib -- so the referencable artifact is this joblib sidecar,
# written beside the checkpoint and naming it. Same shape as
# ``lightning_action_feature``. The name is fixed rather than derived from
# ``weights_name`` because dependency resolution globs it, and the run root also
# holds ``identity_names.joblib`` and ``training_history.joblib``.
_BUNDLE_NAME = "identity_embedding_model.joblib"


class EmbeddingIdentityBundle(TypedDict):
    """Sidecar naming the exported embedding-identity checkpoint.

    Attributes:
        weights: Checkpoint filename, relative to the bundle's directory.
        identity_names: Class order the checkpoint was exported with.
        version: Feature version that wrote the bundle.
    """

    weights: str
    identity_names: list[str]
    version: str


class EmbeddingIdentityArtifact(JoblibArtifact[EmbeddingIdentityBundle]):
    """Fitted embedding-identity bundle (identity_embedding_model.joblib)."""

    feature: str = "global-identity-embedding"
    pattern: str = _BUNDLE_NAME
    load: JoblibLoadSpec = Field(default_factory=JoblibLoadSpec)


# --- Feature class ---


@final
@register_feature
class GlobalIdentityEmbedding:
    """Train an identity model from frozen backbone embeddings + k-NN.

    Takes EgocentricCrop output as input. Each identity is specified as a
    mapping of identity names to lists of sequences containing that
    individual alone. Computes a prototype embedding per identity from the
    training crops and predicts at inference by cosine k-NN against those
    prototypes.

    Example::

        ego_result = dataset.run_feature(EgocentricCrop())

        # The default backbone: MIT-licensed, commercially usable.
        identity_model = GlobalIdentityEmbedding(
            Inputs((Result(feature="egocentric-crop"),)),
            params={
                "identities": {
                    "mouse_A": ["cage1/day1_mouseA_alone"],
                    "mouse_B": ["cage1/day1_mouseB_alone"],
                    "mouse_C": ["cage1/day2_mouseC_alone"],
                    "mouse_D": ["cage1/day1_mouseD_alone"],
                },
            },
        )
        result = dataset.run_feature(identity_model)

        # Wildlife-pretrained weights instead: markedly stronger for animal
        # re-identification, and CC-BY-NC-4.0, so not for commercial use.
        academic = GlobalIdentityEmbedding(
            Inputs((Result(feature="egocentric-crop"),)),
            params={
                "identities": {...},
                "model_name": "BVRA/MegaDescriptor-L-384",
            },
        )

    Params:
        model: Pre-fitted EmbeddingIdentityArtifact to load, skipping the
            fit. Default None (fit from scratch). Pinning one makes an
            inference run's identity carry its training run by reference, so
            the run needs no scope of its own.
        identities: Explicit identity -> sequences mapping.
        group_as_identity: Treat each group name as one identity. Default
            False.
        model_name: A bare timm architecture tag or a Hugging Face hub id for
            the frozen backbone. Default
            ``"timm/swin_large_patch4_window12_384.ms_in22k_ft_in1k"`` (MIT).
            Mosaic ships no weights; whatever is named here is downloaded at
            run time under its own license. ``"BVRA/MegaDescriptor-L-384"`` is
            markedly stronger for animal re-identification and is
            CC-BY-NC-4.0, so it is not available for commercial use.
        image_size: Crop resize target ``(height, width)``. Default None,
            meaning follow the backbone's declared input size -- which is
            almost always what you want, and is the only value correct for
            every backbone.
        channels: Number of channels read from disk (1 = grayscale,
            3 = RGB). Grayscale inputs are replicated to 3 channels for
            the backbone. Default 3.
        batch_size: Embedding batch size. Default 32.
        max_images_per_identity: Cap on training crops per identity.
            Default 2000.
        crop_root: Optional EgocentricCrop output root override.
        weights_name: Stem of the exported ``.pth`` checkpoint. Default
            ``"identity_embedding"``.
    """

    category = "global"
    name: str = "global-identity-embedding"
    version: str = "0.1"
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
    ModelArtifact = EmbeddingIdentityArtifact

    class Inputs(Inputs[Result]):
        _require: ClassVar[InputRequire] = "any"

    class Params(Params):
        """Embedding-identity model parameters."""

        # Pre-fitted model reference: when set (and resolvable), fit is skipped.
        model: EmbeddingIdentityArtifact | None = None

        # Primary: explicit identity -> sequences mapping
        identities: dict[str, list[str]] | None = None
        # Convenience shortcut: treat each group as one identity
        group_as_identity: bool = False

        # Backbone selection. Changing ``model_name`` is a licensing decision as
        # well as an accuracy one -- see the class docstring.
        model_name: str = DEFAULT_MODEL_NAME
        # None means follow the backbone's declared input size.
        image_size: tuple[int, int] | None = None
        channels: int = 3

        # Inference
        batch_size: int = Field(default=32, ge=1)

        # Sampling
        max_images_per_identity: int = Field(default=2000, ge=1)

        # Export
        weights_name: str = "identity_embedding"

        # Path to EgocentricCrop output root.
        crop_root: str | None = None

    def __init__(
        self,
        inputs: GlobalIdentityEmbedding.Inputs,
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
        from mosaic.behavior.model_library.identity_embedding import (
            EmbeddingIdentityNetwork,
        )

        self._network = None
        self._history = None
        self._identity_names = None

        # Branch 1: this run's own cached checkpoint.
        cached_path = run_root / f"{self.params.weights_name}.pth"
        if cached_path.exists():
            self._network = EmbeddingIdentityNetwork.from_checkpoint(cached_path)
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
            self._network = EmbeddingIdentityNetwork.from_checkpoint(
                bundle_path.parent / bundle["weights"]
            )
            self._identity_names = list(bundle["identity_names"])
            return True

        return False

    def fit(self, inputs: InputStream) -> None:
        from mosaic.behavior.model_library.identity_common import (
            build_label_mapping,
            load_crop_frames,
        )
        from mosaic.behavior.model_library.identity_embedding import (
            EmbeddingIdentityNetwork,
        )

        p = self.params

        seq_to_label, identity_names = build_label_mapping(p, inputs)
        self._identity_names = identity_names
        num_classes = len(identity_names)

        if num_classes < 2:
            msg = (
                f"[identity-embedding] Need at least 2 identities, "
                f"got {num_classes}: {identity_names}"
            )
            raise ValueError(msg)

        print(
            f"[identity-embedding] training with {num_classes} identities: "
            f"{identity_names}",
            file=sys.stderr,
        )

        # Collect images per identity
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
                    f"[identity-embedding] WARNING: no images for "
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
                f"[identity-embedding]   {identity_names[label_idx]}: "
                f"{len(imgs)} images",
                file=sys.stderr,
            )
            images_list.extend(imgs)
            labels_list.extend([label_idx] * len(imgs))

        if not images_list:
            msg = (
                "[identity-embedding] No images collected. Check sequence keys "
                "and crop output."
            )
            raise RuntimeError(msg)

        # Crops go to the network at their stored size. The network resizes once,
        # to whatever the backbone declares -- resizing here as well would
        # resample twice, and with ``image_size`` free to follow the backbone
        # this layer no longer knows the target.
        images_arr = np.stack(images_list, axis=0)
        labels_arr = np.array(labels_list, dtype=np.int64)

        # Hold out a small validation slice for top-1 reporting
        val_images: np.ndarray | None = None
        val_labels: np.ndarray | None = None
        if len(images_arr) > 10 * num_classes:
            rng = np.random.default_rng(42)
            n = len(images_arr)
            n_val = max(num_classes, int(n * 0.1))
            perm = rng.permutation(n)
            val_idx = perm[:n_val]
            train_idx = perm[n_val:]
            val_images = images_arr[val_idx]
            val_labels = labels_arr[val_idx]
            images_arr = images_arr[train_idx]
            labels_arr = labels_arr[train_idx]

        self._network = EmbeddingIdentityNetwork(
            model_name=p.model_name,
            image_size=p.image_size,
        )
        self._history = self._network.fit(
            images_arr,
            labels_arr,
            val_images=val_images,
            val_labels=val_labels,
            num_classes=num_classes,
            batch_size=p.batch_size,
        )
        # Stash identity names on the network for checkpoint export.
        self._network._identity_names = identity_names  # pyright: ignore[reportPrivateUsage]

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Passthrough -- identity predictions are consumed downstream."""
        return df

    def save_state(self, run_root: Path) -> None:
        from mosaic.behavior.model_library.identity_embedding import (
            EmbeddingIdentityNetwork,
        )

        if self._network is None:
            return
        run_root.mkdir(parents=True, exist_ok=True)

        if isinstance(self._network, EmbeddingIdentityNetwork):
            weights_name = f"{self.params.weights_name}.pth"
            self._network.export_checkpoint(run_root / weights_name)
            # The sidecar a later run references as ``model``. Written here so
            # this run's output is loadable as the next run's pre-fitted model.
            bundle: EmbeddingIdentityBundle = {
                "weights": weights_name,
                "identity_names": list(self._identity_names or ()),
                "version": self.version,
            }
            joblib.dump(bundle, run_root / _BUNDLE_NAME)

        if self._history is not None:
            joblib.dump(self._history, run_root / "training_history.joblib")

        if self._identity_names is not None:
            joblib.dump(self._identity_names, run_root / "identity_names.joblib")
