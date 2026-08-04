"""Resolve a model reference (artifact path or prior training run_id) to weights + lineage.

A model is not always a file. Ultralytics ships one ``best.pt``, but a Lightning
Pose model is a *directory* -- a ``config.yaml`` beside a checkpoint -- and a
SLEAP top-down model is an ordered *pair* of directories, centroid then
centered-instance, which are not interchangeable. TREx's visual-identification
weights are a third shape again: the path it wants is an extensionless prefix,
``<run_root>/identity_model``, for a file that is actually ``identity_model.pth``.

So the unit is an **artifact**, described by a per-kind spec rather than
discovered by inspection. Each spec names the shape, how many artifacts the
reference carries, and which files inside one are *significant* -- the roles that
identity is allowed to read.

That last point is the design, not an implementation detail. **Identity reads
only declared roles, so whatever a tool writes into a model directory after
training is invisible to it by construction.** Lightning Pose writes
``video_preds/`` into the directory it was loaded from; a digest over the whole
tree would change the moment inference ran, and the same model would stop
matching its own cached output. ``video_preds/`` is not part of the model, and a
spec is how that is said once instead of special-cased everywhere.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal

import pandas as pd
import yaml

from mosaic.core.pipeline._utils import hash_params
from mosaic.core.pipeline.file_digest import file_digest
from mosaic.core.pipeline.models import model_index_path

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset

__all__ = [
    "MODEL_IDENTITY_SCHEME",
    "MODEL_KINDS",
    "Arity",
    "ModelArtifact",
    "ModelKindSpec",
    "ModelShape",
    "ResolvedModel",
    "Role",
    "RoleSpec",
    "resolve_model",
    "resolve_model_set",
    "spec_for",
]

MODEL_IDENTITY_SCHEME: Final = "1"
"""The contract turning a model reference into an identity term.

Its own family rather than a bump of ``OP_IDENTITY_SCHEME``. That constant covers
``frames``, ``transcode``, ``convert``, ``train``, ``infer`` and ``trex`` as one
payload-shape contract, and this change touches only how the last three name a
*model*. Bumping the shared number would mark ``frames`` as re-minted -- and
``frames`` is frozen permanently, because mosaic-api embeds its identifier inside
``AnnotationFrame.image_path`` on version-controlled rows carrying keypoint
labour. A marker that lies is worse than none, inside a family as much as across
families.

Born at "1" with the behaviour below, so nothing has to be retrofitted: before
this, a model reference contributed either a run identity or the *path string*,
and neither was recorded as having been produced under any contract.

Still "1" now that a reference can name a directory, because no identifier moved:
every payload spelling below reproduces what its own resolver already minted.
Note also that nothing reads this constant and nothing writes it into a run root,
unlike ``OP_IDENTITY_SCHEME``, which ``write_identity_scheme`` records on disk.
Wiring it up so ``reconcile`` can compare it is separate work; until then a bump
would be inert -- a reason to distrust one, not a reason to avoid one.
"""

ModelShape = Literal["file", "directory", "prefix"]
"""What a reference points at.

``file``
    The reference is the weights (Ultralytics ``best.pt``).
``directory``
    The reference is a directory holding the significant files (SLEAP,
    Lightning Pose).
``prefix``
    The reference is a path *stem*: the real file is ``<stem>.<ext>``, and the
    stem itself does not exist. TREx's ``visual_identification_model_path`` is
    this, and it is why the shape cannot be inferred from the filesystem -- a
    prefix fails ``Path.exists()`` exactly as a typo does.
"""

Arity = Literal["one", "ordered"]
"""How many artifacts one reference carries.

``ordered`` exists for SLEAP top-down, where the centroid and centered-instance
directories are passed in sequence and swapping them is a different model.
"""

Role = Literal["weights", "config"]
"""What a significant file *is*, so identity can name it without naming a path."""


@dataclass(frozen=True, slots=True)
class RoleSpec:
    """How to find one significant file inside an artifact.

    Attributes:
        role: What the file is. Reaches the identity payload as part of its key.
        names: Candidate filenames, in preference order.
        glob: Searched, sorted, when no candidate name matched. Lightning Pose
            checkpoints live at an unpredictable
            ``tb_logs/<name>/version_*/checkpoints/*.ckpt``.
        fallback_glob: Searched only when *glob* and *names* found nothing.
        prefer: When several matched, prefer the last whose name contains this
            token -- Lightning Pose writes both ``best`` and ``last``.
        required: When False, an absent file is not an error and the role simply
            does not appear. SLEAP's config is provenance, and a model directory
            without one still runs.
        in_identity: Whether this role's digest reaches the identity payload.
            Separate from *required* because SLEAP's config is read, for
            ``model_type``, but deliberately not named.
    """

    role: Role
    names: tuple[str, ...] = ()
    glob: str = ""
    fallback_glob: str = ""
    prefer: str = ""
    required: bool = True
    in_identity: bool = True


@dataclass(frozen=True, slots=True)
class ModelKindSpec:
    """Everything identity needs to know about one kind of model.

    Attributes:
        shape: What a reference points at.
        arity: How many artifacts one reference carries.
        roles: The significant files, in the order errors should be raised in.
        payload_prefix: Names the identity payload's keys, as
            ``<prefix>_<role>``. ``None`` means the identity *is* the weights
            digest, unwrapped -- what a single-file model has always minted, and
            changing it would move every registered model.
        config_names: Candidate files to read ``model_type`` from. The first that
            exists is the one consulted.
        model_types: Recognised ``model_type`` tokens, in preference order,
            matched against what the parsed config *selects* -- a key with a
            non-null value, or a scalar value. Not a text scan: a framework that
            writes its merged config names every head it knows, all but one of
            them null, and a scan cannot tell the configured one from the
            candidates. This is provenance recorded on a row, not identity, so an
            unreadable config gives an empty answer rather than raising.
        label: How the kind is named in an error a human reads.
    """

    shape: ModelShape = "file"
    arity: Arity = "one"
    roles: tuple[RoleSpec, ...] = (RoleSpec(role="weights"),)
    payload_prefix: str | None = None
    config_names: tuple[str, ...] = ()
    model_types: tuple[str, ...] = ()
    label: str = "model"


# SLEAP checkpoint filenames, in preference order. ``best.ckpt`` is current
# sleap-nn; ``best_model.h5`` is a classic (<=1.4) TensorFlow UNet checkpoint,
# which sleap-nn still runs inference on.
_SLEAP_CHECKPOINTS: Final[tuple[str, ...]] = ("best.ckpt", "best_model.h5")

# Longest-first, so a ``multi_class_topdown`` config is not misread as
# ``centered_instance``.
_SLEAP_HEADS: Final[tuple[str, ...]] = (
    "multi_class_topdown",
    "multi_class_bottomup",
    "centered_instance",
    "single_instance",
    "centroid",
    "bottomup",
)

# Longest-first, so ``heatmap_mhcrnn`` is not misread as ``heatmap``.
_LITPOSE_MODEL_TYPES: Final[tuple[str, ...]] = (
    "heatmap_multiview_transformer",
    "heatmap_mhcrnn",
    "regression",
    "heatmap",
)

_SLEAP_SPEC: Final = ModelKindSpec(
    shape="directory",
    arity="ordered",
    roles=(RoleSpec(role="weights", names=_SLEAP_CHECKPOINTS, glob="*.ckpt"),),
    payload_prefix="sleap",
    config_names=("training_config.yaml", "training_config.json"),
    model_types=_SLEAP_HEADS,
    label="SLEAP model",
)

_LITPOSE_SPEC: Final = ModelKindSpec(
    shape="directory",
    # Config first, so a directory missing ``config.yaml`` reports that
    # rather than reporting the checkpoint it also does not have.
    roles=(
        RoleSpec(role="config", names=("config.yaml",)),
        RoleSpec(
            role="weights",
            glob="tb_logs/*/version_*/checkpoints/*.ckpt",
            fallback_glob="**/*.ckpt",
            prefer="best",
        ),
    ),
    payload_prefix="litpose",
    config_names=("config.yaml",),
    model_types=_LITPOSE_MODEL_TYPES,
    label="Lightning Pose model",
)

MODEL_KINDS: Final[Mapping[str, ModelKindSpec]] = {
    "sleap": _SLEAP_SPEC,
    "litpose": _LITPOSE_SPEC,
    # A training op produces its framework's shape, so it names the same spec.
    # Declared rather than left to the ``train-`` fallback: the fallback is a
    # rule about names, and a rule about names cannot tell an intended
    # inheritance from a collision. Saying it here makes the intent checkable,
    # and keeps the fallback for kinds nobody has thought about yet.
    "train-sleap": _SLEAP_SPEC,
    "train-litpose": _LITPOSE_SPEC,
    # TREx's visual-identification weights. The file is ``identity_model.pth``
    # and the path TREx is handed is the extensionless stem beside it, so the
    # reference cannot be probed for -- it has to be declared.
    "train-identity": ModelKindSpec(
        shape="prefix",
        roles=(RoleSpec(role="weights", names=("identity_model.pth",)),),
        label="identity model",
    ),
}
"""Per-kind specs. A kind absent here is a single weights file, which is what
every Ultralytics-backed training op produces."""

_DEFAULT_SPEC: Final = ModelKindSpec()


def spec_for(kind: str) -> ModelKindSpec:
    """The spec for *kind*, or the single-weights-file default.

    ``train-<framework>`` resolves to ``<framework>``'s spec, because a training
    op that produces a framework's model produces that framework's *shape* --
    ``train-litpose`` writes exactly the directory ``litpose`` describes. The two
    names are genuinely different things and both are needed: the op kind says
    which ``models/<kind>/index.csv`` a row lives in, the spec says what the
    artifact looks like. Deriving one from the other beats asking every caller to
    carry both.

    The fallback is narrow by construction: ``train-pose``, ``train-points`` and
    ``train-localizer`` strip to names no spec claims, so they land on the
    single-file default, which is what Ultralytics and the localizer produce.
    """
    spec = MODEL_KINDS.get(kind)
    if spec is not None:
        return spec
    framework = kind.removeprefix("train-")
    return MODEL_KINDS.get(framework, _DEFAULT_SPEC)


@dataclass(frozen=True, slots=True)
class ModelArtifact:
    """One model: where it lives, what inside it counts, and what runs it.

    Attributes:
        root: The reference itself -- the file, the directory, or the prefix.
        files: Each resolved significant file paired with its role, in spec
            order. What identity reads and what ``consumed_source_roots``
            records.
        exec_path: What the tool is handed. Differs from *root* only for a
            prefix, where the root is the stem and the file is the stem plus an
            extension.
        digest: This artifact's weights digest, kept per-artifact because
            SLEAP's payload is an ordered list of them.
    """

    root: Path
    files: tuple[tuple[Role, Path], ...]
    exec_path: Path
    digest: str

    def file_for(self, role: Role) -> Path | None:
        """The resolved file for *role*, or ``None`` when the spec made it optional."""
        for candidate_role, path in self.files:
            if candidate_role == role:
                return path
        return None


@dataclass(frozen=True, slots=True)
class ResolvedModel:
    """Weights, and what names them.

    Returned instead of a widened tuple. There are five unpack sites and the
    added field is a third string beside two that already look like identifiers;
    a positional swap between ``run_id`` and ``digest`` would type-check, run,
    and mint plausible-looking identities from the wrong value.

    ``run_id`` is the training run that produced these weights, or ``""`` for
    weights handed in as a bare path -- there is no run to name. ``digest`` is
    what makes that second case identifiable anyway.
    """

    artifacts: tuple[ModelArtifact, ...]
    run_id: str
    digest: str
    model_type: str = ""

    @property
    def path(self) -> Path:
        """What the tool is handed, for a reference carrying one artifact.

        The first artifact when there are several, which is the centroid stage of
        a SLEAP top-down pair -- callers wanting both want :attr:`paths`.
        """
        return self.artifacts[0].exec_path

    @property
    def paths(self) -> list[Path]:
        """Every artifact's execution path, in reference order."""
        return [artifact.exec_path for artifact in self.artifacts]

    @property
    def significant_files(self) -> tuple[Path, ...]:
        """Every declared file across every artifact, in spec order.

        What ``consumed_roots_for`` is given, so a run records the bytes it
        actually read rather than a directory that may hold anything.
        """
        return tuple(
            path for artifact in self.artifacts for _role, path in artifact.files
        )

    @property
    def model_id(self) -> str:
        """What identity calls this model.

        The training run when there is one: readable, stable across a copy or a
        move, and it already names a directory a human can go and look at. The
        weights digest otherwise. **Never the path** -- a path names a location,
        and two locations hold different weights as readily as the same ones,
        which is the whole defect item 4.6 exists to close.
        """
        return self.run_id or self.digest


# --- Resolving one artifact -------------------------------------------------


def _resolve_role(directory: Path, role: RoleSpec, spec: ModelKindSpec) -> Path | None:
    """Find *role*'s file inside *directory*, or raise when it is required.

    Named candidates first, then the glob, then the fallback glob -- each sorted,
    so a directory holding several checkpoints always resolves to the same one.
    """
    for name in role.names:
        candidate = directory / name
        if candidate.exists():
            return candidate

    matches: list[Path] = []
    if role.glob:
        matches = sorted(directory.glob(role.glob))
    if not matches and role.fallback_glob:
        matches = sorted(directory.glob(role.fallback_glob))
    if matches:
        if role.prefer:
            preferred = [m for m in matches if role.prefer in m.name.lower()]
            return preferred[-1] if preferred else matches[-1]
        return matches[0]

    if not role.required:
        return None
    wanted = " / ".join(role.names) or role.glob
    raise FileNotFoundError(
        f"No {spec.label} {role.role} ({wanted}) found in directory: {directory}"
    )


def _selected_tokens(node: object, into: set[str]) -> None:
    """Collect every token the document *selects*, as opposed to merely mentions.

    A token is selected when it is a mapping key with a non-null value -- the
    shape a framework uses to say "this head, configured so" -- or when it is a
    scalar value, the shape used to say ``model_type: heatmap``. A key whose
    value is null is the opposite of a selection: it is the framework listing a
    candidate it did *not* pick.
    """
    if isinstance(node, dict):
        for key, value in node.items():  # pyright: ignore[reportUnknownVariableType]
            if isinstance(key, str) and value is not None:
                into.add(key)
            _selected_tokens(value, into)
    elif isinstance(node, list):
        for value in node:  # pyright: ignore[reportUnknownVariableType]
            _selected_tokens(value, into)
    elif isinstance(node, str):
        into.add(node)


def _read_model_type(directory: Path, spec: ModelKindSpec) -> str:
    """Best-effort ``model_type`` token from the config, for provenance.

    Parsed rather than scanned as text. A text scan cannot tell
    ``centered_instance: {...}`` from ``multi_class_topdown: null``, and a
    framework that writes its *merged* config names every head it knows -- eight
    of them null and one configured. Every SLEAP model was then recorded as
    whichever token came first in :data:`ModelKindSpec.model_types`, regardless
    of what was trained.

    YAML covers JSON, so one parse serves both ``config_names`` spellings. This
    is provenance recorded on a row, not identity, so an unreadable or
    unparseable config gives an empty string rather than raising -- and a
    document that parses to nothing recognisable falls back to the text scan,
    which is still right for a config that names only what it selected.

    The first config that exists is the one consulted -- a second candidate is a
    different serialisation of the same thing, not a fallback.
    """
    for name in spec.config_names:
        config = directory / name
        if not config.exists():
            continue
        try:
            text = config.read_text()
        except OSError:
            return ""
        selected: set[str] = set()
        try:
            _selected_tokens(yaml.safe_load(text), selected)
        except yaml.YAMLError:
            selected = set()
        for token in spec.model_types:
            if token in selected:
                return token
        for token in spec.model_types:
            if token in text:
                return token
        return ""
    return ""


def _resolve_artifact(reference: Path, spec: ModelKindSpec) -> ModelArtifact:
    """One reference -> one artifact, per *spec*'s shape."""
    if spec.shape == "file":
        if not reference.exists():
            raise FileNotFoundError(f"{spec.label} file does not exist: {reference}")
        return ModelArtifact(
            root=reference,
            files=(("weights", reference),),
            exec_path=reference,
            digest=file_digest(reference),
        )

    if spec.shape == "prefix":
        # The stem does not exist; the file beside it does. Probed only here and
        # never by inspection, so a run_id-shaped reference cannot accidentally
        # glob the working directory.
        named = [reference.parent / name for name in spec.roles[0].names]
        found = [candidate for candidate in named if candidate.exists()]
        if not found:
            found = sorted(reference.parent.glob(reference.name + ".*"))
        if not found:
            raise FileNotFoundError(f"{spec.label} prefix names no file: {reference}.*")
        return ModelArtifact(
            root=reference,
            files=(("weights", found[0]),),
            exec_path=reference,
            digest=file_digest(found[0]),
        )

    if not reference.exists():
        raise FileNotFoundError(f"{spec.label} directory does not exist: {reference}")
    if not reference.is_dir():
        raise NotADirectoryError(
            f"{spec.label} reference is not a directory: {reference}"
        )
    files: list[tuple[Role, Path]] = []
    weights: Path | None = None
    for role in spec.roles:
        resolved = _resolve_role(reference, role, spec)
        if resolved is None:
            continue
        files.append((role.role, resolved))
        if role.role == "weights":
            weights = resolved
    if weights is None:
        raise FileNotFoundError(f"{spec.label} directory has no weights: {reference}")
    return ModelArtifact(
        root=reference,
        files=tuple(files),
        exec_path=reference,
        digest=file_digest(weights),
    )


def _identity(artifacts: Sequence[ModelArtifact], spec: ModelKindSpec) -> str:
    """The digest naming *artifacts*, spelled the way *spec* declares.

    Two spellings, and the split is what keeps every existing identifier where it
    is. With no ``payload_prefix`` the identity *is* the weights digest,
    unwrapped -- what a single-file model has minted since item 4.6. With one, it
    is ``hash_params`` over ``<prefix>_<role>`` keys, which is what the SLEAP and
    Lightning Pose resolvers already minted, down to the list-valued weights key
    SLEAP uses even for a single directory.
    """
    if spec.payload_prefix is None:
        return artifacts[0].digest

    payload: dict[str, object] = {}
    for role in spec.roles:
        if not role.in_identity:
            continue
        digests = [
            file_digest(path)
            for artifact in artifacts
            for candidate_role, path in artifact.files
            if candidate_role == role.role
        ]
        if not digests:
            continue
        payload[f"{spec.payload_prefix}_{role.role}"] = (
            digests if spec.arity == "ordered" else digests[0]
        )
    return hash_params(payload)


def _model_type_of(artifact: ModelArtifact, spec: ModelKindSpec) -> str:
    """Provenance for a directory-shaped artifact; nothing to read otherwise."""
    return _read_model_type(artifact.root, spec) if spec.shape == "directory" else ""


# --- Public entry points ----------------------------------------------------


def _registered_artifact_path(ds: Dataset, ref: str, kind: str) -> Path:
    """The artifact a registered training run left behind, or raise.

    ``artifact_path`` when the row carries one, ``best_model_path`` otherwise --
    rows written before a model could be a directory name only the single file,
    and they must keep resolving.

    Read with ``keep_default_na=False`` so an empty cell arrives as ``""``. Left
    to its defaults, pandas turns it into ``NaN``, which is *truthy*, so an
    absent ``artifact_path`` resolved to the literal path ``nan`` instead of
    falling back.
    """
    idx_path = model_index_path(ds, kind)
    if not idx_path.exists():
        raise FileNotFoundError(
            f"Model reference '{ref}' is not a path and {idx_path} does not "
            f"exist; cannot resolve as a run_id."
        )
    df = pd.read_csv(idx_path, keep_default_na=False)
    match = df[df["run_id"].astype(str) == ref]
    if match.empty:
        raise KeyError(f"No model run_id '{ref}' found in {idx_path}")
    row = match.iloc[0]
    stored = ""
    if "artifact_path" in match.columns:
        stored = str(row["artifact_path"]).strip()
    if not stored:
        stored = str(row["best_model_path"]).strip()
    return ds.resolve_path(stored)


def resolve_model(ds: Dataset, ref: str, kind: str) -> ResolvedModel:
    """Resolve a model reference to its artifact, lineage and content digest.

    *ref* is either a filesystem path to a model artifact or a prior training
    ``run_id`` in ``models/<kind>/index.csv``. This powers
    retrain-from-existing-model and the trained-model -> TREx ``detect_model``
    handoff.

    A bare path is still accepted. Refusing it would break the documented
    ``detect_model=/path/to/best.pt`` workflow, and the digest is precisely the
    answer to "this reference carries no lineage", so refusal buys nothing that
    measuring does not.

    The digest is computed on both branches. For a registered model it never
    reaches identity -- ``model_id`` prefers the run -- but it is what the index
    row records and what a future integrity check would compare.

    Args:
        ds: The dataset whose ``models/`` root holds the run index.
        ref: A path to the artifact, or a registered training ``run_id``.
        kind: Selects the spec in :data:`MODEL_KINDS`. An unregistered kind
            resolves as a single weights file.

    Returns:
        The resolved model, carrying one artifact.

    Raises:
        FileNotFoundError: The reference names nothing, or the artifact is
            missing a required file.
        NotADirectoryError: A directory-shaped kind was given a file.
        KeyError: A ``run_id`` absent from the index.
    """
    spec = spec_for(kind)
    reference = Path(ref)
    # A prefix never exists, so the spec decides before the filesystem does.
    if spec.shape == "prefix" or reference.exists():
        artifact = _resolve_artifact(reference, spec)
        return ResolvedModel(
            artifacts=(artifact,),
            run_id="",
            digest=_identity((artifact,), spec),
            model_type=_model_type_of(artifact, spec),
        )

    artifact = _resolve_artifact(_registered_artifact_path(ds, ref, kind), spec)
    return ResolvedModel(
        artifacts=(artifact,),
        run_id=ref,
        digest=_identity((artifact,), spec),
        model_type=_model_type_of(artifact, spec),
    )


def resolve_model_set(
    ds: Dataset | None, refs: Sequence[str], kind: str
) -> ResolvedModel:
    """Resolve an ordered set of references as one model.

    SLEAP top-down is two directories -- centroid, then centered-instance -- and
    they are not interchangeable, so the order reaches identity. Resolving here,
    before anything is minted, means an unresolvable reference aborts before any
    run root or tracks variant is written.

    A separate function rather than widening *ref* to ``str | Sequence[str]``:
    ``str`` *is* a ``Sequence[str]``, so the union cannot be discriminated and a
    caller passing one path would silently resolve it character by character.

    Args:
        ds: Needed only to resolve a registered ``run_id``. ``None`` is honest
            for an external framework model, which never consults the index.
        refs: One reference per artifact, in the order the tool expects them.
        kind: Selects the spec in :data:`MODEL_KINDS`.

    Returns:
        The resolved model, carrying one artifact per reference.

    Raises:
        ValueError: *refs* is empty.
        FileNotFoundError: A reference names nothing, or names a ``run_id``
            while *ds* is ``None``.
    """
    spec = spec_for(kind)
    if not refs:
        raise ValueError(f"resolving a {spec.label} requires at least one reference")

    artifacts: list[ModelArtifact] = []
    registered: list[str] = []
    model_type = ""
    for ref in refs:
        reference = Path(ref)
        if spec.shape != "prefix" and not reference.exists():
            if ds is None:
                raise FileNotFoundError(
                    f"{spec.label} directory does not exist: {reference}"
                )
            reference = _registered_artifact_path(ds, ref, kind)
            registered.append(ref)
        artifact = _resolve_artifact(reference, spec)
        artifacts.append(artifact)
        if not model_type:
            model_type = _model_type_of(artifact, spec)

    # A run identity names one run. A set assembled from several has no single
    # one to name, so it falls back to the content digest -- exactly as
    # identifying, and it does not pretend otherwise.
    run_id = registered[0] if len(refs) == 1 and len(registered) == 1 else ""
    return ResolvedModel(
        artifacts=tuple(artifacts),
        run_id=run_id,
        digest=_identity(artifacts, spec),
        model_type=model_type,
    )
