"""The heads of ``sequence-classifier``, its parameters and the pieces of its training that need no torch.

The registry, the options of a head, the parameter model that selects them, the
batches the loop draws and the class weights are plain Python, and they run in the
default job. The networks and the fit are in ``test_sequence_classifier.py``.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from importlib.metadata import EntryPoint
from typing import TYPE_CHECKING, ClassVar, override

import numpy as np
import numpy.typing as npt
import pytest
from pydantic import ValidationError

from mosaic.behavior.feature_library import sequence_heads
from mosaic.behavior.feature_library.sequence_classifier import SequenceClassifier
from mosaic.behavior.feature_library.sequence_heads import (
    HEADS,
    GruConfig,
    HeadConfig,
    HeadSpec,
    MlpConfig,
    TcnConfig,
    TransformerConfig,
    UnetConfig,
    get_head,
    head_names,
    head_source_digest,
    network_reach,
    register_head,
)
from mosaic.behavior.feature_library.sequence_training import (
    Sampler,
    class_weights,
    sub_seed,
)
from mosaic.core.pipeline.types import InputStream

if TYPE_CHECKING:
    from pathlib import Path

    from torch import nn


class ToyConfig(HeadConfig):
    """The options of a head registered by a test."""

    width: int = 4


def _toy_build(config: ToyConfig, n_features: int, n_classes: int) -> nn.Module:
    raise NotImplementedError


def _other_build(config: ToyConfig, n_features: int, n_classes: int) -> nn.Module:
    raise LookupError


def _toy() -> HeadSpec[ToyConfig]:
    return HeadSpec(
        name="toy", config=ToyConfig, build=_toy_build, description="A toy head."
    )


# --- The registry ---------------------------------------------------------------


def test_the_built_in_heads_are_registered_with_their_options() -> None:
    assert head_names() == ["gru", "mlp", "tcn", "transformer", "unet"]
    assert get_head("tcn").config is TcnConfig
    assert get_head("gru").config is GruConfig
    assert get_head("mlp").config is MlpConfig
    assert get_head("transformer").config is TransformerConfig
    assert get_head("unet").config is UnetConfig


def test_an_unknown_head_is_refused_with_the_registered_names() -> None:
    with pytest.raises(
        ValueError,
        match=r"'lstm'.*Registered or installed: gru, mlp, tcn, transformer, unet",
    ):
        get_head("lstm")


def test_a_head_registered_in_python_is_found_by_name(
    head_registry: None,
) -> None:
    spec = register_head(_toy())

    assert get_head("toy") is spec
    assert "toy" in head_names()


def test_a_second_head_of_one_name_is_refused(head_registry: None) -> None:
    _ = register_head(_toy())

    with pytest.raises(ValueError, match="a head named 'toy' is already registered"):
        register_head(_toy())


def test_a_head_builds_only_from_its_own_options() -> None:
    with pytest.raises(TypeError, match="head 'gru' takes GruConfig options"):
        get_head("gru").make(TcnConfig(), 3, 2)


def test_the_digest_of_a_head_follows_its_code(head_registry: None) -> None:
    first = register_head(_toy())
    changed = HeadSpec(
        name="toy-2", config=ToyConfig, build=_other_build, description="Changed."
    )

    assert first.source_digest() == _toy().source_digest()
    assert first.source_digest() != changed.source_digest()
    assert head_source_digest("toy") == first.source_digest()
    assert len(first.source_digest()) == 64


# --- Heads from an installed entry point -----------------------------------------

TOY = _toy()
NOT_A_HEAD = "a string"
MISNAMED = HeadSpec(
    name="other", config=ToyConfig, build=_toy_build, description="A toy head."
)


class _CountingEntryPoint(EntryPoint):
    """An entry point that counts its loads."""

    loads: ClassVar[int] = 0

    @override
    def load(self) -> object:
        type(self).loads += 1
        return super().load()


class _FailingEntryPoint(EntryPoint):
    """An entry point whose load raises."""

    @override
    def load(self) -> object:
        msg = "the head module is broken"
        raise ImportError(msg)


def _entry_point(
    name: str, attribute: str, kind: type[EntryPoint] = EntryPoint
) -> EntryPoint:
    return kind(
        name=name,
        value=f"{__name__}:{attribute}",
        group=sequence_heads.ENTRY_POINT_GROUP,
    )


def _install(
    monkeypatch: pytest.MonkeyPatch, *installed: EntryPoint
) -> list[tuple[str, str | None]]:
    """Make *installed* the entry points of the group, and record the lookups."""
    lookups: list[tuple[str, str | None]] = []

    def lookup(*, group: str, name: str | None = None) -> tuple[EntryPoint, ...]:
        lookups.append((group, name))
        return tuple(found for found in installed if name is None or found.name == name)

    monkeypatch.setattr(sequence_heads, "entry_points", lookup)
    return lookups


def test_an_installed_head_is_loaded_registered_and_loaded_once(
    head_registry: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    _CountingEntryPoint.loads = 0
    _ = _install(monkeypatch, _entry_point("toy", "TOY", _CountingEntryPoint))

    params = _params(head="toy")

    assert params.head_options() == ToyConfig()
    assert get_head("toy") is TOY
    assert HEADS["toy"] is TOY
    assert _CountingEntryPoint.loads == 1
    assert get_head("toy") is TOY
    assert _CountingEntryPoint.loads == 1


def test_a_registered_head_takes_precedence_over_an_entry_point_of_its_name(
    head_registry: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    registered = register_head(_toy())
    lookups = _install(monkeypatch, _entry_point("toy", "MISNAMED"))

    assert get_head("toy") is registered
    assert lookups == []


def test_the_names_of_installed_heads_are_listed_without_loading_them(
    head_registry: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    _CountingEntryPoint.loads = 0
    _ = _install(monkeypatch, _entry_point("toy", "TOY", _CountingEntryPoint))

    assert head_names() == ["gru", "mlp", "tcn", "toy", "transformer", "unet"]
    with pytest.raises(
        ValueError,
        match=r"'lstm'.*Registered or installed: gru, mlp, tcn, toy, transformer, unet",
    ):
        get_head("lstm")
    assert _CountingEntryPoint.loads == 0
    assert "toy" not in HEADS


@pytest.mark.parametrize(
    ("installed", "message"),
    [
        (
            [_entry_point("toy", "TOY", _FailingEntryPoint)],
            r"entry point 'toy'.*distribution.*could not be loaded.*"
            r"ImportError: the head module is broken",
        ),
        (
            [_entry_point("toy", "NOT_A_HEAD")],
            r"entry point 'toy'.*distribution.*is a str, not a HeadSpec",
        ),
        (
            [_entry_point("toy", "MISNAMED")],
            r"entry point 'toy'.*distribution.*names its head 'other'",
        ),
        (
            [_entry_point("toy", "TOY"), _entry_point("toy", "TOY")],
            r"2 entry points are named 'toy'.*distribution",
        ),
    ],
    ids=["load-raises", "not-a-spec", "mismatched-name", "two-of-one-name"],
)
def test_a_bad_entry_point_is_refused_and_registers_nothing(
    head_registry: None,
    monkeypatch: pytest.MonkeyPatch,
    installed: list[EntryPoint],
    message: str,
) -> None:
    _ = _install(monkeypatch, *installed)

    with pytest.raises(ValueError, match=message):
        _ = _params(head="toy")
    assert "toy" not in HEADS


def _child(
    tmp_path: Path, entry_points_txt: str, module_source: str, script: str
) -> subprocess.CompletedProcess[str]:
    """Run *script* in a fresh interpreter that sees a distribution ``x``."""
    dist_info = tmp_path / "x.dist-info"
    dist_info.mkdir()
    _ = (dist_info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: x\nVersion: 1.0\n", encoding="utf-8"
    )
    _ = (dist_info / "entry_points.txt").write_text(entry_points_txt, encoding="utf-8")
    _ = (tmp_path / "x_heads.py").write_text(
        textwrap.dedent(module_source), encoding="utf-8"
    )
    inherited = os.environ.get("PYTHONPATH")
    path = os.pathsep.join([str(tmp_path), inherited] if inherited else [str(tmp_path)])
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": path},
    )


_HEAD_MODULE = """
from mosaic.behavior.feature_library.sequence_heads import HeadConfig, HeadSpec


class XConfig(HeadConfig):
    width: int = 4


def build(config, n_features, n_classes):
    raise NotImplementedError


SPEC = HeadSpec(name="x-head", config=XConfig, build=build, description="An x head.")
MISNAMED = HeadSpec(name="y", config=XConfig, build=build, description="Not x.")
"""


def test_a_fresh_interpreter_finds_a_head_that_a_distribution_installs(
    tmp_path: Path,
) -> None:
    script = """
        import sys

        import importlib

        importlib.import_module("mosaic.behavior")
        from mosaic.behavior.feature_library.sequence_classifier import (
            SequenceClassifier,
        )

        assert "x_heads" not in sys.modules, "an entry point loaded on import"
        params = SequenceClassifier.Params.from_overrides(
            {
                "templates": {"feature": "extract-labeled-templates"},
                "labels": {"kind": "behavior"},
                "default_class": 0,
                "head": "x-head",
                "head_config": {"width": 8},
            }
        )
        assert type(params.head_config).__name__ == "XConfig"
        assert params.head_config.width == 8
        assert "x_heads" in sys.modules
        assert "torch" not in sys.modules
        print("OK")
    """

    proc = _child(
        tmp_path,
        "[mosaic.sequence_heads]\nx-head = x_heads:SPEC\n",
        _HEAD_MODULE,
        script,
    )

    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout


def test_a_refusal_names_the_distribution_of_the_entry_point(tmp_path: Path) -> None:
    script = """
        from mosaic.behavior.feature_library.sequence_heads import get_head

        try:
            get_head("x-head")
        except ValueError as error:
            print(error)
    """

    proc = _child(
        tmp_path,
        "[mosaic.sequence_heads]\nx-head = x_heads:MISNAMED\n",
        _HEAD_MODULE,
        script,
    )

    assert proc.returncode == 0, proc.stderr
    assert "entry point 'x-head' of distribution 'x'" in proc.stdout


def test_an_installed_head_enters_the_run_identity_like_a_registered_one(
    head_registry: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    by_python = register_head(_toy())
    expected = _params(head="toy", head_revision=2).identity_dump()
    del HEADS[by_python.name]
    _ = _install(monkeypatch, _entry_point("toy", "TOY"))

    installed = _params(head="toy", head_revision=2).identity_dump()

    assert installed == expected
    assert installed["head"] == "toy"
    assert installed["head_revision"] == 2


# --- The options of a head ------------------------------------------------------


def test_the_defaults_of_the_built_in_heads_are_the_reference_networks() -> None:
    assert TcnConfig().model_dump() == {
        "stages": 1,
        "hidden": 128,
        "blocks": 6,
        "kernel_size": 3,
        "dropout": 0.3,
    }
    assert GruConfig().model_dump() == {
        "stages": 1,
        "hidden": 128,
        "layers": 2,
        "bidirectional": True,
        "dropout": 0.3,
    }
    assert TransformerConfig().model_dump() == {
        "stages": 1,
        "hidden": 128,
        "layers": 4,
        "attention_heads": 4,
        "window": 64,
        "kernel_size": 5,
        "dropout": 0.1,
    }
    assert UnetConfig().model_dump() == {
        "stages": 1,
        "hidden": 64,
        "depth": 4,
        "kernel_size": 5,
        "dropout": 0.3,
    }


def test_every_head_takes_stages_and_a_network_has_one_or_more() -> None:
    assert ToyConfig().stages == 1
    assert ToyConfig(stages=3).stages == 3
    with pytest.raises(ValidationError, match="stages"):
        MlpConfig(stages=0)


def test_an_option_a_head_does_not_have_is_refused() -> None:
    with pytest.raises(ValidationError, match="layers"):
        TcnConfig.model_validate({"layers": 3})


def test_an_even_kernel_is_refused() -> None:
    with pytest.raises(ValidationError, match="kernel_size 4 is even"):
        TcnConfig(kernel_size=4)
    with pytest.raises(ValidationError, match="kernel_size 4 is even"):
        TransformerConfig(kernel_size=4)
    with pytest.raises(ValidationError, match="kernel_size 4 is even"):
        UnetConfig(kernel_size=4)


def test_the_attention_heads_of_a_transformer_divide_its_width() -> None:
    assert TransformerConfig(hidden=12, attention_heads=3).attention_heads == 3
    with pytest.raises(
        ValidationError, match="attention_heads 5 do not divide hidden 128"
    ):
        TransformerConfig(attention_heads=5)


def test_the_options_of_a_head_are_frozen() -> None:
    config = TcnConfig()

    with pytest.raises(ValidationError):
        setattr(config, "hidden", 4)


# --- The reach of a network ---------------------------------------------------


@pytest.mark.parametrize(
    ("head", "config", "reach"),
    [
        ("tcn", TcnConfig(), 63),
        ("tcn", TcnConfig(blocks=2, stages=2), 3 + 63),
        ("mlp", MlpConfig(stages=3), 2 * 63),
        ("gru", GruConfig(stages=2), None),
        ("transformer", TransformerConfig(), 2 + 5 * 64 - 1),
        ("transformer", TransformerConfig(window=2, layers=2, kernel_size=1), 5),
        ("transformer", TransformerConfig(window=2, layers=2, stages=2), 2 + 5 + 63),
        ("unet", UnetConfig(), 9 * 15),
        ("unet", UnetConfig(depth=2, kernel_size=3, stages=2), 5 * 3 + 63),
    ],
)
def test_each_refinement_stage_adds_the_reach_of_a_default_tcn(
    head: str, config: HeadConfig, reach: int | None
) -> None:
    assert network_reach(head, config) == reach


def test_a_classifier_reaches_as_far_as_its_network(head_registry: None) -> None:
    params = _params(head="tcn", head_config={"blocks": 2, "stages": 3})
    _ = register_head(_toy())

    assert SequenceClassifier.temporal_reach(params) == 3 + 2 * 63
    assert SequenceClassifier.temporal_reach(_params(head="toy")) is None


# --- The parameters -------------------------------------------------------------

FIT = {
    "templates": {"feature": "extract-labeled-templates"},
    "labels": {"kind": "behavior"},
    "default_class": 0,
}


def _params(**overrides: object) -> SequenceClassifier.Params:
    return SequenceClassifier.Params.from_overrides({**FIT, **overrides})


def test_the_head_defaults_to_the_gru_with_its_own_defaults() -> None:
    params = _params()

    assert params.head == "gru"
    assert params.head_options() == GruConfig()
    assert params.head_config == GruConfig()


def test_a_mapping_of_options_is_read_as_the_options_of_the_chosen_head() -> None:
    params = _params(head="tcn", head_config={"blocks": 4})

    assert params.head_options() == TcnConfig(blocks=4)


def test_omitted_options_and_spelled_out_defaults_are_one_identity() -> None:
    omitted = _params(head="tcn")
    spelled = _params(head="tcn", head_config=TcnConfig().model_dump())

    assert omitted.identity_dump() == spelled.identity_dump()
    assert omitted.identity_dump()["head_config"] == {
        "hidden": 128,
        "blocks": 6,
        "kernel_size": 3,
        "dropout": 0.3,
    }


def test_one_stage_and_no_label_smoothing_are_the_identity_without_them() -> None:
    omitted = _params(head="tcn")
    spelled = _params(head="tcn", head_config={"stages": 1}, label_smoothing=0.0)

    assert spelled.identity_dump() == omitted.identity_dump()
    assert "label_smoothing" not in omitted.identity_dump()


def test_no_tie_mapping_is_the_identity_without_one_and_a_mapping_changes_it() -> None:
    omitted = _params(cv_folds=3)

    assert _params(cv_folds=3, cv_group_by=None).identity_dump() == (
        omitted.identity_dump()
    )
    assert "cv_group_by" not in omitted.identity_dump()
    spelled = _params(cv_folds=3, cv_group_by={"s0": "a", "g:s1": ["b", "a"]})
    respelled = _params(cv_folds=3, cv_group_by={"g:s1": ["a", "b"], ":s0": ["a"]})
    assert spelled.identity_dump() == respelled.identity_dump()
    assert spelled.identity_dump() != omitted.identity_dump()


@pytest.mark.parametrize(
    "extra, message",
    [
        ({}, "cv_group_by ties entries into the folds of cv_folds"),
        ({"cv_folds": 3, "cv_unit": "block"}, "cv_group_by ties whole entries"),
    ],
)
def test_a_tie_mapping_without_folds_or_by_block_is_refused(
    extra: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _ = _params(cv_group_by={"s0": "a"}, **extra)


def test_a_label_smoothing_of_one_is_refused() -> None:
    with pytest.raises(ValidationError, match="label_smoothing"):
        _ = _params(label_smoothing=1.0)


def test_an_option_of_another_head_is_refused_and_so_are_options_of_a_wrong_head() -> (
    None
):
    with pytest.raises(ValueError, match="layers"):
        _params(head="tcn", head_config={"layers": 2})
    with pytest.raises(ValueError, match="head_config holds TcnConfig options"):
        _params(head="gru", head_config=TcnConfig())


def test_an_unregistered_head_is_refused_with_the_registered_names() -> None:
    with pytest.raises(
        ValueError, match="Registered or installed: gru, mlp, tcn, transformer, unet"
    ):
        _params(head="lstm")


def test_the_dump_of_a_run_reads_back_to_the_same_params() -> None:
    params = _params(head="mlp", head_config={"layers": 3}, cv_folds=3, cv_seed=2)

    again = SequenceClassifier.Params.model_validate(params.model_dump())

    assert again.identity_dump() == params.identity_dump()


@pytest.mark.parametrize(
    "change",
    [
        {"head": "tcn"},
        {"head_config": {"hidden": 64}},
        {"head_revision": 1},
        {"device": "cpu"},
        {"steps": 50},
        {"class_weight": "balanced"},
        {"random_state": 7},
        {"columns": ["a", "b"]},
        {"label_smoothing": 0.1},
        {"head_config": {"stages": 2}},
        {"head": "transformer"},
        {"head": "unet"},
    ],
)
def test_a_setting_that_changes_the_model_changes_the_run_identity(
    change: dict[str, object],
) -> None:
    assert _params(**change).identity_dump() != _params().identity_dump()


@pytest.mark.parametrize(
    "change",
    [
        {"apply_device": "auto"},
        {"apply_device": "cpu"},
        {"apply_device": "cuda"},
        {"n_threads": 4},
    ],
)
def test_a_throughput_setting_is_not_part_of_the_run_identity(
    change: dict[str, object],
) -> None:
    assert _params(**change).identity_dump() == _params().identity_dump()


def test_the_apply_device_defaults_to_the_training_device() -> None:
    assert _params().apply_device == "auto"


def test_the_apply_device_is_one_of_auto_cpu_and_cuda() -> None:
    with pytest.raises(ValidationError):
        _ = _params(apply_device="gpu")


def test_a_schedule_needs_a_step_to_climb_over() -> None:
    with pytest.raises(
        ValueError, match="leaves no step for the schedule to climb over"
    ):
        _params(steps=10, pct_start=0.1)
    _ = _params(steps=11, pct_start=0.1)


def test_a_fit_reads_labels_and_a_model_reads_none() -> None:
    with pytest.raises(ValueError, match="labels is required with templates"):
        SequenceClassifier.Params.from_overrides(
            {"templates": FIT["templates"], "default_class": 0}
        )
    with pytest.raises(ValueError, match="labels is read when fitting"):
        SequenceClassifier.Params.from_overrides(
            {
                "model": {"feature": "sequence-classifier"},
                "labels": FIT["labels"],
                "default_class": 0,
            }
        )


def test_folds_need_templates_and_a_seed_needs_folds() -> None:
    with pytest.raises(ValueError, match="cv_seed.*cv_folds"):
        _params(cv_seed=1)
    with pytest.raises(ValueError, match="cv_folds.*templates.*model"):
        SequenceClassifier.Params.from_overrides(
            {
                "model": {"feature": "sequence-classifier"},
                "default_class": 0,
                "cv_folds": 3,
            }
        )


def test_no_run_hashes_the_scope_only_a_fit_fits_and_none_is_parallel() -> None:
    inputs = SequenceClassifier.Inputs.model_validate([{"feature": "speed-angvel"}])
    fit = SequenceClassifier(inputs, dict(FIT))
    applied = SequenceClassifier(
        inputs,
        {"model": {"feature": "sequence-classifier"}, "default_class": 0},
    )

    assert SequenceClassifier.fit_set_pinned_by == "templates"
    assert fit.scope_dependent is False
    assert applied.scope_dependent is False
    assert fit.fits is True
    assert applied.fits is False
    assert SequenceClassifier.parallelizable is False


def test_a_model_that_did_not_resolve_is_not_replaced_by_a_fit(tmp_path: Path) -> None:
    inputs = SequenceClassifier.Inputs.model_validate([{"feature": "speed-angvel"}])
    feature = SequenceClassifier(
        inputs,
        {"model": {"feature": "sequence-classifier"}, "default_class": 0},
    )

    assert feature.load_state(tmp_path, {}, {}) is False
    with pytest.raises(RuntimeError, match="did not resolve to a stored model"):
        feature.fit(InputStream(lambda: iter(()), n_entries=0))


# --- The batches ----------------------------------------------------------------


def _reference_batch(
    rng: np.random.Generator,
    bounds: list[tuple[int, int]],
    probabilities: npt.NDArray[np.float64],
    crop: int,
    batch: int,
) -> tuple[list[int], int]:
    """The draws of ``train_model`` in the agent's script, in its order."""
    starts: list[int] = []
    lengths: list[int] = []
    chosen: npt.NDArray[np.int64] = rng.choice(len(bounds), batch, p=probabilities)
    for i in chosen:
        a, b = bounds[int(i)]
        length = min(crop, b - a)
        start = a + int(rng.integers(0, b - a - length + 1))
        starts.append(start)
        lengths.append(length)
    return starts, min(lengths)


def test_a_batch_is_drawn_in_the_order_of_the_agents_script() -> None:
    run_lengths = np.array([300, 90, 1500, 64, 700], dtype=np.intp)
    ends = np.cumsum(run_lengths)
    bounds = [(int(e - n), int(e)) for e, n in zip(ends, run_lengths)]
    sampler = Sampler.of(run_lengths, crop_length=256, batch_size=16)
    probabilities = sampler.probabilities()
    ours, reference = np.random.default_rng(5), np.random.default_rng(5)

    for _ in range(40):
        starts, length = sampler.draw(ours, probabilities)
        expected_starts, expected_length = _reference_batch(
            reference, bounds, probabilities, 256, 16
        )
        assert starts.tolist() == expected_starts
        assert length == expected_length


def test_a_crop_stays_inside_its_run_and_a_batch_shares_its_shortest_crop() -> None:
    run_lengths = np.array([100, 40, 400], dtype=np.intp)
    sampler = Sampler.of(run_lengths, crop_length=64, batch_size=8)
    rng = np.random.default_rng(0)

    for _ in range(100):
        starts, length = sampler.draw(rng, sampler.probabilities())
        assert 1 <= length <= 64
        for start in starts.tolist():
            inside = [
                (int(o), int(o + n))
                for o, n in zip(sampler.offsets, sampler.lengths)
                if o <= start < o + n
            ]
            assert len(inside) == 1
            assert start + length <= inside[0][1]


def test_runs_are_drawn_in_proportion_to_their_length() -> None:
    sampler = Sampler.of(np.array([100, 300], dtype=np.intp), 10, 1)

    np.testing.assert_allclose(sampler.probabilities(), [0.25, 0.75])


# --- Weights and seeds ----------------------------------------------------------


def test_the_class_weights_are_the_agents_scaled_to_a_mean_of_one() -> None:
    counts = np.array([14039, 146615, 28615, 318469], dtype=np.intp)
    balanced = counts.sum() / (4 * counts)

    np.testing.assert_allclose(
        class_weights(counts, "balanced"), balanced / balanced.mean()
    )
    np.testing.assert_allclose(
        class_weights(counts, "sqrt"), np.sqrt(balanced) / np.sqrt(balanced).mean()
    )
    # The agent's printed weights, from its notes.
    np.testing.assert_allclose(
        class_weights(counts, "sqrt"), [1.802, 0.558, 1.262, 0.378], atol=1e-3
    )
    np.testing.assert_allclose(class_weights(counts, "none"), np.ones(4))


def test_a_class_without_a_frame_has_no_weight_and_leaves_the_mean_of_the_rest() -> (
    None
):
    counts = np.array([10, 0, 30], dtype=np.intp)

    for how in ("balanced", "sqrt", "none"):
        weights = class_weights(counts, how)
        assert weights[1] == 0.0
        assert weights[[0, 2]].mean() == pytest.approx(1.0)


def test_a_seed_for_each_model_is_stable_and_never_shared() -> None:
    seeds = {sub_seed(seed, "fold", index) for seed in (1, 2, 3) for index in range(5)}

    assert len(seeds) == 15
    assert sub_seed(7, "fold", 2) == sub_seed(7, "fold", 2)
    assert sub_seed(7, "fold", 2) != sub_seed(7, "full", 2)
    assert 0 <= sub_seed(7, "fold", 2) < 2**32
