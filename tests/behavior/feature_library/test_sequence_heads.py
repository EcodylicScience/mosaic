"""The heads of ``sequence-classifier``, its parameters and the pieces of its training that need no torch.

The registry, the options of a head, the parameter model that selects them, the
batches the loop draws and the class weights are plain Python, and they run in the
default job. The networks and the fit are in ``test_sequence_classifier.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import pytest
from pydantic import ValidationError

from mosaic.behavior.feature_library.sequence_classifier import SequenceClassifier
from mosaic.behavior.feature_library.sequence_heads import (
    GruConfig,
    HeadConfig,
    HeadSpec,
    MlpConfig,
    TcnConfig,
    get_head,
    head_names,
    head_source_digest,
    register_head,
)
from mosaic.behavior.feature_library.sequence_training import (
    Sampler,
    class_weights,
    sub_seed,
)

if TYPE_CHECKING:
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
    assert head_names() == ["gru", "mlp", "tcn"]
    assert get_head("tcn").config is TcnConfig
    assert get_head("gru").config is GruConfig
    assert get_head("mlp").config is MlpConfig


def test_an_unknown_head_is_refused_with_the_registered_names() -> None:
    with pytest.raises(ValueError, match=r"'lstm'.*Registered: gru, mlp, tcn"):
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


# --- The options of a head ------------------------------------------------------


def test_the_defaults_of_the_built_in_heads_are_the_reference_networks() -> None:
    assert TcnConfig().model_dump() == {
        "hidden": 128,
        "blocks": 6,
        "kernel_size": 3,
        "dropout": 0.3,
    }
    assert GruConfig().model_dump() == {
        "hidden": 128,
        "layers": 2,
        "bidirectional": True,
        "dropout": 0.3,
    }


def test_an_option_a_head_does_not_have_is_refused() -> None:
    with pytest.raises(ValidationError, match="layers"):
        TcnConfig.model_validate({"layers": 3})


def test_an_even_kernel_is_refused() -> None:
    with pytest.raises(ValidationError, match="kernel_size 4 is even"):
        TcnConfig(kernel_size=4)


def test_the_options_of_a_head_are_frozen() -> None:
    config = TcnConfig()

    with pytest.raises(ValidationError):
        setattr(config, "hidden", 4)


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
    assert omitted.identity_dump()["head_config"] == TcnConfig().model_dump()


def test_an_option_of_another_head_is_refused_and_so_are_options_of_a_wrong_head() -> (
    None
):
    with pytest.raises(ValueError, match="layers"):
        _params(head="tcn", head_config={"layers": 2})
    with pytest.raises(ValueError, match="head_config holds TcnConfig options"):
        _params(head="gru", head_config=TcnConfig())


def test_an_unregistered_head_is_refused_with_the_registered_names() -> None:
    with pytest.raises(ValueError, match="Registered: gru, mlp, tcn"):
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
    ],
)
def test_a_setting_that_changes_the_model_changes_the_run_identity(
    change: dict[str, object],
) -> None:
    assert _params(**change).identity_dump() != _params().identity_dump()


@pytest.mark.parametrize("change", [{"apply_device": "cuda"}, {"n_threads": 4}])
def test_a_throughput_setting_is_not_part_of_the_run_identity(
    change: dict[str, object],
) -> None:
    assert _params(**change).identity_dump() == _params().identity_dump()


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


def test_only_a_fit_is_scope_dependent_and_none_is_parallel() -> None:
    inputs = SequenceClassifier.Inputs.model_validate([{"feature": "speed-angvel"}])
    fit = SequenceClassifier(inputs, dict(FIT))
    applied = SequenceClassifier(
        inputs,
        {"model": {"feature": "sequence-classifier"}, "default_class": 0},
    )

    assert fit.scope_dependent is True
    assert applied.scope_dependent is False
    assert SequenceClassifier.parallelizable is False


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
