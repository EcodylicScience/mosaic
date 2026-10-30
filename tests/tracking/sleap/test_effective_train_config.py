"""The train-sleap document that sleap-nn receives once the overrides apply."""

from __future__ import annotations

import copy

import pytest

from mosaic.core.json_value import JsonValue
from mosaic.tracking.sleap.training import effective_train_config


def _document() -> dict[str, JsonValue]:
    return {
        "data_config": {"train_labels_path": ["train.slp"], "validation_fraction": 0.1},
        "model_config": {
            "head_configs": {
                "embedding": {
                    "embedding": {
                        "embedding_dim": 128,
                        "objective": {
                            "positives": {"scope": "tracklet", "max_gap": 10},
                            "negatives": {"restrict_same_video": True},
                        },
                    }
                }
            }
        },
        "trainer_config": {"max_epochs": 200, "seed": 42},
    }


_POSITIVES = "model_config.head_configs.embedding.embedding.objective.positives"


def test_a_bare_leaf_key_replaces_the_value() -> None:
    effective = effective_train_config(_document(), {"trainer_config.max_epochs": 5})

    assert effective["trainer_config"] == {"max_epochs": 5, "seed": 42}


def test_a_bare_key_under_an_absent_path_adds_it() -> None:
    effective = effective_train_config(
        _document(), {"trainer_config.optimizer.lr": 0.001}
    )

    assert effective["trainer_config"] == {
        "max_epochs": 200,
        "seed": 42,
        "optimizer": {"lr": 0.001},
    }


def test_a_plus_key_adds_under_an_absent_path() -> None:
    effective = effective_train_config(
        _document(), {"+data_config.identity.track_names_are_global": True}
    )

    assert effective["data_config"] == {
        "train_labels_path": ["train.slp"],
        "validation_fraction": 0.1,
        "identity": {"track_names_are_global": True},
    }


def test_a_plus_scalar_over_a_present_value_is_refused() -> None:
    with pytest.raises(ValueError, match=r"\+trainer_config\.seed"):
        _ = effective_train_config(_document(), {"+trainer_config.seed": 1})


def test_a_plus_mapping_merges_into_a_present_mapping() -> None:
    effective = effective_train_config(
        _document(), {f"+{_POSITIVES}": {"scope": "global_id"}}
    )

    assert _dict_at(effective, _POSITIVES) == {"scope": "global_id", "max_gap": 10}


def test_a_plus_list_replaces_a_present_list() -> None:
    effective = effective_train_config(
        _document(), {"+data_config.train_labels_path": ["a.slp"]}
    )

    assert _dict_at(effective, "data_config")["train_labels_path"] == ["a.slp"]


def test_a_plus_key_over_a_present_null_assigns() -> None:
    document = _document()
    _dict_at(document, "trainer_config")["seed"] = None

    effective = effective_train_config(document, {"+trainer_config.seed": 1})

    assert effective["trainer_config"] == {"max_epochs": 200, "seed": 1}


def test_a_double_plus_key_replaces_a_present_value() -> None:
    effective = effective_train_config(
        _document(), {f"++{_POSITIVES}.scope": "global_id"}
    )

    assert _dict_at(effective, _POSITIVES) == {"scope": "global_id", "max_gap": 10}


def test_a_double_plus_key_adds_an_absent_value() -> None:
    effective = effective_train_config(_document(), {"++trainer_config.patience": 3})

    assert effective["trainer_config"] == {"max_epochs": 200, "seed": 42, "patience": 3}


def test_a_mapping_value_merges_and_keeps_sibling_keys() -> None:
    effective = effective_train_config(
        _document(),
        {
            "model_config.head_configs.embedding.embedding.objective": {
                "positives": {"scope": "global_id"},
            }
        },
    )

    objective = _dict_at(
        effective, "model_config.head_configs.embedding.embedding.objective"
    )
    assert objective == {
        "positives": {"scope": "global_id", "max_gap": 10},
        "negatives": {"restrict_same_video": True},
    }


def test_a_list_value_replaces_a_list() -> None:
    effective = effective_train_config(
        _document(), {"data_config.train_labels_path": ["a.slp", "b.slp"]}
    )

    assert _dict_at(effective, "data_config")["train_labels_path"] == ["a.slp", "b.slp"]


def test_a_scalar_value_replaces_a_mapping() -> None:
    effective = effective_train_config(_document(), {_POSITIVES: None})

    objective = _dict_at(
        effective, "model_config.head_configs.embedding.embedding.objective"
    )
    assert objective["positives"] is None


def test_a_tilde_key_deletes_the_path() -> None:
    effective = effective_train_config(
        _document(), {f"~{_POSITIVES}.scope": None, "~data_config": None}
    )

    assert _dict_at(effective, _POSITIVES) == {"max_gap": 10}
    assert "data_config" not in effective


def test_a_tilde_key_on_an_absent_path_is_refused() -> None:
    with pytest.raises(ValueError, match=r"~trainer_config\.patience"):
        _ = effective_train_config(_document(), {"~trainer_config.patience": None})


def test_a_tilde_key_on_a_null_is_refused() -> None:
    document = _document()
    _dict_at(document, "trainer_config")["seed"] = None

    with pytest.raises(ValueError, match=r"~trainer_config\.seed"):
        _ = effective_train_config(document, {"~trainer_config.seed": None})


def test_a_tilde_key_with_the_present_value_deletes_the_path() -> None:
    effective = effective_train_config(_document(), {"~trainer_config.seed": 42})

    assert effective["trainer_config"] == {"max_epochs": 200}


def test_a_tilde_key_with_another_value_is_refused() -> None:
    with pytest.raises(ValueError, match="holds 42 rather than 7"):
        _ = effective_train_config(_document(), {"~trainer_config.seed": 7})


def test_deleting_the_trainer_device_count_is_refused() -> None:
    with pytest.raises(ValueError, match=r"~trainer_config\.trainer_devices"):
        _ = effective_train_config(
            _document(), {"~trainer_config.trainer_devices": None}
        )


def test_deleting_the_mapping_above_the_device_count_is_refused() -> None:
    with pytest.raises(ValueError, match=r"trainer_config\.trainer_devices"):
        _ = effective_train_config(_document(), {"~trainer_config": None})


def test_overrides_apply_in_order() -> None:
    effective = effective_train_config(
        _document(),
        {
            "~trainer_config.seed": None,
            "++trainer_config.seed": 7,
            "trainer_config.seed": 8,
            "+trainer_config.patience": 3,
            "~trainer_config.patience": None,
        },
    )

    assert effective["trainer_config"] == {"max_epochs": 200, "seed": 8}


def test_a_bare_declared_key_after_its_deletion_is_refused() -> None:
    with pytest.raises(ValueError, match=r"'trainer_config\.seed'.*does not hold"):
        _ = effective_train_config(
            _document(), {"~trainer_config.seed": None, "trainer_config.seed": 7}
        )


@pytest.mark.parametrize(
    "key", ["trainer_config.seed.value", "++trainer_config.seed.value"]
)
def test_an_added_key_below_a_scalar_replaces_the_scalar(key: str) -> None:
    """A bare key the document lacks reaches Hydra as ``+``, like ``++`` here."""
    effective = effective_train_config(_document(), {key: 1})

    assert effective["trainer_config"] == {"max_epochs": 200, "seed": {"value": 1}}


def test_a_bare_declared_key_below_a_replaced_scalar_is_refused() -> None:
    with pytest.raises(ValueError, match=r"'model_config'.*does not hold"):
        _ = effective_train_config(
            _document(),
            {"model_config": 5, f"{_POSITIVES}.scope": "global_id"},
        )


def test_a_bare_mapping_that_adds_a_key_is_refused() -> None:
    """Hydra composes in struct mode, where a bare key adds no key."""
    with pytest.raises(ValueError, match=rf"'{_POSITIVES}\.min_gap'.*does not hold"):
        _ = effective_train_config(
            _document(), {_POSITIVES: {"scope": "global_id", "min_gap": 1}}
        )


def test_a_plus_mapping_that_adds_a_key_merges_it() -> None:
    effective = effective_train_config(_document(), {f"++{_POSITIVES}": {"min_gap": 1}})

    assert _dict_at(effective, _POSITIVES) == {
        "scope": "tracklet",
        "max_gap": 10,
        "min_gap": 1,
    }


def test_a_bare_mapping_over_an_added_key_merges_it() -> None:
    """Struct mode checks the document as the earlier overrides left it."""
    effective = effective_train_config(
        _document(),
        {f"++{_POSITIVES}.min_gap": 1, _POSITIVES: {"min_gap": 2}},
    )

    assert _dict_at(effective, _POSITIVES)["min_gap"] == 2


def test_a_bare_key_the_document_lacks_is_appended_once() -> None:
    """Both keys reach Hydra as ``+trainer_config.patience``, and the second is refused."""
    with pytest.raises(ValueError, match=r"'\+trainer_config\.patience'"):
        _ = effective_train_config(
            _document(),
            {"+trainer_config.patience": 1, "trainer_config.patience": 2},
        )


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("data_config.train_labels_path", {"a": 1}),
        (_POSITIVES, ["global_id"]),
        ("++data_config", {"train_labels_path": {"a": 1}}),
    ],
)
def test_a_mapping_and_a_list_do_not_merge(key: str, value: JsonValue) -> None:
    with pytest.raises(
        ValueError, match="mapping into a list or a list into a mapping"
    ):
        _ = effective_train_config(_document(), {key: value})


def test_a_path_through_a_list_is_refused() -> None:
    with pytest.raises(ValueError, match="through the list"):
        _ = effective_train_config(
            _document(), {"++data_config.train_labels_path.0": "a.slp"}
        )


@pytest.mark.parametrize("name", ["a b", "true", "1"])
def test_a_mapping_key_hydra_cannot_spell_is_refused(name: str) -> None:
    with pytest.raises(ValueError, match="cannot spell the mapping key"):
        _ = effective_train_config(_document(), {"++trainer_config.extra": {name: 1}})


def test_the_input_document_and_override_values_are_unchanged() -> None:
    document = _document()
    before = copy.deepcopy(document)
    merged: dict[str, JsonValue] = {"positives": {"scope": "global_id"}}
    merged_before = copy.deepcopy(merged)

    effective = effective_train_config(
        document,
        {
            "model_config.head_configs.embedding.embedding.objective": merged,
            "trainer_config.max_epochs": 1,
            "~data_config.validation_fraction": None,
        },
    )
    _dict_at(effective, _POSITIVES)["scope"] = "aug_view"

    assert document == before
    assert merged == merged_before


def _dict_at(document: dict[str, JsonValue], dotted: str) -> dict[str, JsonValue]:
    node: JsonValue = document
    for name in dotted.split("."):
        assert isinstance(node, dict)
        node = node[name]
    assert isinstance(node, dict)
    return node
