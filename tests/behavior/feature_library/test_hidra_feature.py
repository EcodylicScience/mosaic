"""The hidra feature against a fake ``hidra-predict``.

The fake replaces :func:`run_hidra_predict`. It reads the staged entry the way
HiDRA does and writes ``entry.frames.parquet`` with a probability that is a known
function of the frame and the pair, so each test can say which value lands on which
row. The real package runs in ``test_hidra_real.py``.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from mosaic.behavior.feature_library import hidra_feature
from mosaic.behavior.feature_library.hidra_feature import (
    ALL_CONFIGS,
    DEFAULT_WEIGHTS_REVISION,
    HIDRA_ENV,
    PUBLISHED_HEADS,
    HidraExitError,
    HidraFeature,
    HidraInputError,
    HidraPredictAvailable,
    PROVENANCE_DIR,
    HidraWeightsMissing,
    read_hidra_provenance,
    run_hidra_predict,
)
from mosaic.core.pipeline.job import CancelToken
from mosaic.core.pipeline.resample_tracks import AmbiguousFrameRateError
from mosaic.core.pipeline.run import run_feature
from tests.helpers import (
    add_tracks_variant,
    isolate_tool_location,
    make_dataset,
    result_outputs,
    result_run_root,
    track_table,
)

_VARIANT: Final = "convert-test.0.1-aaaaaaaaaa"
_SELF_ACTIONS: Final = frozenset({"selfgroom", "rear"})
_MAP: Final = {"kp0": "nose", "kp1": "neck", "kp2": "tail_base"}


@dataclass
class _FakePredict:
    """What the fake was given, and the frames table it wrote."""

    argv: list[list[str]] = field(default_factory=list)
    env: list[dict[str, str]] = field(default_factory=list)
    staged: list[pd.DataFrame] = field(default_factory=list)
    metadata: list[pd.DataFrame] = field(default_factory=list)

    def __call__(
        self,
        argv: list[str],
        env: dict[str, str],
        *,
        cancel_token: CancelToken | None = None,
    ) -> tuple[str, str]:
        self.argv.append(argv)
        self.env.append(env)
        folder = Path(argv[1])
        out = Path(argv[argv.index("--out") + 1])
        labs = argv[argv.index("--labs") + 1].split(",")
        actions = argv[argv.index("--actions") + 1].split(",")
        self.metadata.append(pd.read_csv(folder / "metadata.csv"))
        out.mkdir(parents=True, exist_ok=True)
        for path in sorted(folder.glob("*.parquet")):
            staged = pd.read_parquet(path)
            self.staged.append(staged)
            frames = _fake_frames(staged, labs, actions)
            frames.to_parquet(out / f"{path.stem}.frames.parquet")
        return "", ""


def _fake_frames(
    staged: pd.DataFrame, labs: list[str], actions: list[str]
) -> pd.DataFrame:
    """The fake's frames table for one staged scene."""
    mice = sorted(staged["mouse_id"].unique())
    length = int(staged["video_frame"].max()) + 1
    frame = np.arange(length)
    rows: list[pd.DataFrame] = []
    for lab in labs:
        for action in actions:
            if action not in PUBLISHED_HEADS[lab]:
                continue
            for s, subject in enumerate(mice):
                targets = (
                    ["self"]
                    if action in _SELF_ACTIONS
                    else [m for m in mice if m != subject]
                )
                for target in targets:
                    t = 0 if target == "self" else int(target[5:])
                    prob = fake_probability(frame, s + 1, t)
                    rows.append(
                        pd.DataFrame(
                            {
                                "frame": frame,
                                "subject": subject,
                                "target": target,
                                "lab": lab,
                                "action": action,
                                "prob": prob,
                                "call": (prob >= 0.5).astype(np.int8),
                            }
                        )
                    )
    return pd.concat(rows, ignore_index=True)


def fake_probability(frame: np.ndarray, subject: int, target: int) -> np.ndarray:
    """The fake's probability at *frame* for mouse *subject* toward *target*."""
    return ((frame % 10) / 10.0 + 0.01 * subject + 0.001 * target).astype(np.float32)


_INSTALL: Final = '{"install": {"version": "0.2.0"}, "heads": "82 classifier heads"}\n'


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> _FakePredict:
    made = _FakePredict()
    monkeypatch.setattr(hidra_feature, "run_hidra_predict", made)

    def provenance(predict_argv: list[str], env: dict[str, str]) -> str:
        return _INSTALL

    monkeypatch.setattr(hidra_feature, "read_hidra_provenance", provenance)
    return made


@pytest.fixture
def placement(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """A located ``hidra-predict`` and the default revision's weight files."""
    isolate_tool_location(monkeypatch, HIDRA_ENV, found={})
    script = tmp_path / "bin" / "hidra-predict"
    script.parent.mkdir()
    _ = script.write_text("#!/bin/sh\nexit 0\n")
    script.chmod(0o755)
    models = tmp_path / "models"
    revision = models / DEFAULT_WEIGHTS_REVISION
    revision.mkdir(parents=True)
    for config in ALL_CONFIGS:
        for part in ("unsupervised", "supervised_perlab_sniffall"):
            _ = (revision / f"{config}_{part}.safetensors").write_bytes(b"")
    _ = (revision / "thresholds.json").write_text("{}")
    return {"hidra_bin": str(script), "models_root": str(models)}


def _params(placement: dict[str, object], **overrides: object) -> dict[str, object]:
    params: dict[str, object] = {
        "pose": {"pose_n": 3},
        "keypoint_map": dict(_MAP),
        "heads": [
            {"lab": "ReflectiveManatee", "action": "sniff"},
            {"lab": "GroovyShrew", "action": "selfgroom"},
        ],
        "pix_per_cm": 20.0,
        **placement,
    }
    params.update(overrides)
    return params


def _entry(n_ids: int, *, n_rows: int = 12, start_frame: int = 100) -> pd.DataFrame:
    table = track_table("g", "s1", n_rows=n_rows, n_ids=n_ids, start_frame=start_frame)
    for keypoint in range(3):
        table[f"poseX{keypoint}"] = table["X"] + keypoint
        table[f"poseY{keypoint}"] = table["Y"] + keypoint
        table[f"poseP{keypoint}"] = 1.0
    return table


# --- the run ----------------------------------------------------------------


def test_a_run_writes_both_perspectives_of_each_pair_per_frame(
    tmp_path: Path, fake: _FakePredict, placement: dict[str, object]
) -> None:
    dataset = make_dataset(tmp_path / "dataset")
    add_tracks_variant(dataset, _VARIANT, "s1", "s2", n_rows=20, keypoints=3)

    result = run_feature(dataset, HidraFeature(params=_params(placement)))

    assert result.failed_entries == ()
    outputs = result_outputs(dataset, result)
    assert sorted(outputs) == ["s1", "s2"]
    table = outputs["s1"]
    assert len(table) == 2 * 20
    assert sorted(table["perspective"].unique().tolist()) == [0, 1]
    zero = table[table["perspective"] == 0]
    one = table[table["perspective"] == 1]
    assert (zero["id1"] == 0).all() and (zero["id2"] == 1).all()
    assert (one["id1"] == 1).all() and (one["id2"] == 0).all()
    assert {
        "prob_sniff__ReflectiveManatee",
        "call_sniff__ReflectiveManatee",
        "prob_selfgroom__GroovyShrew",
        "predicted_label",
    } <= set(table.columns)
    assert len(fake.argv) == 2


def test_a_run_records_the_hidra_install_of_each_entry(
    tmp_path: Path, fake: _FakePredict, placement: dict[str, object]
) -> None:
    dataset = make_dataset(tmp_path / "dataset")
    add_tracks_variant(dataset, _VARIANT, "s1", "s2", n_rows=20, keypoints=3)

    result = run_feature(dataset, HidraFeature(params=_params(placement)))

    record = result_run_root(dataset, result) / PROVENANCE_DIR
    (stored,) = sorted(record.glob("*.json"))
    assert stored.read_text() == _INSTALL
    entries = sorted(path.name for path in (record / "entries").iterdir())
    assert entries == ["s1.txt", "s2.txt"]
    for path in (record / "entries").iterdir():
        assert path.read_text().strip() == stored.stem


def test_an_unreadable_install_is_recorded_as_an_error(tmp_path: Path) -> None:
    script = tmp_path / "bin" / "hidra-predict"
    script.parent.mkdir()
    _ = script.write_text("#!/bin/sh\necho 'no heads' >&2\nexit 2\n")
    script.chmod(0o755)

    text = read_hidra_provenance([str(script)], {"PATH": "/usr/bin:/bin"})

    assert '"install": {"error": "error:' in text
    assert "exit 2: no heads" in text


# --- the staged input --------------------------------------------------------


def test_the_staged_entry_is_the_mapped_keypoints_from_frame_zero(
    fake: _FakePredict, placement: dict[str, object]
) -> None:
    entry = _entry(2)
    entry.loc[entry.index[0], "poseX1"] = np.nan
    entry.loc[entry.index[1], "poseP2"] = 0.1

    _ = HidraFeature(params=_params(placement, min_confidence=0.5)).apply(entry)

    (staged,) = fake.staged
    assert staged["video_frame"].min() == 0
    assert staged["video_frame"].max() == 11
    assert sorted(staged["mouse_id"].unique()) == ["mouse1", "mouse2"]
    assert sorted(staged["bodypart"].unique()) == ["neck", "nose", "tail_base"]
    first = staged[(staged["video_frame"] == 0) & (staged["mouse_id"] == "mouse1")]
    assert sorted(first["bodypart"]) == ["nose", "tail_base"]
    second = staged[(staged["video_frame"] == 0) & (staged["mouse_id"] == "mouse2")]
    assert sorted(second["bodypart"]) == ["neck", "nose"]
    (metadata,) = fake.metadata
    assert metadata["file"].tolist() == ["scene0.parquet"]
    assert metadata["pix_per_cm"].tolist() == [20.0]
    assert metadata["fps"].tolist() == [pytest.approx(30.0)]


def test_a_keypoint_left_out_of_the_map_is_not_sent(
    fake: _FakePredict, placement: dict[str, object]
) -> None:
    params = _params(placement, keypoint_map={"kp0": "nose", "kp2": "tail_base"})

    _ = HidraFeature(params=params).apply(_entry(2))

    assert sorted(fake.staged[0]["bodypart"].unique()) == ["nose", "tail_base"]


# --- the values --------------------------------------------------------------


def test_each_row_holds_the_probability_of_id1_toward_id2(
    fake: _FakePredict, placement: dict[str, object]
) -> None:
    table = HidraFeature(params=_params(placement)).apply(_entry(3))

    for id1, id2 in itertools.permutations(range(3), 2):
        rows = table[(table["id1"] == id1) & (table["id2"] == id2)]
        assert len(rows) == 12
        expected = fake_probability(rows["frame"].to_numpy() - 100, id1 + 1, id2 + 1)
        np.testing.assert_array_equal(
            rows["prob_sniff__ReflectiveManatee"].to_numpy(), expected
        )
        np.testing.assert_array_equal(
            rows["call_sniff__ReflectiveManatee"].to_numpy(), expected >= 0.5
        )


def test_a_self_directed_head_is_written_on_every_row_of_its_animal(
    fake: _FakePredict, placement: dict[str, object]
) -> None:
    table = HidraFeature(params=_params(placement)).apply(_entry(3))

    assert len(table) == 6 * 12
    for id1 in range(3):
        rows = table[table["id1"] == id1]
        expected = fake_probability(rows["frame"].to_numpy() - 100, id1 + 1, 0)
        np.testing.assert_array_equal(
            rows["prob_selfgroom__GroovyShrew"].to_numpy(), expected
        )


def test_predicted_label_is_the_strongest_called_head(
    fake: _FakePredict, placement: dict[str, object]
) -> None:
    table = HidraFeature(params=_params(placement)).apply(_entry(2))

    sniff = table["prob_sniff__ReflectiveManatee"].to_numpy()
    groom = table["prob_selfgroom__GroovyShrew"].to_numpy()
    called = (sniff >= 0.5) | (groom >= 0.5)
    expected = np.where(
        ~called,
        "none",
        np.where(
            (groom >= 0.5) & ((sniff < 0.5) | (groom > sniff)),
            "selfgroom__GroovyShrew",
            "sniff__ReflectiveManatee",
        ),
    )
    assert table["predicted_label"].tolist() == expected.tolist()


def test_a_frame_missing_from_the_input_keeps_its_neighbors_values(
    fake: _FakePredict, placement: dict[str, object]
) -> None:
    entry = _entry(2)
    entry = entry[entry["frame"] != 105]

    table = HidraFeature(params=_params(placement)).apply(entry)

    rows = table[(table["id1"] == 0) & (table["id2"] == 1)]
    assert 105 not in rows["frame"].tolist()
    np.testing.assert_array_equal(
        rows["prob_sniff__ReflectiveManatee"].to_numpy(),
        fake_probability(rows["frame"].to_numpy() - 100, 1, 2),
    )


def test_each_arena_is_its_own_scene(
    fake: _FakePredict, placement: dict[str, object]
) -> None:
    entry = _entry(6)
    entry["arena"] = entry["id"].map({0: "a", 1: "a", 2: "b", 3: "b", 4: "c", 5: "c"})

    table = HidraFeature(params=_params(placement)).apply(entry)

    assert len(fake.staged) == 3
    for staged in fake.staged:
        assert sorted(staged["mouse_id"].unique()) == ["mouse1", "mouse2"]
    assert fake.metadata[0]["file"].tolist() == [
        "scene0.parquet",
        "scene1.parquet",
        "scene2.parquet",
    ]
    pairs = sorted(set(zip(table["id1"].tolist(), table["id2"].tolist(), strict=True)))
    assert pairs == [(0, 1), (1, 0), (2, 3), (3, 2), (4, 5), (5, 4)]
    rows = table[(table["id1"] == 3) & (table["id2"] == 2)]
    np.testing.assert_array_equal(
        rows["prob_sniff__ReflectiveManatee"].to_numpy(),
        fake_probability(rows["frame"].to_numpy() - 100, 2, 1),
    )


def test_an_arena_of_one_individual_is_left_out(
    fake: _FakePredict, placement: dict[str, object]
) -> None:
    entry = _entry(3)
    entry["arena"] = entry["id"].map({0: "a", 1: "a", 2: "b"})

    table = HidraFeature(params=_params(placement)).apply(entry)

    assert len(fake.staged) == 1
    assert sorted(set(table["id1"].tolist())) == [0, 1]


# --- the command -------------------------------------------------------------


def test_the_command_names_the_labs_actions_and_placement(
    fake: _FakePredict, placement: dict[str, object]
) -> None:
    params = _params(placement, configs=["11fps_4bp"], threshold=0.4, gpu=1)

    _ = HidraFeature(params=params).apply(_entry(2))

    (argv,) = fake.argv
    assert argv[0] == placement["hidra_bin"]
    assert argv[argv.index("--labs") + 1] == "GroovyShrew,ReflectiveManatee"
    assert argv[argv.index("--actions") + 1] == "selfgroom,sniff"
    assert argv[argv.index("--output") + 1] == "probs"
    assert argv[argv.index("--configs") + 1] == "11fps_4bp"
    assert argv[argv.index("--threshold") + 1] == "0.4"
    assert argv[argv.index("--gpu") + 1] == "1"
    (env,) = fake.env
    assert env["HIDRA_MODELS_DIR"] == str(
        Path(str(placement["models_root"])) / DEFAULT_WEIGHTS_REVISION
    )


# --- refusals ----------------------------------------------------------------


@pytest.mark.parametrize("n_ids", [1, 5])
def test_an_entry_outside_two_to_four_individuals_is_refused(
    fake: _FakePredict, placement: dict[str, object], n_ids: int
) -> None:
    with pytest.raises(HidraInputError, match="2 to 4 individuals"):
        _ = HidraFeature(params=_params(placement)).apply(_entry(n_ids))
    assert fake.argv == []


def test_an_entry_of_two_frame_rates_is_refused(
    fake: _FakePredict, placement: dict[str, object]
) -> None:
    entry = _entry(2)
    entry["frame_rate"] = np.where(entry["frame"] < 106, 30.0, 31.0)

    with pytest.raises(AmbiguousFrameRateError):
        _ = HidraFeature(params=_params(placement)).apply(entry)


def test_missing_weights_are_refused_with_the_download_command(
    tmp_path: Path, fake: _FakePredict, placement: dict[str, object]
) -> None:
    params = _params(placement, models_root=str(tmp_path / "empty"))

    with pytest.raises(HidraWeightsMissing, match="hidra-download-models --revision"):
        _ = HidraFeature(params=params).apply(_entry(2))
    assert fake.argv == []


def test_a_keypoint_name_outside_the_pose_is_refused(
    fake: _FakePredict, placement: dict[str, object]
) -> None:
    params = _params(placement, keypoint_map={"snout": "nose"})

    with pytest.raises(HidraInputError, match="snout"):
        _ = HidraFeature(params=params).apply(_entry(2))


def test_an_unpublished_head_is_refused_at_construction(
    placement: dict[str, object],
) -> None:
    with pytest.raises(ValidationError, match="no 'mount' head"):
        _ = HidraFeature(
            params=_params(
                placement, heads=[{"lab": "JovialSwallow", "action": "mount"}]
            )
        )


def test_two_keypoints_on_one_bodypart_are_refused_at_construction(
    placement: dict[str, object],
) -> None:
    with pytest.raises(ValidationError, match="two keypoints"):
        _ = HidraFeature(
            params=_params(placement, keypoint_map={"kp0": "nose", "kp1": "nose"})
        )


def test_without_heads_or_a_scale_apply_is_refused(
    fake: _FakePredict, placement: dict[str, object]
) -> None:
    with pytest.raises(HidraInputError, match="at least one head"):
        _ = HidraFeature(params=_params(placement, heads=[])).apply(_entry(2))
    with pytest.raises(HidraInputError, match="pix_per_cm"):
        _ = HidraFeature(params=_params(placement, pix_per_cm=None)).apply(_entry(2))


def test_a_zero_exit_without_output_is_refused(
    monkeypatch: pytest.MonkeyPatch, placement: dict[str, object]
) -> None:
    def silent(
        argv: list[str],
        env: dict[str, str],
        *,
        cancel_token: CancelToken | None = None,
    ) -> tuple[str, str]:
        return "", "note: nothing to predict"

    monkeypatch.setattr(hidra_feature, "run_hidra_predict", silent)

    with pytest.raises(
        FileNotFoundError, match="(?s)scene0.frames.parquet.*nothing to predict"
    ):
        _ = HidraFeature(params=_params(placement)).apply(_entry(2))


def test_a_nonzero_exit_raises_with_the_tools_output(tmp_path: Path) -> None:
    script = tmp_path / "hidra-predict"
    _ = script.write_text("#!/bin/sh\necho 'ERROR: no fps' >&2\nexit 1\n")
    script.chmod(0o755)

    with pytest.raises(HidraExitError, match="no fps"):
        run_hidra_predict([str(script)], {"PATH": "/usr/bin:/bin"})


def test_a_confidence_threshold_without_confidences_is_refused(
    fake: _FakePredict, placement: dict[str, object]
) -> None:
    entry = _entry(2).drop(columns=["poseP0", "poseP1", "poseP2"])

    with pytest.raises(HidraInputError, match="no confidence column"):
        _ = HidraFeature(params=_params(placement, min_confidence=0.5)).apply(entry)


# --- identity and prerequisite -----------------------------------------------


def test_the_run_identity_holds_the_revision_and_not_the_placement(
    placement: dict[str, object],
) -> None:
    base = HidraFeature(params=_params(placement)).params.identity_dump()
    moved = HidraFeature(
        params=_params(
            {"hidra_bin": "/elsewhere/hidra-predict", "models_root": "/elsewhere"},
            gpu=3,
            hidra_conda_env="hidra",
        )
    ).params.identity_dump()
    revised = HidraFeature(
        params=_params(placement, weights_revision="main")
    ).params.identity_dump()

    assert base == moved
    assert base != revised


def test_the_prerequisite_names_the_install_when_hidra_predict_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    isolate_tool_location(monkeypatch, HIDRA_ENV, found={})

    message = HidraPredictAvailable().missing()

    assert message is not None
    assert "git+https://github.com/neuroethology/HiDRA" in message


def test_the_prerequisite_is_met_when_hidra_predict_is_on_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    isolate_tool_location(
        monkeypatch, HIDRA_ENV, found={"hidra-predict": "/opt/hidra/bin/hidra-predict"}
    )

    assert HidraPredictAvailable().missing() is None
