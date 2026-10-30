"""Exporting an annotation set as a SLEAP ``.slp``.

The conversion itself runs in the SLEAP environment, so what is testable here is
everything around it: that the COCO handed over says what the set said, that the
result is published atomically, and that nothing is left behind either way.
``run_supervised`` is patched, which is the same seam the invocation suites use.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Final

import pytest

from mosaic.core.annotations.readers import read_coco_keypoints
from mosaic.core.annotations import (
    AnnotationFrame,
    AnnotationObject,
    AnnotationSet,
    Keypoint,
    KeypointSchema,
)
from mosaic.tracking.sleap import run as sleap_run
from mosaic.tracking.sleap.labels import (
    SLEAP_VALIDATION_SUFFIX,
    validation_labels_beside,
    write_slp,
)
from mosaic.tracking.sleap.run import SleapError

SCHEMA = KeypointSchema(names=("nose", "thorax", "tail"), skeleton=((0, 1), (1, 2)))


def _set(tmp_path: Path, *, frames: int = 2, tracks: bool = False) -> AnnotationSet:
    """*frames* frames of one instance each, optionally carrying an identity."""
    images = tmp_path / "images"
    images.mkdir(exist_ok=True)
    made: list[AnnotationFrame] = []
    for index in range(frames):
        name = f"f{index}.png"
        _ = (images / name).write_bytes(b"not really a png")
        made.append(
            AnnotationFrame(
                image_path=Path(name),
                width=160,
                height=120,
                objects=(
                    AnnotationObject(
                        keypoints=(
                            Keypoint(40.0, 50.0, 2),
                            Keypoint(52.0, 56.0, 1),
                            Keypoint.absent(),
                        ),
                        # The same animal across every frame, which is what a
                        # multi_class head is trained to recognize.
                        track_id="1" if tracks else "",
                    ),
                ),
            )
        )
    return AnnotationSet(
        schema=SCHEMA, frames=tuple(made), categories=("mouse",), image_root=images
    )


def _stub_sleap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the location ladder at a SLEAP install that never runs."""
    monkeypatch.setenv("MOSAIC_SLEAP_BIN", str(tmp_path / "bin" / "sleap-convert"))
    (tmp_path / "bin").mkdir(exist_ok=True)
    _ = (tmp_path / "bin" / "python").write_text("")


def _hand_over(
    annotations: AnnotationSet, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> str:
    """Run ``write_slp`` and give back the COCO text the subprocess was handed."""
    handed_over: list[str] = []

    def capture(argv: Sequence[str], **kw: object) -> tuple[str, str, int]:
        # Keep the text: the file is deleted once the conversion returns.
        handed_over.append(Path(argv[-3]).read_text())
        _ = Path(argv[-1]).write_bytes(b"slp")
        return ("ok", "", 0)

    _stub_sleap(tmp_path, monkeypatch)
    monkeypatch.setattr(sleap_run, "run_supervised", capture)
    _ = write_slp(annotations, tmp_path / "out.slp")
    assert handed_over, "the subprocess was invoked"
    return handed_over[0]


def _fake_run(
    write: bool = True, returncode: int = 0
) -> tuple[Callable[..., tuple[str, str, int]], list[list[str]]]:
    """A stand-in for the SLEAP subprocess, recording the argv it was given."""
    seen: list[list[str]] = []

    def run(
        argv: Sequence[str],
        *,
        env: Mapping[str, str] | None = None,
        cancel_check: object = None,
        timeout: float | None = None,
        idle_timeout: float | None = None,
        on_output: object = None,
        on_activity: object = None,
    ) -> tuple[str, str, int]:
        seen.append(list(argv))
        if write and returncode == 0:
            _ = Path(argv[-1]).write_bytes(b"slp")
        return ("ok", "", returncode)

    return run, seen


def test_the_coco_handed_over_says_what_the_set_said(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The subprocess only sees the COCO file, so that file is the whole contract."""
    original = _set(tmp_path)
    handed_over = _hand_over(original, tmp_path, monkeypatch)

    # Read the handed-over file back the way sleap-io will, and compare against
    # what went in. Asserting on the parsed set rather than on raw JSON keys is
    # the same question a consumer asks.
    replay = tmp_path / "handed_over.json"
    _ = replay.write_text(handed_over)
    delivered = read_coco_keypoints(replay, tmp_path / "images")

    assert delivered.schema == original.schema, "names and skeleton survive"
    assert len(delivered.frames) == len(original.frames)
    received = delivered.frames[0].objects[0].keypoints
    assert [k.visibility for k in received] == [2, 1, 0], "occlusion included"
    assert not received[2].is_placed, "an unplaced point stays unplaced"


def test_an_identified_instance_hands_over_its_track(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without this key sleap-io builds a .slp with no tracks at all.

    Asserted on the raw JSON rather than on a parsed set, because presence and
    absence of a key is the whole contract and a parsed set cannot express it.
    And asserted on what mosaic writes, never on what sleap-io does with it:
    that library lives in the user's SLEAP environment, so mosaic cannot pin
    its behavior from here.
    """
    handed_over = _hand_over(_set(tmp_path, tracks=True), tmp_path, monkeypatch)
    records = json.loads(handed_over)["annotations"]

    assert records, "an instance was written"
    assert [r["track_id"] for r in records] == ["1", "1"], "one animal, two frames"


FAKE_SLEAP_IO: Final = """
import json
from types import SimpleNamespace

PREFIX = {prefix!r}


def load_coco(path, dataset_root):
    with open(path, encoding="utf-8") as file:
        annotations = json.load(file)["annotations"]
    tracks = {{}}
    for annotation in annotations:
        value = annotation.get("track_id")
        if value is not None and value not in tracks:
            tracks[value] = SimpleNamespace(name=f"{{PREFIX}}{{value}}")
    return SimpleNamespace(tracks=list(tracks.values()), labeled_frames=annotations)


def save_slp(labels, path):
    with open(path, "w", encoding="utf-8") as file:
        json.dump([track.name for track in labels.tracks], file)
"""
"""A ``sleap_io`` package whose COCO reader names each track *prefix* plus its value.

sleap-io 0.7 and 0.9 name a track read from COCO ``track_<value>``. Its ``.slp``
writer here writes the track names as JSON.
"""


def _two_aliases(tmp_path: Path) -> AnnotationSet:
    """Two frames, each holding the animals whose tracks are ``first`` and ``track_7``."""
    images = tmp_path / "images"
    images.mkdir(exist_ok=True)
    frames: list[AnnotationFrame] = []
    for index in range(2):
        name = f"f{index}.png"
        _ = (images / name).write_bytes(b"not really a png")
        frames.append(
            AnnotationFrame(
                image_path=Path(name),
                width=160,
                height=120,
                objects=tuple(
                    AnnotationObject(
                        keypoints=(
                            Keypoint(40.0 + x, 50.0, 2),
                            Keypoint(52.0 + x, 56.0, 2),
                            Keypoint(60.0 + x, 58.0, 2),
                        ),
                        track_id=alias,
                    )
                    for x, alias in ((0.0, "first"), (60.0, "track_7"))
                ),
            )
        )
    return AnnotationSet(
        schema=SCHEMA, frames=tuple(frames), categories=("mouse",), image_root=images
    )


@pytest.mark.parametrize("prefix", ["track_", ""], ids=["prefixed", "bare"])
def test_each_track_of_the_slp_is_named_by_its_track_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prefix: str
) -> None:
    """The conversion runs against a ``sleap_io`` that names tracks *prefix* + value.

    A SLEAP identity model reports the names of the tracks it was trained on, so a
    preparation by alias names each track by the alias. A reader that keeps the
    value as the name is left as it is.
    """
    fake = tmp_path / "environment"
    (fake / "sleap_io").mkdir(parents=True)
    _ = (fake / "sleap_io" / "__init__.py").write_text(
        FAKE_SLEAP_IO.format(prefix=prefix)
    )

    def run(argv: Sequence[str], **kw: object) -> tuple[str, str, int]:
        snippet = argv[argv.index("-c") + 1]
        done = subprocess.run(
            [sys.executable, "-c", snippet, *argv[-3:]],
            cwd=fake,
            capture_output=True,
            text=True,
            check=False,
        )
        return (done.stdout, done.stderr, done.returncode)

    monkeypatch.setattr(sleap_run, "run_supervised", run)
    _stub_sleap(tmp_path, monkeypatch)

    written = write_slp(_two_aliases(tmp_path), tmp_path / "out.slp")

    assert json.loads(written.read_text()) == ["first", "track_7"]


def test_an_unidentified_instance_omits_the_key_entirely(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The guard that keeps a falsy value from reading as an identity.

    sleap-io chains its candidate keys with ``or``, so an empty string present
    in the file is indistinguishable from one absent -- and writing it anyway
    would make every unidentified instance look like a key that failed.
    """
    handed_over = _hand_over(_set(tmp_path), tmp_path, monkeypatch)
    records = json.loads(handed_over)["annotations"]

    assert records, "an instance was written"
    assert all("track_id" not in r for r in records), "absent, not empty"


def test_the_result_is_published_atomically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Existence is how a caller decides the export worked, so it must not lie."""
    observed: dict[str, bool] = {}
    target = tmp_path / "out.slp"

    def run(argv: Sequence[str], **kw: object) -> tuple[str, str, int]:
        # While the tool is writing, the destination must not exist yet.
        observed["target_absent_during_write"] = not target.exists()
        observed["writes_elsewhere"] = argv[-1] != str(target)
        _ = Path(argv[-1]).write_bytes(b"slp")
        return ("", "", 0)

    monkeypatch.setattr(sleap_run, "run_supervised", run)
    _stub_sleap(tmp_path, monkeypatch)

    written = write_slp(_set(tmp_path), target)
    assert observed == {"target_absent_during_write": True, "writes_elsewhere": True}
    assert written.read_bytes() == b"slp"


def test_a_failed_conversion_leaves_nothing_behind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No half-written .slp, and no intermediate COCO for a later run to trip on."""
    run, _ = _fake_run(write=False, returncode=1)
    monkeypatch.setattr(sleap_run, "run_supervised", run)
    _stub_sleap(tmp_path, monkeypatch)

    with pytest.raises(SleapError):
        _ = write_slp(_set(tmp_path), tmp_path / "out.slp")

    leftovers = sorted(p.name for p in tmp_path.iterdir() if p.is_file())
    assert leftovers == [], f"left behind: {leftovers}"


def test_a_silent_failure_is_still_a_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exit zero having written nothing is the case existence-gating misses."""
    run, _ = _fake_run(write=False, returncode=0)
    monkeypatch.setattr(sleap_run, "run_supervised", run)
    _stub_sleap(tmp_path, monkeypatch)

    with pytest.raises(FileNotFoundError, match="did not write its expected output"):
        _ = write_slp(_set(tmp_path), tmp_path / "out.slp")


def test_an_empty_set_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no frames"):
        _ = write_slp(AnnotationSet(schema=SCHEMA), tmp_path / "out.slp")


def test_a_set_with_no_image_root_must_be_told_one(tmp_path: Path) -> None:
    """sleap-io resolves every file_name against a root, so there has to be one."""
    rootless = AnnotationSet(
        schema=SCHEMA,
        frames=(AnnotationFrame(image_path=Path("f.png"), width=1, height=1),),
    )
    with pytest.raises(ValueError, match="images_dir is required"):
        _ = write_slp(rootless, tmp_path / "out.slp")


def test_labels_alone_have_no_validation_file(tmp_path: Path) -> None:
    labels = tmp_path / "labels.slp"
    _ = labels.write_bytes(b"slp")

    assert validation_labels_beside(labels) is None


def test_the_validation_file_beside_the_labels_is_found_by_its_suffix(
    tmp_path: Path,
) -> None:
    labels = tmp_path / "labels.slp"
    _ = labels.write_bytes(b"slp")
    validation = tmp_path / f"labels{SLEAP_VALIDATION_SUFFIX}"
    _ = validation.write_bytes(b"slp")

    assert validation_labels_beside(labels) == validation == tmp_path / "labels.val.slp"
