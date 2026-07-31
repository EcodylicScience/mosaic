"""Exporting an annotation set as a SLEAP ``.slp``.

The conversion itself runs in the SLEAP environment, so what is testable here is
everything around it: that the COCO handed over says what the set said, that the
result is published atomically, and that nothing is left behind either way.
``run_supervised`` is patched, which is the same seam the invocation suites use.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import pytest

from mosaic.core.annotations.readers import read_coco_keypoints
from mosaic.core.annotations import (
    AnnotationFrame,
    AnnotationObject,
    AnnotationSet,
    Keypoint,
    KeypointSchema,
)
from mosaic.tracking.sleap import labels as labels_module
from mosaic.tracking.sleap.labels import write_slp
from mosaic.tracking.sleap.run import SleapError

SCHEMA = KeypointSchema(names=("nose", "thorax", "tail"), skeleton=((0, 1), (1, 2)))


def _set(tmp_path: Path, *, frames: int = 2) -> AnnotationSet:
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
                        )
                    ),
                ),
            )
        )
    return AnnotationSet(
        schema=SCHEMA, frames=tuple(made), categories=("mouse",), image_root=images
    )


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
    handed_over: list[str] = []
    seen: list[list[str]] = []

    def capture(argv: Sequence[str], **kw: object) -> tuple[str, str, int]:
        seen.append(list(argv))
        # Keep the text: the file is deleted once the conversion returns.
        handed_over.append(Path(argv[-3]).read_text())
        _ = Path(argv[-1]).write_bytes(b"slp")
        return ("ok", "", 0)

    monkeypatch.setattr(labels_module, "run_supervised", capture)
    monkeypatch.setenv("MOSAIC_SLEAP_BIN", str(tmp_path / "bin" / "sleap-track"))
    (tmp_path / "bin").mkdir()
    _ = (tmp_path / "bin" / "python").write_text("")

    original = _set(tmp_path)
    _ = write_slp(original, tmp_path / "out.slp")
    assert seen, "the subprocess was invoked"

    # Read the handed-over file back the way sleap-io will, and compare against
    # what went in. Asserting on the parsed set rather than on raw JSON keys is
    # the same question a consumer asks.
    replay = tmp_path / "handed_over.json"
    _ = replay.write_text(handed_over[0])
    delivered = read_coco_keypoints(replay, tmp_path / "images")

    assert delivered.schema == original.schema, "names and skeleton survive"
    assert len(delivered.frames) == len(original.frames)
    received = delivered.frames[0].objects[0].keypoints
    assert [k.visibility for k in received] == [2, 1, 0], "occlusion included"
    assert not received[2].is_placed, "an unplaced point stays unplaced"


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

    monkeypatch.setattr(labels_module, "run_supervised", run)
    monkeypatch.setenv("MOSAIC_SLEAP_BIN", str(tmp_path / "bin" / "sleap-track"))
    (tmp_path / "bin").mkdir()
    _ = (tmp_path / "bin" / "python").write_text("")

    written = write_slp(_set(tmp_path), target)
    assert observed == {"target_absent_during_write": True, "writes_elsewhere": True}
    assert written.read_bytes() == b"slp"


def test_a_failed_conversion_leaves_nothing_behind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No half-written .slp, and no intermediate COCO for a later run to trip on."""
    run, _ = _fake_run(write=False, returncode=1)
    monkeypatch.setattr(labels_module, "run_supervised", run)
    monkeypatch.setenv("MOSAIC_SLEAP_BIN", str(tmp_path / "bin" / "sleap-track"))
    (tmp_path / "bin").mkdir()
    _ = (tmp_path / "bin" / "python").write_text("")

    with pytest.raises(SleapError):
        _ = write_slp(_set(tmp_path), tmp_path / "out.slp")

    leftovers = sorted(p.name for p in tmp_path.iterdir() if p.is_file())
    assert leftovers == [], f"left behind: {leftovers}"


def test_a_silent_failure_is_still_a_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exit zero having written nothing is the case existence-gating misses."""
    run, _ = _fake_run(write=False, returncode=0)
    monkeypatch.setattr(labels_module, "run_supervised", run)
    monkeypatch.setenv("MOSAIC_SLEAP_BIN", str(tmp_path / "bin" / "sleap-track"))
    (tmp_path / "bin").mkdir()
    _ = (tmp_path / "bin" / "python").write_text("")

    with pytest.raises(FileNotFoundError, match="wrote no .slp"):
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
