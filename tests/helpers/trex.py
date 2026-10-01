"""Stand in for TREx in the tests of the TREx tracker, recording each call.

``run_trex`` calls TREx through two module-level seams in
``trex/dataset_runs.py``, one per phase. :func:`install_fake_trex` replaces both
with a :class:`FakeTrex`. A test runs the whole tracker protocol without a TREx
binary: identity, markers, reuse, the conversion cache and the bridge.
"""

from __future__ import annotations

import struct
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from mosaic.core.pipeline._utils import atomic_savez
from mosaic.tracking.trex.run import TRexConvertResult, TRexTrackResult


@dataclass
class FakeTrex:
    """Recording stand-ins for the two TREx phases."""

    converted: list[Path] = field(default_factory=list)
    tracked: list[Path] = field(default_factory=list)
    convert_kwargs: list[dict[str, object]] = field(default_factory=list)
    """The keyword arguments of each conversion, in call order."""
    track_kwargs: list[dict[str, object]] = field(default_factory=list)
    """The keyword arguments of each tracking run, in call order."""
    npz_per_track: int = 1
    npz_frames: int = 4
    """How many frames each per-individual export carries.

    It is four by default, the count that the TREx marker and reuse tests were
    written against. A test of the frame axis sets it. A value short of the
    media's total is the output of TREx's joined conversion when it drops the tail
    of each clip, and it is the only way to exercise that path without a real
    tool.
    """
    extra_fields: Mapping[str, npt.NDArray[np.generic]] = field(
        default_factory=dict[str, npt.NDArray[np.generic]]
    )
    """Fields in each export beyond those that TREx always writes.

    A user adds them to TREx's ``output_fields``. Each array has one value per
    frame.
    """
    pv_frames: int | None = None
    """How many frames each conversion's ``.pv`` header records.

    ``None`` records :attr:`npz_frames`, as though each export spanned every
    frame that TREx read. A test sets it apart from :attr:`npz_frames` to state
    what TREx read separately from what its exports hold.
    """
    write_settings: bool = True
    """Whether a conversion writes its ``.settings`` file, as TREx always does."""
    pv_beside_the_video: bool = False
    on_convert: Callable[[Path], None] | None = None
    """A hook, called with the output directory once a conversion's files are written.

    A hook that raises stands for a conversion that died after writing them.
    """
    sources: list[list[Path]] = field(default_factory=list)
    """Every conversion's *whole* source list, so a joined run is inspectable.

    ``converted`` keeps recording one path per call, the first source, the path
    that a single-video assertion reads.
    """

    def convert(
        self,
        video_path: Path | Sequence[Path],
        seq_dir: Path,
        *,
        output_name: str | None = None,
        **kwargs: object,
    ) -> TRexConvertResult:
        # Mirrors run_trex_convert's own normalisation: one source or many.
        given = (
            [Path(video_path)]
            if isinstance(video_path, (str, Path))
            else [Path(p) for p in video_path]
        )
        self.sources.append(given)
        self.converted.append(given[0])
        self.convert_kwargs.append(dict(kwargs))
        stem = output_name if output_name is not None else given[0].stem
        # `pv_beside_the_video` models TREx choosing its own location, which it
        # only does when nothing pinned the name. Given `filename`, it writes
        # where it was told -- the same order `run_trex_convert` looks in.
        home = (
            given[0].parent
            if self.pv_beside_the_video and output_name is None
            else Path(seq_dir)
        )
        home.mkdir(parents=True, exist_ok=True)
        pv_path = home / f"{stem}.pv"
        write_pv_header(
            pv_path, self.npz_frames if self.pv_frames is None else self.pv_frames
        )
        # TREx writes a settings file beside every conversion, and it is not
        # decorative: re-opening a `.pv` recovers only seven fields from the file
        # itself, so this is the only thing carrying the detection parameters
        # into a later tracking run. Written here for the same reason the npz
        # below is real rather than a stub -- a fake that omits what the tool
        # always produces exercises a path that cannot happen.
        settings_path = home / f"{stem}.settings"
        if self.write_settings:
            _ = settings_path.write_text("detect_type = yolo\n")
        # TREx also writes a results file at the end of every conversion,
        # whatever it is asked for.
        _ = (home / f"{stem}.results").write_bytes(b"conversion results")
        if self.on_convert is not None:
            self.on_convert(Path(seq_dir))
        return TRexConvertResult(
            pv_path=pv_path,
            settings_path=settings_path,
            background_path=None,
            stdout="",
            stderr="",
        )

    def track(self, pv_path: Path, seq_dir: Path, **kwargs: object) -> TRexTrackResult:
        self.tracked.append(Path(pv_path))
        self.track_kwargs.append(dict(kwargs))
        out = Path(seq_dir)
        stem = Path(pv_path).stem
        data_dir = out / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        npz_paths: list[Path] = []
        for i in range(self.npz_per_track):
            # The fake writes a convertible export instead of a stub. The tests
            # that use this fake are about markers and reuse and not conversion,
            # but a stub made every bridge fail, which is recorded as a lost
            # entry. An export in TREx's format keeps them on the real publish
            # path.
            n = self.npz_frames
            fields: dict[str, npt.NDArray[np.generic]] = {
                "frame": np.arange(n),
                "time": np.arange(n) / 30.0,
                "cm_per_pixel": np.array([1.0]),
                "X#wcentroid": np.arange(n, dtype=float),
                "Y#wcentroid": np.arange(n, dtype=float),
                **self.extra_fields,
            }
            # TREx names each export for its individual: `<stem>_fish<i>`.
            npz = data_dir / f"{stem}_fish{i}.npz"
            atomic_savez(npz, **fields)
            npz_paths.append(npz)
        results = out / f"{stem}.results"
        _ = results.write_bytes(b"results")
        return TRexTrackResult(
            npz_paths=npz_paths,
            results_path=results,
            settings_path=out / f"{stem}.settings",
            stdout="",
            stderr="",
        )


def write_pv_header(path: Path, frames: int, *, version: int = 15) -> None:
    """Write the header of a ``.pv`` that records *frames* frames, and nothing after it.

    The fields and their order are those of ``Header::write`` in TREx's
    ``Application/src/ProcessedVideo/pv.cpp`` for *version*, which each version
    extends. An older *version* writes what that version's reader expects.
    """
    header = bytearray(f"PV{version}".encode() + b"\0")
    if version >= 14:
        header += b"gray\0"
    else:
        header += bytes([1] if version < 12 else [1, 0])
    header += struct.pack("<HH", 64, 48)
    if version >= 3:
        header += struct.pack("<4H", 0, 0, 8000, 8000)
    if version >= 15:
        header += struct.pack("<qq", -1, -1) + b"clip.mp4\0"
    header += struct.pack("<BIQQ", 4, frames, 0, 0) + b"clip\0"
    _ = path.write_bytes(bytes(header))


def install_fake_trex(
    monkeypatch: pytest.MonkeyPatch, fake: FakeTrex | None = None
) -> FakeTrex:
    """Replace the tracker's two phase seams with *fake*, or a new :class:`FakeTrex`."""
    import mosaic.tracking.trex.dataset_runs as dataset_runs

    installed = fake if fake is not None else FakeTrex()
    monkeypatch.setattr(dataset_runs, "run_trex_convert", installed.convert)
    monkeypatch.setattr(dataset_runs, "run_trex_track", installed.track)
    return installed
