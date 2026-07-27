"""The read-target gate: verdict-based fitness checks before a reader opens a file."""

from __future__ import annotations

import ast
import dataclasses
import warnings
from collections.abc import Callable
from pathlib import Path

import pytest
from mosaic_media import CHROME_149, MediaFacts, MediaProbeError, derive, probe_media

import mosaic.core.media.read_target as read_target
from mosaic.core.media.video_io import MultiVideoReader
from mosaic.media_probe_config import media_thresholds

WriteVideo = Callable[..., None]


def _playable(clip: Path) -> MediaFacts:
    """Facts that derive to a soft-only stream verdict and a clean analysis axis."""
    return dataclasses.replace(
        probe_media(clip),
        container="mov,mp4,m4a,3gp,3g2,mj2",
        codec_name="h264",
        pixel_format="yuv420p",
        moov_at_start=False,
    )


def _stream_only_defect(clip: Path) -> MediaFacts:
    """Facts that add a hard stream reason (``unsupported_codec``) while the
    analysis axis stays clean."""
    return dataclasses.replace(_playable(clip), codec_name="mpeg4")


def _analysis_only_defect(clip: Path) -> MediaFacts:
    """Facts that add an analysis reason (``unreliable_timing_metadata``) while
    the stream verdict stays ``"recommended"``."""
    return dataclasses.replace(_playable(clip), declared_frame_count=999)


def test_supplied_facts_are_returned_without_a_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, write_cfr_mp4: WriteVideo
) -> None:
    clip = tmp_path / "clip.mp4"
    write_cfr_mp4(clip)
    facts = _playable(clip)

    def _boom(path: Path) -> MediaFacts:
        raise AssertionError("probe_media was called despite supplied facts")

    monkeypatch.setattr(read_target, "probe_media", _boom)

    result = read_target.verified_read_facts(clip, facts, "analysis")
    assert result[0] is facts


def test_absent_facts_are_probed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, write_cfr_mp4: WriteVideo
) -> None:
    clip = tmp_path / "clip.mp4"
    write_cfr_mp4(clip)
    calls: list[Path] = []
    probed: list[MediaFacts] = []
    real_probe = read_target.probe_media

    def _counting(path: Path) -> MediaFacts:
        calls.append(path)
        measured = real_probe(path)
        probed.append(measured)
        return measured

    monkeypatch.setattr(read_target, "probe_media", _counting)

    # The target must be "analysis": this probes the real clip, and
    # write_cfr_mp4 writes mp4v, whose facts carry a hard stream reason. A
    # "playback" target would raise here for a reason unrelated to what this
    # test asserts.
    result = read_target.verified_read_facts(clip, None, "analysis")

    assert calls == [clip.resolve()]
    assert result[0] == probed[0]


def test_analysis_target_raises_on_an_analysis_defect(
    tmp_path: Path, write_cfr_mp4: WriteVideo
) -> None:
    clip = tmp_path / "clip.mp4"
    write_cfr_mp4(clip)
    facts = _analysis_only_defect(clip)
    assert (
        derive(facts, CHROME_149, media_thresholds()).analysis_transcode == "required"
    )

    with pytest.raises(
        MediaProbeError, match="not fit for an analysis read"
    ) as excinfo:
        read_target.verified_read_facts(clip, facts, "analysis")
    assert "unreliable_timing_metadata" in str(excinfo.value)


def test_playback_target_raises_on_a_hard_stream_reason(
    tmp_path: Path, write_cfr_mp4: WriteVideo
) -> None:
    clip = tmp_path / "clip.mp4"
    write_cfr_mp4(clip)
    facts = _stream_only_defect(clip)
    assert derive(facts, CHROME_149, media_thresholds()).stream_transcode == "required"

    with pytest.raises(MediaProbeError, match="not fit for a playback read") as excinfo:
        read_target.verified_read_facts(clip, facts, "playback")
    assert "unsupported_codec" in str(excinfo.value)


def test_playback_target_accepts_a_recommended_stream_transcode(
    tmp_path: Path, write_cfr_mp4: WriteVideo
) -> None:
    """Pins the trap: gating on ``stream_transcode is not None`` would raise on
    an ordinary healthy high-bitrate original."""
    clip = tmp_path / "clip.mp4"
    write_cfr_mp4(clip)
    facts = _playable(clip)
    assert (
        derive(facts, CHROME_149, media_thresholds()).stream_transcode == "recommended"
    )

    result = read_target.verified_read_facts(clip, facts, "playback")
    assert result[0] is facts


def test_analysis_target_ignores_the_stream_axis(
    tmp_path: Path, write_cfr_mp4: WriteVideo
) -> None:
    clip = tmp_path / "clip.mp4"
    write_cfr_mp4(clip)
    facts = _stream_only_defect(clip)
    verdict = derive(facts, CHROME_149, media_thresholds())
    assert verdict.stream_transcode == "required"
    assert verdict.analysis_transcode is None

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = read_target.verified_read_facts(clip, facts, "analysis")
    assert result[0] is facts


def test_playback_target_ignores_the_analysis_axis(
    tmp_path: Path, write_cfr_mp4: WriteVideo
) -> None:
    clip = tmp_path / "clip.mp4"
    write_cfr_mp4(clip)
    facts = _analysis_only_defect(clip)
    verdict = derive(facts, CHROME_149, media_thresholds())
    assert verdict.analysis_transcode == "required"
    assert verdict.stream_transcode == "recommended"

    result = read_target.verified_read_facts(clip, facts, "playback")
    assert result[0] is facts


def test_raw_target_warns_and_proceeds_on_an_analysis_defect(
    tmp_path: Path, write_cfr_mp4: WriteVideo
) -> None:
    clip = tmp_path / "clip.mp4"
    write_cfr_mp4(clip)
    facts = _analysis_only_defect(clip)
    assert (
        derive(facts, CHROME_149, media_thresholds()).analysis_transcode == "required"
    )

    with pytest.warns(UserWarning, match="raw read of") as record:
        result = read_target.verified_read_facts(clip, facts, "raw")
    assert result[0] is facts
    assert "unreliable_timing_metadata" in str(record[0].message)


def test_raw_target_is_silent_on_a_stream_only_defect(
    tmp_path: Path, write_cfr_mp4: WriteVideo
) -> None:
    """A stream-only reason must not produce raw-read noise."""
    clip = tmp_path / "clip.mp4"
    write_cfr_mp4(clip)
    facts = _stream_only_defect(clip)
    verdict = derive(facts, CHROME_149, media_thresholds())
    assert verdict.stream_transcode == "required"
    assert verdict.analysis_transcode is None

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = read_target.verified_read_facts(clip, facts, "raw")
    assert result[0] is facts


def test_a_facts_sequence_of_the_wrong_length_raises_before_probing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, write_cfr_mp4: WriteVideo
) -> None:
    clip_a = tmp_path / "a.mp4"
    clip_b = tmp_path / "b.mp4"
    write_cfr_mp4(clip_a)
    write_cfr_mp4(clip_b)
    facts = _playable(clip_a)

    def _boom(path: Path) -> MediaFacts:
        raise AssertionError("probe_media was called before the length check ran")

    monkeypatch.setattr(read_target, "probe_media", _boom)

    with pytest.raises(ValueError, match="facts length 1 does not match path count 2"):
        read_target.verified_read_facts([clip_a, clip_b], [facts], "analysis")


def test_a_sequence_of_paths_returns_one_facts_entry_per_path(
    tmp_path: Path, write_cfr_mp4: WriteVideo
) -> None:
    clip_a = tmp_path / "a.mp4"
    clip_b = tmp_path / "b.mp4"
    write_cfr_mp4(clip_a)
    write_cfr_mp4(clip_b)
    facts_a = _playable(clip_a)
    facts_b = _playable(clip_b)

    result = read_target.verified_read_facts(
        [clip_a, clip_b], [facts_a, facts_b], "analysis"
    )

    assert result == [facts_a, facts_b]


def test_multi_video_reader_plain_video_branch_is_gated(
    tmp_path: Path, write_cfr_mp4: WriteVideo
) -> None:
    """The plain-video branch of ``open_multi_video_reader`` has no other test:
    the imgstore suite in ``test_imgstore_io.py`` exercises only the store
    branch, and its mixed-sequence test raises before reaching the gate at
    all, so a caller that left ``facts=facts`` ungated there would still pass
    the whole suite.
    """
    clip_a = tmp_path / "a.mp4"
    clip_b = tmp_path / "b.mp4"
    write_cfr_mp4(clip_a)
    write_cfr_mp4(clip_b)

    defective = _analysis_only_defect(clip_a)
    with pytest.raises(MediaProbeError, match="not fit for an analysis read"):
        MultiVideoReader(
            [clip_a, clip_b],
            facts=[defective, _playable(clip_b)],
            target="analysis",
        )

    reader = MultiVideoReader(
        [clip_a, clip_b],
        facts=[_playable(clip_a), _playable(clip_b)],
        target="analysis",
    )
    reader.close()


def _imports_a_mosaic_media_reader(tree: ast.Module) -> bool:
    """True when *tree* imports ``VideoReader`` or ``MultiVideoReader`` from
    ``mosaic_media.io``, in any import form."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        module = node.module or ""
        if module != "mosaic_media.io" and not module.startswith("mosaic_media.io."):
            continue
        if any(
            alias.name in {"VideoReader", "MultiVideoReader"} for alias in node.names
        ):
            return True
    return False


def _imports_the_gate(tree: ast.Module) -> bool:
    """True when *tree* imports ``verified_read_facts`` from the read-target gate
    module, absolute or relative."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        if not any(alias.name == "verified_read_facts" for alias in node.names):
            continue
        module = node.module or ""
        if module == "mosaic.core.media.read_target":
            return True
        if node.level >= 1 and module.endswith("read_target"):
            return True
    return False


def test_only_the_gate_and_its_two_reader_modules_import_mosaic_media_readers() -> None:
    """Every module that constructs a ``mosaic_media`` reader must also import
    the read-target gate.

    This proves only that the import is present: it prevents a new module from
    constructing a reader without also importing the gate, but it does not
    prove the gate is actually called on the read path.
    """
    import mosaic

    root = Path(mosaic.__file__).parent
    reader_importers: set[str] = set()
    for source in root.rglob("*.py"):
        # A vendored virtualenv under this excluded workspace member; not part
        # of mosaic's own tree.
        if "feature_library/external" in source.as_posix():
            continue
        tree = ast.parse(source.read_text())
        if _imports_a_mosaic_media_reader(tree):
            reader_importers.add(str(source.relative_to(root)))

    assert reader_importers == {
        "core/media/video_io.py",
        "core/media/imgstore_native.py",
    }

    gate_importers = {
        module
        for module in reader_importers
        if _imports_the_gate(ast.parse((root / module).read_text()))
    }
    assert gate_importers, "the guard must not pass by matching an empty set"
    assert gate_importers == reader_importers


_UPSTREAM_READER_NAMES = {"VideoReader", "MultiVideoReader"}


def _imports_an_upstream_reader_directly(tree: ast.Module) -> bool:
    """True when *tree* imports ``VideoReader`` or ``MultiVideoReader`` from
    ``mosaic_media``, in any form: a ``from mosaic_media[.submodule...] import``
    naming either name directly (``from mosaic_media import VideoReader``,
    ``from mosaic_media.io import ...``, ``from mosaic_media.io.reader import
    ...``); a submodule bound by name (``from mosaic_media import io`` then
    ``io.VideoReader``); or a module-level ``import mosaic_media[.submodule]``
    (any alias, e.g. ``import mosaic_media.io as mmio``) followed by an
    attribute chain of any depth ending in one of those names.
    """
    aliases = {"mosaic_media"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname is not None and (
                    alias.name == "mosaic_media"
                    or alias.name.startswith("mosaic_media.")
                ):
                    aliases.add(alias.asname)
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            from_upstream = module == "mosaic_media" or module.startswith(
                "mosaic_media."
            )
            if from_upstream:
                for alias in node.names:
                    if alias.name not in _UPSTREAM_READER_NAMES:
                        aliases.add(alias.asname or alias.name)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            from_upstream = module == "mosaic_media" or module.startswith(
                "mosaic_media."
            )
            if from_upstream and any(
                alias.name in _UPSTREAM_READER_NAMES for alias in node.names
            ):
                return True
        if isinstance(node, ast.Attribute) and node.attr in _UPSTREAM_READER_NAMES:
            root_value = node.value
            while isinstance(root_value, ast.Attribute):
                root_value = root_value.value
            if isinstance(root_value, ast.Name) and root_value.id in aliases:
                return True
    return False


def test_only_the_two_reader_modules_may_construct_an_upstream_reader() -> None:
    """No module outside the two reader-construction sites may import
    ``VideoReader`` or ``MultiVideoReader`` from ``mosaic_media`` at all.

    This is a stronger property than the import-co-location guard above: that
    guard only catches a module that constructs a reader without also
    importing the gate, while this one catches a bare upstream reader import
    at the point it is written, in any of the forms ``ast`` can see, whether
    or not the module goes on to import the gate. It does not prove the two
    allowlisted modules call the gate correctly on every reader they
    construct -- the per-site tests in this file and in
    test_media_facts_injection.py / test_imgstore_io.py cover that.
    """
    import mosaic

    allowed = {"core/media/video_io.py", "core/media/imgstore_native.py"}
    root = Path(mosaic.__file__).parent
    matched: set[str] = set()
    for source in root.rglob("*.py"):
        # A vendored virtualenv under this excluded workspace member; not part
        # of mosaic's own tree.
        if "feature_library/external" in source.as_posix():
            continue
        if _imports_an_upstream_reader_directly(ast.parse(source.read_text())):
            matched.add(str(source.relative_to(root)))

    offenders = sorted(matched - allowed)
    message = (
        "these modules import a mosaic_media reader directly instead of "
        "obtaining one through open_frame_reader or open_multi_video_reader, "
        "which gate their facts: " + ", ".join(offenders)
    )
    assert offenders == [], message

    missing = allowed - matched
    assert not missing, (
        "expected both allowlisted modules to import an upstream reader, but "
        f"these did not match: {sorted(missing)} -- a rename may have made "
        "this guard scan for something that no longer exists"
    )
