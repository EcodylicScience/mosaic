"""The resolved media-probe thresholds are real integers.

Thresholds is a slots=True dataclass, so Thresholds.max_gop_bytes read off the
class is a member_descriptor, not the integer default. media_thresholds must
read the defaults from the DEFAULT_THRESHOLDS instance instead; if it reads the
class attribute the value is a member_descriptor and every comparison against it
raises TypeError at verdict time rather than here.
"""

import ast

import pytest
from mosaic_media import DEFAULT_THRESHOLDS

from mosaic.media_probe_config import media_thresholds

from tests.helpers import inside_a_virtualenv, runs_in_an_external_environment


def test_defaults_are_the_documented_integers() -> None:
    thresholds = media_thresholds()
    assert thresholds.max_gop_bytes == 524_288
    assert thresholds.max_keyframe_interval_frames == 200


def test_defaults_are_real_integers_not_member_descriptors() -> None:
    thresholds = media_thresholds()
    # A member_descriptor would make this comparison raise TypeError.
    assert isinstance(thresholds.max_gop_bytes, int)
    assert not 5 > thresholds.max_gop_bytes
    assert thresholds.max_gop_bytes == DEFAULT_THRESHOLDS.max_gop_bytes


def test_env_overrides_are_honored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEDIA_PROBE_MAX_GOP_BYTES", "1048576")
    monkeypatch.setenv("MEDIA_PROBE_MAX_KEYFRAME_INTERVAL_FRAMES", "300")
    thresholds = media_thresholds()
    assert thresholds.max_gop_bytes == 1_048_576
    assert thresholds.max_keyframe_interval_frames == 300


def test_a_blank_value_falls_back_to_the_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MEDIA_PROBE_MAX_GOP_BYTES", "   ")
    thresholds = media_thresholds()
    assert thresholds.max_gop_bytes == DEFAULT_THRESHOLDS.max_gop_bytes


def test_a_non_integer_value_names_the_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MEDIA_PROBE_MAX_GOP_BYTES", "not-a-number")
    with pytest.raises(ValueError, match="MEDIA_PROBE_MAX_GOP_BYTES"):
        _ = media_thresholds()


def test_the_module_reads_no_mosaic_core() -> None:
    # The module is top level rather than under mosaic.core precisely so a
    # light backend consumer does not acquire cv2, numpy, and pandas by
    # importing it. Reading its source and asserting it names no mosaic.core
    # import guards that intent without importing the heavy packages here.
    import sys
    from pathlib import Path

    module = sys.modules["mosaic.media_probe_config"]
    assert module.__file__ is not None
    text = Path(module.__file__).read_text()
    assert "mosaic.core" not in text
    assert "from .core" not in text


def _uses_upstream_default_thresholds(tree: ast.Module) -> bool:
    """True when this module reads `mosaic_media.DEFAULT_THRESHOLDS`.

    Parsed, never grepped. A substring search for the name would flag a comment,
    a docstring, or -- the case that matters in a toolkit full of thresholds --
    an unrelated module defining its own `DEFAULT_THRESHOLDS` for features or
    tracking. That constant is correct and this guard must not touch it; the
    violation is reading the upstream one instead of resolving through
    `media_thresholds`.
    """
    aliases = {"mosaic_media"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "mosaic_media" and alias.asname is not None:
                    aliases.add(alias.asname)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            from_upstream = module == "mosaic_media" or module.startswith(
                "mosaic_media."
            )
            if from_upstream and any(
                alias.name == "DEFAULT_THRESHOLDS" for alias in node.names
            ):
                return True
        if isinstance(node, ast.Attribute) and node.attr == "DEFAULT_THRESHOLDS":
            value = node.value
            if isinstance(value, ast.Name) and value.id in aliases:
                return True
    return False


def test_every_derive_call_in_mosaic_resolves_thresholds() -> None:
    # The failure this guards against is a call site added later that reads the
    # upstream default directly, silently ignoring the deployment's
    # configuration for that one path.
    from pathlib import Path

    import mosaic

    root = Path(mosaic.__file__).parent
    offenders: list[str] = []
    scanned: set[str] = set()
    for source in root.rglob("*.py"):
        if source.name == "media_probe_config.py":
            continue
        # Both exclusions, because both reasons are real here. A program in an
        # external-environment tree may take no import from mosaic at all, so it
        # cannot call `media_thresholds` and this rule cannot reach it; and an
        # installed virtualenv is not mosaic's code wherever it lands, which the
        # first predicate only happens to cover while every such environment is
        # built inside one of the two named trees.
        if inside_a_virtualenv(source, root) or runs_in_an_external_environment(
            source, root
        ):
            continue
        scanned.add(source.relative_to(root).as_posix())
        if _uses_upstream_default_thresholds(ast.parse(source.read_text())):
            offenders.append(str(source.relative_to(root)))
    # A walk that reads nothing reports no offenders and proves nothing, and an
    # exclusion one predicate too broad is how that happens. So it names a module
    # it must have read: `read_target.py` calls `media_thresholds`, which is the
    # very habit this guard is about.
    assert "core/media/read_target.py" in scanned, (
        "the walk did not read core/media/read_target.py, so it is not covering "
        f"the modules this rule is about; it read {len(scanned)} files"
    )
    message = (
        "these modules read mosaic_media.DEFAULT_THRESHOLDS instead of calling "
        "media_thresholds: " + ", ".join(offenders)
    )
    assert offenders == [], message
