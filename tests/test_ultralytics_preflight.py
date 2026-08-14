"""What mosaic declares about Ultralytics tracking, checked against what is installed.

This replaces the invocation suites the three subprocess trackers have. There is
no argv, no ``ToolEnv`` and no location ladder to test, because Ultralytics runs
in mosaic's own process. The equivalent questions are: is the dependency
declared, does the installed library know these six backends, and does mosaic's
declared default table still say what the installed one says.

The drift check **fails** rather than warns. A moved default silently re-means
every run identifier already on disk, so it needs a decision at upgrade time.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import cast

import pytest

from mosaic.core.json_value import JsonValue
from mosaic.tracking.ultralytics_track.tracker_defaults import (
    CLOSED_KEYS,
    TRACKER_KNOBS,
    TRACKER_NAMES,
    TrackerConfigError,
    TrackerName,
    resolve_tracker_config,
)

# Selected by CI's `tracking` job with `-m tracker` rather than by a filename
# list in the workflow, so a new file here is covered the day it lands.
pytestmark = pytest.mark.tracker

_REPO_ROOT = Path(__file__).resolve().parents[1]


# --- the declaration, checkable without Ultralytics ------------------------


def test_mosaic_declares_lap_rather_than_letting_ultralytics_install_it() -> None:
    """``lap`` is named by every extra that can reach a tracker backend.

    Ultralytics imports ``lap`` at module scope in its matching helper and, on
    ImportError, pip-installs it *during the run* -- a network write inside a
    queued job, and an outright failure in a locked environment. With ``lap``
    present the hazard is invisible, so reading the declaration is the only
    thing that can catch its removal.
    """
    pyproject = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text())
    extras = pyproject["project"]["optional-dependencies"]
    for extra in ("pose", "polo", "recommended"):
        declared = cast(list[str], extras[extra])
        assert any(spec.startswith("lap") for spec in declared), (
            f"[{extra}] can reach an Ultralytics tracker but does not declare lap"
        )


def test_every_backend_declares_the_settings_it_selects_on() -> None:
    """``tracker_type`` is present and correct in every resolved table."""
    for name in TRACKER_NAMES:
        assert resolve_tracker_config(name)["tracker_type"] == name


def test_tracktrack_is_not_built_on_the_shared_association_block() -> None:
    """The one table that would break a factored "common settings" row.

    Pinned because the temptation to share it is real and the damage would be a
    wrong default rather than an error.
    """
    tracktrack = resolve_tracker_config("tracktrack")
    bytetrack = resolve_tracker_config("bytetrack")
    assert "fuse_score" not in tracktrack
    assert "fuse_score" in bytetrack
    assert tracktrack["track_high_thresh"] != bytetrack["track_high_thresh"]


def test_an_override_restating_a_default_resolves_to_the_same_table() -> None:
    """The payoff of declaring the defaults instead of reading them."""
    assert resolve_tracker_config("bytetrack", {"track_buffer": 30}) == (
        resolve_tracker_config("bytetrack", None)
    )


def test_an_override_the_backend_lacks_names_the_ones_that_do_take_it() -> None:
    with pytest.raises(TrackerConfigError) as caught:
        _ = resolve_tracker_config("bytetrack", {"delta_t": 5})
    message = str(caught.value)
    assert "bytetrack has no setting 'delta_t'" in message
    assert "deepocsort, ocsort" in message


def test_the_backend_selector_cannot_be_overridden() -> None:
    with pytest.raises(TrackerConfigError, match="selects the backend"):
        _ = resolve_tracker_config("bytetrack", {"tracker_type": "botsort"})


def test_a_reid_checkpoint_cannot_be_named_by_path() -> None:
    for key in CLOSED_KEYS:
        with pytest.raises(TrackerConfigError, match="fixed at"):
            _ = resolve_tracker_config("botsort", {key: "weights/reid.pt"})


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("track_buffer", 30.5),  # a whole number, given a fraction
        ("fuse_score", 1),  # a flag, given an int
        ("match_thresh", True),  # a number, given a flag
        ("gmc_method", "optical"),  # a closed choice, given an unknown name
        ("gmc_method", 3),  # a name, given a number
    ],
)
def test_an_override_of_the_wrong_shape_is_refused_before_minting(
    key: str, value: JsonValue
) -> None:
    with pytest.raises(TrackerConfigError):
        _ = resolve_tracker_config("botsort", {key: value})


def test_a_whole_number_widens_into_a_threshold() -> None:
    """``1`` and ``1.0`` must not be two identifiers for one recipe."""
    widened = resolve_tracker_config("bytetrack", {"match_thresh": 1})["match_thresh"]
    assert isinstance(widened, float)
    assert widened == 1.0


# --- the declaration, checked against the installed library ----------------


def _installed_tracker_config(name: str) -> dict[str, object]:
    """Parse the installed Ultralytics' own YAML for *name*."""
    import yaml

    import ultralytics

    path = Path(ultralytics.__file__).parent / "cfg" / "trackers" / f"{name}.yaml"
    assert path.is_file(), f"the installed ultralytics has no {name}.yaml"
    return cast(dict[str, object], yaml.safe_load(path.read_text()))


@pytest.mark.parametrize("name", TRACKER_NAMES)
def test_each_declared_table_matches_the_installed_yaml(name: TrackerName) -> None:
    """Both directions.

    A setting installed but not declared is an identity gap: it affects the run
    and no identifier names it. A setting declared but not installed is a dead
    knob mosaic offers and nothing reads.
    """
    _ = pytest.importorskip("ultralytics")
    installed = _installed_tracker_config(name)
    declared = resolve_tracker_config(name)

    assert set(installed) == set(declared), (
        f"{name}: only installed={sorted(set(installed) - set(declared))}, "
        f"only declared={sorted(set(declared) - set(installed))}"
    )
    for key in sorted(installed):
        assert installed[key] == declared[key], f"{name}.{key}"
        assert type(installed[key]) is type(declared[key]), f"{name}.{key} type"


def test_the_six_backend_names_are_what_the_installed_ultralytics_knows() -> None:
    _ = pytest.importorskip("ultralytics")
    _ = pytest.importorskip("lap")
    from ultralytics.trackers.track import TRACKER_MAP

    assert set(TRACKER_NAMES) == set(TRACKER_MAP)


@pytest.mark.parametrize("name", TRACKER_NAMES)
def test_a_reset_restarts_the_track_id_counter_for_every_backend(
    name: TrackerName,
) -> None:
    """Each backend's ``reset()`` must return the *class-level* counter to zero.

    Mosaic resets between entries so that a run's identifiers do not depend on
    what ran before it in the same process. The counter lives on ``BaseTrack``,
    shared by every backend, and each backend overrides ``reset()`` -- so this is
    parametrized rather than asserted once: a subclass that clears its own pools
    without resetting the counter would pass a single global check.
    """
    _ = pytest.importorskip("ultralytics")
    _ = pytest.importorskip("lap")
    from ultralytics.trackers.basetrack import BaseTrack
    from ultralytics.trackers.track import TRACKER_MAP
    from ultralytics.utils import IterableSimpleNamespace

    settings = IterableSimpleNamespace(**resolve_tracker_config(name))
    tracker = TRACKER_MAP[name](args=settings)
    for _ in range(41):  # advance the shared counter through the public door
        _ = BaseTrack.next_id()
    tracker.reset()
    assert BaseTrack.next_id() == 1


def test_every_declared_knob_has_a_default_of_its_own_type() -> None:
    """A structural check on the tables themselves, needing nothing installed."""
    for name in TRACKER_NAMES:
        for key, knob in TRACKER_KNOBS[name].items():
            resolved = resolve_tracker_config(name)[key]
            assert type(resolved) is type(knob.default), f"{name}.{key}"
