"""What mosaic declares about Ultralytics tracking, against the environment that runs it.

Most of this file needs nothing installed: the tracker tables are mosaic's own
declaration, and the refusals over them are decided from that declaration alone.
The last section is the one that reaches outside, and it reaches into the
*Ultralytics* environment rather than mosaic's own, because mosaic's own no
longer holds Ultralytics -- that is the whole subject of
``tests/test_ultralytics_separation.py``.

**Why mosaic transcribes those tables at all.** Every detection-affecting setting
is passed explicitly and enters the run identifier, so an upstream retune must
not silently re-mean an identifier already on disk. That only works while the
transcription still says what the installed release says, and the drift check
below is what turns a moved default into a decision at upgrade time. It **fails**
rather than warns, for the same reason.

**What happened to the track-identity reset.** Mosaic used to reset the shared
``BaseTrack`` counter between entries, and a test here asserted that every
backend's ``reset()`` really returned it to zero -- otherwise a run's identifiers
would depend on what ran before it in the same process. Nothing resets it now,
and nothing needs to: each entry is tracked in a subprocess of its own, so the
class-level counter starts at zero by construction. The property did not stop
mattering; it is guaranteed structurally instead of by a call, which is why the
test that pinned the call is gone.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Final

import pytest
from pydantic import TypeAdapter

from mosaic.core.json_value import JsonValue
from mosaic.tracking.common.toolenv import ToolNotFoundError, tool_invocation
from mosaic.tracking.ultralytics_track.params import UltralyticsParams
from mosaic.tracking.ultralytics_track.run import (
    ULTRALYTICS_ENV,
    ultralytics_tracker_defaults,
)
from mosaic.tracking.ultralytics_track.tracker_defaults import (
    CLOSED_KEYS,
    GMC_BACKENDS,
    TRACKER_CONFIG_MODELS,
    TRACKER_NAMES,
    BotsortConfig,
    TrackerConfigError,
    TrackerName,
    TrackerSetting,
    resolve_tracker_config,
)

# Selected by CI's `tracking` job with `-m tracker` rather than by a filename
# list in the workflow, so a new file here is covered the day it lands.
pytestmark = pytest.mark.tracker

_REPO_ROOT = Path(__file__).resolve().parents[1]
_EXTERNAL = _REPO_ROOT / "src" / "mosaic" / "tracking" / "external"
ENVIRONMENT_DIRECTORIES: Final = ("ultralytics-env", "polo-env")
"""Every environment that runs an Ultralytics-family library, by directory.

Two, because POLO ships under the distribution name ``ultralytics`` and so cannot
share one with upstream. Every claim below is made of both: they run the same
program, so a declaration that landed in one and not the other is a difference
nothing else would report.
"""

_REQUIREMENTS: Final = TypeAdapter(list[str])
"""Reads a ``[project] dependencies`` array as the list of strings it is.

Validated rather than asserted so the shape is checked where it is read, and so
nothing here has to narrow ``tomllib``'s untyped document by hand.
"""


# --- the declaration, checkable with nothing installed ---------------------


@pytest.mark.parametrize("directory", ENVIRONMENT_DIRECTORIES)
def test_an_external_environment_declares_what_its_runner_needs(
    directory: str,
) -> None:
    """Every requirement whose absence is invisible until a run is under way.

    ``lap`` is the linear-assignment solver every one of the six backends reaches
    from module scope in ``ultralytics.trackers.utils.matching``, and it appears
    in no Ultralytics extra. Undeclared, Ultralytics pip-installs it *during the
    run* -- a network write inside a queued job, and an outright failure in a
    locked environment.

    ``opencv-python-headless`` is the environment's only ``cv2``, and it has to be
    declared here rather than arriving with the ``augment`` extra: the override
    below excludes the GUI wheel ``ultralytics`` asks for, so without this
    declaration a plain ``uv sync`` resolves no ``cv2`` provider at all and
    ``import ultralytics`` fails before any subcommand runs.

    Asked of the environments' own manifests rather than of mosaic's, because
    mosaic's environment is no longer where any of this runs. With each of them
    present the hazard is invisible, so reading the declaration is the only thing
    that can catch a removal.
    """
    manifest = _EXTERNAL / directory / "pyproject.toml"
    document = tomllib.loads(manifest.read_text())
    declared = _REQUIREMENTS.validate_python(document["project"]["dependencies"])

    assert any(spec.startswith("ultralytics") for spec in declared), (
        f"{manifest} declares no ultralytics; this test has lost its subject"
    )
    assert any(spec.startswith("lap") for spec in declared), (
        f"{manifest} runs Ultralytics tracking but does not declare lap"
    )
    assert any(spec.startswith("opencv-python-headless") for spec in declared), (
        f"{manifest} declares no cv2 provider, so `import ultralytics` fails "
        "there: the GUI wheel it requires is excluded by this environment's own "
        "override"
    )
    assert not any(spec.startswith("opencv-python>") for spec in declared), (
        f"{manifest} declares the GUI OpenCV wheel beside the headless one; two "
        "distributions shipping one import package overwrite each other and leave "
        "two vendored ffmpeg builds in a single cv2"
    )


@pytest.mark.parametrize("directory", ENVIRONMENT_DIRECTORIES)
def test_an_external_environment_offers_augmentation_without_a_second_cv2(
    directory: str,
) -> None:
    """The albumentations opt-in follows the process that reads it, and costs no cv2.

    Ultralytics builds its ``Albumentations`` transform whenever the package is
    importable, so this changes what a training run does and stays an extra. What
    it must not do is bring ``opencv-python-headless``'s rival with it: albucore
    requires the headless build and Ultralytics requires the GUI one, and pip and
    uv both install the pair without complaint.
    """
    manifest = _EXTERNAL / directory / "pyproject.toml"
    document = tomllib.loads(manifest.read_text())

    extras = document["project"]["optional-dependencies"]
    augment = _REQUIREMENTS.validate_python(extras["augment"])
    assert [spec.split(">")[0] for spec in augment] == ["albumentations"]

    overrides = _REQUIREMENTS.validate_python(
        document["tool"]["uv"]["override-dependencies"]
    )
    assert any(
        spec.startswith("opencv-python ") and "never" in spec for spec in overrides
    ), (
        f"{manifest} does not exclude the GUI OpenCV wheel, so a build with the "
        "augment extra installs two cv2 providers"
    )


@pytest.mark.parametrize("directory", ENVIRONMENT_DIRECTORIES)
def test_a_locked_external_environment_resolves_one_cv2(directory: str) -> None:
    """The declaration above, confirmed against what the lock actually pins.

    A manifest says what was asked for; the lock says what a build gets, and it
    is the lock ``uv sync`` reads. The GUI wheel is expected to be *present* and
    unbuildable -- the override records it rather than removing it -- so this
    asserts the marker rather than the absence.
    """
    lock = (_EXTERNAL / directory / "uv.lock").read_text()

    assert 'name = "opencv-python-headless"' in lock, (
        f"{directory}/uv.lock pins no headless OpenCV, so a synced environment "
        "has no cv2 for ultralytics to import"
    )
    assert '{ name = "opencv-python", marker = "sys_platform == \'never\'" }' in lock, (
        f"{directory}/uv.lock does not record the GUI OpenCV wheel as excluded, "
        "so a build can install it beside the headless one"
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


def test_a_closed_setting_restated_at_its_fixed_value_is_accepted() -> None:
    """Refusing this is what broke reading a recorded parameter set back.

    The resolved table names every setting, so a caller that submits what it was
    given submits the closed ones too. Only a value that would change one is
    refused.
    """
    for key in CLOSED_KEYS:
        fixed = resolve_tracker_config("botsort")[key]
        assert resolve_tracker_config("botsort", {key: fixed}) == (
            resolve_tracker_config("botsort")
        )


@pytest.mark.parametrize("name", TRACKER_NAMES)
def test_a_parameter_set_reads_back_as_the_one_that_was_written(
    name: TrackerName,
) -> None:
    """``model_dump()`` must construct, through Python objects and through JSON.

    ``run_op`` validates a parameter dictionary, ``build_op_params`` builds one
    for a recipe, and a client submits the fields the published schema showed
    it. Each of those reads back what an earlier dump wrote, so a dump the model
    refuses is a recorded run that cannot be replayed.
    """
    written = UltralyticsParams(
        model_path="models/pose/best.pt",
        tracker=name,
        tracker_overrides=TRACKER_CONFIG_MODELS[name](),
    )
    dumped = written.model_dump()
    assert UltralyticsParams.model_validate(dumped).model_dump() == dumped

    as_json = written.model_dump_json()
    assert UltralyticsParams.model_validate_json(as_json).model_dump_json() == as_json


def test_gmc_backends_names_the_models_that_declare_a_motion_estimator() -> None:
    """The set and the tables must not drift apart.

    ``bytetrack`` is the default backend because the alternatives can estimate
    camera motion, and that claim is only as good as this set.
    """
    declaring = {
        name
        for name in TRACKER_NAMES
        if "gmc_method" in TRACKER_CONFIG_MODELS[name].model_fields
    }
    assert declaring == GMC_BACKENDS


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("track_buffer", 30.5),  # a whole number, given a fraction
        ("track_buffer", True),  # a whole number, given a flag
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


# --- the declaration, checked against the Ultralytics environment ----------


@pytest.fixture(scope="module")
def installed_tracker_tables() -> dict[str, dict[str, TrackerSetting]]:
    """Every backend's shipped table, read in the environment that runs them.

    Module-scoped because the answer costs one subprocess and a cold torch
    import, and every test below asks it of the same environment. The runner
    reports all six in one response for the same reason.

    Whether an environment resolves is decided by :func:`tool_invocation` -- the
    same five-step ladder a real run walks -- rather than by reading
    ``MOSAIC_ULTRALYTICS_CONDA_ENV`` and ``MOSAIC_ULTRALYTICS_BIN`` here, so the
    skip condition and the lookup cannot come to disagree about what "the
    environment is there" means.
    """
    try:
        _ = tool_invocation(ULTRALYTICS_ENV, executable="python")
    except ToolNotFoundError as absent:
        pytest.skip(f"no Ultralytics environment resolves: {absent}")
    return ultralytics_tracker_defaults().tables


def test_the_environment_knows_exactly_the_backends_mosaic_declares(
    installed_tracker_tables: dict[str, dict[str, TrackerSetting]],
) -> None:
    """A backend either side has and the other does not.

    Mosaic naming one the release does not have is a run that fails inside
    Ultralytics; the release having one mosaic does not name is a backend nobody
    can select through mosaic, which is a decision rather than an accident and so
    belongs at upgrade time.
    """
    assert set(installed_tracker_tables) == set(TRACKER_NAMES)


@pytest.mark.parametrize("name", TRACKER_NAMES)
def test_each_declared_table_matches_the_installed_one(
    name: TrackerName,
    installed_tracker_tables: dict[str, dict[str, TrackerSetting]],
) -> None:
    """Both directions.

    A setting installed but not declared is an identity gap: it affects the run
    and no identifier names it. A setting declared but not installed is a dead
    setting mosaic offers and nothing reads.
    """
    assert name in installed_tracker_tables, (
        f"the ultralytics in this environment knows "
        f"{sorted(installed_tracker_tables)}, not {name!r}"
    )
    installed = installed_tracker_tables[name]
    declared = resolve_tracker_config(name)

    assert set(installed) == set(declared), (
        f"{name}: only installed={sorted(set(installed) - set(declared))}, "
        f"only declared={sorted(set(declared) - set(installed))}"
    )
    for key in sorted(installed):
        assert installed[key] == declared[key], f"{name}.{key}"
        assert type(installed[key]) is type(declared[key]), f"{name}.{key} type"


def test_the_resolved_table_is_the_declared_defaults() -> None:
    """A structural check on the models themselves, needing nothing installed.

    The type is asserted beside the value, because ``1 == 1.0`` in Python and a
    whole number where a threshold belongs is a second identifier for one recipe.
    """
    for name in TRACKER_NAMES:
        resolved = resolve_tracker_config(name)
        declared = TRACKER_CONFIG_MODELS[name].model_fields
        assert set(resolved) == set(declared), name
        for key, field in declared.items():
            assert resolved[key] == field.default, f"{name}.{key}"
            assert type(resolved[key]) is type(field.default), f"{name}.{key} type"


def test_a_configuration_model_resolves_to_the_same_table_as_its_overrides() -> None:
    """The two doors into ``resolve_tracker_config`` agree.

    ``UltralyticsParams`` validates ``tracker_overrides`` into a model, and the
    raw mapping is what a direct caller and the golden corpus pass.
    """
    assert resolve_tracker_config("botsort", BotsortConfig(track_buffer=90)) == (
        resolve_tracker_config("botsort", {"track_buffer": 90})
    )


def test_a_configuration_for_another_backend_is_refused() -> None:
    """The resolved table must not describe a tracker the identifier does not name."""
    with pytest.raises(TrackerConfigError, match="cannot configure"):
        _ = resolve_tracker_config("bytetrack", BotsortConfig())
