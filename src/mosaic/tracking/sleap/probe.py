"""Asking a SLEAP environment what it holds, before a run pays for it.

Three questions a run cannot answer for itself, and each of them looks like
success until hours later.

**Is the trainer actually installed?** ``pip install sleap`` without the ``nn``
extra puts SLEAP's console scripts on ``$PATH`` anyway: they belong to the
``sleap`` distribution itself, while ``sleap_nn`` arrives only through
``sleap[nn]`` and its CUDA variants. So mosaic's location ladder finds the tool,
launches it, and the subprocess dies with ``ModuleNotFoundError: No module named
'sleap_nn'``. A script-presence test cannot tell the two installs apart, which
is the whole reason this exists.

**Do the labels carry identity?** The ``multi_class_`` heads classify identity,
so a ``.slp`` whose instances have no track trains one of them against no
classes at all. That run succeeds, writes a loadable model and registers a row,
and the model learned nothing.

**Can the environment write the analysis export?** A tracker run exports each
entry's predictions through ``sleap_io.save_analysis_h5`` after its inference.
An environment whose sleap-io lacks that function runs the inference, which is
the expensive part, and fails at the export.

**The probe reports; every refusal is mosaic's.** Nothing here raises on a
finding, so :func:`require_sleap_nn` and :func:`require_identity_labels` are
testable against a hand-built response with no SLEAP installed at all. That is
the arrangement
:class:`~mosaic.tracking.external.runner.ultralytics_protocol.ProbeResponse`
already states, for the same reason.

**The snippet takes only argv**, so no path is interpolated into its source and
mosaic never imports ``sleap_io`` or ``sleap_nn``. The answer comes back as a
file rather than as parsed stdout, which is how ``write_slp`` publishes its
result too.
"""

from __future__ import annotations

import json
import logging
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel

from mosaic.core.pipeline.subprocess_util import command_summary, run_supervised
from mosaic.core.pipeline.toolenv import (
    PROBE_DEADLINE_FLOOR_SECONDS,
    missing_output_error,
    subprocess_env,
    tool_invocation,
)
from mosaic.tracking.sleap.run import (
    SLEAP_ENV,
    SleapError,
    SleapNotFoundError,
)

if TYPE_CHECKING:
    from mosaic.tracking.sleap.training import EmbeddingPositiveScope

logger = logging.getLogger(__name__)

__all__ = [
    "SleapProbeResponse",
    "probe_sleap",
    "require_analysis_export",
    "require_embedding_labels",
    "require_identity_labels",
    "require_sleap_nn",
]

SLEAP_NN_BOOTSTRAP: str = (
    "sleap-nn ships as an extra, not with the base distribution: install "
    "'sleap[nn]' (or a CUDA variant such as 'sleap[nn-cuda128]') into the "
    "environment MOSAIC_SLEAP_CONDA_ENV / MOSAIC_SLEAP_BIN names."
)
"""How to turn an environment that cannot train into one that can."""

SLEAP_IO_BOOTSTRAP: str = (
    "install a release of sleap-io that provides it ('pip install -U "
    "sleap-io'), or 'sleap[nn]', whose sleap-io does, into the environment "
    "MOSAIC_SLEAP_CONDA_ENV / MOSAIC_SLEAP_BIN names."
)
"""How to turn an environment that cannot export into one that can."""

# Run by the SLEAP interpreter. Its only inputs are argv -- the labels file, or
# an empty string for none, and where to write the answer -- so no path is ever
# interpolated into this source. It reports and never refuses: an import that
# fails and a labels file that will not load are both findings, written out for
# mosaic to decide about.
_PROBE_SNIPPET: str = """
import json
import sys

_labels_path, _out = sys.argv[1], sys.argv[2]
_found = {
    "has_sleap_nn": False,
    "sleap_nn_version": "",
    "sleap_nn_import_error": "",
    "sleap_io_version": "",
    "has_analysis_export": False,
    "analysis_export_error": "",
    "n_labeled_frames": 0,
    "n_tracks": 0,
    "labels_load_error": "",
}

try:
    import sleap_nn

    _found["has_sleap_nn"] = True
    _found["sleap_nn_version"] = str(getattr(sleap_nn, "__version__", ""))
except Exception as exc:
    _found["sleap_nn_import_error"] = f"{type(exc).__name__}: {exc}"

_sleap_io = None
try:
    import sleap_io as _sleap_io

    _found["sleap_io_version"] = str(getattr(_sleap_io, "__version__", ""))
    if callable(getattr(_sleap_io, "save_analysis_h5", None)):
        _found["has_analysis_export"] = True
    else:
        _found["analysis_export_error"] = "sleap_io has no save_analysis_h5"
except Exception as exc:
    _found["analysis_export_error"] = f"{type(exc).__name__}: {exc}"

if _labels_path:
    try:
        if _sleap_io is None:
            raise ImportError(_found["analysis_export_error"])
        _labels = _sleap_io.load_slp(_labels_path)
        _found["n_labeled_frames"] = len(_labels.labeled_frames)
        _found["n_tracks"] = len(_labels.tracks)
    except Exception as exc:
        _found["labels_load_error"] = f"{type(exc).__name__}: {exc}"

with open(_out, "w", encoding="utf-8") as _fh:
    json.dump(_found, _fh)
print("probed")
"""


class SleapProbeResponse(BaseModel):
    """What a SLEAP environment holds. Findings only; every refusal is mosaic's.

    Attributes:
        has_sleap_nn: Whether ``import sleap_nn`` succeeded, which is what
            separates a ``sleap`` install from a ``sleap[nn]`` one.
        sleap_nn_version: The installed sleap-nn's version, empty when it did
            not import.
        sleap_nn_import_error: Why it did not import, empty when it did. A
            missing extra and a broken torch read the same at the console
            script and differently here.
        sleap_io_version: The installed sleap-io's version, empty when it did
            not import.
        has_analysis_export: Whether ``import sleap_io`` succeeded and holds
            ``save_analysis_h5``, the call that writes a tracker run's analysis
            export.
        analysis_export_error: Why the export is not available, empty when it
            is.
        n_labeled_frames: Labeled frames in the probed file.
        n_tracks: Distinct tracks in the probed file. Zero means no instance
            carries identity, which is what the ``multi_class_`` heads need.
        labels_load_error: Why the labels would not load, empty when they did
            or when none was probed.
    """

    has_sleap_nn: bool = False
    sleap_nn_version: str = ""
    sleap_nn_import_error: str = ""
    sleap_io_version: str = ""
    has_analysis_export: bool = False
    analysis_export_error: str = ""
    n_labeled_frames: int = 0
    n_tracks: int = 0
    labels_load_error: str = ""


def probe_sleap(
    labels_path: Path | str | None = None,
    *,
    idle_timeout: float = PROBE_DEADLINE_FLOOR_SECONDS,
    max_runtime: float | None = None,
    sleap_conda_env: str | None = None,
    sleap_bin: str | Path | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> SleapProbeResponse:
    """What the located SLEAP environment holds, and what *labels_path* carries.

    One subprocess answers both, because both cost the same launch and the
    caller that asks one always wants the other. Against a training run of
    hours, a cold import is the right price for not starting one that cannot
    finish.

    The request and the answer live in a temporary directory: this runs before
    a run root exists, and nothing reads either file afterwards.

    Args:
        labels_path: A ``.slp`` to report on, or ``None`` to ask about the
            environment alone.
        idle_timeout: Kill the subprocess after this long with no output.
            Floored at :data:`~mosaic.core.pipeline.toolenv.PROBE_DEADLINE_FLOOR_SECONDS`,
            because a probe prints nothing between spawn and answer.
        max_runtime: Optional absolute ceiling.
        sleap_conda_env: Run in this conda env, overriding the environment.
        sleap_bin: A SLEAP console script naming the install, overriding the
            environment.
        cancel_check: Polled while the subprocess runs.

    Returns:
        What was found, with nothing refused.

    Raises:
        SleapNotFoundError: No SLEAP install could be located.
        SleapError: The probe itself exited non-zero.
        FileNotFoundError: It exited cleanly and wrote no answer.
    """
    invocation = tool_invocation(
        SLEAP_ENV.placed(conda_env=sleap_conda_env, bin_path=sleap_bin),
        executable="python",
    )
    with tempfile.TemporaryDirectory(prefix="mosaic-sleap-probe-") as scratch:
        response_path = Path(scratch) / "probe-response.json"
        cmd = [
            *invocation,
            "-c",
            _PROBE_SNIPPET,
            str(labels_path) if labels_path is not None else "",
            str(response_path),
        ]
        logger.info("Running: %s", command_summary(cmd, SleapError.head))
        stdout, stderr, returncode = run_supervised(
            cmd,
            env=subprocess_env(),
            cancel_check=cancel_check,
            timeout=max_runtime,
            idle_timeout=max(idle_timeout, PROBE_DEADLINE_FLOOR_SECONDS),
        )
        if returncode != 0:
            raise SleapError(cmd, returncode, stdout, stderr)
        if not response_path.is_file():
            raise missing_output_error("SLEAP", response_path, stdout, stderr)
        return SleapProbeResponse.model_validate(
            json.loads(response_path.read_text(encoding="utf-8"))
        )


def require_sleap_nn(probe: SleapProbeResponse) -> None:
    """Refuse an environment that holds SLEAP's verbs but not its trainer.

    The failure this replaces is a ``ModuleNotFoundError`` raised inside a
    subprocess minutes after a queue accepted the job, from a console script
    that was present on ``$PATH`` all along.
    """
    if probe.has_sleap_nn:
        return
    reason = probe.sleap_nn_import_error or "it is not installed"
    raise SleapNotFoundError(
        "the SLEAP environment resolved, and sleap_nn does not import there "
        f"({reason}), so sleap-nn train cannot train anything. "
        f"{SLEAP_NN_BOOTSTRAP}"
    )


def require_analysis_export(probe: SleapProbeResponse) -> None:
    """Refuse an environment that cannot write the analysis export.

    Without the export, a tracker run fails at its first entry's export, after
    the inference that is most of its cost.
    """
    if probe.has_analysis_export:
        return
    reason = probe.analysis_export_error or "it is not available"
    raise SleapNotFoundError(
        "the SLEAP environment resolved, and sleap_io.save_analysis_h5 is not "
        f"available there ({reason}), so the analysis export of a tracker run "
        f"cannot be written. {SLEAP_IO_BOOTSTRAP}"
    )


def require_identity_labels(
    probe: SleapProbeResponse, head: str, labels_path: Path | str
) -> None:
    """Refuse a *head* that classifies identity against labels that carry none.

    Unreadable labels are refused here too, and first: a file that did not load
    reports no tracks, so treating that as "no identity" would answer a question
    the probe never reached.

    Nothing falls back to another head. A run asked for identity, and a model
    trained without it is not a smaller version of what was asked for.
    """
    if probe.labels_load_error:
        raise ValueError(
            f"the labels at {labels_path} would not load in the SLEAP "
            f"environment: {probe.labels_load_error}"
        )
    if not head.startswith("multi_class"):
        return
    if probe.n_tracks == 0:
        raise ValueError(
            f"head {head!r} classifies identity, and no instance in "
            f"{labels_path} carries a track, so it would train against no "
            "classes at all and produce a model that learned nothing. Either "
            "give labels whose instances are tracked, or train a head that "
            "does not classify identity."
        )


def require_embedding_labels(
    probe: SleapProbeResponse, *, scope: EmbeddingPositiveScope, labels_kind: str
) -> None:
    """Refuse labels with too few tracks for a contrastive embedding *scope*.

    ``global_id`` and ``tracklet`` draw their negatives from other tracks, so
    the labels need two individuals. ``aug_view`` pairs two views of one crop
    and needs no track.

    Args:
        probe: The probe of the labels.
        scope: The positive scope of the effective training document.
        labels_kind: The op kind of the run that wrote the labels, empty for a
            labels path. It names the labels in the message.
    """
    if scope == "aug_view" or probe.n_tracks >= 2:
        return
    named = f"The {labels_kind} labels" if labels_kind else "The labels"
    noun = "track" if probe.n_tracks == 1 else "tracks"
    msg = (
        f"{named} hold {probe.n_tracks} {noun}, and an embedding run with "
        f"positive scope {scope!r} draws contrastive negatives from other "
        "tracks, which needs two individuals."
    )
    raise ValueError(msg)
