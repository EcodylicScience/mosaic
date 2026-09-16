"""Asking a SLEAP environment what it holds, before a training run pays for it.

Two questions a training run cannot answer for itself, and both of them look
like success until hours later.

**Is the trainer actually installed?** ``pip install sleap`` without the ``nn``
extra puts ``sleap-nn-train`` on ``$PATH`` anyway: the console script belongs to
the ``sleap`` distribution itself, while ``sleap_nn`` arrives only through
``sleap[nn]`` and its CUDA variants. So mosaic's location ladder finds the tool,
launches it, and the subprocess dies with ``ModuleNotFoundError: No module named
'sleap_nn'``. A script-presence test cannot tell the two installs apart, which
is the whole reason this exists.

**Do the labels carry identity?** The ``multi_class_`` heads classify identity,
so a ``.slp`` whose instances have no track trains one of them against no
classes at all. That run succeeds, writes a loadable model and registers a row,
and the model learned nothing.

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

from pydantic import BaseModel

from mosaic.core.pipeline.subprocess_util import run_supervised
from mosaic.tracking.common.toolenv import (
    PROBE_DEADLINE_FLOOR_SECONDS,
    missing_output_error,
    subprocess_env,
    tool_invocation,
)
from mosaic.tracking.sleap.run import (
    SLEAP_PYTHON_ENV,
    SleapError,
    SleapNotFoundError,
)

logger = logging.getLogger(__name__)

__all__ = [
    "SleapProbeResponse",
    "probe_sleap",
    "require_identity_labels",
    "require_sleap_nn",
]

SLEAP_NN_BOOTSTRAP: str = (
    "sleap-nn ships as an extra, not with the base distribution: install "
    "'sleap[nn]' (or a CUDA variant such as 'sleap[nn-cuda128]') into the "
    "environment MOSAIC_SLEAP_CONDA_ENV / MOSAIC_SLEAP_BIN names."
)
"""How to turn an environment that cannot train into one that can."""

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

if _labels_path:
    try:
        import sleap_io

        _found["sleap_io_version"] = str(getattr(sleap_io, "__version__", ""))
        _labels = sleap_io.load_slp(_labels_path)
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
        sleap_io_version: The installed sleap-io's version, empty when no
            labels file was probed or it did not import.
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
            Floored at :data:`~mosaic.tracking.common.toolenv.PROBE_DEADLINE_FLOOR_SECONDS`,
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
        SLEAP_PYTHON_ENV.placed(conda_env=sleap_conda_env, bin_path=sleap_bin),
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
        logger.info("Running: %s", " ".join(cmd[:4]) + " ...")
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
        f"({reason}), so sleap-nn-train cannot train anything. "
        f"{SLEAP_NN_BOOTSTRAP}"
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
