"""
Keypoint-MoSeq model application and syllable extraction feature.

Applies a fitted keypoint-MoSeq model (from kpms-fit) to track data and
extracts per-frame syllable labels. Invokes keypoint-moseq in a subprocess
— the kpms package does NOT need to be installed in the mosaic environment.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterable, final

import numpy as np
import pandas as pd
from pydantic import Field

from .spec import register_feature
from mosaic.core.helpers import entry_key

from .helpers import (
    PartialIndexRow,
    _build_index_row,
    _get_feature_run_root,
)
from .kpms_fit import (
    _collect_and_serialize_tracks,
    _run_kpms_subprocess,
)
from .spec import COLUMNS, Inputs, OutputType, Params, TrackInput
from mosaic.core.pipeline._utils import Scope


@final
@register_feature
class KpmsApply:
    """
    Global feature that applies a fitted keypoint-MoSeq model and extracts
    per-frame syllable labels via subprocess.

    keypoint-moseq runs in a separate Python environment to avoid dependency
    conflicts. Pass the interpreter path via ``kpms_python``.

    Parameters
    ----------
    kpms_python : str
        Path to Python interpreter with keypoint-moseq installed.
        Example: "~/miniforge3/envs/kpms/bin/python"
    kpms_fit_feature : str
        Name of the fitting feature (default: "kpms-fit").
    kpms_fit_run_id : str or None
        Run ID of the fitted model. None picks the latest.
    pose_prefix_x, pose_prefix_y : str
        Column prefixes for keypoint x/y coordinates.
    pose_confidence_prefix : str
        Column prefix for confidence scores.
    num_iters_apply : int
        Number of Gibbs sampling iterations for inference (default: 500).
    apply_batch_size : int or None
        Process recordings in batches of this size to avoid OOM (default: 10).
        Set to None or 0 to process all at once.
    """

    name = "kpms-apply"
    version = "0.1"
    parallelizable = False
    output_type: OutputType = "per_frame"
    skip_transform_phase = True

    class Inputs(Inputs[TrackInput]):
        pass

    class Params(Params):
        kpms_python: str | None = None
        kpms_fit_feature: str = "kpms-fit"
        kpms_fit_run_id: str | None = None
        pose_prefix_x: str = "poseX"
        pose_prefix_y: str = "poseY"
        pose_confidence_prefix: str = "poseP"
        num_iters_apply: int = Field(default=500, ge=1)
        apply_batch_size: int = Field(default=10, ge=1)

    def __init__(
        self,
        inputs: KpmsApply.Inputs = Inputs(("tracks",)),
        params: dict[str, object] | None = None,
    ):
        self.inputs = inputs
        self.params = self.Params.from_overrides(params)
        self.storage_feature_name = self.name
        self.storage_use_input_suffix = True
        self._ds = None
        self._run_root: Path | None = None
        self._additional_index_rows: list[PartialIndexRow] = []
        self._scope: Scope = Scope()

    # ----------------------- Dataset hooks -----------------------

    def bind_dataset(self, ds):
        self._ds = ds

    def set_run_root(self, run_root: Path) -> None:
        self._run_root = Path(run_root)

    def set_scope(self, scope: Scope) -> None:
        self._scope = scope

    def get_additional_index_rows(self) -> list[PartialIndexRow]:
        return list(self._additional_index_rows)

    # ----------------------- Feature protocol --------------------

    def needs_fit(self) -> bool:
        return True

    def supports_partial_fit(self) -> bool:
        return False

    def partial_fit(self, df: pd.DataFrame) -> None:
        raise NotImplementedError

    def finalize_fit(self) -> None:
        pass

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(index=[])

    # ----------------------- Helpers -----------------------------

    @staticmethod
    def _parse_recording_name(name: str) -> tuple[str, str, int | None]:
        """Parse a recording name back into (group, sequence, id).

        Keys use safe-encoded names (``to_safe_name``), so slashes appear
        as ``%2F``.  This method decodes them back to canonical form.

        Handles formats:
          - "{group}__{sequence}__id{id}"
          - "{group}__{sequence}"
          - "{sequence}__id{id}"
          - "{sequence}"
        """
        from urllib.parse import unquote

        group = ""
        sequence = name
        ind_id = None

        # Check for __id{N} suffix
        if "__id" in name:
            parts = name.rsplit("__id", 1)
            base = parts[0]
            try:
                ind_id = int(parts[1])
            except (ValueError, IndexError):
                base = name
                ind_id = None
        else:
            base = name

        # Split group__sequence
        if "__" in base:
            parts = base.split("__", 1)
            group = unquote(parts[0])
            sequence = unquote(parts[1])
        else:
            sequence = unquote(base)

        return group, sequence, ind_id

    # ----------------------- Fit (= Apply + Extract) -------------

    def fit(self, X_iter: Iterable[pd.DataFrame]) -> None:
        if self._ds is None:
            raise RuntimeError("[kpms-apply] Dataset not bound.")

        p = self.params
        kpms_python = p.kpms_python
        if not kpms_python:
            raise ValueError(
                "[kpms-apply] 'kpms_python' param is required. "
                "Set it to the path of a Python interpreter with keypoint-moseq installed. "
                "Example: '~/miniforge3/envs/kpms/bin/python'"
            )
        kpms_python = str(Path(kpms_python).expanduser())

        self._additional_index_rows = []

        if self._run_root is None:
            raise RuntimeError("[kpms-apply] run_root not set.")

        # 1. Locate the fitted model from kpms-fit
        fit_feature = p.kpms_fit_feature
        fit_run_id = p.kpms_fit_run_id

        resolved_run_id, fit_run_root = _get_feature_run_root(
            self._ds, fit_feature, fit_run_id
        )

        model_dir = fit_run_root / "_kpms_output"
        model_file = model_dir / "kpms_model.joblib"
        if not model_file.exists():
            # Fall back to the standard model.joblib (copied there by kpms-fit save_model)
            model_dir = fit_run_root
            model_file = fit_run_root / "model.joblib"
        if not model_file.exists():
            raise FileNotFoundError(
                f"[kpms-apply] No kpms_model.joblib found in {fit_run_root}. "
                f"Run kpms-fit first."
            )

        print(
            f"[kpms-apply] Using model from {fit_feature} run={resolved_run_id}",
            file=sys.stderr,
        )

        # 2. Serialize track data
        data_dir = self._run_root / "_kpms_data"

        # Extract group/sequence scope
        scope_groups = sorted(self._scope.groups) if self._scope.entries else None
        scope_sequences = sorted(self._scope.sequences) if self._scope.entries else None

        print("[kpms-apply] Collecting and serializing track data...", file=sys.stderr)
        _collect_and_serialize_tracks(
            self._ds,
            data_dir,
            p.pose_prefix_x,
            p.pose_prefix_y,
            p.pose_confidence_prefix,
            COLUMNS.id_col,
            None,
            groups=scope_groups,
            sequences=scope_sequences,
        )

        # 3. Write apply config JSON
        apply_config = {
            "num_iters_apply": p.num_iters_apply,
        }
        config_path = self._run_root / "kpms_apply_config.json"
        config_path.write_text(json.dumps(apply_config, indent=2, default=str))

        # 4. Run kpms_runner.py apply in subprocess
        output_dir = self._run_root / "_kpms_output"
        apply_args = [
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(output_dir),
            "--model-dir",
            str(model_dir),
            "--config",
            str(config_path),
        ]
        batch_size = p.apply_batch_size
        if batch_size and batch_size > 0:
            apply_args += ["--batch-size", str(batch_size)]

        _run_kpms_subprocess(
            kpms_python,
            "apply",
            apply_args,
            label="kpms-apply",
        )

        # 5. Read results and write per-sequence parquets
        print("[kpms-apply] Reading results and writing parquets...", file=sys.stderr)
        self._collect_results(output_dir)

        print(
            f"[kpms-apply] Done. Wrote {len(self._additional_index_rows)} sequence outputs.",
            file=sys.stderr,
        )

    def _collect_results(self, output_dir: Path) -> None:
        """Read syllable npz files from kpms_runner output and write parquets."""
        processed_path = output_dir / "processed_recordings.json"
        if not processed_path.exists():
            raise RuntimeError(
                f"[kpms-apply] No processed_recordings.json in {output_dir}. "
                "kpms_runner apply may have failed."
            )

        with open(processed_path) as f:
            processed = json.load(f)

        for recording_name in processed:
            npz_path = output_dir / f"syllables__{recording_name}.npz"
            if not npz_path.exists():
                print(
                    f"[kpms-apply] WARN: missing {npz_path}",
                    file=sys.stderr,
                )
                continue

            data = np.load(npz_path)
            syllables = data["syllables"]
            T = len(syllables)

            # Parse recording name back to mosaic identifiers
            group, sequence, ind_id = self._parse_recording_name(recording_name)

            effective_seq = sequence or recording_name

            # Build output DataFrame
            df_out = pd.DataFrame(
                {
                    "frame": np.arange(T, dtype=np.int64),
                    "syllable": syllables,
                    "sequence": effective_seq,
                    "group": group,
                }
            )
            if ind_id is not None:
                df_out["id"] = ind_id

            # Write parquet
            out_name = entry_key(group, effective_seq)
            if ind_id is not None:
                out_name += f"__id{ind_id}"
            out_name += ".parquet"
            out_path = self._run_root / out_name
            df_out.to_parquet(out_path, index=False)

            self._additional_index_rows.append(
                _build_index_row(group, sequence or recording_name, out_path, T)
            )

    # ----------------------- Save / Load -------------------------

    def save_model(self, path: Path) -> None:
        run_root = path.parent
        run_root.mkdir(parents=True, exist_ok=True)

        import joblib

        joblib.dump(
            {
                "params": self.params.model_dump(),
                "kpms_fit_feature": self.params.kpms_fit_feature,
                "kpms_fit_run_id": self.params.kpms_fit_run_id,
                "version": self.version,
            },
            path,
        )

    def load_model(self, path: Path) -> None:
        import joblib

        bundle = joblib.load(path)
        saved = bundle.get("params", {})
        if isinstance(saved, dict):
            merged = {**self.params.model_dump(), **saved}
            self.params = self.Params.from_overrides(merged)
        elif hasattr(saved, "model_dump"):
            self.params = saved
