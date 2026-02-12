"""
GlobalTSNE feature.

Extracted from features.py as part of feature_library modularization.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Any, Iterable, List, Tuple
import gc
import sys

import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler

from openTSNE import TSNEEmbedding, affinity, initialization

from mosaic.core.dataset import register_feature
from .helpers import (
    StreamingFeatureHelper, _load_array_from_spec,
    _parse_scope_filter, _build_sequence_lookup, _resolve_sequence_identity,
    _get_feature_run_root, _build_index_row,
)
from mosaic.core.dataset import _resolve_inputs
from mosaic.core.helpers import to_safe_name


@register_feature
class GlobalTSNE: 
    """
    Global t-SNE over multiple prior features discovered from the dataset's feature indexes.
    - inputs: a list of dicts describing which feature outputs to include.
      Each input supports:
        {
          "feature": "pair-wavelet",  # prior feature name
          "run_id":  None,                         # optional, pick latest if None
          "pattern": "wavelet_social_seq=*persp=*.npz",  # files to glob inside the run folder
          "load": { "kind": "npz", "key": "spectrogram", "transpose": True }  # loader spec
        }
    The loaded arrays are treated as (T x D); arrays from all inputs with the same sequence
    are length-aligned (min T) and concatenated horizontally.

    Heavy work happens in fit(); transform() returns a tiny stub DF.
    """
    name: str = "global-tsne"
    version: str = "0.2"
    output_type: str = "global"
    skip_transform_phase: bool = True
    params: dict

    def __init__(self, params: Optional[dict] = None):
        defaults = dict(
            # Multi-input spec (good defaults for your current pipeline):
            inputs=[
                {
                    "feature": "pair-wavelet",
                    "run_id": None,  # pick latest
                    "pattern": "wavelet_social_seq=*persp=*.npz",
                    "load": {"kind": "npz", "key": "spectrogram", "transpose": True}
                },
                {
                    "feature": "pair-egocentric-wavelet",
                    "run_id": None,
                    "pattern": "wavelet_ego_seq=*persp=*.npz",
                    "load": {"kind": "npz", "key": "spectrogram", "transpose": True}
                },
            ],
            inputset=None,
            map_existing_inputs=False,
            reuse_embedding=None,
            artifact=None,
            scaler=None,
            random_state=42,
            r_scaler=200_000,          # cap for standardizer fit sample
            total_templates=2000,      # farthest-first target
            pre_quota_per_key=50,      # pre-sample per key
            perplexity=50,
            n_jobs=8,
            partial_k=25,
            partial_iters=100,
            partial_lr=1.0,
            map_chunk=50_000,  # Larger chunks = fewer prepare_partial calls (major memory saver)
        )
        self.params = {**defaults, **(params or {})}
        self._inputs_overridden = bool(params and "inputs" in params)
        self._rng = np.random.default_rng(self.params["random_state"])
        self._scaler: Optional[StandardScaler] = None
        self._embedding: Optional[TSNEEmbedding] = None
        self._artifacts: Dict[str, Any] = {}
        self._ds = None  # set by bind_dataset
        self._inputs_meta: Dict[str, Any] = {}
        self._resolved_inputs: List[dict] = []
        self._scope_filter: Optional[dict] = None
        self._allowed_safe_sequences: Optional[set[str]] = None
        self._run_root: Optional[Path] = None
        self._pair_map: dict[str, tuple[str, str]] = {}
        self._sequence_lookup_cache: Optional[dict[str, tuple[str, str]]] = None
        self._additional_index_rows: list[dict] = []

    # dataset hook
    def bind_dataset(self, ds) -> None:
        self._ds = ds

    def set_scope_filter(self, scope: Optional[dict]) -> None:
        self._scope_filter = scope or {}
        self._allowed_safe_sequences, self._pair_map = _parse_scope_filter(scope)
        self._sequence_lookup_cache = None

    def set_run_root(self, run_root: Path) -> None:
        self._run_root = Path(run_root)

    def get_additional_index_rows(self) -> list[dict]:
        if self._additional_index_rows:
            return list(self._additional_index_rows)
        return self._discover_existing_coord_rows()

    # ---- Feature protocol ----
    def needs_fit(self) -> bool: return True
    def supports_partial_fit(self) -> bool: return False
    def partial_fit(self, X: pd.DataFrame) -> None: raise NotImplementedError
    def loads_own_data(self) -> bool: return True  # Skip run_feature pre-loading; we stream from disk

    def fit(self, X_iter: Iterable[pd.DataFrame]) -> None:
        if self._ds is None:
            raise RuntimeError("GlobalTSNE requires dataset binding. Ensure Dataset.run_feature calls bind_dataset().")
        self._artifacts = {}
        self._additional_index_rows = []

        # 0) Gather inputs from feature indexes
        inputset_name = self.params.get("inputset")
        explicit_inputs = self.params["inputs"] if (self._inputs_overridden or not inputset_name) else None
        inputs, inputs_meta = _resolve_inputs(
            self._ds, explicit_inputs, inputset_name,
            explicit_override=self._inputs_overridden
        )
        self._resolved_inputs = inputs
        self._inputs_meta = inputs_meta

        # Use StreamingFeatureHelper for manifest building
        helper = StreamingFeatureHelper(self._ds, "global-tsne")
        if inputs_meta.get("pair_filter"):
            helper.set_pair_filter(inputs_meta["pair_filter"])
        scope_filter = {"safe_sequences": self._allowed_safe_sequences} if self._allowed_safe_sequences else None
        key_file_manifest = helper.build_manifest(inputs, scope_filter=scope_filter)

        if not key_file_manifest:
            raise RuntimeError("GlobalTSNE found no usable inputs from the specified prior features.")

        keys = list(key_file_manifest.keys())

        # Handle map_existing_inputs mode (reuse existing embedding)
        if bool(self.params.get("map_existing_inputs")):
            self._prepare_reuse_artifacts()
            mapped = self._map_sequences_streaming(key_file_manifest, helper)
            self._artifacts["mapped_coords"] = mapped if self._run_root is None else {}
            self._artifacts["keys"] = keys
            self._artifacts["inputs_meta"] = inputs_meta
            return

        # === PASS 1: Sample for scaler fitting and template selection ===
        # Stream through data, sampling without holding everything in memory
        r_scaler = int(self.params["r_scaler"])
        T_target = int(self.params["total_templates"])
        quota = max(int(self.params["pre_quota_per_key"]), T_target // max(1, len(keys)))
        samples_per_key = max(1000, r_scaler // max(1, len(keys)))

        scaler_samples = []
        template_samples = []

        for ki, key in enumerate(keys):
            X, _ = helper.load_key_data(key_file_manifest[key], extract_frames=False, key=key)
            if X is None or X.shape[0] == 0:
                continue

            # Sample for scaler
            take_scaler = min(X.shape[0], samples_per_key)
            idx_scaler = self._rng.choice(X.shape[0], size=take_scaler, replace=False)
            scaler_samples.append(X[idx_scaler].copy())  # copy to decouple from X

            # Sample for templates (larger sample for farthest-first)
            take_templ = min(X.shape[0], quota * 3)
            idx_templ = self._rng.choice(X.shape[0], size=take_templ, replace=False)
            template_samples.append(X[idx_templ].copy())  # copy to decouple from X

            # Explicitly free memory
            del X
            if ki % 10 == 0:
                gc.collect()

        if not scaler_samples:
            raise RuntimeError("No combined feature frames after alignment.")

        # 2) Fit global standardizer
        Xsamp = np.vstack(scaler_samples)
        del scaler_samples  # free memory
        if Xsamp.shape[0] > r_scaler:
            idx = self._rng.choice(Xsamp.shape[0], size=r_scaler, replace=False)
            Xsamp = Xsamp[idx]
        scaler = StandardScaler().fit(Xsamp)
        self._scaler = scaler
        del Xsamp  # free memory

        # 3) Template selection (farthest-first over pre-sample)
        X_pre = np.vstack([scaler.transform(s) for s in template_samples])
        del template_samples  # free memory

        sel = [int(self._rng.integers(0, X_pre.shape[0]))]
        d2 = np.sum((X_pre - X_pre[sel[0]])**2, axis=1)
        while len(sel) < min(T_target, X_pre.shape[0]):
            i = int(np.argmax(d2))
            sel.append(i)
            d2 = np.minimum(d2, np.sum((X_pre - X_pre[i])**2, axis=1))
        templates = X_pre[np.array(sel)]
        del X_pre, d2  # free memory

        # 4) Fit openTSNE on templates
        aff = affinity.PerplexityBasedNN(
            templates,
            perplexity=int(self.params["perplexity"]),
            metric="euclidean",
            method="annoy",
            n_jobs=int(self.params["n_jobs"]),
            random_state=int(self.params["random_state"]),
        )
        init = initialization.pca(templates, random_state=int(self.params["random_state"]))
        emb = TSNEEmbedding(
            init, aff,
            negative_gradient_method="fft",
            n_jobs=int(self.params["n_jobs"]),
            random_state=int(self.params["random_state"]),
        )
        emb.optimize(n_iter=250, exaggeration=12, momentum=0.5, inplace=True, verbose=False)
        emb.optimize(n_iter=750, momentum=0.8, inplace=True, verbose=False)
        self._embedding = emb

        # === PASS 2: Map sequences one at a time (streaming) ===
        mapped = self._map_sequences_streaming(key_file_manifest, helper)

        # hold artifacts for save_model
        self._artifacts["keys"] = keys
        self._artifacts["templates"] = templates
        self._artifacts["template_indices"] = np.array(sel)
        self._artifacts["mapped_coords"] = mapped if self._run_root is None else {}
        self._artifacts["inputs_meta"] = inputs_meta

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # Just a stub so Dataset.run_feature can index a parquet per (group,sequence)
        return pd.DataFrame({"global_tsne_done": [True]})

    def save_model(self, path: Path) -> None:
        run_root = path.parent
        run_root.mkdir(parents=True, exist_ok=True)

        # Save templates
        templates = self._artifacts.get("templates")
        if templates is not None:
            np.savez_compressed(run_root / "global_templates_features.npz", templates=templates)

        # Save tsne coords of templates + selection indices
        if self._embedding is not None:
            Y_templates = np.asarray(self._embedding)
            np.savez_compressed(run_root / "global_tsne_templates.npz",
                                Y=Y_templates, sel=self._artifacts.get("template_indices", np.array([], int)))

        # Save embedding + scaler
        joblib.dump(
            {
                "embedding": self._embedding,
                "scaler": self._scaler,
                "params": self.params,
            },
            run_root / "global_opentsne_embedding.joblib"
        )

        # Save per-key coords
        mapped = self._artifacts.get("mapped_coords", {})
        for safe_seq, Y in mapped.items():
            out_name = f"global_tsne_coords_seq={safe_seq}.npz"
            np.savez_compressed(run_root / out_name, Y=Y)

    def load_model(self, path: Path) -> None:
        bundle = joblib.load(path)
        self._embedding = bundle.get("embedding", None)
        self._scaler = bundle.get("scaler", None)

    def _load_embedding_from_spec(self, spec: dict) -> tuple[Any, Any]:
        feature = spec.get("feature")
        if not feature:
            raise ValueError("reuse_embedding spec requires 'feature'.")
        resolved_run_id, run_root = _get_feature_run_root(self._ds, feature, spec.get("run_id"))
        pattern = spec.get("pattern", "global_opentsne_embedding.joblib")
        files = sorted(run_root.glob(pattern))
        if not files:
            raise FileNotFoundError(f"reuse_embedding: no files matching '{pattern}' in {run_root}")
        bundle = joblib.load(files[0])
        embedding = bundle.get("embedding")
        scaler = bundle.get("scaler")
        if embedding is None:
            raise ValueError(f"reuse_embedding bundle missing 'embedding' object: {files[0]}")
        self._artifacts.setdefault("reuse_sources", {})["embedding"] = {
            "feature": feature,
            "run_id": resolved_run_id,
            "path": str(files[0]),
        }
        return embedding, scaler

    def _load_scaler_from_spec(self, spec: dict):
        feature = spec.get("feature")
        if not feature:
            raise ValueError("scaler spec requires 'feature'.")
        resolved_run_id, run_root = _get_feature_run_root(self._ds, feature, spec.get("run_id"))
        pattern = spec.get("pattern", "global_opentsne_embedding.joblib")
        files = sorted(run_root.glob(pattern))
        if not files:
            raise FileNotFoundError(f"scaler spec: no files matching '{pattern}' in {run_root}")
        obj = joblib.load(files[0])
        key = spec.get("key")
        scaler = obj if key is None else obj[key]
        self._artifacts.setdefault("reuse_sources", {})["scaler"] = {
            "feature": feature,
            "run_id": resolved_run_id,
            "path": str(files[0]),
        }
        return scaler

    def _load_templates_from_spec(self, spec: dict) -> Optional[np.ndarray]:
        feature = spec.get("feature")
        if not feature:
            raise ValueError("artifact spec requires 'feature'.")
        resolved_run_id, run_root = _get_feature_run_root(self._ds, feature, spec.get("run_id"))
        pattern = spec.get("pattern", "global_templates_features.npz")
        files = sorted(run_root.glob(pattern))
        if not files:
            raise FileNotFoundError(f"artifact spec: no files matching '{pattern}' in {run_root}")
        load_spec = spec.get("load", {"kind": "npz", "key": "templates", "transpose": False})
        arr, _ = _load_array_from_spec(files[0], load_spec)
        if arr is None:
            return None
        if arr.ndim == 1:
            arr = arr[None, :]
        self._artifacts.setdefault("reuse_sources", {})["templates"] = {
            "feature": feature,
            "run_id": resolved_run_id,
            "path": str(files[0]),
        }
        return arr

    def _prepare_reuse_artifacts(self) -> None:
        emb_spec = self.params.get("reuse_embedding")
        if not emb_spec:
            raise ValueError("map_existing_inputs=True requires params['reuse_embedding'].")
        embedding, scaler = self._load_embedding_from_spec(emb_spec)
        self._embedding = embedding
        self._scaler = scaler
        scaler_override = self.params.get("scaler")
        if scaler_override:
            self._scaler = self._load_scaler_from_spec(scaler_override)
        if self._scaler is None:
            raise ValueError("Reusable embedding bundle missing scaler; provide params['scaler'].")
        art_spec = self.params.get("artifact")
        if art_spec:
            templates = self._load_templates_from_spec(art_spec)
            if templates is not None:
                self._artifacts["templates"] = templates

    def _get_sequence_lookup(self) -> dict[str, tuple[str, str]]:
        if self._sequence_lookup_cache is not None:
            return self._sequence_lookup_cache
        self._sequence_lookup_cache = _build_sequence_lookup(
            self._ds, self._allowed_safe_sequences
        )
        return self._sequence_lookup_cache

    def _append_index_row(self, safe_seq: str, out_path: Path, n_rows: Optional[int]) -> None:
        group, sequence = _resolve_sequence_identity(
            safe_seq, self._pair_map, self._get_sequence_lookup()
        )
        self._additional_index_rows = [
            r for r in self._additional_index_rows
            if not (str(r.get("group", "")) == group and str(r.get("sequence", "")) == sequence)
        ]
        self._additional_index_rows.append(
            _build_index_row(safe_seq, group, sequence, out_path, n_rows)
        )

    def _persist_mapped_coords(self, safe_seq: str, Y: np.ndarray) -> None:
        if self._run_root is None:
            return
        out_name = f"global_tsne_coords_seq={safe_seq}.npz"
        out_path = self._run_root / out_name
        Y_arr = np.asarray(Y)
        if Y_arr.dtype != np.float32:
            Y_arr = Y_arr.astype(np.float32, copy=False)
        np.savez_compressed(out_path, Y=Y_arr)
        self._append_index_row(safe_seq, out_path, int(Y.shape[0]))

    def _discover_existing_coord_rows(self) -> list[dict]:
        if self._run_root is None or not self._run_root.exists():
            return []
        rows = []
        for fp in sorted(self._run_root.glob("global_tsne_coords_seq=*.npz")):
            stem = fp.stem
            if "seq=" not in stem:
                continue
            safe_seq = stem.split("seq=", 1)[1].strip()
            if not safe_seq:
                continue
            group, sequence = _resolve_sequence_identity(
                safe_seq, self._pair_map, self._get_sequence_lookup()
            )
            rows.append(_build_index_row(safe_seq, group, sequence, fp))
        return rows

    def _map_sequences_streaming(
        self,
        key_file_manifest: Dict[str, List[Tuple[Path, dict]]],
        helper: StreamingFeatureHelper
    ) -> Dict[str, np.ndarray]:
        """Map sequences one at a time to minimize memory usage."""
        if self._embedding is None or self._scaler is None:
            raise RuntimeError("GlobalTSNE mapping requires both embedding and scaler.")

        embedding = self._embedding
        scaler = self._scaler
        CHUNK = int(self.params["map_chunk"])
        Kp = int(self.params["partial_k"])
        Pp = int(self.params["perplexity"])
        It = int(self.params["partial_iters"])
        Lr = float(self.params["partial_lr"])

        def map_chunk_block(X_chunk_std: np.ndarray) -> np.ndarray:
            part = embedding.prepare_partial(X_chunk_std, initialization="median", k=Kp, perplexity=Pp)
            # Use inplace=True to avoid creating extra objects
            part.optimize(
                n_iter=It,
                learning_rate=Lr,
                exaggeration=2.0,
                momentum=0.0,
                inplace=True,
                verbose=False,
            )
            # Extract coordinates before deleting
            coords = np.asarray(part)
            if coords.dtype != np.float32:
                coords = coords.astype(np.float32, copy=False)
            coords = coords.copy()
            # Explicitly delete partial embedding internals to free openTSNE memory
            if hasattr(part, 'affinities'):
                del part.affinities
            if hasattr(part, '_P'):
                del part._P
            del part
            gc.collect()
            return coords

        import pyarrow as pa
        mapped: Dict[str, np.ndarray] = {}
        keys = list(key_file_manifest.keys())
        n_keys = len(keys)

        # Use load_key_data directly (NOT iter_sequences generator) to avoid holding extra references
        for i, key in enumerate(keys):
            X, _ = helper.load_key_data(key_file_manifest[key], extract_frames=False, key=key)
            if X is None or X.shape[0] == 0:
                continue

            # Scale and map in chunks
            Xs = scaler.transform(X)
            if Xs.dtype != np.float32:
                Xs = Xs.astype(np.float32, copy=False)
            del X  # free raw data immediately - no generator holding a reference
            gc.collect()
            pa.default_memory_pool().release_unused()

            Y_seq = np.empty((Xs.shape[0], 2), dtype=np.float32)
            for j in range(0, Xs.shape[0], CHUNK):
                block = map_chunk_block(Xs[j:j + CHUNK])
                Y_seq[j:j + block.shape[0], :] = block
                del block
                # Aggressive gc every few chunks to prevent openTSNE memory buildup
                if (j // CHUNK) % 5 == 4:
                    gc.collect()

            if self._run_root is not None:
                self._persist_mapped_coords(key, Y_seq)
                del Y_seq
            else:
                mapped[key] = Y_seq
            del Xs  # free memory

            # Force garbage collection after each sequence to prevent openTSNE memory buildup
            gc.collect()
            pa.default_memory_pool().release_unused()

            if (i + 1) % 10 == 0 or i == n_keys - 1:
                print(f"[global-tsne] Mapped {i + 1}/{n_keys} sequences", file=sys.stderr)

        return mapped
