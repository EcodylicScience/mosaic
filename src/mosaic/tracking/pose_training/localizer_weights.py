"""Keras → PyTorch weight conversion for the localizer model.

Converts weights from Keras H5 format to a PyTorch state_dict that can
be loaded directly into :class:`~.localizer_model.LocalizerEncoder`.

Requires: ``h5py >= 3.0`` (optional dependency).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _require_h5py():
    try:
        import h5py
        return h5py
    except ImportError:
        raise ImportError(
            "h5py is required for Keras weight conversion. "
            "Install with: pip install h5py"
        )


# --------------------------------------------------------------------------- #
# H5 weight extraction
# --------------------------------------------------------------------------- #

def _extract_keras_weights(h5_path: Path) -> list[np.ndarray]:
    """Extract weight arrays from a Keras H5 file in layer order.

    Handles both ``model.save_weights()`` and ``model.save()`` formats.

    Returns
    -------
    list of ndarray
        Flat list of weight arrays in model layer order: conv kernels,
        biases, BN gamma/beta/mean/var, etc.
    """
    h5py = _require_h5py()

    weights: list[np.ndarray] = []

    with h5py.File(h5_path, "r") as f:
        # Find the weights group
        if "model_weights" in f:
            root = f["model_weights"]
        else:
            root = f

        # Get layer names in order
        if "layer_names" in root.attrs:
            layer_names = [
                n.decode("utf-8") if isinstance(n, bytes) else n
                for n in root.attrs["layer_names"]
            ]
        else:
            layer_names = list(root.keys())

        for layer_name in layer_names:
            if layer_name not in root:
                continue
            layer_group = root[layer_name]

            # Get weight names for this layer
            if "weight_names" in layer_group.attrs:
                weight_names = [
                    n.decode("utf-8") if isinstance(n, bytes) else n
                    for n in layer_group.attrs["weight_names"]
                ]
            else:
                weight_names = []

                def _collect(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        weight_names.append(name)

                layer_group.visititems(_collect)

            for wn in weight_names:
                # weight_names may be 'layer_name/kernel:0' or nested
                try:
                    if wn in layer_group:
                        weights.append(np.array(layer_group[wn]))
                    else:
                        parts = wn.split("/")
                        g = layer_group
                        for p in parts:
                            g = g[p]
                        weights.append(np.array(g))
                except (KeyError, TypeError):
                    # Skip unreadable entries
                    continue

    return weights


# --------------------------------------------------------------------------- #
# Weight mapping
# --------------------------------------------------------------------------- #

def _transpose_conv_kernel(kernel: np.ndarray) -> np.ndarray:
    """Keras (H, W, C_in, C_out) → PyTorch (C_out, C_in, H, W)."""
    return np.transpose(kernel, (3, 2, 0, 1)).copy()


def _build_state_dict(
    keras_weights: list[np.ndarray],
    num_classes: int = 4,
    initial_channels: int = 32,
) -> dict[str, Any]:
    """Map a flat list of Keras weight arrays to a PyTorch state_dict.

    Expected order (matching ``get_conv_model``)::

        conv0:       kernel(3,3,1,C), bias(C)
        conv1:       kernel(3,3,C,2C), bias(2C)
        bn1:         gamma(2C), beta(2C), mean(2C), var(2C)
        conv2:       kernel(3,3,2C,4C), bias(4C)
        bn2:         gamma(4C), beta(4C), mean(4C), var(4C)
        conv3–6:     [kernel, bias, gamma, beta, mean, var] × 4
        conv_expand: kernel(3,3,4C,8C), bias(8C)
        bn_expand:   gamma(8C), beta(8C), mean(8C), var(8C)
        conv_reduce: kernel(1,1,8C,C), bias(C)
        conv_out:    kernel(1,1,C,N), bias(N)
    """
    import torch

    idx = 0

    def take():
        nonlocal idx
        arr = keras_weights[idx]
        idx += 1
        return arr

    state_dict: dict[str, Any] = {}

    def add_conv(name: str):
        kernel, bias = take(), take()
        state_dict[f"{name}.weight"] = torch.from_numpy(_transpose_conv_kernel(kernel))
        state_dict[f"{name}.bias"] = torch.from_numpy(bias.copy())

    def add_bn(name: str):
        gamma, beta, mean, var = take(), take(), take(), take()
        state_dict[f"{name}.weight"] = torch.from_numpy(gamma.copy())
        state_dict[f"{name}.bias"] = torch.from_numpy(beta.copy())
        state_dict[f"{name}.running_mean"] = torch.from_numpy(mean.copy())
        state_dict[f"{name}.running_var"] = torch.from_numpy(var.copy())
        state_dict[f"{name}.num_batches_tracked"] = torch.tensor(0, dtype=torch.long)

    # Conv0 (no BN)
    add_conv("conv0")

    # Conv1 + BN1
    add_conv("conv1")
    add_bn("bn1")

    # Conv2 + BN2
    add_conv("conv2")
    add_bn("bn2")

    # Bottleneck layers 3–6
    for i in range(3, 7):
        add_conv(f"conv{i}")
        add_bn(f"bn{i}")

    # Expand
    add_conv("conv_expand")
    add_bn("bn_expand")

    # Head
    add_conv("conv_reduce")
    add_conv("conv_out")

    if idx != len(keras_weights):
        raise ValueError(
            f"Weight count mismatch: consumed {idx} arrays, "
            f"but file has {len(keras_weights)}"
        )

    return state_dict


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def convert_keras_weights(
    keras_h5_path: str | Path,
    output_pt_path: str | Path | None = None,
    *,
    num_classes: int = 4,
    initial_channels: int = 32,
) -> Path:
    """Convert Keras H5 weights to a PyTorch state_dict ``.pt`` file.

    Parameters
    ----------
    keras_h5_path : path
        Path to Keras ``.h5`` weight file.
    output_pt_path : path, optional
        Where to save the ``.pt`` file.  Defaults to the same directory
        with a ``.pt`` extension.
    num_classes : int
        Number of output classes in the model.
    initial_channels : int
        Base channel width.

    Returns
    -------
    Path
        Path to the saved ``.pt`` file.
    """
    import torch

    h5_path = Path(keras_h5_path)
    if output_pt_path is None:
        output_pt_path = h5_path.with_suffix(".pt")
    else:
        output_pt_path = Path(output_pt_path)

    keras_weights = _extract_keras_weights(h5_path)
    state_dict = _build_state_dict(keras_weights, num_classes, initial_channels)

    torch.save(state_dict, str(output_pt_path))
    print(f"[localizer_weights] Converted {h5_path.name} → {output_pt_path.name}")
    print(f"  {len(state_dict)} parameter tensors")

    return output_pt_path


def load_localizer_weights(
    model: Any,
    weights_path: str | Path,
    *,
    strict: bool = True,
) -> Any:
    """Load weights into a :class:`~.localizer_model.LocalizerEncoder`.

    Auto-detects format: ``.h5`` (Keras — auto-converted) or ``.pt``
    (PyTorch state_dict).

    Parameters
    ----------
    model : LocalizerEncoder
        Model instance to load weights into.
    weights_path : path
        Path to ``.h5`` or ``.pt`` weight file.
    strict : bool
        If True, requires exact match between state dict keys and model.

    Returns
    -------
    The model with loaded weights.
    """
    import torch

    path = Path(weights_path)

    if path.suffix in (".h5", ".hdf5"):
        # Auto-convert Keras weights to .pt, then load
        pt_path = path.with_suffix(".pt")
        if not pt_path.exists():
            convert_keras_weights(
                path,
                pt_path,
                num_classes=model.num_classes,
                initial_channels=model.initial_channels,
            )
        state_dict = torch.load(str(pt_path), map_location="cpu", weights_only=True)
    elif path.suffix == ".pt":
        state_dict = torch.load(str(path), map_location="cpu", weights_only=True)
    else:
        raise ValueError(f"Unsupported weight format: {path.suffix}")

    model.load_state_dict(state_dict, strict=strict)
    return model
