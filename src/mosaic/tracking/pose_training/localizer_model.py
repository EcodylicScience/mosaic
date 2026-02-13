"""BeesBook localizer model — PyTorch port.

Fully convolutional heatmap encoder for animal location detection.
Ported from the TensorFlow/Keras implementation in bb_pipeline_models
(``get_conv_model``).

Architecture
------------
3 strided (stride=2) convolutions for downsampling, 4 bottleneck
convolutions (stride=1), 1 expansion convolution, and a 2-layer 1×1 head.
All 3×3 convolutions use **valid padding** (``padding=0``).

For 128×128 input the output is a 5×5 heatmap with *num_classes* channels.
The center pixel at [2, 2] corresponds to the center of the input patch.
Effective stride: 8.  Offset: 47.  For output position *q* in the heatmap,
the corresponding input coordinate is  ``q * 8 + 47``.

Requires: ``torch >= 2.0``
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LocalizerEncoder(nn.Module):
    """Fully convolutional heatmap encoder (~248 K params, default config).

    Parameters
    ----------
    num_classes : int
        Number of output heatmap channels (sigmoid-activated).
    initial_channels : int
        Base channel width (channels double at each strided layer).
    """

    STRIDE = 8
    OFFSET = 47

    def __init__(self, num_classes: int = 4, initial_channels: int = 32):
        super().__init__()
        C = initial_channels
        self.num_classes = num_classes
        self.initial_channels = C

        # ---- downsampling ------------------------------------------------
        # Layer 0: Conv + ReLU + Dropout  (NO BatchNorm)
        self.conv0 = nn.Conv2d(1, C, 3, stride=2, padding=0)
        self.drop0 = nn.Dropout2d(0.1)

        # Layer 1: Conv + ReLU + BN + Dropout
        self.conv1 = nn.Conv2d(C, C * 2, 3, stride=2, padding=0)
        self.bn1 = nn.BatchNorm2d(C * 2)
        self.drop1 = nn.Dropout2d(0.1)

        # Layer 2: Conv + ReLU + BN + Dropout
        self.conv2 = nn.Conv2d(C * 2, C * 4, 3, stride=2, padding=0)
        self.bn2 = nn.BatchNorm2d(C * 4)
        self.drop2 = nn.Dropout2d(0.1)

        # ---- bottleneck (stride=1) ---------------------------------------
        # Layers 3–5: Conv + ReLU + BN + Dropout
        self.conv3 = nn.Conv2d(C * 4, C * 4, 3, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(C * 4)
        self.drop3 = nn.Dropout2d(0.1)

        self.conv4 = nn.Conv2d(C * 4, C * 4, 3, stride=1, padding=0)
        self.bn4 = nn.BatchNorm2d(C * 4)
        self.drop4 = nn.Dropout2d(0.1)

        self.conv5 = nn.Conv2d(C * 4, C * 4, 3, stride=1, padding=0)
        self.bn5 = nn.BatchNorm2d(C * 4)
        self.drop5 = nn.Dropout2d(0.1)

        # Layer 6: Conv + ReLU + BN  (NO Dropout)
        self.conv6 = nn.Conv2d(C * 4, C * 4, 3, stride=1, padding=0)
        self.bn6 = nn.BatchNorm2d(C * 4)

        # ---- expansion ---------------------------------------------------
        # Layer 7: Conv + ReLU + BN  (NO Dropout)
        self.conv_expand = nn.Conv2d(C * 4, C * 8, 3, stride=1, padding=0)
        self.bn_expand = nn.BatchNorm2d(C * 8)

        # ---- 1×1 head ----------------------------------------------------
        self.conv_reduce = nn.Conv2d(C * 8, C, 1)
        self.conv_out = nn.Conv2d(C, num_classes, 1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor, shape (B, 1, H, W)
            Grayscale input in [0, 1].

        Returns
        -------
        Tensor, shape (B, num_classes, H', W')
            Sigmoid-activated heatmap.
        """
        # Downsampling
        x = self.drop0(self.relu(self.conv0(x)))
        x = self.drop1(self.bn1(self.relu(self.conv1(x))))
        x = self.drop2(self.bn2(self.relu(self.conv2(x))))

        # Bottleneck
        x = self.drop3(self.bn3(self.relu(self.conv3(x))))
        x = self.drop4(self.bn4(self.relu(self.conv4(x))))
        x = self.drop5(self.bn5(self.relu(self.conv5(x))))
        x = self.bn6(self.relu(self.conv6(x)))

        # Expansion
        x = self.bn_expand(self.relu(self.conv_expand(x)))

        # Head
        x = self.relu(self.conv_reduce(x))
        x = torch.sigmoid(self.conv_out(x))
        return x


class LocalizerTrainWrapper(nn.Module):
    """Wraps :class:`LocalizerEncoder` for patch-based training.

    Takes 128×128 patches, runs the full encoder, and extracts the center
    pixel ``[2, 2]`` of the 5×5 output — producing a single
    ``num_classes``-element prediction vector per patch.
    """

    def __init__(self, encoder: LocalizerEncoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape (B, 1, 128, 128)

        Returns
        -------
        Tensor, shape (B, num_classes)
        """
        heatmap = self.encoder(x)  # (B, C, 5, 5)
        return heatmap[:, :, 2, 2]  # (B, C)


def build_localizer(
    num_classes: int = 4,
    initial_channels: int = 32,
    *,
    mode: str = "inference",
) -> LocalizerEncoder | LocalizerTrainWrapper:
    """Factory for building a localizer model.

    Parameters
    ----------
    num_classes : int
        Number of output heatmap channels.
    initial_channels : int
        Base channel width.
    mode : str
        ``"inference"`` — return :class:`LocalizerEncoder`.
        ``"train"`` — return :class:`LocalizerTrainWrapper`.
    """
    encoder = LocalizerEncoder(num_classes, initial_channels)
    if mode == "train":
        return LocalizerTrainWrapper(encoder)
    return encoder
