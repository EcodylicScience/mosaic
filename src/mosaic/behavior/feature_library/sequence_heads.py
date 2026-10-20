"""The heads of ``sequence-classifier``: networks that map frames to class logits.

A head is a :class:`HeadSpec`: a name, the model of its options and a function that
builds a torch module. The module reads a batch of windows ``(B, T, F)`` of float32
frames and returns logits ``(B, T, C)``. It does not mix the items of a batch and
keeps no state between calls, because the classifier predicts a whole series in one
pass and trains on a batch of crops.

``tcn``, ``gru`` and ``mlp`` are built in. A head of your own is registered in
Python with :func:`register_head`, in the process that runs the step, and is
named by ``head`` in the step's params. A head registered in a notebook is not
visible to the command line or to a queue worker, which start their own process.

This module imports no torch. Each ``build`` function imports it when it is called,
and defines its ``nn.Module`` there.
"""

from __future__ import annotations

import hashlib
import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, ClassVar, Final, Protocol

from pydantic import ConfigDict, Field, field_validator

from mosaic.core.params import Declared, DeclaredModel

if TYPE_CHECKING:
    from torch import nn

__all__ = [
    "HEADS",
    "GruConfig",
    "HeadConfig",
    "HeadSpec",
    "MlpConfig",
    "TcnConfig",
    "assemble_network",
    "get_head",
    "head_names",
    "head_source_digest",
    "register_head",
]


class HeadConfig(DeclaredModel):
    """The options of a head. A head subclasses this and declares its fields."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)


class _Spec(Protocol):
    """What the registry reads of a head, whatever its options are."""

    @property
    def name(self) -> str: ...

    @property
    def config(self) -> type[HeadConfig]: ...

    @property
    def description(self) -> str: ...

    def make(
        self, config: HeadConfig, n_features: int, n_classes: int
    ) -> nn.Module: ...

    def source_digest(self) -> str: ...


@dataclass(frozen=True, slots=True)
class HeadSpec[C: HeadConfig]:
    """One head: its name, its options and the function that builds it.

    Attributes:
        name: The value of the step's ``head`` param.
        config: The model of the head's options. Its defaults are the head's
            defaults, and an option that changes the network is a field.
        build: Returns the module for *config*, with ``n_features`` input
            columns and ``n_classes`` outputs. It imports torch itself.
        description: One sentence on the head, for the catalog.
    """

    name: str
    config: type[C]
    build: Callable[[C, int, int], nn.Module]
    description: str

    def make(self, config: HeadConfig, n_features: int, n_classes: int) -> nn.Module:
        """Build the head for *config*, which must be an instance of :attr:`config`."""
        if not isinstance(config, self.config):
            msg = (
                f"head {self.name!r} takes {self.config.__name__} options, "
                f"not {type(config).__name__}"
            )
            raise TypeError(msg)
        return self.build(config, n_features, n_classes)

    def source_digest(self) -> str:
        """Return the sha256 of the source of :attr:`build`, empty when it has none."""
        try:
            source = inspect.getsource(self.build)
        except (OSError, TypeError):
            return ""
        return hashlib.sha256(source.encode("utf-8")).hexdigest()


HEADS: Final[dict[str, _Spec]] = {}
"""The registered heads by name."""


def register_head[C: HeadConfig](spec: HeadSpec[C]) -> HeadSpec[C]:
    """Register *spec* under its name, and return it.

    Raises:
        ValueError: A head of that name is registered.
    """
    if spec.name in HEADS:
        msg = f"a head named {spec.name!r} is already registered"
        raise ValueError(msg)
    HEADS[spec.name] = spec
    return spec


def head_names() -> list[str]:
    """Return the names of the registered heads, sorted."""
    return sorted(HEADS)


def get_head(name: str) -> _Spec:
    """Return the head registered as *name*.

    Raises:
        ValueError: No head has that name. The message lists those that do.
    """
    try:
        return HEADS[name]
    except KeyError:
        msg = (
            f"no head is registered as {name!r}. Registered: {', '.join(head_names())}"
        )
        raise ValueError(msg) from None


def head_source_digest(name: str) -> str:
    """Return the digest of the source of the head *name*, as a model records it."""
    return get_head(name).source_digest()


# --- The built-in heads ---


class TcnConfig(HeadConfig):
    """Dilated residual convolutions over time, the same at every frame."""

    hidden: Annotated[int, Field(ge=1), Declared("Channels of every block.")] = 128
    blocks: Annotated[
        int,
        Field(ge=1),
        Declared(
            "Residual blocks. Block i convolves at dilation 2**i, so the network "
            "reads (kernel_size - 1) * (2**blocks - 1) + 1 frames in all around "
            "each frame."
        ),
    ] = 6
    kernel_size: Annotated[
        int,
        Field(ge=1),
        Declared("Width of the convolution in a block, odd so that it is centered."),
    ] = 3
    dropout: Annotated[
        float, Field(ge=0.0, lt=1.0), Declared("Dropout inside each block.")
    ] = 0.3

    @field_validator("kernel_size")
    @classmethod
    def _kernel_is_centered(cls, value: int) -> int:
        if value % 2 == 0:
            msg = f"kernel_size {value} is even, and a block's convolution is centered"
            raise ValueError(msg)
        return value


class GruConfig(HeadConfig):
    """A recurrent network over the frames, read in both directions."""

    hidden: Annotated[
        int, Field(ge=1), Declared("Units of the input layer and the GRU.")
    ] = 128
    layers: Annotated[int, Field(ge=1), Declared("Stacked GRU layers.")] = 2
    bidirectional: Annotated[
        bool,
        Declared(
            "Read each series forward and backward. A unidirectional head "
            "sees only earlier frames."
        ),
    ] = True
    dropout: Annotated[
        float,
        Field(ge=0.0, lt=1.0),
        Declared("Dropout between GRU layers and before the output layer."),
    ] = 0.3


class MlpConfig(HeadConfig):
    """A feed-forward network applied to each frame by itself."""

    hidden: Annotated[int, Field(ge=1), Declared("Units of every hidden layer.")] = 128
    layers: Annotated[int, Field(ge=1), Declared("Hidden layers.")] = 2
    dropout: Annotated[
        float, Field(ge=0.0, lt=1.0), Declared("Dropout after each hidden layer.")
    ] = 0.3


def _build_tcn(config: TcnConfig, n_features: int, n_classes: int) -> nn.Module:
    import torch.nn as nn
    import torch.nn.functional as functional
    from torch import Tensor

    class Tcn(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.inp = nn.Conv1d(n_features, config.hidden, 1)
            padding = config.kernel_size // 2
            self.blocks = nn.ModuleList(
                nn.Sequential(
                    nn.Conv1d(
                        config.hidden,
                        config.hidden,
                        config.kernel_size,
                        padding=padding * 2**block,
                        dilation=2**block,
                    ),
                    nn.BatchNorm1d(config.hidden),
                    nn.ReLU(),
                    nn.Dropout(config.dropout),
                    nn.Conv1d(config.hidden, config.hidden, 1),
                )
                for block in range(config.blocks)
            )
            self.out = nn.Conv1d(config.hidden, n_classes, 1)

        def forward(self, x: Tensor) -> Tensor:
            h: Tensor = self.inp(x.transpose(1, 2))
            for block in self.blocks:
                h = h + block(h)
            out: Tensor = self.out(functional.relu(h))
            return out.transpose(1, 2)

    return Tcn()


def _build_gru(config: GruConfig, n_features: int, n_classes: int) -> nn.Module:
    import torch.nn as nn
    from torch import Tensor

    class Gru(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.inp = nn.Sequential(nn.Linear(n_features, config.hidden), nn.ReLU())
            self.rnn = nn.GRU(
                config.hidden,
                config.hidden,
                num_layers=config.layers,
                batch_first=True,
                bidirectional=config.bidirectional,
                dropout=config.dropout if config.layers > 1 else 0.0,
            )
            width = config.hidden * (2 if config.bidirectional else 1)
            self.out = nn.Sequential(
                nn.Dropout(config.dropout), nn.Linear(width, n_classes)
            )

        def forward(self, x: Tensor) -> Tensor:
            recurrent, _ = self.rnn(self.inp(x))
            out: Tensor = self.out(recurrent)
            return out

    return Gru()


def _build_mlp(config: MlpConfig, n_features: int, n_classes: int) -> nn.Module:
    import torch.nn as nn
    from torch import Tensor

    class Mlp(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            layers: list[nn.Module] = []
            width = n_features
            for _ in range(config.layers):
                layers += [
                    nn.Linear(width, config.hidden),
                    nn.ReLU(),
                    nn.Dropout(config.dropout),
                ]
                width = config.hidden
            layers.append(nn.Linear(width, n_classes))
            self.net = nn.Sequential(*layers)

        def forward(self, x: Tensor) -> Tensor:
            out: Tensor = self.net(x)
            return out

    return Mlp()


_ = register_head(
    HeadSpec(
        name="tcn",
        config=TcnConfig,
        build=_build_tcn,
        description="Dilated residual convolutions over time, centered on each frame.",
    )
)
_ = register_head(
    HeadSpec(
        name="gru",
        config=GruConfig,
        build=_build_gru,
        description="A bidirectional GRU over the frames of a series.",
    )
)
_ = register_head(
    HeadSpec(
        name="mlp",
        config=MlpConfig,
        build=_build_mlp,
        description="A feed-forward network on each frame, with no temporal context.",
    )
)


def assemble_network(
    head: str,
    config: HeadConfig,
    *,
    input_dropout: float,
    n_features: int,
    n_classes: int,
) -> nn.Module:
    """Return the module of a model: input dropout, then the head.

    The head is built after the dropout layer, which holds no parameters, so a
    run of the network draws its initial weights in the order the head's own
    layers are constructed.
    """
    import torch.nn as nn
    from torch import Tensor

    inner = get_head(head).make(config, n_features, n_classes)

    class Network(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_dropout = nn.Dropout(input_dropout)
            self.head = inner

        def forward(self, x: Tensor) -> Tensor:
            out: Tensor = self.head(self.input_dropout(x))
            return out

    return Network()
