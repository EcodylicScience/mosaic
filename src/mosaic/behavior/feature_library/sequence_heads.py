"""The heads of ``sequence-classifier``: networks that map frames to class logits.

A head is a :class:`HeadSpec`: a name, the model of its options and a function that
builds a torch module. The module reads a batch of windows ``(B, T, F)`` of float32
frames and returns logits ``(B, T, C)``. It does not mix the items of a batch and
keeps no state between calls, because the classifier predicts a whole series in one
pass and trains on a batch of crops.

A model may refine the head's logits in further stages: each stage is a ``tcn`` at its
defaults that reads the softmax of the logits before it, and the model predicts with
the last. Every head takes ``stages``.

``tcn``, ``gru``, ``mlp`` and ``transformer`` are built in. A head of your own is
named by ``head`` in the step's params, and a recipe names an installed head: a package
that declares an entry point in the ``mosaic.sequence_heads`` group, whose name is the
head's name and whose object is a :class:`HeadSpec`. :func:`get_head` loads that entry
point when the head is first named, in any process that has the package, so the command
line and a queue worker find it. A head registered with :func:`register_head` exists
only in the process that registered it, and a notebook's head is not visible to another
process.

This module imports no torch. Each ``build`` function imports it when it is called,
and defines its ``nn.Module`` there.
"""

from __future__ import annotations

import hashlib
import inspect
from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import EntryPoint, entry_points
from typing import TYPE_CHECKING, Annotated, ClassVar, Final, Protocol, Self

from pydantic import ConfigDict, Field, field_validator, model_validator

from mosaic.core.params import Declared, DeclaredModel

if TYPE_CHECKING:
    from torch import nn

__all__ = [
    "ENTRY_POINT_GROUP",
    "HEADS",
    "GruConfig",
    "HeadConfig",
    "HeadSpec",
    "MlpConfig",
    "TcnConfig",
    "TransformerConfig",
    "assemble_network",
    "get_head",
    "head_names",
    "head_source_digest",
    "network_reach",
    "register_head",
]


class HeadConfig(DeclaredModel):
    """The options of a head. A head subclasses this and declares its fields."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    stages: Annotated[
        int,
        Field(ge=1),
        Declared(
            "Stages of the network. Each stage after the head is a tcn at its "
            "defaults that reads the softmax of the logits before it, adding 63 "
            "frames of reach on each side. Every stage trains on a loss of its own, "
            "and the last one predicts."
        ),
    ] = 1


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

    def temporal_reach(self, config: HeadConfig) -> int | None: ...


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
        reach: Returns the frames on each side of a frame that the head reads,
            for *config*: ``0`` for a head that reads each frame by itself, and
            ``None`` for a reach that no option bounds, such as a recurrent
            network's. Unset, the reach is unknown, which is what a head
            registered in Python says until its author declares one.
    """

    name: str
    config: type[C]
    build: Callable[[C, int, int], nn.Module]
    description: str
    reach: Callable[[C], int | None] | None = None

    def make(self, config: HeadConfig, n_features: int, n_classes: int) -> nn.Module:
        """Build the head for *config*, which must be an instance of :attr:`config`."""
        if not isinstance(config, self.config):
            msg = (
                f"head {self.name!r} takes {self.config.__name__} options, "
                f"not {type(config).__name__}"
            )
            raise TypeError(msg)
        return self.build(config, n_features, n_classes)

    def temporal_reach(self, config: HeadConfig) -> int | None:
        """Return the frames on each side that the head reads, ``None`` when unbounded."""
        if self.reach is None or not isinstance(config, self.config):
            return None
        return self.reach(config)

    def source_digest(self) -> str:
        """Return the sha256 of the source of :attr:`build`, empty when it has none."""
        try:
            source = inspect.getsource(self.build)
        except (OSError, TypeError):
            return ""
        return hashlib.sha256(source.encode("utf-8")).hexdigest()


ENTRY_POINT_GROUP: Final = "mosaic.sequence_heads"
"""The entry-point group of installed heads: a name, and a :class:`HeadSpec` as its object."""

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
    """Return the names of the registered heads and of the installed entry points, sorted.

    An entry point is listed by its name and is not loaded.
    """
    installed = {found.name for found in entry_points(group=ENTRY_POINT_GROUP)}
    return sorted(HEADS.keys() | installed)


def _origin(found: EntryPoint) -> str:
    """Return the entry point and its distribution, as a refusal names them."""
    distribution = "an unknown distribution"
    if found.dist is not None:
        distribution = f"distribution {found.dist.name!r}"
    return f"entry point {found.name!r} of {distribution}"


def _load_installed(name: str) -> _Spec | None:
    """Load, check and register the entry point named *name*, ``None`` when none is.

    Raises:
        ValueError: The entry point cannot be loaded, is not a :class:`HeadSpec`,
            carries another name than its entry point's, or has a twin of its name.
    """
    found = entry_points(group=ENTRY_POINT_GROUP, name=name)
    if not found:
        return None
    if len(found) > 1:
        origins = ", ".join(_origin(each) for each in found)
        msg = f"{len(found)} entry points are named {name!r} in {ENTRY_POINT_GROUP}: {origins}"
        raise ValueError(msg)
    (installed,) = found
    try:
        loaded = installed.load()
    except Exception as error:
        msg = (
            f"{_origin(installed)} could not be loaded: {type(error).__name__}: {error}"
        )
        raise ValueError(msg) from error
    if not isinstance(loaded, HeadSpec):
        msg = f"{_origin(installed)} is a {type(loaded).__name__}, not a HeadSpec"
        raise ValueError(msg)
    if loaded.name != name:
        msg = (
            f"{_origin(installed)} names its head {loaded.name!r}, and an entry "
            "point carries the name of its head"
        )
        raise ValueError(msg)
    HEADS[name] = loaded
    return HEADS[name]


def get_head(name: str) -> _Spec:
    """Return the head named *name*: a registered one, else an installed entry point's.

    A registered or built-in head takes precedence over an entry point of its
    name. The entry point of the group ``mosaic.sequence_heads`` with that name is
    loaded on the first lookup and registered.

    Raises:
        ValueError: No head has that name, with the registered and installed names
            listed. Or the entry point of that name cannot be loaded, is not a
            :class:`HeadSpec`, carries another name, or has a twin of its name.
    """
    registered = HEADS.get(name)
    if registered is not None:
        return registered
    loaded = _load_installed(name)
    if loaded is not None:
        return loaded
    msg = (
        f"no head is named {name!r}. Registered or installed: {', '.join(head_names())}"
    )
    raise ValueError(msg)


def head_source_digest(name: str) -> str:
    """Return the digest of the source of the head *name*, as a model records it."""
    return get_head(name).source_digest()


# --- The built-in heads ---


def _require_odd_kernel(value: int) -> int:
    """Return the width *value* of a convolution, refusing an even one."""
    if value % 2 == 0:
        msg = f"kernel_size {value} is even, and the head's convolution is centered"
        raise ValueError(msg)
    return value


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
        return _require_odd_kernel(value)


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


class TransformerConfig(HeadConfig):
    """Self-attention within blocks of frames, after a convolution giving position."""

    hidden: Annotated[
        int,
        Field(ge=1),
        Declared(
            "Width of the attention and of each layer's output, split among the "
            "attention heads."
        ),
    ] = 128
    layers: Annotated[
        int,
        Field(ge=1),
        Declared(
            "Encoder layers. Each layer lets a block read the blocks on either side, "
            "so the network reads layers blocks on each side of a frame's block."
        ),
    ] = 4
    attention_heads: Annotated[
        int, Field(ge=1), Declared("Attention heads of a layer, a divisor of hidden.")
    ] = 4
    window: Annotated[
        int,
        Field(ge=1),
        Declared(
            "Frames of a block. A frame attends to the frames of its block and of "
            "the blocks on either side, so cost grows with the series, not its square."
        ),
    ] = 64
    kernel_size: Annotated[
        int,
        Field(ge=1),
        Declared(
            "Width of the input convolution, odd so that it is centered. It is the "
            "network's only sense of position."
        ),
    ] = 5
    dropout: Annotated[
        float,
        Field(ge=0.0, lt=1.0),
        Declared("Dropout of the attention weights and after each sublayer."),
    ] = 0.1

    @field_validator("kernel_size")
    @classmethod
    def _kernel_is_centered(cls, value: int) -> int:
        return _require_odd_kernel(value)

    @model_validator(mode="after")
    def _heads_divide_the_width(self) -> Self:
        if self.hidden % self.attention_heads:
            msg = (
                f"attention_heads {self.attention_heads} do not divide hidden "
                f"{self.hidden}, and each head reads an equal share of the width"
            )
            raise ValueError(msg)
        return self


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


def _build_transformer(
    config: TransformerConfig, n_features: int, n_classes: int
) -> nn.Module:
    import math

    import torch
    import torch.nn as nn
    import torch.nn.functional as functional
    from torch import Tensor

    window = config.window
    heads = config.attention_heads
    width = config.hidden // heads

    def neighborhoods(blocks: Tensor) -> Tensor:
        """Join each block of ``(B, N, W, ...)`` to its neighbors, zero past an end."""
        shape = list(blocks.shape)
        shape[1] = 1
        edge = blocks.new_zeros(shape)
        padded = torch.cat([edge, blocks, edge], dim=1)
        return torch.cat([padded[:, :-2], padded[:, 1:-1], padded[:, 2:]], dim=2)

    class BlockAttention(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.qkv = nn.Linear(config.hidden, 3 * config.hidden)
            self.out = nn.Linear(config.hidden, config.hidden)
            self.dropout = nn.Dropout(config.dropout)

        def forward(self, x: Tensor, keys_valid: Tensor) -> Tensor:
            batch, length, _ = x.shape
            n_blocks = length // window
            q, k, v = (
                self.qkv(x).view(batch, n_blocks, window, 3, heads, width).unbind(3)
            )
            # (B, N, heads, W, width) queries against (B, N, heads, 3W, width) keys.
            q = q.transpose(2, 3)
            k = neighborhoods(k).transpose(2, 3)
            v = neighborhoods(v).transpose(2, 3)
            scores = q @ k.transpose(-1, -2) / math.sqrt(width)
            scores = scores.masked_fill(~keys_valid, float("-inf"))
            weights = self.dropout(torch.softmax(scores, dim=-1))
            mixed: Tensor = (weights @ v).transpose(2, 3).reshape(batch, length, -1)
            out: Tensor = self.out(mixed)
            return out

    class EncoderLayer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attention_norm = nn.LayerNorm(config.hidden)
            self.attention = BlockAttention()
            self.feedforward_norm = nn.LayerNorm(config.hidden)
            self.feedforward = nn.Sequential(
                nn.Linear(config.hidden, 4 * config.hidden),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(4 * config.hidden, config.hidden),
            )
            self.dropout = nn.Dropout(config.dropout)

        def forward(self, x: Tensor, keys_valid: Tensor) -> Tensor:
            attended = self.attention(self.attention_norm(x), keys_valid)
            x = x + self.dropout(attended)
            fed: Tensor = self.feedforward(self.feedforward_norm(x))
            return x + self.dropout(fed)

    class Transformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.inp = nn.Conv1d(
                n_features,
                config.hidden,
                config.kernel_size,
                padding=config.kernel_size // 2,
            )
            self.layers = nn.ModuleList(EncoderLayer() for _ in range(config.layers))
            self.norm = nn.LayerNorm(config.hidden)
            self.out = nn.Linear(config.hidden, n_classes)

        def forward(self, x: Tensor) -> Tensor:
            length = x.shape[1]
            h: Tensor = self.inp(x.transpose(1, 2)).transpose(1, 2)
            padded = -length % window
            h = functional.pad(h, (0, 0, 0, padded))
            n_blocks = h.shape[1] // window
            valid = torch.arange(h.shape[1], device=h.device) < length
            # Which keys each block reads, (1, N, 1, 1, 3W): the padded frames of the
            # last block and the zero blocks beyond either end are masked out.
            keys_valid = neighborhoods(valid.view(1, n_blocks, window)).view(
                1, n_blocks, 1, 1, 3 * window
            )
            for layer in self.layers:
                h = layer(h, keys_valid)
            out: Tensor = self.out(self.norm(h[:, :length]))
            return out

    return Transformer()


def _tcn_reach(config: TcnConfig) -> int:
    """Frames on each side that a TCN reads: its receptive field less the frame itself.

    Block ``i`` convolves at dilation ``2**i``, each side of its kernel reaching
    ``kernel_size // 2`` taps of that dilation, and the blocks stack.
    """
    return (config.kernel_size // 2) * (2**config.blocks - 1)


def _transformer_reach(config: TransformerConfig) -> int:
    """Frames on each side that a transformer reads.

    A layer lets each block read the blocks on either side, so after ``layers``
    layers a block has read the ``layers`` blocks on each side. A frame at the
    first position of its block is then ``layers * window + window - 1`` frames
    from the last frame of the farthest block ahead, and a frame at the last
    position as far from the first frame of the farthest block behind. The input
    convolution adds ``kernel_size // 2``.
    """
    return config.kernel_size // 2 + (config.layers + 1) * config.window - 1


def _unbounded_reach(config: HeadConfig) -> None:
    return None


def _frame_reach(config: HeadConfig) -> int:
    return 0


_ = register_head(
    HeadSpec(
        name="tcn",
        config=TcnConfig,
        build=_build_tcn,
        description="Dilated residual convolutions over time, centered on each frame.",
        reach=_tcn_reach,
    )
)
_ = register_head(
    HeadSpec(
        name="gru",
        config=GruConfig,
        build=_build_gru,
        description="A bidirectional GRU over the frames of a series.",
        # Recurrent state carries information from every earlier frame, and from
        # every later one when bidirectional, so no option bounds the reach.
        reach=_unbounded_reach,
    )
)
_ = register_head(
    HeadSpec(
        name="mlp",
        config=MlpConfig,
        build=_build_mlp,
        description="A feed-forward network on each frame, with no temporal context.",
        reach=_frame_reach,
    )
)
_ = register_head(
    HeadSpec(
        name="transformer",
        config=TransformerConfig,
        build=_build_transformer,
        description=(
            "Self-attention within blocks of frames and their neighbors, after a "
            "convolution that gives position."
        ),
        reach=_transformer_reach,
    )
)


def network_reach(head: str, config: HeadConfig) -> int | None:
    """Return the frames on each side that the network of *head* at *config* reads.

    The head's reach plus that of a default ``tcn`` for each refinement stage, and
    ``None`` when the head's reach is unbounded or unknown.
    """
    reach = get_head(head).temporal_reach(config)
    if reach is None:
        return None
    return reach + (config.stages - 1) * _tcn_reach(TcnConfig())


def assemble_network(
    head: str,
    config: HeadConfig,
    *,
    input_dropout: float,
    n_features: int,
    n_classes: int,
) -> nn.Module:
    """Return the module of a model: input dropout, the head, then its refinement stages.

    The head is built after the dropout layer, which holds no parameters, and
    before the refinement stages, so a run of the network draws the head's
    initial weights in the order its own layers are constructed. With one stage
    the refinements are empty and the parameters are the head's alone.

    The module's ``stage_logits`` returns the logits of every stage in order, and
    its ``forward`` the last of them.
    """
    import torch
    import torch.nn as nn
    from torch import Tensor

    inner = get_head(head).make(config, n_features, n_classes)
    refinements = [
        _build_tcn(TcnConfig(), n_classes, n_classes) for _ in range(config.stages - 1)
    ]

    class Network(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_dropout = nn.Dropout(input_dropout)
            self.head = inner
            self.refinements = nn.ModuleList(refinements)

        def stage_logits(self, x: Tensor) -> list[Tensor]:
            logits: list[Tensor] = [self.head(self.input_dropout(x))]
            for stage in self.refinements:
                logits.append(stage(torch.softmax(logits[-1], dim=-1)))
            return logits

        def forward(self, x: Tensor) -> Tensor:
            return self.stage_logits(x)[-1]

    return Network()
