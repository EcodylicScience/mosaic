"""The GPU card that a local driver leases to one step, and the lock that holds it.

A driver names a card by the UUID that ``nvidia-smi`` reports, since UUIDs do not
depend on ``CUDA_DEVICE_ORDER``. :func:`leasable_cards` lists the cards the
driver may lease: those ``nvidia-smi`` lists, narrowed by the operator's
``CUDA_VISIBLE_DEVICES`` and by the operator's ``gpu_cards``. Anything that keeps
the cards from being known, or that would let an unleased step reach a leased
card, leaves no card to lease and a warning naming the reason.

A lease is an exclusive ``flock`` on ``<uuid>.lock`` in the lease directory
(:func:`lease_directory`). :func:`try_lease` takes the first free card. The driver
passes the locked descriptor to the step's process and closes its own copy, so
the lock lives exactly as long as the processes that hold the descriptor.
:func:`step_environment` narrows the step's ``CUDA_VISIBLE_DEVICES`` to the card
and records the lease in two variables, which :func:`leased_card` and
:func:`inherited_lease_fds` read back inside the step. A leased step has one card,
which its ``device`` index ``0`` names, so :func:`names_another_card` tells a
``device`` that asks for another card or several, and :func:`leases_cards_here`
tells a plan whether the host is one that leases.

Imports the standard library alone. ``fcntl`` is imported where a lock is taken,
and Windows takes no lease.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Final

from ..error_kind import ClassifiedError, ErrorKind

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from mosaic.core.json_value import JsonValue

__all__ = [
    "GPU_CARD_VAR",
    "LEASE_DIR_VAR",
    "LEASE_FDS_VAR",
    "LEASE_RETRY_SECONDS",
    "QUERY_TIMEOUT_SECONDS",
    "CardLease",
    "CardQueryFailed",
    "GpuCard",
    "GpuCardsRefused",
    "LeasableCards",
    "LeaseDirectoryUnusable",
    "card_warning",
    "default_lease_directory",
    "inherited_lease_fds",
    "leasable_cards",
    "lease_directory",
    "leased_card",
    "leases_cards_here",
    "names_another_card",
    "no_lease_warning",
    "query_cards",
    "step_environment",
    "try_lease",
    "visible_cards",
]

GPU_CARD_VAR: Final = "MOSAIC_GPU_CARD"
"""The variable that names a leased step's card, as :attr:`GpuCard.label`."""

LEASE_FDS_VAR: Final = "MOSAIC_GPU_LEASE_FDS"
"""The variable that names a leased step's lock descriptors.

A comma-separated list of ``<fd>:<st_dev>:<st_ino>``, the descriptor and the
identity of the lock file it holds open.
"""

LEASE_DIR_VAR: Final = "MOSAIC_GPU_LEASE_DIR"
"""The variable that names the lease directory in place of the per-user default."""

QUERY_TIMEOUT_SECONDS: Final = 5.0
"""How long ``nvidia-smi`` may take to list the cards."""

LEASE_RETRY_SECONDS: Final = 2.0
"""How often a driver with no free card tries again."""

_QUERY_FIELDS: Final = "index,uuid,name,mig.mode.current"
_MIG_STATES: Final = {"Enabled": True, "Disabled": False, "[N/A]": False}
_NONE_VISIBLE: Final = frozenset({"", "-1"})
_PCI_BUS_ORDER: Final = "PCI_BUS_ID"
_INDEX_PREFIXES: Final = ("cuda:", "gpu:")


@dataclass(frozen=True, slots=True)
class GpuCard:
    """One card that ``nvidia-smi`` lists.

    Attributes:
        index: The ``nvidia-smi`` index, which is PCI bus order.
        uuid: The card's UUID, ``GPU-`` followed by hex digits.
        name: The model name.
        mig_enabled: Whether MIG mode is on.
    """

    index: int
    uuid: str
    name: str
    mig_enabled: bool

    @property
    def label(self) -> str:
        """Return ``<index>:<uuid>``, how a step records its card."""
        return f"{self.index}:{self.uuid}"


@dataclass(frozen=True, slots=True)
class LeasableCards:
    """The cards a driver may lease, and what it should warn about.

    No cards with a warning means no lease: GPU steps start with the driver's
    environment. Cards with a warning keep the lease.

    Attributes:
        cards: The cards, in the order a lease tries them.
        warning: Empty, or one sentence for the driver's log.
    """

    cards: tuple[GpuCard, ...]
    warning: str


def no_lease_warning(reason: str) -> str:
    """Return the warning line for GPU steps that start with the caller's environment."""
    return f"[mosaic] warning: GPU steps take no card lease: {reason}"


def card_warning(cards: LeasableCards) -> str:
    """Return the warning line for the non-empty ``warning`` of *cards*."""
    if not cards.cards:
        return no_lease_warning(cards.warning)
    return f"[mosaic] warning: {cards.warning}"


class CardQueryFailed(Exception):
    """``nvidia-smi`` is absent, failed, timed out or printed a row it cannot read.

    Attributes:
        reason: One sentence naming what happened.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason: str = reason


class LeaseDirectoryUnusable(Exception):
    """The lease directory cannot be created, or another user owns it.

    Attributes:
        reason: One sentence naming what happened.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason: str = reason


class GpuCardsRefused(ClassifiedError, ValueError):
    """The operator's card limit names a card that is not visible, or no card."""

    error_kind: ClassVar[ErrorKind] = "invalid"

    def __init__(
        self, message: str, *, card: str | None, visible: Sequence[str]
    ) -> None:
        super().__init__(message)
        self.card: str | None = card
        self.visible: tuple[str, ...] = tuple(visible)

    def error_detail(self) -> dict[str, JsonValue] | None:
        return {"card": self.card, "visible": list(self.visible)}


def _nvidia_smi() -> str | None:
    """Return the path of ``nvidia-smi``, or ``None`` when it is not on ``PATH``."""
    return shutil.which("nvidia-smi")


def _parse_row(row: str) -> GpuCard:
    parts = [part.strip() for part in row.split(",")]
    if len(parts) < 4:
        raise CardQueryFailed(f"nvidia-smi printed a row it should not: {row!r}")
    index, uuid, mig = parts[0], parts[1], parts[-1]
    name = ", ".join(parts[2:-1])
    if not index.isdigit() or not uuid.startswith("GPU-") or mig not in _MIG_STATES:
        raise CardQueryFailed(f"nvidia-smi printed a row it should not: {row!r}")
    return GpuCard(index=int(index), uuid=uuid, name=name, mig_enabled=_MIG_STATES[mig])


def query_cards(timeout: float = QUERY_TIMEOUT_SECONDS) -> tuple[GpuCard, ...]:
    """List the host's cards through ``nvidia-smi``, without CUDA.

    Raises:
        CardQueryFailed: When ``nvidia-smi`` is absent, exits non-zero, takes
            longer than *timeout* seconds, or prints a row it cannot read.
    """
    program = _nvidia_smi()
    if program is None:
        raise CardQueryFailed("nvidia-smi is not on PATH")
    argv = [program, f"--query-gpu={_QUERY_FIELDS}", "--format=csv,noheader"]
    try:
        completed = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        raise CardQueryFailed(
            f"nvidia-smi did not answer within {timeout:g} s"
        ) from None
    except OSError as error:
        raise CardQueryFailed(f"nvidia-smi could not start: {error}") from None
    if completed.returncode != 0:
        said = completed.stderr.strip() or completed.stdout.strip()
        detail = f": {said.splitlines()[0]}" if said else ""
        raise CardQueryFailed(f"nvidia-smi exited {completed.returncode}{detail}")
    rows = [row for row in completed.stdout.splitlines() if row.strip()]
    return tuple(_parse_row(row) for row in rows)


def _card_named(cards: Iterable[GpuCard], token: str) -> GpuCard | None:
    """Return the card that *token*, an index or a UUID, names among *cards*."""
    for card in cards:
        if token.isdigit() and card.index == int(token):
            return card
        if token.startswith("GPU-") and card.uuid == token:
            return card
    return None


def _resolve_visible(
    listed: Sequence[GpuCard], visible: str | None
) -> tuple[GpuCard, ...] | str:
    """Return the cards that *visible* shows, or the reason it shows no known set."""
    if visible is None:
        return tuple(listed)
    if visible.strip() in _NONE_VISIBLE:
        return ()
    shown: list[GpuCard] = []
    for raw in visible.split(","):
        token = raw.strip()
        if token.startswith("MIG-"):
            return f"CUDA_VISIBLE_DEVICES names a MIG device, {token}"
        card = _card_named(listed, token)
        if card is None:
            return (
                f"CUDA_VISIBLE_DEVICES names {token!r}, which nvidia-smi does not list"
            )
        if card not in shown:
            shown.append(card)
    return tuple(shown)


def _lease_rules(
    listed: Sequence[GpuCard],
    shown: tuple[GpuCard, ...],
    visible: str | None,
    device_order: str | None,
) -> LeasableCards:
    if not shown:
        return LeasableCards((), "no card is visible to this driver")
    mig = [card.label for card in shown if card.mig_enabled]
    if mig:
        return LeasableCards((), f"MIG mode is on for {', '.join(mig)}")
    integer_tokens = visible is not None and any(
        token.strip().isdigit() for token in visible.split(",")
    )
    mixed_models = len({card.name for card in listed}) > 1
    if integer_tokens and mixed_models and device_order != _PCI_BUS_ORDER:
        warning = (
            "CUDA_VISIBLE_DEVICES numbers cards as nvidia-smi does, in PCI bus "
            "order, which CUDA's default order need not match on mixed models; "
            "set CUDA_DEVICE_ORDER=PCI_BUS_ID or name cards by UUID"
        )
        return LeasableCards(shown, warning)
    return LeasableCards(shown, "")


def visible_cards(
    listed: Sequence[GpuCard], visible: str | None, device_order: str | None
) -> LeasableCards:
    """Return the cards of *listed* that a driver may lease under *visible*.

    Args:
        listed: The cards ``nvidia-smi`` lists.
        visible: ``CUDA_VISIBLE_DEVICES``, or ``None`` when unset. Unset shows
            every card, and empty or ``-1`` none. A ``GPU-`` token is the card of
            that UUID and an integer token the card of that ``nvidia-smi`` index.
            A ``MIG-`` token or one that names no card is no lease.
        device_order: ``CUDA_DEVICE_ORDER``, or ``None`` when unset.

    Returns:
        The cards in the order *visible* gives them. No lease when a visible
        card has MIG mode on, since an unleased step could then reach a leased
        card. A warning, and the lease kept, when integer tokens name cards of
        several models outside ``PCI_BUS_ID`` order.
    """
    shown = _resolve_visible(listed, visible)
    if isinstance(shown, str):
        return LeasableCards((), shown)
    return _lease_rules(listed, shown, visible, device_order)


def _narrow(
    shown: tuple[GpuCard, ...], gpu_cards: Sequence[str]
) -> tuple[GpuCard, ...]:
    labels = [card.label for card in shown]
    allowed: set[GpuCard] = set()
    for name in gpu_cards:
        card = _card_named(shown, name.strip())
        if card is None:
            visible_text = ", ".join(labels) or "none"
            message = (
                f"the card limit names {name!r}, which is not a visible card "
                f"(visible: {visible_text})"
            )
            raise GpuCardsRefused(message, card=name, visible=labels)
        allowed.add(card)
    return tuple(card for card in shown if card in allowed)


def leasable_cards(
    gpu_cards: Sequence[str] | None = None,
    environ: Mapping[str, str] = os.environ,
) -> LeasableCards:
    """Return the cards a driver may lease on this host.

    Args:
        gpu_cards: The operator's limit, as ``nvidia-smi`` indexes or UUIDs among
            the visible cards. ``None`` allows every visible card.
        environ: The driver's environment, read for ``CUDA_VISIBLE_DEVICES``,
            ``CUDA_DEVICE_ORDER`` and the lease directory.

    Returns:
        The cards, or no cards and the reason when the query fails, the visible
        cards cannot be leased, or the lease directory is unusable. The rules of
        :func:`visible_cards` judge every visible card before *gpu_cards*
        narrows them, so a MIG card it leaves out still means no lease.

    Raises:
        GpuCardsRefused: When *gpu_cards* is empty, or names a card that a
            successful query shows is not visible. A failed query refuses only
            an empty *gpu_cards*.
    """
    if gpu_cards is not None and not gpu_cards:
        raise GpuCardsRefused(
            "the card limit names no card; leave it unset to lease every visible card",
            card=None,
            visible=(),
        )
    if sys.platform == "win32":
        return LeasableCards((), "Windows takes no GPU card lease")
    try:
        listed = query_cards(timeout=QUERY_TIMEOUT_SECONDS)
    except CardQueryFailed as failed:
        return LeasableCards((), failed.reason)
    visible = environ.get("CUDA_VISIBLE_DEVICES")
    shown = _resolve_visible(listed, visible)
    if isinstance(shown, str):
        return LeasableCards((), shown)
    result = _lease_rules(listed, shown, visible, environ.get("CUDA_DEVICE_ORDER"))
    if gpu_cards is not None:
        narrowed = _narrow(shown, gpu_cards)
        if result.cards:
            result = LeasableCards(narrowed, result.warning)
    if not result.cards:
        return result
    try:
        _ = lease_directory(environ)
    except LeaseDirectoryUnusable as unusable:
        return LeasableCards((), unusable.reason)
    return result


def default_lease_directory() -> Path:
    """Return ``/tmp/mosaic-gpu-leases-<uid>``, one fixed path per user, on POSIX."""
    return Path(f"/tmp/mosaic-gpu-leases-{os.getuid()}")


def lease_directory(environ: Mapping[str, str] = os.environ) -> Path:
    """Return the lease directory, created with mode ``0700`` when absent.

    It is ``MOSAIC_GPU_LEASE_DIR`` when set, else :func:`default_lease_directory`.

    Raises:
        LeaseDirectoryUnusable: On Windows, or when it cannot be created, is not
            a directory, or another user owns it, who could hold or fake every
            lock in it.
    """
    if sys.platform == "win32":
        raise LeaseDirectoryUnusable("Windows takes no GPU card lease")
    configured = environ.get(LEASE_DIR_VAR)
    directory = Path(configured) if configured else default_lease_directory()
    try:
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        owner = directory.stat().st_uid
    except OSError as error:
        reason = f"the GPU lease directory {directory} is unusable: {error}"
        raise LeaseDirectoryUnusable(reason) from None
    if not directory.is_dir():
        raise LeaseDirectoryUnusable(
            f"the GPU lease directory {directory} is not a directory"
        )
    if owner != os.getuid():
        reason = f"the GPU lease directory {directory} belongs to another user"
        raise LeaseDirectoryUnusable(reason)
    return directory


@dataclass(slots=True)
class CardLease:
    """An exclusive lock on one card's lock file, held through an open descriptor.

    Attributes:
        card: The leased card.
        fd: The descriptor that holds the lock.
        device: The lock file's ``st_dev``.
        inode: The lock file's ``st_ino``.
    """

    card: GpuCard
    fd: int
    device: int
    inode: int
    _closed: bool = field(default=False, init=False, repr=False)

    def close(self) -> None:
        """Close this process's descriptor. The lock ends once no process holds one."""
        if not self._closed:
            self._closed = True
            os.close(self.fd)


def try_lease(
    cards: Sequence[GpuCard], environ: Mapping[str, str] = os.environ
) -> CardLease | None:
    """Lock the first free card of *cards*, without waiting.

    Returns:
        The lease, or ``None`` when another process holds every card's lock. No
        other descriptor stays open.

    Raises:
        LeaseDirectoryUnusable: When the host is Windows, the lease directory is
            unusable, or a card's lock file cannot be opened or locked for a
            reason other than another holder. Waiting would not end any of them.
    """
    if sys.platform == "win32":
        raise LeaseDirectoryUnusable("Windows takes no GPU card lease")
    import fcntl

    directory = lease_directory(environ)
    for card in cards:
        path = directory / f"{card.uuid}.lock"
        try:
            fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
        except OSError as error:
            reason = f"the GPU lock file {path} cannot be opened: {error}"
            raise LeaseDirectoryUnusable(reason) from None
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            status = os.fstat(fd)
        except BlockingIOError:
            os.close(fd)
            continue
        except OSError as error:
            os.close(fd)
            reason = f"the GPU lock file {path} cannot be locked: {error}"
            raise LeaseDirectoryUnusable(reason) from None
        return CardLease(card=card, fd=fd, device=status.st_dev, inode=status.st_ino)
    return None


def step_environment(
    base: Mapping[str, str], lease: CardLease | None
) -> dict[str, str]:
    """Return a step's environment: *base* narrowed to *lease*'s card.

    Without a lease, ``CUDA_VISIBLE_DEVICES`` stays as *base* has it, so a step
    that uses a GPU without declaring one keeps it. Either way the lease
    variables *base* carries are dropped, so a step records no card it did not
    lease.
    """
    environment = {
        key: value
        for key, value in base.items()
        if key not in (GPU_CARD_VAR, LEASE_FDS_VAR)
    }
    if lease is not None:
        environment["CUDA_VISIBLE_DEVICES"] = lease.card.uuid
        environment[GPU_CARD_VAR] = lease.card.label
        environment[LEASE_FDS_VAR] = f"{lease.fd}:{lease.device}:{lease.inode}"
    return environment


def leased_card(environ: Mapping[str, str] = os.environ) -> str:
    """Return the card this process was leased, as :attr:`GpuCard.label`, or ``""``."""
    return environ.get(GPU_CARD_VAR, "")


def names_another_card(device: object) -> bool:
    """Whether *device* asks for a card other than the one a leased step has.

    A leased step sees one card, and a ``device`` index counts within the cards it
    sees, so ``0`` names that card. A ``device`` is another card when each of its
    comma-separated tokens is an integer, ``cuda:<integer>`` or ``gpu:<integer>``,
    and it is not the single token ``0``, ``cuda:0`` or ``gpu:0``. ``0,1`` and
    ``0,0`` are lists, so they count. ``None``, ``cpu``, ``cuda``, ``gpu``,
    ``auto``, ``mps``, any other text and anything that is not a string do not.
    """
    if not isinstance(device, str):
        return False
    tokens = [token.strip() for token in device.split(",")]
    indexes: list[str] = []
    for token in tokens:
        index = token.split(":", 1)[1] if token.startswith(_INDEX_PREFIXES) else token
        if not index.isdigit():
            return False
        indexes.append(index)
    return len(tokens) > 1 or int(indexes[0]) != 0


def leases_cards_here() -> bool:
    """Whether this host is one whose drivers lease cards, without querying them.

    True on POSIX with ``nvidia-smi`` on ``PATH``, which is cheap enough for a plan
    to ask. The driver's own query still decides which cards, if any, it leases.
    """
    return sys.platform != "win32" and _nvidia_smi() is not None


def inherited_lease_fds(environ: Mapping[str, str] = os.environ) -> tuple[int, ...]:
    """Return the lock descriptors this process inherited that still hold their file.

    An item of ``MOSAIC_GPU_LEASE_FDS`` that does not parse, whose descriptor is
    closed, or whose descriptor now names another file is skipped.
    """
    held: list[int] = []
    for item in environ.get(LEASE_FDS_VAR, "").split(","):
        parts = item.strip().split(":")
        if len(parts) != 3 or not all(part.isdigit() for part in parts):
            continue
        fd, device, inode = (int(part) for part in parts)
        try:
            status = os.fstat(fd)
        except OSError:
            continue
        if (status.st_dev, status.st_ino) == (device, inode):
            held.append(fd)
    return tuple(held)
