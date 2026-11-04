"""Which GPU cards a driver may lease, the lock it takes on one, and a step's view of it."""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from mosaic.core.pipeline import gpu_lease
from mosaic.core.pipeline.gpu_lease import (
    GPU_CARD_VAR,
    LEASE_DIR_VAR,
    LEASE_FDS_VAR,
    CardLease,
    CardQueryFailed,
    GpuCard,
    GpuCardsRefused,
    LeasableCards,
    LeaseDirectoryUnusable,
    inherited_lease_fds,
    leasable_cards,
    lease_directory,
    leased_card,
    query_cards,
    step_environment,
    try_lease,
    visible_cards,
)
from tests.helpers import (
    TWO_CARDS,
    install_fake_nvidia_smi,
    modules_loaded_by,
    wait_for,
)

UUID_0 = "GPU-11111111-1111-1111-1111-111111111111"
UUID_1 = "GPU-22222222-2222-2222-2222-222222222222"
CARD_0 = GpuCard(index=0, uuid=UUID_0, name="NVIDIA RTX A4000", mig_enabled=False)
CARD_1 = GpuCard(index=1, uuid=UUID_1, name="NVIDIA RTX A4000", mig_enabled=False)
MIXED = (
    CARD_0,
    GpuCard(index=1, uuid=UUID_1, name="NVIDIA RTX A6000", mig_enabled=False),
)

posix_only = pytest.mark.skipif(
    sys.platform == "win32", reason="Windows takes no lease"
)


def _no_lease(result: LeasableCards) -> bool:
    return result.cards == () and result.warning != ""


def test_the_module_imports_the_standard_library_alone() -> None:
    assert (
        modules_loaded_by(
            "import mosaic.core.pipeline.gpu_lease",
            ("pydantic", "psutil", "numpy"),
        )
        == []
    )


def test_a_card_is_labelled_by_index_and_uuid() -> None:
    assert CARD_0.label == f"0:{UUID_0}"


def test_the_query_reads_each_row(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    rows = (
        *TWO_CARDS,
        "2, GPU-33333333-3333-3333-3333-333333333333, NVIDIA A100, Enabled",
    )
    rows = (*rows, "3, GPU-44444444-4444-4444-4444-444444444444, NVIDIA T4, [N/A]")
    _ = install_fake_nvidia_smi(monkeypatch, tmp_path, rows)

    cards = query_cards()

    assert cards[:2] == (CARD_0, CARD_1)
    assert [card.mig_enabled for card in cards] == [False, False, True, False]
    assert cards[2].name == "NVIDIA A100"


def test_a_row_the_query_cannot_read_fails_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _ = install_fake_nvidia_smi(monkeypatch, tmp_path, ["0, GPU-1, NVIDIA"])

    with pytest.raises(CardQueryFailed) as raised:
        _ = query_cards()

    assert "0, GPU-1, NVIDIA" in raised.value.reason


# -- visible cards -----------------------------------------------------------


def test_unset_visible_devices_is_every_listed_card() -> None:
    assert visible_cards((CARD_0, CARD_1), None, None) == LeasableCards(
        cards=(CARD_0, CARD_1), warning=""
    )


@pytest.mark.parametrize("visible", ["", "-1", " "])
def test_empty_visible_devices_is_no_card(visible: str) -> None:
    assert _no_lease(visible_cards((CARD_0, CARD_1), visible, None))


def test_an_index_token_is_the_card_of_that_index() -> None:
    assert visible_cards((CARD_0, CARD_1), "1", None).cards == (CARD_1,)


def test_a_uuid_token_is_that_card() -> None:
    assert visible_cards((CARD_0, CARD_1), f"{UUID_1},0", None).cards == (
        CARD_1,
        CARD_0,
    )


@pytest.mark.parametrize(
    "visible",
    ["MIG-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa", "5", "GPU-ffff", "0,7"],
)
def test_a_mig_or_unmatched_token_is_no_lease(visible: str) -> None:
    result = visible_cards((CARD_0, CARD_1), visible, None)

    assert _no_lease(result)
    assert "CUDA_VISIBLE_DEVICES" in result.warning


def test_a_visible_card_with_mig_on_is_no_lease_for_the_host() -> None:
    mig = GpuCard(index=1, uuid=UUID_1, name="NVIDIA A100", mig_enabled=True)

    result = visible_cards((CARD_0, mig), None, None)

    assert _no_lease(result)
    assert "MIG" in result.warning


def test_a_hidden_card_with_mig_on_leaves_the_lease() -> None:
    mig = GpuCard(index=1, uuid=UUID_1, name="NVIDIA A100", mig_enabled=True)

    assert visible_cards((CARD_0, mig), "0", None).cards == (CARD_0,)


def test_integer_tokens_on_mixed_models_keep_the_lease_with_a_warning() -> None:
    result = visible_cards(MIXED, "1,0", None)

    assert result.cards == (MIXED[1], CARD_0)
    assert "nvidia-smi" in result.warning


@pytest.mark.parametrize(
    ("listed", "visible", "order"),
    [
        (MIXED, "1,0", "PCI_BUS_ID"),
        ((CARD_0, CARD_1), "1,0", None),
        ((CARD_0, CARD_1), "1,0", "FASTEST_FIRST"),
        (MIXED, f"{UUID_1}", None),
    ],
)
def test_no_numbering_warning_under_pci_order_one_model_or_uuids(
    listed: tuple[GpuCard, ...], visible: str, order: str | None
) -> None:
    assert visible_cards(listed, visible, order).warning == ""


# -- the query ---------------------------------------------------------------


def test_a_query_that_exits_nonzero_is_no_lease(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _ = install_fake_nvidia_smi(monkeypatch, tmp_path, TWO_CARDS, exit_code=1)

    result = leasable_cards(environ={})

    assert _no_lease(result)
    assert "exited 1" in result.warning


def test_a_query_that_hangs_is_no_lease(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _ = install_fake_nvidia_smi(monkeypatch, tmp_path, TWO_CARDS, hang_seconds=30.0)
    monkeypatch.setattr(gpu_lease, "QUERY_TIMEOUT_SECONDS", 0.3)

    result = leasable_cards(environ={})

    assert _no_lease(result)
    assert "did not answer" in result.warning


def test_no_nvidia_smi_is_no_lease() -> None:
    result = leasable_cards(environ={})

    assert _no_lease(result)
    assert "nvidia-smi" in result.warning


def test_the_suite_reads_no_card_with_nvidia_smi_on_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The autouse fixture's lookup stub hides a real ``nvidia-smi``."""
    on_path = tmp_path / "bin"
    on_path.mkdir()
    program = on_path / "nvidia-smi"
    _ = program.write_text(
        "#!/bin/sh\n" + "".join(f"echo '{row}'\n" for row in TWO_CARDS)
    )
    program.chmod(0o755)
    monkeypatch.setenv("PATH", f"{on_path}{os.pathsep}{os.environ['PATH']}")
    assert shutil.which("nvidia-smi") == str(program)

    assert leasable_cards().cards == ()


def test_leasable_cards_follow_the_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _ = install_fake_nvidia_smi(monkeypatch, tmp_path, TWO_CARDS)

    assert leasable_cards(environ={}).cards == (CARD_0, CARD_1)
    assert leasable_cards(environ={"CUDA_VISIBLE_DEVICES": "1"}).cards == (CARD_1,)


# -- the operator's gpu_cards ------------------------------------------------


def test_a_mig_card_that_gpu_cards_excludes_still_blocks_the_lease(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    rows = (TWO_CARDS[0], TWO_CARDS[1].replace("Disabled", "Enabled"))
    _ = install_fake_nvidia_smi(monkeypatch, tmp_path, rows)

    result = leasable_cards(["0"], environ={})

    assert _no_lease(result)
    assert "MIG" in result.warning


@pytest.mark.parametrize("name", ["1", UUID_1])
def test_gpu_cards_narrow_by_index_or_uuid(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, name: str
) -> None:
    _ = install_fake_nvidia_smi(monkeypatch, tmp_path, TWO_CARDS)

    assert leasable_cards([name], environ={}) == LeasableCards(
        cards=(CARD_1,), warning=""
    )


@pytest.mark.parametrize(
    ("names", "environ"),
    [
        (["0"], {"CUDA_VISIBLE_DEVICES": "1"}),
        (["7"], {}),
        (["GPU-ffff"], {}),
        ([], {}),
    ],
)
def test_gpu_cards_naming_no_visible_card_are_refused(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    names: list[str],
    environ: dict[str, str],
) -> None:
    _ = install_fake_nvidia_smi(monkeypatch, tmp_path, TWO_CARDS)

    with pytest.raises(GpuCardsRefused) as raised:
        _ = leasable_cards(names, environ=environ)

    assert isinstance(raised.value, ValueError)
    assert raised.value.error_kind == "invalid"


def test_gpu_cards_after_a_failed_query_are_no_lease() -> None:
    assert _no_lease(leasable_cards(["0"], environ={}))


def test_empty_gpu_cards_are_refused_without_a_query() -> None:
    with pytest.raises(GpuCardsRefused):
        _ = leasable_cards([], environ={})


# -- the lease directory -----------------------------------------------------


@posix_only
def test_the_lease_directory_is_created_private(tmp_path: Path) -> None:
    directory = tmp_path / "leases" / "here"

    assert lease_directory({LEASE_DIR_VAR: str(directory)}) == directory
    assert stat.S_IMODE(directory.stat().st_mode) == 0o700


@posix_only
def test_the_default_lease_directory_is_per_user() -> None:
    assert gpu_lease.default_lease_directory() == Path(
        f"/tmp/mosaic-gpu-leases-{os.getuid()}"
    )


@posix_only
def test_a_directory_owned_by_another_user_is_no_lease(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _ = install_fake_nvidia_smi(monkeypatch, tmp_path, TWO_CARDS)
    environ = {LEASE_DIR_VAR: str(tmp_path / "theirs")}
    (tmp_path / "theirs").mkdir(mode=0o700)
    uid = os.getuid()
    monkeypatch.setattr(os, "getuid", lambda: uid + 1)

    with pytest.raises(LeaseDirectoryUnusable):
        _ = lease_directory(environ)
    assert _no_lease(leasable_cards(environ=environ))
    with pytest.raises(LeaseDirectoryUnusable):
        _ = try_lease((CARD_0,), environ=environ)


# -- the lock ----------------------------------------------------------------


@posix_only
def test_an_index_token_and_its_uuid_take_one_lock(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _ = install_fake_nvidia_smi(monkeypatch, tmp_path, TWO_CARDS)
    by_index = leasable_cards(environ={"CUDA_VISIBLE_DEVICES": "0"}).cards
    by_uuid = leasable_cards([UUID_0], environ={}).cards

    lease = try_lease(by_index)
    assert lease is not None
    try:
        assert try_lease(by_uuid) is None
    finally:
        lease.close()
    second = try_lease(by_uuid)
    assert second is not None
    second.close()


@posix_only
def test_a_lock_file_that_cannot_be_opened_is_raised(tmp_path: Path) -> None:
    directory = tmp_path / "leases"
    (directory / f"{UUID_0}.lock").mkdir(parents=True)

    with pytest.raises(LeaseDirectoryUnusable) as raised:
        _ = try_lease((CARD_0,), environ={LEASE_DIR_VAR: str(directory)})

    assert f"{UUID_0}.lock" in raised.value.reason


@posix_only
def test_a_second_lease_takes_the_next_free_card() -> None:
    first = try_lease((CARD_0, CARD_1))
    second = try_lease((CARD_0, CARD_1))
    try:
        assert first is not None and first.card == CARD_0
        assert second is not None and second.card == CARD_1
        assert try_lease((CARD_0, CARD_1)) is None
    finally:
        for lease in (first, second):
            if lease is not None:
                lease.close()


@posix_only
def test_a_child_that_holds_the_descriptor_holds_the_card(tmp_path: Path) -> None:
    ready = tmp_path / "ready"
    lease = try_lease((CARD_0,))
    assert lease is not None
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import sys; open(sys.argv[1], 'w').close(); sys.stdin.read()",
            str(ready),
        ],
        stdin=subprocess.PIPE,
        pass_fds=(lease.fd,),
    )
    lease.close()
    try:
        wait_for(ready.exists, 30.0, "the child never wrote its ready file")
        assert try_lease((CARD_0,)) is None
    finally:
        assert child.stdin is not None
        child.stdin.close()
        _ = child.wait(30.0)

    taken: list[CardLease] = []

    def leased() -> bool:
        found = try_lease((CARD_0,))
        if found is not None:
            taken.append(found)
        return found is not None

    wait_for(leased, 30.0, "the card stayed locked after the child exited")
    taken[0].close()


# -- a step's environment ----------------------------------------------------

STALE = {
    "PATH": "/bin",
    "CUDA_VISIBLE_DEVICES": "0,1",
    GPU_CARD_VAR: "9:GPU-stale",
    LEASE_FDS_VAR: "99:1:2",
}


@posix_only
def test_a_leased_step_sees_its_card_alone() -> None:
    lease = try_lease((CARD_1,))
    assert lease is not None
    try:
        environment = step_environment(STALE, lease)
    finally:
        lease.close()

    assert environment == {
        "PATH": "/bin",
        "CUDA_VISIBLE_DEVICES": UUID_1,
        GPU_CARD_VAR: CARD_1.label,
        LEASE_FDS_VAR: f"{lease.fd}:{lease.device}:{lease.inode}",
    }
    assert leased_card(environment) == CARD_1.label


def test_an_unleased_step_keeps_the_visible_devices_and_drops_stale_variables() -> None:
    environment = step_environment(STALE, None)

    assert environment == {"PATH": "/bin", "CUDA_VISIBLE_DEVICES": "0,1"}
    assert leased_card(environment) == ""
    assert STALE[GPU_CARD_VAR] == "9:GPU-stale"


# -- inherited descriptors ---------------------------------------------------


@posix_only
def test_inherited_lease_fds_keep_only_descriptors_that_name_their_file(
    tmp_path: Path,
) -> None:
    lease = try_lease((CARD_0,))
    assert lease is not None
    reused = os.open(tmp_path / "reused", os.O_RDWR | os.O_CREAT)
    reused_stat = os.fstat(reused)
    other = os.open(tmp_path / "other", os.O_RDWR | os.O_CREAT)
    _ = os.dup2(other, reused)
    os.close(other)
    closed = os.open(tmp_path / "closed", os.O_RDWR | os.O_CREAT)
    os.close(closed)
    items = [
        f"{lease.fd}:{lease.device}:{lease.inode}",
        f"{closed}:{lease.device}:{lease.inode}",
        f"{reused}:{reused_stat.st_dev}:{reused_stat.st_ino}",
        "not-an-item",
    ]
    try:
        assert inherited_lease_fds({LEASE_FDS_VAR: ",".join(items)}) == (lease.fd,)
        assert inherited_lease_fds({}) == ()
    finally:
        lease.close()
        os.close(reused)
