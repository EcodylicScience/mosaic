"""The op registry in ``mosaic.core.pipeline.ops`` and the domain each op declares.

The registry is generic. The tracking ops enter it when ``mosaic.tracking`` is
imported, so this module registers them before asserting on their domain.
"""

from __future__ import annotations

from mosaic.tracking import register_ops

register_ops()


def test_registry_lives_in_core_pipeline():
    from mosaic.core.pipeline.ops import (
        OPS,
        Op,
        describe_op,
        list_ops,
        op_resource_class,
        register_op,
        run_op,
    )

    assert callable(run_op) and callable(register_op)
    assert isinstance(OPS, dict)
    assert callable(describe_op) and callable(list_ops) and callable(op_resource_class)
    assert isinstance(Op, type)


# The op domains this codebase recognizes. Extend it only when a genuinely new op
# domain is introduced (a deliberate act) -- new ops within an existing domain need
# no edit. Deliberately NOT imported from the source, so a stray new value fails here.
KNOWN_OP_DOMAINS = {"tracking", "media"}


def test_every_op_declares_a_known_domain():
    from mosaic.core.pipeline.ops import OPS

    for kind, op_cls in OPS.items():
        assert op_cls.domain in KNOWN_OP_DOMAINS, (kind, op_cls.domain)


def test_tracking_package_ops_declare_tracking_domain():
    from mosaic.core.pipeline.ops import OPS

    for kind, op_cls in OPS.items():
        if op_cls.__module__.startswith("mosaic.tracking"):
            assert op_cls.domain == "tracking", (kind, op_cls.__module__)


def test_list_ops_filters_by_domain_and_carries_domain():
    from mosaic.core.pipeline.ops import list_ops

    tracking = list_ops(domain="tracking")
    assert tracking, "expected registered tracking ops"
    assert all(entry["domain"] == "tracking" for entry in tracking)
    assert list_ops(domain="nonexistent") == []


def test_describe_op_includes_domain():
    from mosaic.core.pipeline.ops import describe_op

    info = describe_op("infer-pose")
    assert info["domain"] == "tracking"
    assert "params_schema" in info


def test_register_ops_populates_registry_in_a_fresh_interpreter():
    import subprocess
    import sys

    script = (
        "from mosaic.core.pipeline.ops import list_ops\n"
        "assert not list_ops(domain='tracking'), 'tracking ops registered too early'\n"
        "from mosaic.tracking import register_ops\n"
        "register_ops()\n"
        "kinds = {e['kind'] for e in list_ops(domain='tracking')}\n"
        "assert 'infer-pose' in kinds and 'trex' in kinds, sorted(kinds)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
