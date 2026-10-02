"""The test suite, a package so that its modules import ``tests.helpers``.

pytest imports this package before ``tests.conftest`` and before any test module,
so the helpers are registered here for assertion rewriting ahead of their first
import: a failing assertion inside a helper then reports its operands, as one in
a test module does.
"""

import pytest

pytest.register_assert_rewrite("tests.helpers")
