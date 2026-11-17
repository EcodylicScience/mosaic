"""Two facts about a feature's params that a report of a step carries.

Both read :meth:`~mosaic.core.params.Params.identity_dump`, the dump a run id
hashes, so a field that leaves a run's identity (``HASH_EXCLUDE``) leaves both.
"""

from __future__ import annotations

from pydantic import ValidationError

from mosaic.core.params import Params

from ._utils import hash_params

__all__ = ["params_equal_defaults", "params_identity_digest"]


def params_identity_digest(params: Params) -> str:
    """Return the params term of a run id: the hash of their identity dump.

    Args:
        params: The feature's params.

    Returns:
        The digest, 10 hex characters.
    """
    return hash_params(params.identity_dump())


def params_equal_defaults(params: Params) -> bool:
    """Return whether *params* hold the defaults of their class.

    Args:
        params: The feature's params.

    Returns:
        True when their identity dump equals that of ``type(params)()``. False
        when the class cannot be built without arguments, because it declares a
        required field.
    """
    try:
        defaults = type(params)()
    except ValidationError:
        return False
    return params.identity_dump() == defaults.identity_dump()
