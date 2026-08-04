"""The Lightning Pose configuration mosaic trains from when a caller names none.

Lightning Pose composes its configuration with Hydra from the single file it is
handed and merges no defaults of its own, unlike sleap-nn. Its validator then
walks a complete document and raises on the first absent key, so training needs a
*complete* config -- and the package ships none: there is no YAML anywhere in the
installed distribution and no defaults factory to ask. Upstream's own `litpose
train` tells the user to download `scripts/configs/config_default.yaml` from the
Lightning Pose repository.

Requiring every caller to fetch that file put the burden in the wrong place.
mosaic is driven through an API and an app as well as a shell, and neither has a
sensible way to ask a user for a path to a file they must first find on GitHub.
So the template is carried here and used when `base_config` names nothing.

**Fetched once and committed, never downloaded at run time.** A network call in
the training path breaks offline, air-gapped and containerised runs, and a run
that trains against whatever upstream served that day is not reproducible later.
The copy is a fact about this release of mosaic, refreshed deliberately.

**It does not decide the run's identity.** What reaches the identifier is the
config's *content digest*, so a run trained from this template and one trained
from a caller's own file are told apart by what they contain rather than by where
they came from -- and refreshing this copy moves the identifier honestly, because
the settings really did change.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

__all__ = ["DEFAULT_CONFIG", "LIGHTNING_POSE_VERSION", "default_config_path"]

LIGHTNING_POSE_VERSION: Final = "2.3.1"
"""The Lightning Pose release :data:`DEFAULT_CONFIG` was taken from.

Recorded so a mismatch with the installed version is visible rather than
inferred. A config from a different release still trains -- Lightning Pose
validates what it is given -- but a key added upstream since will be missing, and
this is what says where to look.
"""

DEFAULT_CONFIG: Final = "config_default.yaml"


def default_config_path() -> Path:
    """Where the vendored template lives on disk.

    A function rather than a module constant so the path is resolved against this
    file at call time, which keeps it right for a zip-safe install and for a
    developer running from a checkout.
    """
    return Path(__file__).parent / DEFAULT_CONFIG
