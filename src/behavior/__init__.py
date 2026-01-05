"""Public package surface for behavior utilities."""

from .dataset import Dataset, register_feature
from .helpers import from_safe_name, to_safe_name

# Optional convenience: expose top-level feature_library as behavior.feature_library
feature_library = None
try:
    import feature_library as feature_library  # type: ignore
except Exception:
    try:
        import sys
        from pathlib import Path
        repo_root = Path(__file__).resolve().parents[2]
        if (repo_root / "feature_library").exists() and str(repo_root) not in sys.path:
            sys.path.append(str(repo_root))
            import feature_library as feature_library  # type: ignore
    except Exception:
        feature_library = None

__all__ = [
    "Dataset",
    "register_feature",
    "to_safe_name",
    "from_safe_name",
    "feature_library",
]
