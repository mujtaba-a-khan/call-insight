# scripts/__init__.py
# Lightweight package init for CLI helpers.
# Exposes:
#   - __version__
#   - convenience wrappers: index(), validate(), rebuild()
# These mirror the console_scripts defined in pyproject.toml.

from __future__ import annotations

from typing import List, Optional

try:
    # When installed as a package, pull version from metadata
    from importlib.metadata import version  # py3.8+
    __version__ = version("call-insights")
except Exception:  # pragma: no cover
    __version__ = "0.0.0"

# Re-export CLI entry points as callable functions
from . import build_vector_index as _build_vector_index  # noqa: E402
from . import validate_configs as _validate_configs  # noqa: E402
from . import rebuild_caches as _rebuild_caches  # noqa: E402


def index(argv: Optional[List[str]] = None) -> int:
    """Programmatic wrapper for `callinsights-index`."""
    return _build_vector_index.main(argv)


def validate(argv: Optional[List[str]] = None) -> int:
    """Programmatic wrapper for `callinsights-validate`."""
    return _validate_configs.main(argv)


def rebuild(argv: Optional[List[str]] = None) -> int:
    """Programmatic wrapper for `callinsights-rebuild-caches`."""
    return _rebuild_caches.main(argv)
