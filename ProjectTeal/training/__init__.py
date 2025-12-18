"""Training package for ProjectTeal."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

__all__ = ["evaluation", "losses", "models", "qualitative", "reporting"]


def __getattr__(name: str) -> Any:  # pragma: no cover - thin lazy loader
    if name in __all__:
        return import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if TYPE_CHECKING:  # pragma: no cover
    from . import evaluation, losses, models, qualitative, reporting  # noqa: F401
