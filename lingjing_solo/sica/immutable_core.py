"""R5 immutable-core path checks.

Paths are repository-relative POSIX paths. The check is deliberately small and
fail-closed so an automated candidate cannot silently alter the safety boundary.
"""
from __future__ import annotations

from pathlib import PurePosixPath
from typing import Iterable

DEFAULT_IMMUTABLE_PATHS: tuple[str, ...] = (
    "env_interface/",
    "safety_gate.py",
    "lingjing_solo/sica/immutable_core.py",
    "lingjing_solo/sica/tabu_store.py",
    "scripts/benchmark_holdout.py",
    "tabu_store.py",
    ".immutable",
)


class ImmutablePathError(PermissionError):
    """Raised when a candidate touches a protected path."""


def _normalise(path: str) -> str:
    if not isinstance(path, str) or not path.strip():
        raise ImmutablePathError("modified path must be a non-empty string")
    value = path.replace(chr(92), "/")
    if value.startswith("/") or ".." in PurePosixPath(value).parts:
        raise ImmutablePathError(f"path must be repository-relative: {path!r}")
    value = value.lstrip("./")
    if not value:
        raise ImmutablePathError(f"path must be repository-relative: {path!r}")
    return value.rstrip("/")


def _matches(path: str, protected: str) -> bool:
    protected = protected.replace("\\", "/").lstrip("./").rstrip("/")
    return path == protected or path.startswith(protected + "/")


def immutable_paths(paths: Iterable[str] | None = None) -> tuple[str, ...]:
    values = tuple(paths or DEFAULT_IMMUTABLE_PATHS)
    if not values:
        raise ImmutablePathError("immutable path list must not be empty")
    return tuple(_normalise(value) for value in values)


def is_mutable(path: str, *, protected_paths: Iterable[str] | None = None) -> bool:
    normalised = _normalise(path)
    return not any(_matches(normalised, protected) for protected in immutable_paths(protected_paths))


def pre_commit_check(modified_files: Iterable[str], *, protected_paths: Iterable[str] | None = None) -> None:
    """Reject the complete change set if any file is protected."""
    protected = immutable_paths(protected_paths)
    touched = [_normalise(path) for path in modified_files]
    blocked = [path for path in touched if any(_matches(path, item) for item in protected)]
    if blocked:
        raise ImmutablePathError(
            "candidate touches immutable core: " + ", ".join(sorted(blocked))
        )
