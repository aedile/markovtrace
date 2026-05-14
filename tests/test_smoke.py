"""Smoke test. Confirms the package imports and exposes a version."""

from __future__ import annotations

import markovtrace


def test_package_imports() -> None:
    assert hasattr(markovtrace, "__version__")


def test_version_is_string() -> None:
    assert isinstance(markovtrace.__version__, str)
    assert markovtrace.__version__ != ""
