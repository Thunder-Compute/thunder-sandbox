"""Installed distribution version."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("thunder-sandbox")
except PackageNotFoundError:  # Source tree imported before installation.
    __version__ = "0+unknown"

__all__ = ["__version__"]
