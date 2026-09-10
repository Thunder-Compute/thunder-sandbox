"""Local publication primitives for reconnectable file transfers."""

from __future__ import annotations

import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path

from .._common.exceptions import SandboxFailedError


@dataclass(frozen=True)
class ParsedTransferPath:
    path: str
    contents_only: bool
    name: str


def parse_transfer_path(path: str, *, separator: str) -> ParsedTransferPath:
    """Parse the SCP-style trailing ``/.`` convention once."""

    contents_only = path.endswith(separator + ".")
    stripped = path[:-2] if contents_only else path
    name = (
        "contents"
        if contents_only
        else Path(stripped.rstrip(separator)).name
    )
    return ParsedTransferPath(stripped, contents_only, name)


def remove_local_transfer_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


def publish_local_transfer(stage: Path, target: Path) -> None:
    """Publish a complete local download without exposing its staging path."""

    if not stage.exists() and not stage.is_symlink():
        raise SandboxFailedError("download completed without producing its staged path")
    if stage.is_dir() and target.is_dir():
        publish_local_contents(stage, target)
        return
    _replace_local_transfer(stage, target)


def _replace_local_transfer(stage: Path, target: Path) -> None:
    backup = target.parent / f".{target.name}.thunder-backup-{uuid.uuid4().hex}"
    if target.is_dir() and not target.is_symlink():
        os.replace(target, backup)
        try:
            os.replace(stage, target)
        except BaseException:
            os.replace(backup, target)
            raise
        remove_local_transfer_path(backup)
    else:
        os.replace(stage, target)


def publish_local_contents(stage: Path, target: Path) -> None:
    """Atomically overlay staged directory contents on a local destination."""

    if not stage.is_dir():
        raise SandboxFailedError("directory download produced an invalid staged path")
    combined = target.parent / f".{target.name}.thunder-combined-{uuid.uuid4().hex}"
    try:
        if target.exists() or target.is_symlink():
            if not target.is_dir():
                raise SandboxFailedError(
                    "cannot download directory contents into a non-directory"
                )
            shutil.copytree(target, combined)
        else:
            combined.mkdir()
        shutil.copytree(stage, combined, dirs_exist_ok=True)
        _replace_local_transfer(combined, target)
    finally:
        remove_local_transfer_path(combined)


__all__ = [
    "ParsedTransferPath",
    "parse_transfer_path",
    "publish_local_contents",
    "publish_local_transfer",
    "remove_local_transfer_path",
]
