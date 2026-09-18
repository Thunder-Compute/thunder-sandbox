"""Local publication primitives for reconnectable file transfers."""

from __future__ import annotations

import io
import os
import queue
import shutil
import tarfile
import threading
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
        else stripped.rstrip(separator).rsplit(separator, 1)[-1]
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


class ArchiveStream(io.RawIOBase):
    """A blocking byte stream that one thread feeds and another reads.

    The event loop receives archive bytes from SSH while ``tarfile`` consumes
    them in a worker thread. The bounded queue gives backpressure, so a large
    download never accumulates in memory. Both sides poll rather than block
    forever: either can disappear (a lost connection, a malformed archive)
    without stranding the other.
    """

    _POLL_SECONDS = 0.1

    def __init__(self, *, max_chunks: int = 8) -> None:
        super().__init__()
        self._chunks: queue.Queue[bytes] = queue.Queue(maxsize=max_chunks)
        self._pending = memoryview(b"")
        self._finished = threading.Event()
        self._abandoned = threading.Event()

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: bytearray | memoryview) -> int:  # type: ignore[override]
        while not self._pending:
            try:
                chunk = self._chunks.get(timeout=self._POLL_SECONDS)
            except queue.Empty:
                if not self._finished.is_set():
                    continue
                # The feeder enqueues every chunk before it finishes, so one
                # last look cannot miss data.
                try:
                    chunk = self._chunks.get_nowait()
                except queue.Empty:
                    return 0
            self._pending = memoryview(chunk)
        count = min(len(buffer), len(self._pending))
        buffer[:count] = self._pending[:count]
        self._pending = self._pending[count:]
        return count

    def feed(self, chunk: bytes) -> bool:
        """Queue ``chunk``; return False once the reader has gone away."""

        while not self._abandoned.is_set():
            try:
                self._chunks.put(chunk, timeout=self._POLL_SECONDS)
                return True
            except queue.Full:
                continue
        return False

    def finish(self) -> None:
        """Signal end of data. Never blocks, so it is safe in ``finally``."""

        self._finished.set()

    def abandon(self) -> None:
        """Release a feeder blocked on a reader that stopped early."""

        self._abandoned.set()


def extract_container_archive(
    archive: io.RawIOBase,
    stage: Path,
    *,
    contents_only: bool,
    recursive: bool,
) -> None:
    """Unpack a ``docker cp CONTAINER:PATH -`` tar stream into ``stage``.

    Docker names the archive's top level after the source (``result.txt``,
    ``logs/...``) or ``./`` for the ``PATH/.`` contents form. That top level
    maps onto ``stage`` itself, which is the layout the publishers expect.

    Reading the archive needs no permission on the source files: the stream
    was produced as root, and everything written here belongs to the caller
    with default modes. That matches ``scp`` without ``preserve``.

    Links are materialized as copies when their target is part of the
    download, as ``scp`` does, and skipped otherwise. A link pointing outside
    the payload names a container path that means nothing on this machine.
    Devices and FIFOs are skipped.
    """

    links: list[tuple[Path, Path]] = []
    produced = False
    try:
        with tarfile.open(fileobj=archive, mode="r|") as tar:
            for member in tar:
                relative = _archive_member_path(member.name, contents_only=contents_only)
                if relative is None:
                    raise SandboxFailedError(
                        f"download archive has an unsafe member path: {member.name!r}"
                    )
                destination = stage.joinpath(*relative)
                if member.isdir():
                    if not recursive:
                        raise SandboxFailedError(
                            "download source is a directory; pass recursive=True"
                        )
                    destination.mkdir(parents=True, exist_ok=True)
                    produced = True
                elif member.isreg():
                    source = tar.extractfile(member)
                    if source is None:
                        continue
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with source, open(destination, "wb") as output:
                        shutil.copyfileobj(source, output)
                    produced = True
                elif member.issym() or member.islnk():
                    target = _archive_link_target(
                        member, relative, stage, contents_only=contents_only
                    )
                    if target is not None:
                        links.append((destination, target))
    except tarfile.TarError as exc:
        raise SandboxFailedError(f"download archive could not be read: {exc}") from exc
    finally:
        if isinstance(archive, ArchiveStream):
            archive.abandon()
    # A stream can only be read forwards, and a link may precede its target.
    for destination, target in links:
        if target.is_file() and not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(target, destination)
            produced = True
    if not produced:
        raise SandboxFailedError("download archive was empty")


def _archive_member_path(name: str, *, contents_only: bool) -> tuple[str, ...] | None:
    """Return ``name`` relative to the stage, or None if it could escape it."""

    if name.startswith("/"):
        return None
    parts = [part for part in name.split("/") if part not in ("", ".")]
    if ".." in parts:
        return None
    # Outside the contents form, the first component is the source's own name,
    # which the stage already stands for.
    return tuple(parts if contents_only else parts[1:])


def _archive_link_target(
    member: tarfile.TarInfo,
    relative: tuple[str, ...],
    stage: Path,
    *,
    contents_only: bool,
) -> Path | None:
    if member.islnk():
        # Hard link names are archive paths, like member names.
        target_parts = _archive_member_path(
            member.linkname, contents_only=contents_only
        )
        return None if target_parts is None else stage.joinpath(*target_parts)
    if member.linkname.startswith("/"):
        return None
    # Symlink targets are relative to the link's own directory.
    resolved: list[str] = list(relative[:-1])
    for part in member.linkname.split("/"):
        if part in ("", "."):
            continue
        if part == "..":
            if not resolved:
                return None
            resolved.pop()
        else:
            resolved.append(part)
    return stage.joinpath(*resolved)


__all__ = [
    "ArchiveStream",
    "ParsedTransferPath",
    "extract_container_archive",
    "parse_transfer_path",
    "publish_local_contents",
    "publish_local_transfer",
    "remove_local_transfer_path",
]
