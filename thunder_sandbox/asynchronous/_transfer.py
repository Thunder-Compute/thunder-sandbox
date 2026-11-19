"""Local publication primitives for reconnectable file transfers."""

from __future__ import annotations

import io
import os
import posixpath
import queue
import shutil
import tarfile
import threading
import uuid
from collections.abc import Iterator
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
    if contents_only and (not stripped or stripped.endswith(":")):
        # The root's contents: the separator is the whole path, not a suffix.
        stripped += separator
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


class TransferStreamAborted(Exception):
    """The other end of a :class:`ByteChannel` gave up on the stream."""


class ByteChannel:
    """A bounded hand-off of byte chunks from the event loop to a thread.

    The SSH reader puts chunks from a worker thread and the tar extractor reads
    them in another, so neither blocks the event loop and a slow disk applies
    back-pressure to the SSH channel. Either side can :meth:`abort`, after
    which the other stops waiting instead of hanging on a full or empty queue.
    """

    _POLL_SECONDS = 0.05

    def __init__(self, max_chunks: int = 64) -> None:
        self._queue: queue.Queue[bytes | None] = queue.Queue(max_chunks)
        self._aborted = threading.Event()

    def abort(self) -> None:
        self._aborted.set()

    def put(self, chunk: bytes | None) -> bool:
        """Queue ``chunk`` (``None`` ends the stream); False once aborted."""

        while not self._aborted.is_set():
            try:
                self._queue.put(chunk, timeout=self._POLL_SECONDS)
                return True
            except queue.Full:
                continue
        return False

    def reader(self) -> io.BufferedReader:
        return io.BufferedReader(_ChannelReader(self), buffer_size=1 << 16)

    def _get(self) -> bytes | None:
        while True:
            try:
                return self._queue.get(timeout=self._POLL_SECONDS)
            except queue.Empty:
                if self._aborted.is_set():
                    raise TransferStreamAborted("the download stream was abandoned")


class _ChannelReader(io.RawIOBase):
    def __init__(self, channel: ByteChannel) -> None:
        self._channel = channel
        self._pending = memoryview(b"")
        self._eof = False

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: memoryview | bytearray) -> int:  # type: ignore[override]
        while not self._pending and not self._eof:
            chunk = self._channel._get()
            if chunk is None:
                self._eof = True
            else:
                self._pending = memoryview(chunk)
        count = min(len(buffer), len(self._pending))
        buffer[:count] = self._pending[:count]
        self._pending = self._pending[count:]
        return count


def _archive_member_parts(name: str) -> tuple[str, ...]:
    """Return a member's path components, refusing names that could escape."""

    if not name or "\x00" in name:
        raise SandboxFailedError("download archive contains an empty member name")
    if name.startswith("/"):
        raise SandboxFailedError(f"download archive contains an absolute path: {name!r}")
    parts = tuple(part for part in name.split("/") if part not in ("", "."))
    for part in parts:
        if part == "..":
            raise SandboxFailedError(
                f"download archive contains a parent reference: {name!r}"
            )
        # A separator or drive in a component would land elsewhere on Windows.
        if any(sep and sep != "/" and sep in part for sep in (os.sep, os.altsep)) or (
            os.name == "nt" and ":" in part
        ):
            raise SandboxFailedError(
                f"download archive name is not valid on this system: {name!r}"
            )
    return parts


def extract_download_archive(
    stream: io.BufferedReader, destination: Path, *, single_root: bool
) -> str | None:
    """Extract a downloaded tar stream into the empty ``destination``.

    The archive comes from inside the sandbox, so it is untrusted. Every member
    must stay inside ``destination`` and appear only once, and a hard link must
    name a file earlier in the archive. The sender follows symlinks (tar -h),
    so a symlink in the archive is refused: extracted here, its target would
    name the caller's own files, which a later reader would then follow.
    Entries are owned by the local user and readable by them, as an SCP
    download was. With ``single_root`` every member must sit under one
    top-level name, which is returned.
    """

    links: set[tuple[str, ...]] = set()
    seen: set[tuple[str, ...]] = set()
    files: set[tuple[str, ...]] = set()
    roots: set[str] = set()
    uid = os.getuid() if hasattr(os, "getuid") else None
    gid = os.getgid() if hasattr(os, "getgid") else None

    def members(archive: tarfile.TarFile) -> Iterator[tarfile.TarInfo]:
        for member in archive:
            parts = _archive_member_parts(member.name)
            if not parts:
                # The "." entry of a contents archive is destination itself.
                continue
            if member.issym():
                raise SandboxFailedError(
                    f"download archive contains a symlink: {member.name!r}"
                )
            if not (member.isfile() or member.isdir() or member.islnk()):
                # Devices, FIFOs and sockets cannot be downloaded usefully.
                continue
            if parts in seen:
                raise SandboxFailedError(
                    f"download archive repeats the member {member.name!r}"
                )
            for depth in range(1, len(parts)):
                if parts[:depth] in links:
                    raise SandboxFailedError(
                        f"download archive member {member.name!r} passes through a link"
                    )
            if member.islnk():
                target = _archive_member_parts(member.linkname)
                if target not in files:
                    raise SandboxFailedError(
                        f"download archive hard link {member.name!r} has no target "
                        "earlier in the archive"
                    )
            seen.add(parts)
            roots.add(parts[0])
            if single_root and len(roots) > 1:
                raise SandboxFailedError("download archive has more than one top-level entry")
            if member.islnk():
                links.add(parts)
            elif member.isfile():
                files.add(parts)
            member.name = posixpath.join(*parts)
            if member.islnk():
                member.linkname = posixpath.join(*_archive_member_parts(member.linkname))
            member.mode = (member.mode & 0o777) | (0o700 if member.isdir() else 0o600)
            if uid is not None and gid is not None:
                member.uid, member.gid = uid, gid
            member.uname = member.gname = ""
            yield member

    try:
        with tarfile.open(fileobj=stream, mode="r|") as archive:
            if hasattr(tarfile, "tar_filter"):
                # The checks above also cover Pythons without filters.
                archive.extractall(
                    os.fspath(destination),
                    members=members(archive),
                    numeric_owner=True,
                    filter="data",
                )
            else:
                archive.extractall(
                    os.fspath(destination),
                    members=members(archive),
                    numeric_owner=True,
                )
    except tarfile.TarError as exc:
        raise SandboxFailedError(f"download archive is invalid: {exc}") from exc
    if single_root:
        if not roots:
            raise SandboxFailedError("download archive is empty")
        return next(iter(roots))
    return None


__all__ = [
    "ByteChannel",
    "ParsedTransferPath",
    "TransferStreamAborted",
    "extract_download_archive",
    "parse_transfer_path",
    "publish_local_contents",
    "publish_local_transfer",
    "remove_local_transfer_path",
]
