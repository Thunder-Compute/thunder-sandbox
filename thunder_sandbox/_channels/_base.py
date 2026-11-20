"""The execution channel: how a sandbox's processes and files are reached.

A sandbox is bound to one channel when its handle is built, and keeps it.
``exec``, ``upload`` and ``download`` check their request against the
channel's support table before anything reaches the sandbox, so a request
either does exactly what it says or fails up front with
``UnsupportedFeatureError``.

A support table names every optional field of a request type, so no channel
can silently ignore one. Each field is:

- ``SUPPORTED``: honoured as given; unset applies the channel's default.
- ``Fixed(value)``: the channel always behaves as ``value``. Unset, or
  exactly ``value``, is accepted; anything else is refused.
- ``Unsupported(...)``: only unset is accepted.

Channels and handles raise the SDK's own exceptions, never those of their
transport.
"""

from __future__ import annotations

import abc
import dataclasses
import types
import typing
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from typing import Any, ClassVar, Union

from .._common.exceptions import UnsupportedFeatureError
from ._spec import DownloadRequest, ExecRequest, ExecStatus, TransferResult, UploadRequest


@dataclass(frozen=True)
class _Supported:
    def __repr__(self) -> str:
        return "SUPPORTED"


SUPPORTED = _Supported()


@dataclass(frozen=True)
class Fixed:
    """The channel always behaves as ``value``."""

    value: object


@dataclass(frozen=True)
class Unsupported:
    reason: str = ""


Rule = Union[_Supported, Fixed, Unsupported]
# Dotted field path, relative to the request: "umask", "termination.signal".
SupportTable = Mapping[str, Rule]


# ============================================================ process handle


class ExecHandle(abc.ABC):
    """A process started by ``ExecutionChannel.exec`` or found by ``attach``.

    The handle deals only in raw bytes. Reading behaviour (``read(n)``,
    ``readline``, line iteration, text decoding) is built once on top of
    ``stdout()``/``stderr()``, so it is identical on every channel.
    """

    @property
    @abc.abstractmethod
    def exec_id(self) -> str: ...

    @abc.abstractmethod
    def stdout(self) -> AsyncIterator[bytes]:
        """The captured stdout, as non-empty chunks in order from its first byte.

        The iterator ends exactly at end of stream: once the process has ended
        and every byte has been delivered. A reconnectable process's stream
        survives a lost connection with no gaps or duplicates. A stream the
        request discarded ends at once. Each handle's
        stream can be iterated once; a second call raises ``RuntimeError``. A step
        that raises can be retried: the stream resumes where it was.
        """

    @abc.abstractmethod
    def stderr(self) -> AsyncIterator[bytes]:
        """The captured stderr; see ``stdout``."""

    @abc.abstractmethod
    async def write_stdin(self, data: bytes) -> None:
        """Return once the process's stdin has accepted ``data``.

        Only a PTY process has stdin; any other raises ``io.UnsupportedOperation``.
        """

    @abc.abstractmethod
    async def close_stdin(self) -> None:
        """Send end of file on stdin."""

    def last_status(self) -> ExecStatus | None:
        """The most recent status observed, without asking the sandbox."""
        return None

    @abc.abstractmethod
    async def poll(self) -> ExecStatus:
        """The process's current status."""

    @abc.abstractmethod
    async def wait(self) -> ExecStatus:
        """Return the final status once the process has ended."""

    @abc.abstractmethod
    async def terminate(self) -> ExecStatus:
        """Apply the request's termination policy and return the final status."""

    @abc.abstractmethod
    async def cleanup(self) -> None:
        """Wait for the process to end, then remove its state and captured output.

        A PTY process keeps no state; it raises ``io.UnsupportedOperation``.
        """


# ============================================================ channel


class ExecutionChannel(abc.ABC):
    """Runs processes in, and moves files to and from, one sandbox."""

    #: Short name used in error messages, such as "ssh" or "ssh-docker".
    name: ClassVar[str]
    exec_support: ClassVar[SupportTable]
    upload_support: ClassVar[SupportTable]
    download_support: ClassVar[SupportTable]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # ABCMeta records __abstractmethods__ only after this runs, so look
        # for abstract methods directly.
        if any(getattr(getattr(cls, name, None), "__isabstractmethod__", False) for name in dir(cls)):
            return  # An intermediate base; its concrete subclasses are checked.
        for request_type, attribute in _TABLES:
            _check_table_covers(cls, request_type, getattr(cls, attribute))

    async def exec(self, request: ExecRequest) -> ExecHandle:
        """Start a process. Sending the same ``exec_id`` again returns a handle
        to the process it already started."""
        self._check(request, self.exec_support)
        return await self._exec(request)

    async def attach(self, exec_id: str) -> ExecHandle:
        """A handle to a process started earlier, possibly by another client.

        Raises ``NotFoundError`` if there's no such process, or it can't be
        reached again (as with a PTY process whose connection was lost).
        """
        return await self._attach(exec_id)

    async def upload(self, request: UploadRequest) -> TransferResult:
        self._check(request, self.upload_support)
        return await self._upload(request)

    async def download(self, request: DownloadRequest) -> TransferResult:
        self._check(request, self.download_support)
        return await self._download(request)

    @abc.abstractmethod
    async def after_network_policy_change(self) -> None:
        """Run once the API has confirmed a new network policy is enforced."""

    @abc.abstractmethod
    async def close(self) -> None:
        """Release the transport. Reconnectable processes keep running."""

    # ------------------------------------------------------------ for subclasses

    @abc.abstractmethod
    async def _exec(self, request: ExecRequest) -> ExecHandle: ...

    @abc.abstractmethod
    async def _attach(self, exec_id: str) -> ExecHandle: ...

    @abc.abstractmethod
    async def _upload(self, request: UploadRequest) -> TransferResult: ...

    @abc.abstractmethod
    async def _download(self, request: DownloadRequest) -> TransferResult: ...

    # ------------------------------------------------------------ checking

    def _check(self, request: object, table: SupportTable) -> None:
        for path, rule in table.items():
            if rule is SUPPORTED:
                continue
            value = _lookup(request, path)
            if _is_unset(value):
                continue
            if isinstance(rule, Fixed):
                if value == rule.value:
                    continue
                raise UnsupportedFeatureError(
                    f"the {self.name} channel only supports {path}={rule.value!r}, got {value!r}"
                )
            assert isinstance(rule, Unsupported), rule
            raise UnsupportedFeatureError(
                f"the {self.name} channel does not support {path}"
                + (f": {rule.reason}" if rule.reason else "")
            )


_TABLES: tuple[tuple[type, str], ...] = (
    (ExecRequest, "exec_support"),
    (UploadRequest, "upload_support"),
    (DownloadRequest, "download_support"),
)


def _is_unset(value: object) -> bool:
    if value is None:
        return True
    # Empty collections mean unset; False is a real value, so it isn't.
    return isinstance(value, (tuple, list, dict, Mapping)) and not value


def _lookup(request: object, path: str) -> object:
    value: object = request
    for part in path.split("."):
        if value is None:
            return None
        value = getattr(value, part)
    return value


def _unwrap_optional(annotation: object) -> object:
    origin = typing.get_origin(annotation)
    if origin in (Union, types.UnionType):
        members = [arg for arg in typing.get_args(annotation) if arg is not type(None)]
        if len(members) == 1:
            return members[0]
    return annotation


def optional_fields(request_type: type, prefix: str = "") -> list[str]:
    """Every field path a support table for ``request_type`` must name.

    A field holding a single dataclass, such as ``identity`` or ``termination``,
    is named by its subfields. A union of several types, such as transfer
    ``mode`` or ``owner``, is named as one field. Core fields are left out.
    """
    hints = typing.get_type_hints(request_type)
    paths: list[str] = []
    for item in dataclasses.fields(request_type):
        if item.metadata.get("core"):
            continue
        annotation = _unwrap_optional(hints[item.name])
        path = f"{prefix}{item.name}"
        if isinstance(annotation, type) and dataclasses.is_dataclass(annotation):
            paths.extend(optional_fields(annotation, prefix=f"{path}."))
        else:
            paths.append(path)
    return paths


def _check_table_covers(cls: type, request_type: type, table: SupportTable) -> None:
    expected = set(optional_fields(request_type))
    named = set(table)
    missing, unknown = expected - named, named - expected
    if missing or unknown:
        problems = []
        if missing:
            problems.append(f"missing {', '.join(sorted(missing))}")
        if unknown:
            problems.append(f"unknown {', '.join(sorted(unknown))}")
        raise TypeError(
            f"{cls.__name__} support table for {request_type.__name__} is incomplete: "
            + "; ".join(problems)
        )
    for path, rule in table.items():
        if not isinstance(rule, (_Supported, Fixed, Unsupported)):
            raise TypeError(f"{cls.__name__}: invalid rule for {path}: {rule!r}")
