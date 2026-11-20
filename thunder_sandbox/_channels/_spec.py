"""Requests and results shared by every execution channel.

Every field names a behaviour the SDK has today. A channel either honours it
as given (``SUPPORTED`` in its support table) or always behaves one way
(``Fixed(value)``), which makes the difference between channels explicit
instead of implicit. ``None`` (or an empty collection) means unset, and the
channel applies its documented default.

Strings stand in for the wire's bytes: a channel encodes them as UTF-8 with
``surrogateescape``, so any Linux path or argument survives the round trip.
Durations are milliseconds, as on the wire.
"""

from __future__ import annotations

import enum
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

from .._common.exceptions import InvalidRequestError

# A field every channel must accept, which its support table therefore leaves
# out: the identity of the request and the operation itself.
CORE = {"core": True}

_ID = re.compile(r"[0-9a-f]{32}")


def _check_id(value: str, what: str) -> None:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise InvalidRequestError(f"{what} must be 32 lowercase hexadecimal characters")


def _check_text(value: str, what: str, *, empty: bool = True) -> None:
    if not isinstance(value, str):
        raise InvalidRequestError(f"{what} must be a string")
    if not empty and not value:
        raise InvalidRequestError(f"{what} cannot be empty")
    if "\x00" in value:
        raise InvalidRequestError(f"{what} cannot contain NUL")


def _check_absolute(value: str, what: str) -> None:
    _check_text(value, what, empty=False)
    if not value.startswith("/"):
        raise InvalidRequestError(f"{what} must be an absolute path, got {value!r}")


def _check_ms(value: int | None, what: str, *, zero: bool = False) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidRequestError(f"{what} must be an integer number of milliseconds")
    if value < 0 or (value == 0 and not zero):
        raise InvalidRequestError(f"{what} must be {'non-negative' if zero else 'positive'}")


def _check_bool(value: bool | None, what: str) -> None:
    if value is not None and not isinstance(value, bool):
        raise InvalidRequestError(f"{what} must be a boolean")


# ============================================================ shared


class Signal(enum.IntEnum):
    SIGKILL = 9
    SIGTERM = 15


@dataclass(frozen=True)
class Identity:
    """An account inside the sandbox, resolved like ``docker exec --user``.

    ``user`` and ``group`` are each a name or a numeric ID. A named user gets
    its primary group and supplementary groups from the sandbox's
    ``/etc/passwd`` and ``/etc/group``; a ``group`` replaces the primary group
    and clears the supplementary ones. Unset means root (uid 0, gid 0). A
    string is looked up as a name first and, failing that, taken as a numeric
    ID if it is all digits, as Docker does: ``"1000"`` is uid 1000 unless an
    account is named ``1000``.

    Privilege follows the uid alone. Root has every capability, in a container
    sandbox as in a VM sandbox. Another user has none, but nothing sets
    ``no_new_privs``, so it can still escalate through ``sudo`` or setuid
    programs if the image provides them.
    """

    user: int | str | None = None
    group: int | str | None = None

    def __post_init__(self) -> None:
        if self.group is not None and self.user is None:
            raise InvalidRequestError("group requires user")
        for value, what in ((self.user, "user"), (self.group, "group")):
            if value is None:
                continue
            if isinstance(value, bool):
                raise InvalidRequestError(f"{what} must be a name or a numeric ID")
            if isinstance(value, int):
                if not 0 <= value <= 0xFFFFFFFF:
                    raise InvalidRequestError(f"{what} ID is out of range")
            else:
                _check_text(value, what, empty=False)
                if ":" in value:
                    raise InvalidRequestError(f"{what} name cannot contain ':'")


# ============================================================ exec


class OutputMode(enum.Enum):
    CAPTURE = "capture"
    DISCARD = "discard"


@dataclass(frozen=True)
class TerminationPolicy:
    """What ``ExecHandle.terminate`` does: ``signal`` to the process group,
    then SIGKILL after ``graceful_termination_timeout_ms``."""

    signal: Signal | None = None
    graceful_termination_timeout_ms: int | None = None
    process_group: bool | None = None  # signal the whole process group, not just the leader

    def __post_init__(self) -> None:
        if self.signal is not None and not isinstance(self.signal, Signal):
            raise InvalidRequestError("termination signal must be a Signal")
        _check_ms(self.graceful_termination_timeout_ms, "graceful_termination_timeout_ms", zero=True)
        _check_bool(self.process_group, "process_group")


class Retention(enum.Enum):
    # State and captured output are removed once the process has ended and
    # every captured stream has been read to the end.
    AUTO = "auto"
    # Kept until ``ExecHandle.cleanup``.
    RETAIN = "retain"


@dataclass(frozen=True)
class ExecRequest:
    """One command. ``executable`` and ``args`` are argv; no shell interprets them.

    Without a PTY the process has no stdin and is reconnectable: it keeps
    running and its output stays readable across lost connections, and
    ``ExecutionChannel.attach`` finds it again. With a PTY, stdin is the
    terminal, stdout carries all output, stderr is empty, and the process lives
    only as long as the connection that started it.
    """

    exec_id: str = field(metadata=CORE)
    executable: str = field(metadata=CORE)
    args: tuple[str, ...] = field(default=(), metadata=CORE)
    working_dir: str | None = None
    inherit_env: bool | None = None  # start from the sandbox's environment
    env: Mapping[str, str] = field(default_factory=dict)
    unset_env: tuple[str, ...] = ()
    identity: Identity | None = None
    umask: int | None = None
    stdout: OutputMode | None = None
    stderr: OutputMode | None = None
    pty: bool | None = None
    termination: TerminationPolicy | None = None
    retention: Retention | None = None

    def __post_init__(self) -> None:
        _check_id(self.exec_id, "exec_id")
        _check_text(self.executable, "executable", empty=False)
        object.__setattr__(self, "args", tuple(self.args))
        for argument in self.args:
            _check_text(argument, "argument")
        if self.working_dir is not None:
            _check_absolute(self.working_dir, "working_dir")
        _check_bool(self.inherit_env, "inherit_env")
        env = dict(self.env)
        for name, value in env.items():
            _check_env_name(name)
            _check_text(value, f"environment variable {name}")
        object.__setattr__(self, "env", MappingProxyType(env))
        object.__setattr__(self, "unset_env", tuple(self.unset_env))
        for name in self.unset_env:
            _check_env_name(name)
        overlap = set(env) & set(self.unset_env)
        if overlap:
            raise InvalidRequestError(
                f"env and unset_env name the same variables: {', '.join(sorted(overlap))}"
            )
        if self.umask is not None and (
            isinstance(self.umask, bool) or not isinstance(self.umask, int) or not 0 <= self.umask <= 0o777
        ):
            raise InvalidRequestError("umask must be between 0 and 0o777")
        _check_bool(self.pty, "pty")
        if self.pty and (
            self.stdout not in (None, OutputMode.CAPTURE)
            or self.stderr not in (None, OutputMode.CAPTURE)
            or self.retention not in (None, Retention.AUTO)
        ):
            raise InvalidRequestError("a PTY process captures its output and cannot be retained")


def _check_env_name(name: str) -> None:
    _check_text(name, "environment variable name", empty=False)
    if "=" in name:
        raise InvalidRequestError(f"environment variable name cannot contain '=': {name!r}")


class ExecState(enum.Enum):
    STARTING = "starting"
    RUNNING = "running"
    ENDED = "ended"


class ExitReason(enum.Enum):
    # The process ended on its own. A shell's convention applies: killed by
    # signal N reads as exit code 128 + N.
    EXITED = "exited"
    # ``ExecHandle.terminate`` ended it: 143 after the termination signal, 137
    # after the SIGKILL that follows the grace period.
    TERMINATED = "terminated"
    # It stopped running, but nothing observed how it ended (the sandbox
    # restarted, for example), so there is no exit code.
    INTERRUPTED = "interrupted"


@dataclass(frozen=True)
class ExecStatus:
    """One observation of a process.

    ``reason`` is set exactly when the process has ended, and ``returncode``
    then too, except for ``INTERRUPTED``.
    """

    exec_id: str
    state: ExecState
    reason: ExitReason | None = None
    returncode: int | None = None

    def __post_init__(self) -> None:
        ended = self.state is ExecState.ENDED
        if ended != (self.reason is not None):
            raise ValueError("reason is set exactly when a process has ended")
        if (ended and self.reason is not ExitReason.INTERRUPTED) != (self.returncode is not None):
            raise ValueError("returncode is set exactly when a process has ended, unless interrupted")


# ============================================================ transfers


class ConflictPolicy(enum.Enum):
    # A directory merges into an existing directory, whose own mode, owner and
    # modification time become the source directory's; a file replaces a file.
    MERGE = "merge"


class SymlinkPolicy(enum.Enum):
    # Copy what each link points to; a dangling or looping link fails.
    FOLLOW = "follow"


@dataclass(frozen=True)
class TransferOptions:
    """How a transfer lands. The destination is the result itself, unless it is
    an existing directory, in which case the source goes inside it under its
    own name. ``contents_only`` instead copies a directory's contents into the
    destination directory (the ``/.`` suffix of ``cp`` and ``scp``).

    Fixed rules for every channel:

    - Nothing appears at the destination until the whole transfer has
      arrived, and a failure or timeout before then leaves the destination as
      it was.
    - The sandbox side reads and writes as root, so permissions there never
      block a transfer; a read-only filesystem still does.
    - A destination that is a symlink is followed.
    """

    recursive: bool = False  # required when the source is a directory
    contents_only: bool = False
    on_conflict: ConflictPolicy | None = None
    create_parents: bool | None = None
    preserve_times: bool | None = None
    symlinks: SymlinkPolicy | None = None
    timeout_ms: int | None = None  # the whole operation

    def __post_init__(self) -> None:
        if not isinstance(self.recursive, bool):
            raise InvalidRequestError("recursive must be a boolean")
        if not isinstance(self.contents_only, bool):
            raise InvalidRequestError("contents_only must be a boolean")
        if self.contents_only and not self.recursive:
            raise InvalidRequestError("contents_only copies a directory and requires recursive=True")
        _check_bool(self.create_parents, "create_parents")
        _check_bool(self.preserve_times, "preserve_times")
        _check_ms(self.timeout_ms, "timeout_ms")


@dataclass(frozen=True)
class UploadRequest:
    """Uploaded files keep their source permission bits, masked by the umask of
    the sandbox's login session (normally 0o022)."""

    local_path: Path = field(metadata=CORE)
    destination: str = field(metadata=CORE)
    options: TransferOptions = field(default_factory=TransferOptions)
    owner: Identity | None = None  # who owns what lands in the sandbox

    def __post_init__(self) -> None:
        object.__setattr__(self, "local_path", Path(self.local_path))
        _check_absolute(self.destination, "upload destination")
        if self.owner is not None and not isinstance(self.owner, Identity):
            raise InvalidRequestError("owner must be an Identity")


@dataclass(frozen=True)
class DownloadRequest:
    """Downloaded files belong to the local user and keep their source
    permission bits (0o777 only), plus read and write for that user, and
    execute too on directories. Devices, FIFOs and sockets in the source are
    left out."""

    source: str = field(metadata=CORE)
    local_path: Path = field(metadata=CORE)
    options: TransferOptions = field(default_factory=TransferOptions)

    def __post_init__(self) -> None:
        _check_absolute(self.source, "download source")
        object.__setattr__(self, "local_path", Path(self.local_path))


@dataclass(frozen=True)
class TransferResult:
    path: str  # what was published
