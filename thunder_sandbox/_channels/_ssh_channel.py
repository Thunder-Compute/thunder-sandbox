"""The SSH channel: processes and files of a VM sandbox, reached over SSH.

A VM sandbox boots its image as the whole VM, and the SDK logs in as root.
Commands run directly on the VM; there is no container in between.

Behaviour this channel fixes, beyond its support table:

- A command starts in the image's working directory (the sandbox's effective
  workdir, else ``/``) unless ``working_dir`` is given.
- Its environment is root's SSH login: ``/etc/environment`` (the image's
  ``ENV`` overlaid with the sandbox's env), ``HOME``, ``USER``, ``LOGNAME``,
  ``SHELL`` and the ``SSH_*`` variables. With an ``identity``, ``HOME``,
  ``USER`` and ``LOGNAME`` become that user's.
- A command given by bare name is looked up in that environment's ``PATH``.
- It runs with umask 0o022, with or without a PTY.
- Terminating a process without a PTY sends SIGTERM to its process group and
  SIGKILL 5 seconds later if it is still running. A PTY process gets one
  SIGTERM through its SSH session, with no SIGKILL.
- Uploads land owned by root:root, with the source's permission bits masked
  by the login umask (normally 0o022) and the time of the transfer as their
  modification time. Downloads keep modification times.
"""

from __future__ import annotations

import asyncio
import io
import os
import posixpath
import shlex
import shutil
import tempfile
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Container
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, TypeVar, cast

import asyncssh

from .._common.exceptions import (
    ConnectionError,
    InvalidRequestError,
    NotFoundError,
    SandboxFailedError,
    SandboxTimeoutError,
)
from .._common.types import SandboxInfo, SSHConnection
from ..asynchronous import credentials
from ..asynchronous._jobs import (
    JobSpec,
    JobState,
    JobStatus,
    RemoteJobPaths,
    can_transition,
    cleanup_command,
    signal_command,
    status_command,
    submission_command,
    validate_job_id,
)
from ..asynchronous._remote import (
    _VM_IMAGE_WORKDIR,
    _download_archive_command,
    _pinned_host_key,
    _remember_host_key,
    _remote_command,
    _stream_download_archive,
)
from ..asynchronous._ssh import (
    SSH_CONNECT_TIMEOUT_SECONDS,
    JumpSSHClient,
    RetryableSSHOperationError,
    SSHConnectionManager,
)
from ..asynchronous._transfer import (
    publish_local_contents,
    publish_local_transfer,
    remove_local_transfer_path,
)
from ._base import SUPPORTED, ExecHandle, ExecutionChannel, Fixed, Unsupported
from ._spec import (
    ConflictPolicy,
    DownloadRequest,
    ExecRequest,
    ExecState,
    ExecStatus,
    ExitReason,
    Identity,
    OutputMode,
    Retention,
    SymlinkPolicy,
    TransferResult,
    UploadRequest,
)

if TYPE_CHECKING:
    from ..asynchronous.client import Client

_T = TypeVar("_T")

SSH_KEEPALIVE_INTERVAL_SECONDS = 15
SSH_KEEPALIVE_COUNT_MAX = 4
# How long staging cleanup may take before it is abandoned.
TRANSFER_CLEANUP_GRACE_SECONDS = 5.0
# Bytes asked for in one read of a job's captured output.
OUTPUT_READ_BYTES = 64 * 1024
STATUS_INITIAL_DELAY_SECONDS = 0.1
STATUS_MAX_DELAY_SECONDS = 1.0
TERMINATION_GRACE_SECONDS = 5.0
# How long one poll keeps retrying a lost status read before it gives up.
POLL_RETRY_SECONDS = 5.0
COMMAND_UMASK = 0o022
# sshd serves at most MaxSessions (10 by default) sessions on one connection
# and drops the connection when more are opened, so every operation would
# reconnect and hit the limit again. Each session takes a slot instead.
SESSION_SLOTS = 9

_TERMINATION_UNSUPPORTED = Unsupported(
    "termination is SIGTERM to the process group, then SIGKILL after 5 seconds; "
    "a PTY process gets one SIGTERM"
)


def _identity_text(identity: Identity | None) -> str | None:
    if identity is None or identity.user is None:
        return None
    user = str(identity.user)
    return user if identity.group is None else f"{user}:{identity.group}"


def _exec_status(exec_id: str, status: JobStatus) -> ExecStatus:
    if status.state in (JobState.PREPARED, JobState.STARTING):
        return ExecStatus(exec_id, ExecState.STARTING)
    if status.state is JobState.RUNNING:
        return ExecStatus(exec_id, ExecState.RUNNING)
    if status.interrupted:
        return ExecStatus(exec_id, ExecState.ENDED, reason=ExitReason.INTERRUPTED)
    reason = (
        ExitReason.TERMINATED if status.state is JobState.TERMINATED else ExitReason.EXITED
    )
    return ExecStatus(
        exec_id, ExecState.ENDED, reason=reason, returncode=status.returncode
    )


def _decode_status(value: str, *, context: str) -> JobStatus:
    try:
        return JobStatus.from_json(value.strip())
    except ValueError as exc:
        raise SandboxFailedError(context) from exc


def _timeout_seconds(timeout_ms: int) -> str:
    seconds = timeout_ms / 1000
    return f"{seconds:g}"


class SSHChannel(ExecutionChannel):
    """Commands and files of a VM sandbox, over one reusable SSH connection."""

    name = "ssh"
    exec_support: ClassVar = {
        "working_dir": SUPPORTED,
        "inherit_env": Fixed(True),
        "env": SUPPORTED,
        "unset_env": SUPPORTED,
        "identity.user": SUPPORTED,
        "identity.group": SUPPORTED,
        "umask": Fixed(COMMAND_UMASK),
        "stdout": SUPPORTED,
        "stderr": SUPPORTED,
        "pty": SUPPORTED,
        "termination.signal": _TERMINATION_UNSUPPORTED,
        "termination.graceful_termination_timeout_ms": _TERMINATION_UNSUPPORTED,
        "termination.process_group": _TERMINATION_UNSUPPORTED,
        "retention": SUPPORTED,
    }
    upload_support: ClassVar = {
        "options.recursive": SUPPORTED,
        "options.contents_only": SUPPORTED,
        "options.on_conflict": Fixed(ConflictPolicy.MERGE),
        "options.create_parents": Fixed(False),
        "options.preserve_times": Fixed(False),
        "options.symlinks": Fixed(SymlinkPolicy.FOLLOW),
        "options.timeout_ms": SUPPORTED,
        "owner.user": Unsupported("uploads are owned by root:root"),
        "owner.group": Unsupported("uploads are owned by root:root"),
    }
    download_support: ClassVar = {
        "options.recursive": SUPPORTED,
        "options.contents_only": SUPPORTED,
        "options.on_conflict": Fixed(ConflictPolicy.MERGE),
        "options.create_parents": Fixed(False),
        "options.preserve_times": Fixed(True),
        "options.symlinks": Fixed(SymlinkPolicy.FOLLOW),
        "options.timeout_ms": SUPPORTED,
    }

    def __init__(self, client: "Client", info: Callable[[], SandboxInfo]) -> None:
        self._client = client
        self._info = info
        self._manager = SSHConnectionManager(self._open_connection)
        self._slots = asyncio.Semaphore(SESSION_SLOTS)

    # ------------------------------------------------------------ identity

    @property
    def _sandbox_id(self) -> str:
        return self._info().id

    def _deadline(self) -> float | None:
        """When the sandbox's TTL ends, on the monotonic clock: SSH retries stop there."""
        expires_at = self._info().expires_at
        if expires_at is None:
            return None
        remaining = (expires_at - datetime.now(expires_at.tzinfo)).total_seconds()
        return time.monotonic() + max(0.0, remaining)

    # ------------------------------------------------------------ connection

    def _ssh(self) -> SSHConnection:
        ssh = self._info().ssh
        if ssh is None:
            raise SandboxFailedError("sandbox SSH connection details are not available")
        return ssh

    async def _open_connection(self) -> asyncssh.SSHClientConnection:
        ssh = self._ssh()
        sandbox_id = self._sandbox_id
        # The node reuses forwarded ports across sandboxes, so a pin keyed by
        # host and port would reject the next sandbox that lands on a finished
        # one's port. Pin per sandbox instead.
        pinned = _pinned_host_key(sandbox_id)
        known_hosts = ([pinned], [], []) if pinned is not None else None
        store = self._client._credentials_for(self._info().workspace_id)
        credential = await store.ensure(self._client)
        try:
            connection = await self._open(ssh, credential, known_hosts)
        except (asyncssh.PermissionDenied, asyncssh.ChannelOpenError) as exc:
            if isinstance(exc, asyncssh.ChannelOpenError) and (
                ssh.proxy_jump is None or exc.code != asyncssh.OPEN_ADMINISTRATIVELY_PROHIBITED
            ):
                raise
            # A cached certificate can authenticate and still be refused, by a
            # rotated authority or one from before ProxyJump. Renew it once.
            credential = await store.renew(self._client, rejected=credential)
            connection = await self._open(ssh, credential, known_hosts)
        if pinned is None:
            try:
                _remember_host_key(sandbox_id, connection.get_server_host_key())
            except BaseException:
                connection.close()
                await connection.wait_closed()
                raise
        return connection

    async def _open(
        self,
        ssh: SSHConnection,
        credential: credentials.SSHCredential,
        known_hosts: object,
    ) -> asyncssh.SSHClientConnection:
        options = dict(
            client_keys=[(credential.key, credential.certificate)],
            agent_path=None,
            preferred_auth=["publickey"],
            config=None,
            connect_timeout=SSH_CONNECT_TIMEOUT_SECONDS,
            keepalive_interval=SSH_KEEPALIVE_INTERVAL_SECONDS,
            keepalive_count_max=SSH_KEEPALIVE_COUNT_MAX,
        )
        jump = None
        if ssh.proxy_jump is not None:
            proxy = ssh.proxy_jump
            pinned = asyncssh.import_public_key(proxy.host_key)
            jump = await asyncssh.connect(
                proxy.host, proxy.port, username=proxy.user,
                known_hosts=([pinned], [], []), **options,
            )
        try:
            return await asyncssh.connect(
                ssh.host, ssh.port, username=ssh.user, known_hosts=known_hosts,
                **options,
                **({"tunnel": jump, "client_factory": lambda: JumpSSHClient(jump)}
                   if jump is not None else {}),
            )
        except BaseException:
            if jump is not None:
                jump.close()
                await jump.wait_closed()
            raise

    async def close(self) -> None:
        await self._manager.close()

    async def _connect(self) -> asyncssh.SSHClientConnection:
        try:
            return await self._manager.connect(
                name=f"sandbox {self._sandbox_id} SSH connection"
            )
        except (OSError, asyncssh.Error) as exc:
            raise ConnectionError(f"could not connect to sandbox over SSH: {exc}") from exc

    async def _session(
        self,
        operation: Callable[[asyncssh.SSHClientConnection], Awaitable[_T]],
        *,
        name: str,
        deadline: float | None,
    ) -> _T:
        """Run one SSH session's work in a free slot, reconnecting after a loss."""
        async with self._slots:
            return await self._manager.run(operation, name=name, deadline=deadline)

    async def _run(
        self,
        command: str,
        *,
        name: str,
        deadline: float | None = None,
        retry_exit_codes: Container[int] = (),
        check: bool = True,
    ) -> asyncssh.SSHCompletedProcess[str]:
        """Run an idempotent command, reconnecting and resending it after a loss."""

        async def run(
            connection: asyncssh.SSHClientConnection,
        ) -> asyncssh.SSHCompletedProcess[str]:
            result = await connection.run(command, check=False, encoding="utf-8")
            if result.returncode is None or result.returncode in retry_exit_codes:
                raise RetryableSSHOperationError(f"{name} acknowledgement was lost")
            return result

        result = await self._session(
            run, name=name, deadline=self._deadline() if deadline is None else deadline
        )
        if check and result.returncode != 0:
            detail = str(result.stderr or "").strip()
            raise SandboxFailedError(
                f"{name} failed with exit code {result.returncode}"
                + (f": {detail}" if detail else "")
            )
        return result

    # ------------------------------------------------------------ commands

    def _command(self, request: ExecRequest) -> str:
        """The shell command that runs ``request`` on the VM."""
        env: dict[str, str | None] = dict(request.env)
        env.update(dict.fromkeys(request.unset_env))
        command = _remote_command(
            (request.executable, *request.args),
            workdir=request.working_dir,
            env=env or None,
            user=_identity_text(request.identity),
        )
        if request.working_dir is None:
            command = _VM_IMAGE_WORKDIR + command
        # The job launcher keeps its own files private with umask 077; the
        # command itself gets the usual 022 instead of inheriting that.
        return f"umask {COMMAND_UMASK:04o}; " + command

    async def _exec(self, request: ExecRequest) -> ExecHandle:
        if request.pty:
            return await self._start_attached(request)
        return await self._start_job(request)

    async def _start_attached(self, request: ExecRequest) -> ExecHandle:
        # A PTY process holds its session, and so its slot, for its whole life.
        await self._slots.acquire()
        try:
            connection = await self._connect()
            try:
                process = await connection.create_process(
                    self._command(request), encoding=None, term_type="xterm"
                )
            except (OSError, asyncssh.Error) as exc:
                await self._manager.discard(connection)
                raise ConnectionError(f"could not open a sandbox SSH session: {exc}") from exc
        except BaseException:
            self._slots.release()
            raise
        closed = asyncio.ensure_future(process.wait_closed())
        closed.add_done_callback(lambda _: self._slots.release())
        return _AttachedHandle(request.exec_id, process)

    async def _start_job(self, request: ExecRequest) -> ExecHandle:
        job_id = request.exec_id
        paths = RemoteJobPaths(job_id)
        env: dict[str, str | None] = dict(request.env)
        env.update(dict.fromkeys(request.unset_env))
        spec = JobSpec(
            job_id,
            (request.executable, *request.args),
            workdir=request.working_dir,
            user=_identity_text(request.identity),
            env=env,
            container=None,
            stdout=(request.stdout or OutputMode.CAPTURE).value,  # type: ignore[arg-type]
            stderr=(request.stderr or OutputMode.CAPTURE).value,  # type: ignore[arg-type]
            retain=request.retention is Retention.RETAIN,
        )
        script = submission_command(spec, self._command(request), paths)

        async def submit(
            connection: asyncssh.SSHClientConnection,
        ) -> asyncssh.SSHCompletedProcess[str]:
            # Sent over stdin: as an exec request the script would carry the
            # command inside one shell argument, which Linux caps at 128 KiB.
            # Resending it is safe; the job ID admits one launch.
            result = await connection.run(
                "sh -s", input=script, check=False, encoding="utf-8"
            )
            if result.returncode is None or result.returncode == 75:
                raise RetryableSSHOperationError(
                    "durable job submission acknowledgement was lost"
                )
            return result

        try:
            result = await self._session(
                submit,
                name=f"durable sandbox job {job_id} submission",
                deadline=self._deadline(),
            )
        except (OSError, asyncssh.Error) as exc:
            raise ConnectionError(f"could not submit a durable sandbox job: {exc}") from exc
        if result.returncode != 0:
            detail = str(result.stderr or "").strip()
            raise SandboxFailedError(
                f"could not submit durable sandbox job (exit code {result.returncode})"
                + (f": {detail}" if detail else "")
            )
        status = _decode_status(
            str(result.stdout or ""),
            context="durable sandbox job returned an invalid launch acknowledgement",
        )
        return _JobHandle(self, spec, status)

    async def _attach(self, exec_id: str) -> ExecHandle:
        try:
            validate_job_id(exec_id)
        except ValueError as exc:
            raise InvalidRequestError(str(exc)) from exc
        spec = await self._read_spec(exec_id)
        status = await self._read_status(exec_id, deadline=self._deadline())
        return _JobHandle(self, spec, status)

    async def _read_spec(self, job_id: str) -> JobSpec:
        paths = RemoteJobPaths(job_id)
        result = await self._run(
            f"cat -- {shlex.quote(str(paths.specification))}",
            name=f"durable sandbox job {job_id} specification",
            check=False,
        )
        if result.returncode != 0:
            raise NotFoundError(f"durable sandbox job {job_id} was not found")
        try:
            spec = JobSpec.from_json(str(result.stdout or "").strip())
        except ValueError as exc:
            raise SandboxFailedError(
                f"durable sandbox job {job_id} has an invalid specification"
            ) from exc
        if spec.job_id != job_id:
            raise SandboxFailedError(f"durable sandbox job {job_id} has a mismatched specification")
        if spec.container is not None:
            raise NotFoundError(f"durable sandbox job {job_id} belongs to a container")
        return spec

    async def _read_status(self, job_id: str, *, deadline: float | None) -> JobStatus:
        result = await self._run(
            status_command(RemoteJobPaths(job_id)),
            name=f"durable sandbox job {job_id} status",
            deadline=deadline,
            check=False,
        )
        if result.returncode != 0:
            detail = str(result.stderr or "").strip()
            raise NotFoundError(
                f"durable sandbox job {job_id} was not found" + (f": {detail}" if detail else "")
            )
        return _decode_status(
            str(result.stdout or ""),
            context=f"durable sandbox job {job_id} has an invalid status",
        )

    async def _read_output(self, job_id: str, stream: str, offset: int) -> bytes:
        paths = RemoteJobPaths(job_id)
        path = paths.stdout if stream == "stdout" else paths.stderr
        # dd on the VM, not SFTP: opening an SFTP session per read costs over a
        # second, while this is one exec round trip like every status read.
        command = (
            f"dd if={shlex.quote(str(path))} bs=64K status=none "
            f"iflag=skip_bytes,count_bytes,fullblock skip={offset} count={OUTPUT_READ_BYTES}"
        )

        async def read(connection: asyncssh.SSHClientConnection) -> bytes:
            result = await connection.run(command, check=False, encoding=None)
            if result.returncode is None:
                raise RetryableSSHOperationError(f"durable sandbox job {job_id} {stream} read was lost")
            if result.returncode != 0:
                raise SandboxFailedError(f"durable sandbox job {job_id} {stream} is unavailable")
            return bytes(result.stdout or b"")

        return await self._session(
            read, name=f"durable sandbox job {job_id} {stream} read", deadline=self._deadline()
        )

    async def _signal(self, job_id: str, signal: str, pid: int) -> JobStatus:
        result = await self._run(
            signal_command(RemoteJobPaths(job_id), signal=signal, pid=pid, container=None),
            name=f"durable sandbox job {job_id} SIG{signal}",
            retry_exit_codes={75},
            check=False,
        )
        if result.returncode != 0:
            detail = str(result.stderr or "").strip()
            raise SandboxFailedError(
                f"could not send SIG{signal} to durable sandbox job {job_id}"
                + (f": {detail}" if detail else "")
            )
        return _decode_status(
            str(result.stdout or ""),
            context=f"durable sandbox job {job_id} returned invalid status after SIG{signal}",
        )

    async def _cleanup(self, job_id: str) -> None:
        result = await self._run(
            cleanup_command(RemoteJobPaths(job_id), container=None),
            name=f"durable sandbox job {job_id} cleanup",
            check=False,
        )
        if result.returncode != 0:
            detail = str(result.stderr or "").strip()
            raise SandboxFailedError(
                f"could not clean up durable sandbox job {job_id}" + (f": {detail}" if detail else "")
            )

    # ------------------------------------------------------------ files

    async def _within(self, operation: Awaitable[TransferResult], timeout_ms: int | None,
                      name: str) -> TransferResult:
        if timeout_ms is None:
            return await operation
        try:
            return await asyncio.wait_for(operation, timeout_ms / 1000)
        except asyncio.TimeoutError as exc:
            raise SandboxTimeoutError(
                f"{name} did not finish within {_timeout_seconds(timeout_ms)} seconds"
            ) from exc

    async def _upload(self, request: UploadRequest) -> TransferResult:
        return await self._within(
            self._upload_now(request),
            request.options.timeout_ms,
            f"upload to sandbox {self._sandbox_id}",
        )

    async def _upload_now(self, request: UploadRequest) -> TransferResult:
        source_path = request.local_path
        options = request.options
        if not source_path.exists():
            raise InvalidRequestError(f"upload source does not exist: {source_path}")
        if source_path.is_dir() and not options.recursive:
            raise InvalidRequestError("uploading a directory requires recursive=True")
        if options.contents_only:
            if not source_path.is_dir():
                raise InvalidRequestError("a contents-only upload needs a directory source")
            await self._upload_contents(source_path, request.destination)
            return TransferResult(request.destination)
        source_name = source_path.name
        if not source_name:
            raise InvalidRequestError("upload source must have a file or directory name")
        target = await self._upload_target(request.destination, source_name)
        transfer_id = uuid.uuid4().hex
        stage = f"{target}.thunder-transfer-{transfer_id}.stage"
        backup = f"{target}.thunder-transfer-{transfer_id}.backup"
        try:
            async def transfer(connection: asyncssh.SSHClientConnection) -> None:
                await self._reset_remote_path(connection, stage)
                await asyncssh.scp(
                    os.fspath(source_path), (connection, stage), recurse=options.recursive
                )

            await self._retry_scp(transfer, name=f"upload {source_name} to sandbox {self._sandbox_id}")
            await self._publish(stage, target, backup)
        finally:
            await self._cleanup_remote_paths(stage, backup)
        return TransferResult(target)

    async def _upload_target(self, destination: str, source_name: str) -> str:
        quoted = shlex.quote(destination)
        result = await self._run(
            f"if [ -d {quoted} ]; then printf directory; else printf exact; fi",
            name=f"inspect upload destination in sandbox {self._sandbox_id}",
        )
        if str(result.stdout or "") == "directory":
            return posixpath.join(destination, source_name)
        return destination

    async def _upload_contents(self, source_directory: Path, destination: str) -> None:
        transfer_id = uuid.uuid4().hex
        stage = f"{destination.rstrip('/') or '/'}.thunder-transfer-{transfer_id}.stage"
        try:
            async def transfer(connection: asyncssh.SSHClientConnection) -> None:
                await self._reset_remote_path(connection, stage, directory=True)
                for child in source_directory.iterdir():
                    await asyncssh.scp(os.fspath(child), (connection, stage + "/"), recurse=True)

            await self._retry_scp(
                transfer, name=f"upload directory contents to sandbox {self._sandbox_id}"
            )
            await self._run(
                f"mkdir -p -- {shlex.quote(destination)} && "
                f"cp -a -- {shlex.quote(stage + '/.')} {shlex.quote(destination + '/')}",
                name=f"publish directory upload in sandbox {self._sandbox_id}",
            )
        finally:
            await self._cleanup_remote_paths(stage)

    async def _retry_scp(
        self,
        transfer: Callable[[asyncssh.SSHClientConnection], Awaitable[None]],
        *,
        name: str,
    ) -> None:
        """Restart a complete SCP transfer after a lost connection."""

        async def attempt(connection: asyncssh.SSHClientConnection) -> None:
            try:
                await transfer(connection)
            except (OSError, asyncssh.SFTPError) as exc:
                # On a live connection the transfer itself failed, and repeating
                # it cannot help. Only a lost connection is worth retrying.
                if not connection.is_closed() and not isinstance(
                    exc, (asyncssh.SFTPConnectionLost, asyncssh.SFTPNoConnection)
                ):
                    raise SandboxFailedError(f"could not complete {name}: {exc}") from exc
                raise

        try:
            await self._session(attempt, name=name, deadline=self._deadline())
        except ConnectionError:
            raise
        except (OSError, asyncssh.Error) as exc:
            raise SandboxFailedError(f"could not complete {name}: {exc}") from exc

    async def _reset_remote_path(
        self, connection: asyncssh.SSHClientConnection, path: str, *, directory: bool = False
    ) -> None:
        command = f"rm -rf -- {shlex.quote(path)}"
        if directory:
            command += f" && mkdir -- {shlex.quote(path)}"
        result = await connection.run(command, check=False, encoding="utf-8")
        if result.returncode is None:
            raise RetryableSSHOperationError("transfer staging reset was lost")
        if result.returncode != 0:
            detail = str(result.stderr or "").strip()
            raise SandboxFailedError(
                "could not reset remote transfer staging" + (f": {detail}" if detail else "")
            )

    async def _publish(self, stage: str, target: str, backup: str) -> None:
        command = f"""set -eu
stage={shlex.quote(stage)}
target={shlex.quote(target)}
backup={shlex.quote(backup)}
if [ ! -e "$stage" ] && [ ! -L "$stage" ]; then
    rm -rf -- "$backup"
    exit 0
fi
if [ -d "$stage" ] && [ -d "$target" ]; then
    cp -a -- "$stage/." "$target/"
    rm -rf -- "$stage" "$backup"
else
    mv -T -f -- "$stage" "$target"
    rm -rf -- "$backup"
fi
"""
        await self._run(command, name=f"publish upload in sandbox {self._sandbox_id}")

    async def _cleanup_remote_paths(self, *paths: str) -> None:
        deadline = time.monotonic() + TRANSFER_CLEANUP_GRACE_SECONDS
        expires = self._deadline()
        if expires is not None:
            deadline = min(deadline, expires)
        with suppress(Exception):
            await self._run(
                "rm -rf -- " + " ".join(shlex.quote(path) for path in paths),
                name=f"clean up transfer staging in sandbox {self._sandbox_id}",
                deadline=deadline,
            )

    async def _download(self, request: DownloadRequest) -> TransferResult:
        return await self._within(
            self._download_now(request),
            request.options.timeout_ms,
            f"download from sandbox {self._sandbox_id}",
        )

    async def _download_now(self, request: DownloadRequest) -> TransferResult:
        """Stream the source as one tar archive and extract it locally.

        A tar stream costs one session whatever the file count, where SCP pays
        round trips for every file, and nothing is staged on the VM.
        """
        options = request.options
        contents_only = options.contents_only
        source = request.source
        if contents_only:
            source_name = "contents"
        else:
            source_name = source.rstrip("/").rsplit("/", 1)[-1]
            if not source_name:
                raise InvalidRequestError("download source must have a file or directory name")
        destination = request.local_path
        target = (
            destination
            if contents_only
            else destination / source_name if destination.is_dir() else destination
        )
        transfer_root = Path(
            tempfile.mkdtemp(prefix=f".{target.name}.thunder-transfer-", dir=os.fspath(target.parent))
        )
        extracted = transfer_root / "archive"
        command = _download_archive_command(
            source, recursive=options.recursive, contents_only=contents_only, container=False
        )
        name = f"download {source} from sandbox {self._sandbox_id}"
        try:
            async def transfer(connection: asyncssh.SSHClientConnection) -> str | None:
                def reset() -> None:
                    remove_local_transfer_path(extracted)
                    extracted.mkdir()

                try:
                    await asyncio.to_thread(reset)
                except OSError as exc:
                    # A local disk error: final, not an SSH loss to retry.
                    raise SandboxFailedError(f"could not prepare local staging for {name}: {exc}") from exc
                return await _stream_download_archive(
                    connection, command, extracted, single_root=not contents_only, name=name
                )

            root = await self._session(transfer, name=name, deadline=self._deadline())
            if contents_only:
                await asyncio.to_thread(publish_local_contents, extracted, target)
            else:
                await asyncio.to_thread(publish_local_transfer, extracted / cast(str, root), target)
        except ConnectionError:
            raise
        except (OSError, asyncssh.Error) as exc:
            raise SandboxFailedError(f"could not complete {name}: {exc}") from exc
        finally:
            await asyncio.to_thread(shutil.rmtree, transfer_root, True)
        return TransferResult(os.fspath(target))

    # ------------------------------------------------------------ network

    async def after_network_policy_change(self) -> None:
        # Best effort: a resolver without resolvectl keeps its cache.
        await self._run(
            "sudo --non-interactive resolvectl flush-caches",
            name=f"sandbox {self._sandbox_id} VM DNS cache flush",
            check=False,
        )


# ============================================================ handles


class _JobHandle(ExecHandle):
    """A process launched as a durable job: it outlives the connection that started it.

    Its status and output live in files on the VM, so every read is a short
    command that reconnects and resends after a loss.
    """

    def __init__(self, channel: SSHChannel, spec: JobSpec, status: JobStatus) -> None:
        self._channel = channel
        self._spec = spec
        self._status = status
        self._status_lock = asyncio.Lock()
        self._termination_lock = asyncio.Lock()
        self._cleanup_lock = asyncio.Lock()
        self._cleanup_task: asyncio.Task[None] | None = None
        self._cleaned = False
        self._retain = spec.retain
        captured = {"stdout": spec.stdout == "capture", "stderr": spec.stderr == "capture"}
        self._eof = {name: not captured[name] for name in captured}
        self._streamed: set[str] = set()

    @property
    def exec_id(self) -> str:
        return self._spec.job_id

    # ------------------------------------------------------------ status

    async def _refresh(self, *, deadline: float | None = None) -> JobStatus:
        async with self._status_lock:
            current = self._status
            if current.state.terminal:
                return current
            following = await self._channel._read_status(self.exec_id, deadline=deadline)
            self._record(current, following)
            return following

    def _record(self, current: JobStatus, following: JobStatus) -> None:
        if following.state != current.state and not can_transition(current.state, following.state):
            raise SandboxFailedError(
                f"durable sandbox job {self.exec_id} moved backward from "
                f"{current.state.value} to {following.state.value}"
            )
        self._status = following

    async def _accept(self, following: JobStatus) -> JobStatus:
        async with self._status_lock:
            if self._status.state.terminal:
                return self._status
            self._record(self._status, following)
            return following

    async def poll(self) -> ExecStatus:
        deadline = time.monotonic() + POLL_RETRY_SECONDS
        return _exec_status(self.exec_id, await self._refresh(deadline=deadline))

    async def wait(self) -> ExecStatus:
        delay = STATUS_INITIAL_DELAY_SECONDS
        while True:
            status = await self._refresh()
            if status.state.terminal:
                self._maybe_cleanup()
                return _exec_status(self.exec_id, status)
            await asyncio.sleep(delay)
            delay = min(STATUS_MAX_DELAY_SECONDS, delay * 2.0)

    # ------------------------------------------------------------ streams

    def stdout(self) -> AsyncIterator[bytes]:
        return self._stream("stdout")

    def stderr(self) -> AsyncIterator[bytes]:
        return self._stream("stderr")

    def _stream(self, name: str) -> AsyncIterator[bytes]:
        if name in self._streamed:
            raise RuntimeError(f"{name} of process {self.exec_id} is already being read")
        self._streamed.add(name)
        return self._chunks(name)

    async def _chunks(self, name: str) -> AsyncIterator[bytes]:
        offset = 0
        while not self._eof[name]:
            # End of file is conclusive only when the job had already ended
            # before the read began: a job exiting during the read may flush
            # more, which the next read collects.
            ended = (await self._refresh()).state.terminal
            chunk = await self._channel._read_output(self.exec_id, name, offset)
            if chunk:
                # The offset moves only once the whole chunk is here, so a
                # resent read starts at the same place.
                offset += len(chunk)
                yield chunk
            elif ended:
                self._eof[name] = True
            else:
                await asyncio.sleep(STATUS_INITIAL_DELAY_SECONDS)
        self._maybe_cleanup()

    async def write_stdin(self, data: bytes) -> None:
        raise _no_stdin()

    async def close_stdin(self) -> None:
        raise _no_stdin()

    # ------------------------------------------------------------ termination

    async def terminate(self) -> ExecStatus:
        async with self._termination_lock:
            status = self._status
            if not status.state.terminal:
                status = await self._refresh()
            while not status.state.terminal and status.pid is None:
                await asyncio.sleep(STATUS_INITIAL_DELAY_SECONDS)
                status = await self._refresh()
            if status.state.terminal:
                self._maybe_cleanup()
                return _exec_status(self.exec_id, status)
            assert status.pid is not None
            status = await self._accept(await self._channel._signal(self.exec_id, "TERM", status.pid))
            if not status.state.terminal:
                deadline = time.monotonic() + TERMINATION_GRACE_SECONDS
                delay = STATUS_INITIAL_DELAY_SECONDS
                while time.monotonic() < deadline:
                    try:
                        status = await self._refresh(deadline=deadline)
                    except ConnectionError:
                        break
                    if status.state.terminal:
                        break
                    await asyncio.sleep(min(delay, max(0.0, deadline - time.monotonic())))
                    delay = min(STATUS_MAX_DELAY_SECONDS, delay * 2.0)
            if not status.state.terminal:
                assert status.pid is not None
                status = await self._accept(
                    await self._channel._signal(self.exec_id, "KILL", status.pid)
                )
                if not status.state.terminal:
                    raise SandboxFailedError(
                        f"durable sandbox job {self.exec_id} did not stop after SIGKILL"
                    )
            self._maybe_cleanup()
            return _exec_status(self.exec_id, status)

    # ------------------------------------------------------------ cleanup

    async def cleanup(self) -> None:
        await self.wait()
        await self._cleanup_now()

    async def _cleanup_now(self) -> None:
        async with self._cleanup_lock:
            if self._cleaned:
                return
            await self._channel._cleanup(self.exec_id)
            self._cleaned = True

    def _maybe_cleanup(self) -> None:
        """Remove the job once it has ended and its captured output is all read."""
        if self._retain or self._cleaned or not self._status.state.terminal:
            return
        if not all(self._eof.values()):
            return
        task = self._cleanup_task
        if task is None or task.done():
            task = asyncio.ensure_future(self._cleanup_now())
            task.add_done_callback(_consume)
            self._cleanup_task = task

    def __del__(self) -> None:
        # Finalizers cannot make cleanup reliable, but a handle dropped while
        # its event loop runs schedules the same idempotent cleanup that
        # ``cleanup`` performs; ``cleanup`` stays the deterministic way.
        if getattr(self, "_retain", True) or getattr(self, "_cleaned", True):
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        task = loop.create_task(self.cleanup())
        task.add_done_callback(_consume)


class _AttachedHandle(ExecHandle):
    """A PTY process, attached to the SSH session that started it.

    Its output is drained as it arrives, so an unread stream never stalls the
    session; losing the connection loses the process.
    """

    def __init__(self, exec_id: str, process: asyncssh.SSHClientProcess[bytes]) -> None:
        self._exec_id = exec_id
        self._process = process
        self._streams = {
            "stdout": _Drain(process.stdout),
            "stderr": _Drain(process.stderr),
        }
        self._streamed: set[str] = set()

    @property
    def exec_id(self) -> str:
        return self._exec_id

    def _status(self) -> ExecStatus:
        process = self._process
        if process.exit_status is None and process.exit_signal is None:
            return ExecStatus(self._exec_id, ExecState.RUNNING)
        signal = process.exit_signal
        if signal is not None:
            number = _SIGNAL_NUMBERS.get(str(signal[0]), 1)
            return ExecStatus(
                self._exec_id, ExecState.ENDED, reason=ExitReason.EXITED, returncode=128 + number
            )
        return ExecStatus(
            self._exec_id, ExecState.ENDED, reason=ExitReason.EXITED,
            returncode=int(process.exit_status or 0),
        )

    async def poll(self) -> ExecStatus:
        return self._status()

    async def wait(self) -> ExecStatus:
        try:
            await self._process.wait_closed()
            await asyncio.gather(*(drain.finished() for drain in self._streams.values()))
        except (OSError, asyncssh.Error) as exc:
            raise _attached_error(exc) from exc
        status = self._status()
        if status.state is not ExecState.ENDED:
            raise ConnectionError("sandbox SSH process ended without reporting an exit status")
        return status

    def stdout(self) -> AsyncIterator[bytes]:
        return self._stream("stdout")

    def stderr(self) -> AsyncIterator[bytes]:
        return self._stream("stderr")

    def _stream(self, name: str) -> AsyncIterator[bytes]:
        if name in self._streamed:
            raise RuntimeError(f"{name} of process {self._exec_id} is already being read")
        self._streamed.add(name)
        return self._streams[name].chunks()

    async def write_stdin(self, data: bytes) -> None:
        try:
            self._process.stdin.write(data)
            await self._process.stdin.drain()
        except (OSError, asyncssh.Error) as exc:
            raise _attached_error(exc) from exc

    async def close_stdin(self) -> None:
        try:
            self._process.stdin.write_eof()
            await self._process.stdin.drain()
        except (OSError, asyncssh.Error) as exc:
            raise _attached_error(exc) from exc

    async def terminate(self) -> ExecStatus:
        try:
            self._process.terminate()
        except (OSError, asyncssh.Error) as exc:
            raise _attached_error(exc) from exc
        return self._status()

    async def cleanup(self) -> None:
        raise io.UnsupportedOperation("a PTY process keeps no state to clean up")


class _Drain:
    """Reads an SSH stream as it arrives and hands chunks to one consumer."""

    def __init__(self, stream: asyncssh.SSHReader[bytes]) -> None:
        self._stream = stream
        self._chunks: list[bytes] = []
        self._condition = asyncio.Condition()
        self._done = False
        self._error: BaseException | None = None
        self._task = asyncio.ensure_future(self._pump())

    async def _pump(self) -> None:
        try:
            while chunk := await self._stream.read(OUTPUT_READ_BYTES):
                async with self._condition:
                    self._chunks.append(chunk)
                    self._condition.notify_all()
        except (OSError, asyncssh.Error) as exc:
            self._error = _attached_error(exc)
        finally:
            async with self._condition:
                self._done = True
                self._condition.notify_all()

    async def finished(self) -> None:
        await asyncio.wait({self._task})

    async def chunks(self) -> AsyncIterator[bytes]:
        while True:
            async with self._condition:
                await self._condition.wait_for(lambda: bool(self._chunks) or self._done)
                pending, self._chunks = self._chunks, []
                done, error = self._done, self._error
            for chunk in pending:
                yield chunk
            if done and not pending:
                if error is not None:
                    raise error
                return


_SIGNAL_NUMBERS = {
    "HUP": 1, "INT": 2, "QUIT": 3, "ILL": 4, "ABRT": 6, "FPE": 8, "KILL": 9,
    "USR1": 10, "SEGV": 11, "USR2": 12, "PIPE": 13, "ALRM": 14, "TERM": 15,
}


def _no_stdin() -> io.UnsupportedOperation:
    return io.UnsupportedOperation(
        "stdin is unavailable for durable commands; use files, arguments, "
        "environment variables, or pty=True"
    )


def _attached_error(error: BaseException) -> ConnectionError:
    return ConnectionError(
        "attached SSH process connection was lost; its remote state is unknown. "
        "Use pty=False for reconnectable unattended commands"
        + (f": {error}" if str(error) else "")
    )


def _consume(task: "asyncio.Future[None]") -> None:
    with suppress(asyncio.CancelledError, Exception):
        task.result()


__all__ = ["SSHChannel"]
