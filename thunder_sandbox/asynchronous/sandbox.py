"""Native asynchronous sandbox lifecycle, SSH execution, and transfers."""

from __future__ import annotations

import asyncio
import os
import posixpath
import random
import shlex
import shutil
import tempfile
import time
import uuid
from collections.abc import Awaitable, Callable, Container, Mapping, Sequence
from contextlib import suppress
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Literal, cast, overload
from urllib.parse import quote

import asyncssh

from .._common.config import ThunderPaths
from .._common.exceptions import (
    ConflictError,
    ConnectionError,
    InvalidRequestError,
    NotFoundError,
    RetryableError,
    SandboxFailedError,
    SandboxTimeoutError,
    _WaitWindowElapsedError,
)
from .._common.types import (
    GPUType,
    NetworkPolicy,
    OutputMode,
    Resources,
    SandboxInfo,
    SandboxStatus,
    SSHConnection,
)
from ..image import Image
from . import credentials
from ._jobs import (
    JobSpec,
    JobState,
    JobStatus,
    RemoteJobPaths,
    cleanup_command,
    container_job_directory,
    new_job_id,
    signal_command,
    submission_command,
    validate_job_id,
)
from ._ssh import (
    RetryableSSHOperationError,
    SSHConnectionManager,
    SSH_CONNECT_TIMEOUT_SECONDS,
)
from ._transfer import (
    parse_transfer_path as _parse_transfer_path,
    publish_local_contents as _publish_local_contents,
    publish_local_transfer as _publish_local_transfer,
    remove_local_transfer_path as _remove_local_transfer_path,
)
from .client import Client
from .process import Process

OUTAGE_GRACE_SECONDS = 30.0
# The API refuses a readiness wait held open longer than this per request.
WAIT_WINDOW_MAX_SECONDS = 30.0
# How much longer than its window one wait request may take to answer before
# the client gives up on it, covering transit and a server that has stalled.
WAIT_REPLY_GRACE_SECONDS = 15.0
_CONTAINER_NAME = "thunder-sandbox"
_CONTAINER_BUSYBOX = "/busybox"


SSH_KEEPALIVE_INTERVAL_SECONDS = 15
SSH_KEEPALIVE_COUNT_MAX = 4
PROCESS_CLEANUP_GRACE_SECONDS = 5.0


class Sandbox:
    """A native asynchronous handle to a Thunder sandbox."""

    def __init__(
        self, client: Client, info: SandboxInfo, *, owns_client: bool = False
    ) -> None:
        self._client = client
        self._owns_client = owns_client
        self._info = info
        self._main_process: Process[str] | None = None
        self._ssh_manager = SSHConnectionManager(self._open_connection)

    @staticmethod
    async def create(
        *args: str,
        name: str | None = None,
        env: Mapping[str, str | None] | None = None,
        timeout: int | None = 300,
        cpu: int | None = None,
        memory: int | None = None,
        storage: int | None = None,
        gpu_type: GPUType | None = None,
        gpu_count: int | None = None,
        image: Image | None = None,
        block_network: bool = False,
        outbound_cidr_allowlist: Sequence[str] | None = None,
        outbound_domain_allowlist: Sequence[str] | None = None,
        client: Client | None = None,
    ) -> "Sandbox":
        """Create a sandbox, resolving an optional image before allocation.

        An image-backed sandbox starts a long-running container without running
        the image's ENTRYPOINT or CMD. Positional arguments start the first
        process through ``exec`` after the sandbox becomes ready.
        """
        _validate_create_options(
            timeout=timeout,
            gpu_type=gpu_type,
            gpu_count=gpu_count,
            image=image,
            block_network=block_network,
            outbound_cidr_allowlist=outbound_cidr_allowlist,
            outbound_domain_allowlist=outbound_domain_allowlist,
        )
        owns_client = client is None
        resolved_client = client or Client.from_cli()
        try:
            resolved_image = (
                await resolved_client.resolve_image(image) if image is not None else None
            )
            internet_access, cidrs, domains = _network_policy_request(
                block_network,
                outbound_cidr_allowlist,
                outbound_domain_allowlist,
            )
            request = {
                "spec": {
                    "cpu_count": cpu if cpu is not None else 4,
                    "memory_gib": memory if memory is not None else 32,
                    "storage_gib": storage if storage is not None else 50,
                    **(
                        {"gpu_type": gpu_type.value, "gpu_count": gpu_count}
                        if gpu_type is not None
                        else {}
                    ),
                },
                "env": {
                    key: value
                    for key, value in (env or {}).items()
                    if value is not None
                },
                "lifetime": {
                    "enforce_ttl": timeout is not None,
                    **({"max_ttl_seconds": timeout} if timeout is not None else {}),
                },
                "network_policy": {
                    "internet_access": internet_access,
                    "cidr_allowlist": cidrs,
                    "domain_allowlist": domains,
                },
                **({"name": name} if name is not None else {}),
                **(
                    {"image_id": resolved_image.id}
                    if resolved_image is not None
                    else {}
                ),
            }
        except BaseException:
            if owns_client:
                await resolved_client.close()
            raise
        sandbox_id: str | None = None
        try:
            response = await resolved_client._request("POST", "/sandboxes/start", request)
            sandbox_id = str(response.get("id", ""))
            if not sandbox_id:
                raise SandboxFailedError("Thunder did not return a sandbox ID")
            sandbox = await Sandbox.from_id(sandbox_id, client=resolved_client)
            sandbox._owns_client = owns_client
            if args:
                await sandbox.wait_until_ready(timeout=timeout)
                sandbox._main_process = await sandbox.exec(*args)
            return sandbox
        except BaseException:
            if sandbox_id:
                with suppress(BaseException):
                    await _stop_sandbox(
                        resolved_client,
                        sandbox_id,
                        deadline=time.monotonic() + OUTAGE_GRACE_SECONDS,
                    )
            if owns_client:
                await resolved_client.close()
            raise

    @staticmethod
    async def from_id(
        sandbox_id: str, *, client: Client | None = None
    ) -> "Sandbox":
        owns_client = client is None
        resolved_client = client or Client.from_cli()
        try:
            response = await resolved_client._request(
                "GET", f"/sandboxes/{_path_segment(sandbox_id)}"
            )
            sandbox = Sandbox._from_response(resolved_client, response)
            sandbox._owns_client = owns_client
            return sandbox
        except BaseException:
            if owns_client:
                await resolved_client.close()
            raise

    @staticmethod
    async def from_name(
        name: str, *, client: Client | None = None
    ) -> "Sandbox":
        owns_client = client is None
        resolved_client = client or Client.from_cli()
        try:
            matches = [
                sandbox
                async for sandbox in resolved_client.list_sandboxes(status="active")
                if sandbox.name == name
            ]
            if not matches:
                raise NotFoundError(f"no live sandbox is named {name!r}")
            if len(matches) > 1:
                raise ConflictError(
                    f"{len(matches)} live sandboxes are named {name!r}; address one by ID"
                )
            matches[0]._owns_client = owns_client
            return matches[0]
        except BaseException:
            if owns_client:
                await resolved_client.close()
            raise

    @staticmethod
    def _from_response(
        client: Client, response: dict[str, object]
    ) -> "Sandbox":
        return Sandbox(client, _info_from_response(client.config.paths, response))

    @property
    def id(self) -> str:
        return self._info.id

    @property
    def name(self) -> str:
        return self._info.name

    @property
    def status(self) -> SandboxStatus:
        return self._info.status

    @property
    def image_id(self) -> str | None:
        return self._info.image_id

    @property
    def info(self) -> SandboxInfo:
        return self._info

    @property
    def ssh(self) -> SSHConnection:
        if self._info.ssh is None:
            raise SandboxFailedError(
                "sandbox SSH connection details are not available"
            )
        return self._info.ssh

    @property
    def ssh_command(self) -> tuple[str, ...]:
        return self.ssh.command

    @overload
    async def exec(
        self,
        *args: str,
        timeout: float | None = None,
        workdir: str | None = None,
        env: Mapping[str, str | None] | None = None,
        text: Literal[True] = True,
        pty: bool = False,
        stdout: OutputMode = "capture",
        stderr: OutputMode = "capture",
        retain: bool = False,
    ) -> Process[str]: ...

    @overload
    async def exec(
        self,
        *args: str,
        timeout: float | None = None,
        workdir: str | None = None,
        env: Mapping[str, str | None] | None = None,
        text: Literal[False] = False,
        pty: bool = False,
        stdout: OutputMode = "capture",
        stderr: OutputMode = "capture",
        retain: bool = False,
    ) -> Process[bytes]: ...

    @overload
    async def exec(
        self,
        *args: str,
        timeout: float | None = None,
        workdir: str | None = None,
        env: Mapping[str, str | None] | None = None,
        text: bool,
        pty: bool = False,
        stdout: OutputMode = "capture",
        stderr: OutputMode = "capture",
        retain: bool = False,
    ) -> Process[str] | Process[bytes]: ...

    async def exec(
        self,
        *args: str,
        timeout: float | None = None,
        workdir: str | None = None,
        env: Mapping[str, str | None] | None = None,
        text: bool = True,
        pty: bool = False,
        stdout: OutputMode = "capture",
        stderr: OutputMode = "capture",
        retain: bool = False,
    ) -> Process[str] | Process[bytes]:
        if not args:
            raise InvalidRequestError("exec requires a command")
        if stdout not in {"capture", "discard"}:
            raise InvalidRequestError("stdout must be 'capture' or 'discard'")
        if stderr not in {"capture", "discard"}:
            raise InvalidRequestError("stderr must be 'capture' or 'discard'")
        if pty and (stdout != "capture" or stderr != "capture" or retain):
            raise InvalidRequestError(
                "stdout, stderr, and retain durable-job options require pty=False"
            )
        if not pty:
            spec, status = await self._launch_detached_job(
                args,
                workdir=workdir,
                env=env,
                stdout=stdout,
                stderr=stderr,
                retain=retain,
            )
            return self._durable_process(
                spec, status=status, timeout=timeout, text=text
            )
        command = (
            _container_remote_command(args, workdir=workdir, env=env, pty=pty)
            if self.image_id is not None
            else _remote_command(args, workdir=workdir, env=env)
        )
        return await self._create_process(
            command, timeout=timeout, text=text, pty=pty
        )

    @overload
    async def get_process(
        self, process_id: str, *, text: Literal[True] = True
    ) -> Process[str]: ...

    @overload
    async def get_process(
        self, process_id: str, *, text: Literal[False]
    ) -> Process[bytes]: ...

    @overload
    async def get_process(
        self, process_id: str, *, text: bool
    ) -> Process[str] | Process[bytes]: ...

    async def get_process(
        self, process_id: str, *, text: bool = True
    ) -> Process[str] | Process[bytes]:
        """Recover a durable process by its client-generated job ID."""

        try:
            validate_job_id(process_id)
        except ValueError as exc:
            raise InvalidRequestError(str(exc)) from exc
        spec = await self._read_job_spec(
            process_id, deadline=self._ssh_deadline()
        )
        status = await self._read_job_status(process_id, deadline=self._ssh_deadline())
        return self._durable_process(
            spec, status=status, timeout=None, text=text
        )

    async def _create_process(
        self,
        command: str,
        *,
        timeout: float | None,
        text: bool,
        pty: bool,
    ) -> Process[str] | Process[bytes]:
        connection = await self._connect()
        try:
            process = await connection.create_process(
                command,
                encoding="utf-8" if text else None,
                term_type="xterm" if pty else None,
            )
        except (OSError, asyncssh.Error) as exc:
            await self._discard_connection(connection)
            raise ConnectionError(f"could not open a sandbox SSH session: {exc}") from exc
        return Process(process, timeout=timeout, text=text)

    async def _launch_detached_job(
        self,
        args: Sequence[str],
        *,
        workdir: str | None,
        env: Mapping[str, str | None] | None,
        stdout: OutputMode = "capture",
        stderr: OutputMode = "capture",
        retain: bool = False,
    ) -> tuple[JobSpec, JobStatus]:
        """Stage and start one durable non-PTY job.

        The returned acknowledgement seeds the process status cache, avoiding
        an extra SSH round trip immediately after submission.
        """

        job_id = new_job_id()
        paths = RemoteJobPaths(job_id)
        spec = JobSpec(
            job_id,
            tuple(args),
            workdir=workdir,
            env=env,
            container=_CONTAINER_NAME if self.image_id is not None else None,
            stdout=stdout,
            stderr=stderr,
            retain=retain,
        )
        remote_command = (
            _container_remote_command(
                args, workdir=workdir, env=env, pty=False, job_id=job_id
            )
            if self.image_id is not None
            else _remote_command(args, workdir=workdir, env=env)
        )

        async def submit(
            connection: asyncssh.SSHClientConnection,
        ) -> asyncssh.SSHCompletedProcess[str]:
            # A fresh staging directory avoids colliding with a first remote
            # shell which may still be unwinding after its connection was lost.
            # The durable job ID and immutable contents remain identical.
            command = submission_command(spec, remote_command, paths)
            result = await connection.run(command, check=False, encoding="utf-8")
            if result.returncode is None or result.returncode == 75:
                raise RetryableSSHOperationError(
                    "durable job submission acknowledgement was lost"
                )
            return result

        try:
            result = await self._ssh_manager.run(
                submit,
                name=f"durable sandbox job {job_id} submission",
                deadline=self._ssh_deadline(),
            )
        except (OSError, asyncssh.Error) as exc:
            raise ConnectionError(
                f"could not submit a durable sandbox job: {exc}"
            ) from exc
        if result.returncode is None:
            raise AssertionError("SSH retry manager returned an ambiguous result")
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
        return spec, status

    def _durable_process(
        self,
        spec: JobSpec,
        *,
        status: JobStatus,
        timeout: float | None,
        text: bool,
    ) -> Process[str] | Process[bytes]:
        job_id = spec.job_id
        async def read_status(deadline: float | None) -> JobStatus:
            return await self._read_job_status(
                job_id, deadline=_earliest_deadline(deadline, self._ssh_deadline())
            )

        async def read_output(stream: str, offset: int, size: int) -> bytes:
            return await self._read_job_output(job_id, stream, offset=offset, size=size)

        async def cleanup_job() -> None:
            await self._cleanup_job(job_id, container=spec.container)

        async def signal_job(signal: str, pid: int) -> JobStatus:
            return await self._signal_job(
                job_id, signal=signal, pid=pid, container=spec.container
            )

        process = Process.durable(
            job_id,
            status=status,
            read_status=read_status,
            read_output=read_output,
            cleanup_job=cleanup_job,
            signal_job=signal_job,
            timeout=timeout,
            text=text,
            stdout=spec.stdout,
            stderr=spec.stderr,
            retain=spec.retain,
        )
        return cast(Process[str] | Process[bytes], process)

    async def _read_job_status(
        self, job_id: str, *, deadline: float | None
    ) -> JobStatus:
        paths = RemoteJobPaths(job_id)
        command = f"cat -- {shlex.quote(str(paths.status))}"

        result = await self._run_idempotent_command(
            command,
            name=f"durable sandbox job {job_id} status",
            deadline=deadline,
            check=False,
        )
        if result.returncode != 0:
            detail = str(result.stderr or "").strip()
            raise NotFoundError(
                f"durable sandbox job {job_id} was not found"
                + (f": {detail}" if detail else "")
            )
        return _decode_status(
            str(result.stdout or ""),
            context=f"durable sandbox job {job_id} has an invalid status",
        )

    async def _read_job_spec(
        self, job_id: str, *, deadline: float | None
    ) -> JobSpec:
        paths = RemoteJobPaths(job_id)
        command = f"cat -- {shlex.quote(str(paths.specification))}"

        result = await self._run_idempotent_command(
            command,
            name=f"durable sandbox job {job_id} specification",
            deadline=deadline,
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
            raise SandboxFailedError(
                f"durable sandbox job {job_id} has a mismatched specification"
            )
        return spec

    async def _read_job_output(
        self, job_id: str, stream: str, *, offset: int, size: int
    ) -> bytes:
        paths = RemoteJobPaths(job_id)
        if stream not in {"stdout", "stderr"}:
            raise InvalidRequestError(f"invalid durable job stream: {stream!r}")
        path = paths.stdout if stream == "stdout" else paths.stderr

        async def read(
            connection: asyncssh.SSHClientConnection,
        ) -> bytes:
            try:
                async with connection.start_sftp_client() as sftp:
                    async with sftp.open(str(path), "rb") as remote:
                        value = await remote.read(size, offset)
            except asyncssh.SFTPNoSuchFile as exc:
                raise SandboxFailedError(
                    f"durable sandbox job {job_id} {stream} is unavailable"
                ) from exc
            return bytes(value)

        return await self._ssh_manager.run(
            read,
            name=f"durable sandbox job {job_id} {stream} at byte {offset}",
            deadline=self._ssh_deadline(),
        )

    async def _cleanup_job(self, job_id: str, *, container: str | None = None) -> None:
        paths = RemoteJobPaths(job_id)
        command = cleanup_command(
            paths, container=container, container_busybox=_CONTAINER_BUSYBOX
        )

        result = await self._run_idempotent_command(
            command,
            name=f"durable sandbox job {job_id} cleanup",
            deadline=self._ssh_deadline(),
            check=False,
        )
        if result.returncode != 0:
            detail = str(result.stderr or "").strip()
            raise SandboxFailedError(
                f"could not clean up durable sandbox job {job_id}"
                + (f": {detail}" if detail else "")
            )

    async def _signal_job(
        self,
        job_id: str,
        *,
        signal: str,
        pid: int,
        container: str | None = None,
    ) -> JobStatus:
        if signal not in {"TERM", "KILL"}:
            raise InvalidRequestError(f"invalid durable job signal: {signal!r}")
        if pid <= 0:
            raise InvalidRequestError("durable job PID must be positive")
        paths = RemoteJobPaths(job_id)
        command = signal_command(
            paths,
            signal=signal,
            pid=pid,
            container=container,
            container_busybox=_CONTAINER_BUSYBOX,
        )

        result = await self._run_idempotent_command(
            command,
            name=f"durable sandbox job {job_id} SIG{signal}",
            deadline=self._ssh_deadline(),
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
            context=(
                f"durable sandbox job {job_id} returned invalid status after "
                f"SIG{signal}"
            ),
        )

    def _ssh_deadline(self) -> float | None:
        expires_at = self._info.expires_at
        if expires_at is None:
            return None
        remaining = (expires_at - datetime.now(expires_at.tzinfo)).total_seconds()
        return time.monotonic() + max(0.0, remaining)

    async def upload(
        self,
        local_path: str | os.PathLike[str],
        remote_path: str,
        *,
        recursive: bool = False,
    ) -> None:
        source = os.fspath(local_path)
        parsed_source = _parse_transfer_path(source, separator=os.sep)
        source_path = Path(parsed_source.path)
        if not source_path.exists():
            raise InvalidRequestError(f"upload source does not exist: {source}")
        if source_path.is_dir() and not recursive:
            raise InvalidRequestError("uploading a directory requires recursive=True")
        if self.image_id is not None:
            await self._upload_to_container(local_path, remote_path, recursive=recursive)
            return
        if parsed_source.contents_only:
            await self._upload_guest_contents(source_path, remote_path)
            return
        source_name = parsed_source.name
        if not source_name:
            raise InvalidRequestError("upload source must have a file or directory name")
        target = await self._resolve_remote_upload_target(remote_path, source_name)
        transfer_id = uuid.uuid4().hex
        stage = f"{target}.thunder-transfer-{transfer_id}.stage"
        backup = f"{target}.thunder-transfer-{transfer_id}.backup"
        try:
            async def transfer(connection: asyncssh.SSHClientConnection) -> None:
                await self._reset_remote_transfer_path(connection, stage)
                await asyncssh.scp(source, (connection, stage), recurse=recursive)

            await self._retry_scp(
                transfer,
                name=f"upload {source_name} to sandbox {self.id}",
            )
            await self._publish_remote_transfer(stage, target, backup)
        finally:
            await self._cleanup_remote_transfer_paths(stage, backup)

    async def download(
        self,
        remote_path: str,
        local_path: str | os.PathLike[str],
        *,
        recursive: bool = False,
    ) -> None:
        if self.image_id is not None:
            await self._download_from_container(
                remote_path, local_path, recursive=recursive
            )
            return
        await self._download_from_guest(remote_path, local_path, recursive=recursive)

    async def _download_from_guest(
        self,
        remote_path: str,
        local_path: str | os.PathLike[str],
        *,
        recursive: bool,
    ) -> None:
        if not remote_path or "\x00" in remote_path:
            raise InvalidRequestError("download source must be a non-empty remote path")
        parsed_source = _parse_transfer_path(remote_path, separator="/")
        contents_only = parsed_source.contents_only
        source_name = parsed_source.name
        if not source_name:
            raise InvalidRequestError("download source must have a file or directory name")
        destination = Path(local_path)
        target = (
            destination
            if contents_only
            else destination / source_name if destination.is_dir() else destination
        )
        transfer_root = Path(
            tempfile.mkdtemp(
                prefix=f".{target.name}.thunder-transfer-",
                dir=os.fspath(target.parent),
            )
        )
        stage = transfer_root / ("contents" if contents_only else source_name)
        try:
            async def transfer(connection: asyncssh.SSHClientConnection) -> None:
                await asyncio.to_thread(_remove_local_transfer_path, stage)
                await asyncssh.scp(
                    (connection, remote_path), os.fspath(stage), recurse=recursive
                )

            await self._retry_scp(
                transfer,
                name=f"download {remote_path} from sandbox {self.id}",
            )
            publisher = (
                _publish_local_contents if contents_only else _publish_local_transfer
            )
            await asyncio.to_thread(publisher, stage, target)
        finally:
            await asyncio.to_thread(shutil.rmtree, transfer_root, True)

    async def _retry_scp(
        self,
        transfer: Callable[[asyncssh.SSHClientConnection], Awaitable[None]],
        *,
        name: str,
    ) -> None:
        """Restart a complete SCP transfer after a transient connection loss."""

        async def attempt(connection: asyncssh.SSHClientConnection) -> None:
            try:
                await transfer(connection)
            except (
                asyncssh.SFTPFailure,
                asyncssh.SFTPNoSuchFile,
                asyncssh.SFTPNoSuchPath,
                asyncssh.SFTPPermissionDenied,
            ) as exc:
                raise SandboxFailedError(f"could not complete {name}: {exc}") from exc
            except OSError as exc:
                if not connection.is_closed():
                    raise SandboxFailedError(
                        f"could not complete {name}: {exc}"
                    ) from exc
                raise

        try:
            await self._ssh_manager.run(
                attempt, name=name, deadline=self._ssh_deadline()
            )
        except ConnectionError:
            raise
        except (OSError, asyncssh.Error) as exc:
            raise SandboxFailedError(f"could not complete {name}: {exc}") from exc

    async def _reset_remote_transfer_path(
        self,
        connection: asyncssh.SSHClientConnection,
        path: str,
        *,
        directory: bool = False,
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
                "could not reset remote transfer staging"
                + (f": {detail}" if detail else "")
            )

    async def _resolve_remote_upload_target(
        self, remote_path: str, source_name: str
    ) -> str:
        if not remote_path or "\x00" in remote_path:
            raise InvalidRequestError("upload destination must be a non-empty remote path")
        destination = shlex.quote(remote_path)
        command = f"if [ -d {destination} ]; then printf directory; else printf exact; fi"
        result = await self._run_idempotent_command(
            command, name=f"inspect upload destination in sandbox {self.id}"
        )
        if str(result.stdout or "") == "directory":
            return posixpath.join(remote_path, source_name)
        return remote_path

    async def _upload_guest_contents(
        self, source_directory: Path, remote_path: str
    ) -> None:
        if not remote_path or "\x00" in remote_path:
            raise InvalidRequestError("upload destination must be a non-empty remote path")
        transfer_id = uuid.uuid4().hex
        destination = remote_path.rstrip("/") or "/"
        stage = f"{destination}.thunder-transfer-{transfer_id}.stage"

        try:
            await self._upload_directory_entries(
                source_directory,
                stage,
                name=f"upload directory contents to sandbox {self.id}",
            )
            command = (
                f"mkdir -p -- {shlex.quote(remote_path)} && "
                f"cp -a -- {shlex.quote(stage + '/.')} {shlex.quote(remote_path + '/')}"
            )
            await self._run_idempotent_command(
                command, name=f"publish directory upload in sandbox {self.id}"
            )
        finally:
            await self._cleanup_remote_transfer_paths(stage)

    async def _upload_directory_entries(
        self, source_directory: Path, stage: str, *, name: str
    ) -> None:
        async def transfer(connection: asyncssh.SSHClientConnection) -> None:
            await self._reset_remote_transfer_path(connection, stage, directory=True)
            for child in source_directory.iterdir():
                await asyncssh.scp(
                    os.fspath(child), (connection, stage + "/"), recurse=True
                )

        await self._retry_scp(transfer, name=name)

    async def _publish_remote_transfer(
        self, stage: str, target: str, backup: str
    ) -> None:
        quoted_stage = shlex.quote(stage)
        quoted_target = shlex.quote(target)
        quoted_backup = shlex.quote(backup)
        command = f"""set -eu
stage={quoted_stage}
target={quoted_target}
backup={quoted_backup}
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
        await self._run_idempotent_command(
            command, name=f"publish upload in sandbox {self.id}"
        )

    async def _cleanup_remote_transfer_paths(self, *paths: str) -> None:
        command = "rm -rf -- " + " ".join(shlex.quote(path) for path in paths)
        deadline = _earliest_deadline(
            time.monotonic() + PROCESS_CLEANUP_GRACE_SECONDS, self._ssh_deadline()
        )
        with suppress(Exception):
            await self._run_idempotent_command(
                command,
                name=f"clean up transfer staging in sandbox {self.id}",
                deadline=deadline,
            )

    async def _run_idempotent_command(
        self,
        command: str,
        *,
        name: str,
        deadline: float | None = None,
        retry_exit_codes: Container[int] = (),
        check: bool = True,
    ) -> asyncssh.SSHCompletedProcess[str]:
        async def run(
            connection: asyncssh.SSHClientConnection,
        ) -> asyncssh.SSHCompletedProcess[str]:
            result = await connection.run(command, check=False, encoding="utf-8")
            if (
                result.returncode is None
                or result.returncode in retry_exit_codes
            ):
                raise RetryableSSHOperationError(f"{name} acknowledgement was lost")
            return result

        result = await self._ssh_manager.run(
            run,
            name=name,
            deadline=self._ssh_deadline() if deadline is None else deadline,
        )
        if check and result.returncode != 0:
            detail = str(result.stderr or "").strip()
            raise SandboxFailedError(
                f"{name} failed with exit code {result.returncode}"
                + (f": {detail}" if detail else "")
            )
        return result

    async def _upload_to_container(
        self,
        local_path: str | os.PathLike[str],
        remote_path: str,
        *,
        recursive: bool,
    ) -> None:
        stage = f"/tmp/thunder-sandbox-transfer-{uuid.uuid4().hex}"
        try:
            raw_source = os.fspath(local_path)
            parsed_source = _parse_transfer_path(raw_source, separator=os.sep)
            contents_only = parsed_source.contents_only
            if contents_only:
                source_directory = Path(parsed_source.path)
                await self._upload_directory_entries(
                    source_directory,
                    stage,
                    name=f"upload directory contents to sandbox {self.id}",
                )
                guest_source = stage + "/."
            else:
                async def transfer_path(
                    connection: asyncssh.SSHClientConnection,
                ) -> None:
                    await self._reset_remote_transfer_path(
                        connection, stage, directory=True
                    )
                    await asyncssh.scp(
                        raw_source,
                        (connection, stage + "/"),
                        recurse=recursive,
                    )

                await self._retry_scp(
                    transfer_path,
                    name=f"upload {parsed_source.name} to sandbox {self.id}",
                )
                guest_source = stage + "/" + parsed_source.name
            await self._run_guest_command(
                "sudo",
                "--non-interactive",
                "docker",
                "cp",
                guest_source,
                f"{_CONTAINER_NAME}:{remote_path}",
            )
        except (OSError, asyncssh.Error) as exc:
            raise SandboxFailedError(f"could not upload to sandbox container: {exc}") from exc
        finally:
            await self._cleanup_remote_transfer_paths(stage)

    async def _download_from_container(
        self,
        remote_path: str,
        local_path: str | os.PathLike[str],
        *,
        recursive: bool,
    ) -> None:
        stage = f"/tmp/thunder-sandbox-transfer-{uuid.uuid4().hex}"
        await self._run_guest_command("mkdir", "-p", "--", stage)
        try:
            parsed_source = _parse_transfer_path(remote_path, separator="/")
            contents_only = parsed_source.contents_only
            await self._run_guest_command(
                "sudo",
                "--non-interactive",
                "docker",
                "cp",
                f"{_CONTAINER_NAME}:{remote_path}",
                stage + "/",
            )
            guest_source = (
                stage + "/."
                if contents_only
                else stage + "/" + parsed_source.name
            )
            await self._download_from_guest(
                guest_source, local_path, recursive=recursive
            )
        except (OSError, asyncssh.Error) as exc:
            raise SandboxFailedError(
                f"could not download from sandbox container: {exc}"
            ) from exc
        finally:
            await self._cleanup_remote_transfer_paths(stage)

    async def _run_guest_command(self, *args: str) -> None:
        await self._run_idempotent_command(
            _remote_command(args, workdir=None, env=None),
            name=f"sandbox {self.id} guest command",
        )

    async def update_network_policy(
        self,
        *,
        block_network: bool = False,
        outbound_cidr_allowlist: Sequence[str] | None = None,
        outbound_domain_allowlist: Sequence[str] | None = None,
    ) -> None:
        """Replace this running sandbox's outbound network policy.

        The call returns once Thunder accepts the desired policy. Enforcement on
        the sandbox's node converges asynchronously. ``None`` leaves an allowlist
        dimension unrestricted; an empty sequence blocks that dimension.
        """
        _validate_network_policy_options(
            block_network=block_network,
            outbound_cidr_allowlist=outbound_cidr_allowlist,
            outbound_domain_allowlist=outbound_domain_allowlist,
        )
        internet_access, cidrs, domains = _network_policy_request(
            block_network,
            outbound_cidr_allowlist,
            outbound_domain_allowlist,
        )
        response = await self._client._request(
            "PATCH",
            f"/sandboxes/{_path_segment(self.id)}/network-policy",
            {
                "network_policy": {
                    "internet_access": internet_access,
                    "cidr_allowlist": cidrs,
                    "domain_allowlist": domains,
                }
            },
        )
        policy = response.get("network_policy")
        if not isinstance(policy, dict) or not policy.get("internet_access"):
            raise SandboxFailedError(
                "Thunder did not return the accepted sandbox network policy"
            )
        self._info = replace(
            self._info,
            network_policy=_network_policy_from_response(policy),
        )

    async def _connect(self) -> asyncssh.SSHClientConnection:
        try:
            return await self._ssh_manager.connect(
                name=f"sandbox {self.id} SSH connection"
            )
        except (OSError, asyncssh.Error) as exc:
            raise ConnectionError(
                f"could not connect to sandbox over SSH: {exc}"
            ) from exc

    async def _open_connection(self) -> asyncssh.SSHClientConnection:
        ssh = self.ssh
        # The node reuses forwarded ports across sandboxes, so a pin keyed by
        # host and port would reject the next sandbox that lands on a finished
        # one's port. Pin per sandbox instead.
        pinned = _pinned_host_key(self.id)
        known_hosts = ([pinned], [], []) if pinned is not None else None
        credential = await self._client._credentials.ensure(self._client)
        try:
            connection = await self._open(ssh, credential, known_hosts)
        except asyncssh.PermissionDenied:
            # An unexpired certificate may belong to another environment or a
            # rotated authority. Renew the rejected credential exactly once.
            credential = await self._client._credentials.renew(
                self._client, rejected=credential
            )
            connection = await self._open(ssh, credential, known_hosts)
        if pinned is None:
            try:
                _remember_host_key(self.id, connection.get_server_host_key())
            except BaseException:
                connection.close()
                await connection.wait_closed()
                raise
        return connection

    async def _open(
        self,
        ssh: SSHConnection,
        credential: "credentials.SSHCredential",
        known_hosts: object,
    ) -> asyncssh.SSHClientConnection:
        return await asyncssh.connect(
            ssh.host,
            ssh.port,
            username=ssh.user,
            # The sandbox trusts the authority that signed this, not the key
            # itself, so the same credential opens every sandbox the
            # organization owns.
            client_keys=[(credential.key, credential.certificate)],
            known_hosts=known_hosts,
            agent_path=None,
            preferred_auth=["publickey"],
            config=None,
            connect_timeout=SSH_CONNECT_TIMEOUT_SECONDS,
            # A command that writes to a file rather than the terminal sends
            # nothing over the channel, so a long build looks idle and is cut
            # by a NAT or idle timeout. wait() then reports a closed channel
            # with no exit status. Keep the connection warm and notice a dead
            # peer within a minute instead of blocking on it forever.
            keepalive_interval=SSH_KEEPALIVE_INTERVAL_SECONDS,
            keepalive_count_max=SSH_KEEPALIVE_COUNT_MAX,
        )

    async def _discard_connection(
        self, connection: asyncssh.SSHClientConnection
    ) -> None:
        await self._ssh_manager.discard(connection)

    async def _close_connection(self) -> None:
        await self._ssh_manager.close()

    async def refresh(self) -> "Sandbox":
        response = await self._client._request(
            "GET", f"/sandboxes/{_path_segment(self.id)}"
        )
        self._info = _info_from_response(self._client.config.paths, response)
        return self

    async def poll(self) -> int | None:
        if self._main_process is not None:
            return await self._main_process.poll()
        await self.refresh()
        if self.status == SandboxStatus.FINISHED:
            return 0
        if self.status == SandboxStatus.FAILED:
            return 1
        return None

    async def _refresh_while_waiting(
        self, deadline: float | None, failing_since: float | None
    ) -> tuple[bool, float | None, float | None]:
        try:
            await self.refresh()
        except (ConnectionError, RetryableError) as exc:
            now = time.monotonic()
            started = now if failing_since is None else failing_since
            if now - started >= OUTAGE_GRACE_SECONDS:
                raise
            if deadline is not None and now >= deadline:
                raise
            return False, started, exc.retry_after
        return True, None, None

    async def wait(self, *, timeout: float | None = None) -> int | None:
        if self._main_process is not None:
            try:
                return await self._main_process.wait(timeout=timeout)
            except asyncio.TimeoutError as exc:
                raise SandboxTimeoutError(
                    f"sandbox command did not finish within {timeout} seconds"
                ) from exc
        deadline = None if timeout is None else time.monotonic() + timeout
        failing_since: float | None = None
        poll_delay = 1.0
        while True:
            ok, failing_since, retry_after = await self._refresh_while_waiting(
                deadline, failing_since
            )
            if ok:
                if self.status == SandboxStatus.FINISHED:
                    return 0
                if self.status == SandboxStatus.FAILED:
                    return 1
            if deadline is not None and time.monotonic() >= deadline:
                raise SandboxTimeoutError(
                    f"sandbox {self.id} did not stop within {timeout} seconds"
                )
            await _sleep_until_next_poll(deadline, poll_delay, retry_after)
            poll_delay = min(5.0, poll_delay * 2.0)

    async def wait_until_ready(
        self, *, timeout: float | None = 300
    ) -> "Sandbox":
        deadline = None if timeout is None else time.monotonic() + timeout
        if not await self._wait_for_startup(deadline):
            raise SandboxTimeoutError(
                f"sandbox {self.id} did not become ready within {timeout} seconds"
            )
        if self.status == SandboxStatus.READY:
            return self
        raise SandboxFailedError(
            f"sandbox {self.id} did not become ready (status: {self.status.value})"
        )

    async def _wait_for_startup(self, deadline: float | None) -> bool:
        """Read the sandbox until it has left ``created``.

        Prefers the API's blocking wait, which answers the moment the sandbox
        is ready, and polls when the API predates it. Every attempt is bounded
        on this side: each wait request carries its own window and HTTP
        timeout, faults retry with backoff inside the outage grace period, and
        ``deadline`` caps the whole. Returns ``False`` when the deadline passes
        with the sandbox still starting; the caller owns the message.
        """

        def expired() -> bool:
            return deadline is not None and time.monotonic() >= deadline

        failing_since: float | None = None
        delay = 1.0
        while True:
            try:
                await self._read_startup_state(deadline)
            except _WaitWindowElapsedError:
                # A full server-side window passed with the sandbox still
                # starting. That is the wait's normal answer, not a fault, so
                # open the next window at once: a pause here would be the one
                # place readiness could arrive unnoticed.
                failing_since = None
                delay = 1.0
                if expired():
                    return False
            except (ConnectionError, RetryableError) as exc:
                now = time.monotonic()
                failing_since = now if failing_since is None else failing_since
                if now - failing_since >= OUTAGE_GRACE_SECONDS:
                    raise
                if deadline is not None and now >= deadline:
                    raise
                await _sleep_until_next_poll(deadline, delay, exc.retry_after)
                delay = min(5.0, delay * 2.0)
            else:
                failing_since = None
                if self.status != SandboxStatus.CREATED:
                    return True
                if expired():
                    return False
                # Only a poll answers while the sandbox is still starting, so
                # pace the next one. The sleep stops at the deadline and one
                # last read follows it, so readiness arriving during the
                # pause is still seen.
                await _sleep_until_next_poll(deadline, delay)
                delay = min(5.0, delay * 2.0)

    async def _read_startup_state(self, deadline: float | None) -> None:
        """Refresh once, holding the request open server-side where the API allows.

        Raises ``_WaitWindowElapsedError`` when the sandbox is still starting
        at the end of the window.
        """
        client = self._client
        if not client._wait_endpoint_available:
            await self.refresh()
            return
        window = _wait_window(deadline)
        try:
            response = await client._request(
                "GET",
                f"/sandboxes/{_path_segment(self.id)}/wait",
                query={"timeout_seconds": window},
                timeout=window + WAIT_REPLY_GRACE_SECONDS,
            )
        except NotFoundError:
            # An API without the endpoint answers 404 exactly as it does for
            # a missing sandbox, so let a plain read settle it: that raises
            # the same NotFoundError if the sandbox is gone, and otherwise
            # doubles as the first poll of the fallback.
            await self.refresh()
            client._wait_endpoint_available = False
            return
        self._info = _info_from_response(client.config.paths, response)

    async def terminate(self, *, timeout: float | None = 300) -> None:
        try:
            deadline = None if timeout is None else time.monotonic() + timeout
            if self.status == SandboxStatus.CREATED:
                if not await self._wait_for_startup(deadline):
                    raise SandboxTimeoutError(
                        f"sandbox {self.id} did not become ready to stop within {timeout} seconds"
                    )
            if self.status.terminal:
                return
            await _stop_sandbox(self._client, self.id, deadline=deadline)
            await self.refresh()
        finally:
            await self._close_connection()
            if self._owns_client:
                await self._client.close()


def _path_segment(value: str) -> str:
    if not value:
        raise InvalidRequestError("sandbox ID cannot be empty")
    return quote(value, safe="")


def _earliest_deadline(first: float | None, second: float | None) -> float | None:
    if first is None:
        return second
    if second is None:
        return first
    return min(first, second)


def _decode_status(value: str, *, context: str) -> JobStatus:
    try:
        return JobStatus.from_json(value.strip())
    except ValueError as exc:
        raise SandboxFailedError(context) from exc


@overload
def _datetime(value: object, optional: Literal[False] = False) -> datetime: ...


@overload
def _datetime(value: object, optional: Literal[True]) -> datetime | None: ...


def _datetime(value: object, optional: bool = False) -> datetime | None:
    if value in (None, ""):
        return None if optional else datetime.fromtimestamp(0).astimezone()
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def _info_from_response(paths: ThunderPaths, response: dict[str, object]) -> SandboxInfo:
    sandbox_id = str(response.get("id", ""))
    name = str(response.get("name", ""))
    if not sandbox_id:
        raise SandboxFailedError("Thunder did not return a sandbox ID")
    spec_value = response.get("spec")
    spec: dict[str, object] = spec_value if isinstance(spec_value, dict) else {}
    policy_value = response.get("network_policy")
    gpu_type = GPUType(str(spec["gpu_type"])) if spec.get("gpu_type") else None
    ssh_value = response.get("ssh")
    ssh = None
    if isinstance(ssh_value, dict) and ssh_value.get("host"):
        ssh = SSHConnection(
            host=str(ssh_value["host"]),
            port=int(ssh_value.get("port", 22)),
            user=str(ssh_value.get("user", "ubuntu")),
            private_key_path=paths.ssh_key,
            certificate_path=paths.ssh_certificate,
        )
        host_key = ssh_value.get("host_key")
        if host_key:
            _remember_host_key(sandbox_id, str(host_key))
    return SandboxInfo(
        id=sandbox_id,
        name=name,
        status=SandboxStatus(str(response.get("status", "created"))),
        resources=Resources(
            cpu=int(str(spec.get("cpu_count", 0))),
            memory=int(str(spec.get("memory_gib", 0))),
            storage=int(str(spec.get("storage_gib", 0))),
            gpu_type=gpu_type,
            gpu_count=int(str(spec.get("gpu_count", 0))),
        ),
        network_policy=_network_policy_from_response(policy_value),
        created_at=_datetime(response.get("created_at")),
        expires_at=_datetime(response.get("expires_at"), optional=True),
        ssh=ssh,
        failure_code=(
            str(response["failure_code"]) if response.get("failure_code") else None
        ),
        failure=str(response["failure"]) if response.get("failure") else None,
        image_id=str(response["image_id"]) if response.get("image_id") else None,
    )


def _string_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(str(item) for item in value)


def _network_policy_from_response(value: object) -> NetworkPolicy:
    policy: dict[str, object] = value if isinstance(value, dict) else {}
    return NetworkPolicy(
        internet_access=str(policy.get("internet_access", "closed")),
        outbound_cidr_allowlist=_string_tuple(policy.get("cidr_allowlist")),
        outbound_domain_allowlist=_string_tuple(policy.get("domain_allowlist")),
    )


def _known_host_name(ssh: SSHConnection) -> str:
    return ssh.host if ssh.port == 22 else f"[{ssh.host}]:{ssh.port}"
# Host keys are remembered only for the life of this process. A sandbox is
# short-lived and nothing verifies the key on the first connection, so a file
# would add staleness and an unwritable-home failure mode without buying any
# trust the process does not already have.
_REMEMBERED_HOST_KEYS: dict[str, str] = {}


def _host_key_text(key: object) -> str:
    if hasattr(key, "export_public_key"):
        exported = key.export_public_key()  # type: ignore[union-attr]
        text = exported.decode("ascii") if isinstance(exported, bytes) else str(exported)
    else:
        text = str(key)
    text = text.strip()
    if not text:
        raise SandboxFailedError("Thunder returned an empty SSH host key")
    try:
        return asyncssh.import_public_key(text).export_public_key().decode("ascii").strip()
    except (asyncssh.Error, UnicodeError, ValueError) as exc:
        raise SandboxFailedError("Thunder returned an invalid SSH host key") from exc


def _pinned_host_key(sandbox_id: str) -> "asyncssh.SSHKey | None":
    text = _REMEMBERED_HOST_KEYS.get(sandbox_id)
    if text is None:
        return None
    try:
        return asyncssh.import_public_key(text)
    except (asyncssh.Error, UnicodeError, ValueError):
        return None


def _remember_host_key(sandbox_id: str, key: object) -> None:
    _REMEMBERED_HOST_KEYS[sandbox_id] = _host_key_text(key)


def _validate_create_options(
    *,
    timeout: int | None,
    gpu_type: GPUType | None,
    gpu_count: int | None,
    image: Image | None,
    block_network: bool,
    outbound_cidr_allowlist: Sequence[str] | None,
    outbound_domain_allowlist: Sequence[str] | None,
) -> None:

    if timeout is not None and timeout < 0:
        raise InvalidRequestError("timeout cannot be negative")
    if gpu_type is not None and not isinstance(gpu_type, GPUType):
        raise InvalidRequestError("gpu_type must be a GPUType")
    if (gpu_type is None) != (gpu_count is None):
        raise InvalidRequestError("gpu_type and gpu_count must be provided together")
    if gpu_count is not None and gpu_count not in (1, 2, 4, 8):
        raise InvalidRequestError("gpu_count must be one of 1, 2, 4, or 8")
    if image is not None and not isinstance(image, Image):
        raise InvalidRequestError("image must be an Image")
    _validate_network_policy_options(
        block_network=block_network,
        outbound_cidr_allowlist=outbound_cidr_allowlist,
        outbound_domain_allowlist=outbound_domain_allowlist,
    )


def _validate_network_policy_options(
    *,
    block_network: bool,
    outbound_cidr_allowlist: Sequence[str] | None,
    outbound_domain_allowlist: Sequence[str] | None,
) -> None:
    if block_network and (
        outbound_cidr_allowlist is not None or outbound_domain_allowlist is not None
    ):
        raise InvalidRequestError(
            "network allowlists cannot be combined with block_network"
        )


def _network_policy_request(
    block_network: bool,
    outbound_cidr_allowlist: Sequence[str] | None,
    outbound_domain_allowlist: Sequence[str] | None,
) -> tuple[str, list[str], list[str]]:
    if block_network:
        return "closed", [], []
    if outbound_cidr_allowlist is None and outbound_domain_allowlist is None:
        return "open", [], []
    cidrs = (
        list(outbound_cidr_allowlist)
        if outbound_cidr_allowlist is not None
        else ["0.0.0.0/0"]
    )
    domains = (
        list(outbound_domain_allowlist)
        if outbound_domain_allowlist is not None
        else ["*"]
    )
    return "restricted", cidrs, domains


def _wait_window(deadline: float | None) -> float:
    """Seconds to ask the API to hold one readiness wait open.

    The client deadline, not the server's maximum, is the binding bound once
    it is nearer. Rounded to the millisecond so the query string stays plain,
    and never below one: the API rejects a window of zero, and a deadline that
    has just passed is reported by the caller, not by a 400.
    """
    window = WAIT_WINDOW_MAX_SECONDS
    if deadline is not None:
        window = min(window, deadline - time.monotonic())
    window = max(0.001, round(window, 3))
    assert 0.0 < window <= WAIT_WINDOW_MAX_SECONDS, window
    return window


async def _sleep_until_next_poll(
    deadline: float | None, delay: float, retry_after: float | None = None
) -> None:
    delay = max(delay, retry_after or 0.0)
    delay += random.uniform(0.0, min(delay * 0.2, 1.0))
    if deadline is not None:
        delay = max(0.0, min(delay, deadline - time.monotonic()))
    await asyncio.sleep(delay)


async def _stop_sandbox(
    client: Client, sandbox_id: str, *, deadline: float | None
) -> None:
    failing_since: float | None = None
    delay = 1.0
    while True:
        try:
            await client._request(
                "POST", f"/sandboxes/{_path_segment(sandbox_id)}/stop"
            )
            return
        except (ConnectionError, RetryableError) as exc:
            now = time.monotonic()
            failing_since = now if failing_since is None else failing_since
            if now - failing_since >= OUTAGE_GRACE_SECONDS:
                raise
            if deadline is not None and now >= deadline:
                raise
            await _sleep_until_next_poll(deadline, delay, exc.retry_after)
            delay = min(5.0, delay * 2.0)


def _remote_command(
    args: Sequence[str],
    *,
    workdir: str | None,
    env: Mapping[str, str | None] | None,
) -> str:
    parts: list[str] = []
    if workdir is not None:
        parts.extend(("cd", shlex.quote(workdir), "&&"))
    if env:
        parts.append("env")
        for key, value in env.items():
            if value is None:
                parts.extend(("-u", shlex.quote(key)))
            else:
                parts.append(f"{shlex.quote(key)}={shlex.quote(value)}")
    parts.extend(shlex.quote(arg) for arg in args)
    return " ".join(parts)


def _container_remote_command(
    args: Sequence[str],
    *,
    workdir: str | None,
    env: Mapping[str, str | None] | None,
    pty: bool,
    job_id: str | None = None,
) -> str:
    parts = ["sudo", "--non-interactive", "docker", "exec", "--interactive"]
    if pty:
        parts.append("--tty")
    if workdir is not None:
        parts.extend(("--workdir", workdir))
    unset: list[str] = []
    for key, value in (env or {}).items():
        if value is None:
            unset.append(key)
        else:
            parts.extend(("--env", f"{key}={value}"))
    parts.append(_CONTAINER_NAME)
    payload: list[str] = []
    if unset:
        payload.extend((_CONTAINER_BUSYBOX, "env"))
        for key in unset:
            payload.extend(("-u", key))
    payload.extend(args)
    if job_id is None:
        parts.extend(payload)
    else:
        validate_job_id(job_id)
        container_job = str(container_job_directory(job_id))
        inner = (
            "set -eu\n"
            f"job={shlex.quote(container_job)}\n"
            'mkdir -p -- "$job"\n'
            'printf \'%s\\n\' "$$" >"$job/pid"\n'
            'exec "$@"\n'
        )
        parts.extend(
            (_CONTAINER_BUSYBOX, "setsid", _CONTAINER_BUSYBOX, "sh", "-c", inner, "sh")
        )
        parts.extend(payload)
    return " ".join(shlex.quote(part) for part in parts)
__all__ = ["Sandbox"]
