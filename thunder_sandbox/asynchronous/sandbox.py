"""Native asynchronous sandbox lifecycle, SSH execution, and transfers."""

from __future__ import annotations

import asyncio
import os
import shlex
import tempfile
import time
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Literal, overload
from urllib.parse import quote

import asyncssh

from .client import Client
from .process import Process
from .._common.config import ThunderPaths
from .._common.exceptions import (
    ConflictError,
    ConnectionError,
    InvalidRequestError,
    NotFoundError,
    SandboxFailedError,
    SandboxTimeoutError,
)
from .._common.types import (
    GPUType,
    NetworkPolicy,
    Resources,
    SandboxInfo,
    SandboxStatus,
    SSHConnection,
)

OUTAGE_GRACE_SECONDS = 30.0


class Sandbox:
    """A native asynchronous handle to an immutable Thunder sandbox."""

    def __init__(self, client: Client, info: SandboxInfo) -> None:
        self._client = client
        self._info = info
        self._main_process: Process[str] | None = None
        self._connection: asyncssh.SSHClientConnection | None = None
        self._connection_lock = asyncio.Lock()

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
        block_network: bool = False,
        outbound_cidr_allowlist: Sequence[str] | None = None,
        outbound_domain_allowlist: Sequence[str] | None = None,
        ssh_public_key: str | None = None,
        ssh_private_key: str | None = None,
        client: Client | None = None,
    ) -> "Sandbox":
        _validate_create_options(
            timeout=timeout,
            gpu_type=gpu_type,
            gpu_count=gpu_count,
            block_network=block_network,
            outbound_cidr_allowlist=outbound_cidr_allowlist,
            outbound_domain_allowlist=outbound_domain_allowlist,
            ssh_public_key=ssh_public_key,
            ssh_private_key=ssh_private_key,
        )
        resolved_client = client or Client.from_cli()
        paths = resolved_client.config.paths
        paths.sandbox_keys.mkdir(mode=0o700, parents=True, exist_ok=True)
        temporary_directory: tempfile.TemporaryDirectory[str] | None = None
        if ssh_public_key is None:
            temporary_directory = tempfile.TemporaryDirectory(
                prefix=".creating-", dir=paths.sandbox_keys
            )
            temporary_key = Path(temporary_directory.name) / "key"
            await _generate_key_pair(temporary_key)
            public_key = temporary_key.with_suffix(".pub").read_text(
                encoding="utf-8"
            ).strip()
            private_key_source = temporary_key
            public_key_source = temporary_key.with_suffix(".pub")
        else:
            public_key = ssh_public_key.strip()
            if not public_key:
                raise InvalidRequestError("ssh_public_key cannot be empty")
            private_key_source = public_key_source = None

        internet_access, cidrs, domains = _network_policy(
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
                key: value for key, value in (env or {}).items() if value is not None
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
            "ssh_public_key": public_key,
            **({"name": name} if name is not None else {}),
        }
        try:
            response = await resolved_client._request("POST", "/sandboxes/start", request)
            sandbox_id = str(response.get("id", ""))
            if not sandbox_id:
                raise SandboxFailedError("Thunder did not return a sandbox ID")
            private_destination = paths.sandbox_private_key(sandbox_id)
            public_destination = paths.sandbox_public_key(sandbox_id)
            if private_destination.exists() or public_destination.exists():
                raise InvalidRequestError(
                    f"SSH key already exists for sandbox {sandbox_id}"
                )
            if private_key_source is not None and public_key_source is not None:
                os.replace(private_key_source, private_destination)
                os.replace(public_key_source, public_destination)
            else:
                _write_key(private_destination, ssh_private_key or "", 0o600)
                _write_key(public_destination, public_key + "\n", 0o644)
        finally:
            if temporary_directory is not None:
                temporary_directory.cleanup()

        sandbox = await Sandbox.from_id(sandbox_id, client=resolved_client)
        if args:
            await sandbox.wait_until_ready(timeout=timeout)
            sandbox._main_process = await sandbox.exec(*args)
        return sandbox

    @staticmethod
    async def from_id(
        sandbox_id: str, *, client: Client | None = None
    ) -> "Sandbox":
        resolved_client = client or Client.from_cli()
        response = await resolved_client._request(
            "GET", f"/sandboxes/{_path_segment(sandbox_id)}"
        )
        return Sandbox._from_response(resolved_client, response)

    @staticmethod
    async def from_name(
        name: str, *, client: Client | None = None
    ) -> "Sandbox":
        resolved_client = client or Client.from_cli()
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
        return matches[0]

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
    def info(self) -> SandboxInfo:
        return self._info

    @property
    def ssh(self) -> SSHConnection:
        if self._info.ssh is None:
            raise SandboxFailedError(
                "sandbox SSH connection details are not available"
            )
        if not self._info.ssh.private_key_path.is_file():
            raise SandboxFailedError(
                f"SSH private key does not exist for immutable sandbox {self.id}"
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
    ) -> Process[bytes]: ...

    async def exec(
        self,
        *args: str,
        timeout: float | None = None,
        workdir: str | None = None,
        env: Mapping[str, str | None] | None = None,
        text: bool = True,
        pty: bool = False,
    ) -> Process[str] | Process[bytes]:
        if not args:
            raise InvalidRequestError("exec requires a command")
        connection = await self._connect()
        command = _remote_command(args, workdir=workdir, env=env)
        try:
            process = await connection.create_process(
                command,
                encoding="utf-8" if text else None,
                term_type="xterm" if pty else None,
            )
        except (OSError, asyncssh.Error) as exc:
            await self._discard_connection(connection)
            raise ConnectionError(f"could not open a sandbox SSH session: {exc}") from exc
        return Process(process, timeout=timeout)

    async def upload(
        self,
        local_path: str | os.PathLike[str],
        remote_path: str,
        *,
        recursive: bool = False,
    ) -> None:
        connection = await self._connect()
        try:
            await asyncssh.scp(local_path, (connection, remote_path), recurse=recursive)
        except (OSError, asyncssh.Error) as exc:
            if connection.is_closed():
                await self._discard_connection(connection)
            raise SandboxFailedError(f"could not upload with SCP: {exc}") from exc

    async def download(
        self,
        remote_path: str,
        local_path: str | os.PathLike[str],
        *,
        recursive: bool = False,
    ) -> None:
        connection = await self._connect()
        try:
            await asyncssh.scp((connection, remote_path), local_path, recurse=recursive)
        except (OSError, asyncssh.Error) as exc:
            if connection.is_closed():
                await self._discard_connection(connection)
            raise SandboxFailedError(f"could not download with SCP: {exc}") from exc

    async def _connect(self) -> asyncssh.SSHClientConnection:
        connection = self._connection
        if connection is not None and not connection.is_closed():
            return connection
        async with self._connection_lock:
            connection = self._connection
            if connection is not None and not connection.is_closed():
                return connection
            ssh = self.ssh
            try:
                connection = await asyncssh.connect(
                    ssh.host,
                    ssh.port,
                    username=ssh.user,
                    client_keys=[ssh.private_key_path],
                    known_hosts=(
                        ssh.known_hosts_path
                        if ssh.known_hosts_path is not None
                        else None
                    ),
                    agent_path=None,
                    preferred_auth=["publickey"],
                    config=None,
                )
            except (OSError, asyncssh.Error) as exc:
                raise ConnectionError(
                    f"could not connect to sandbox over SSH: {exc}"
                ) from exc
            self._connection = connection
            return connection

    async def _discard_connection(
        self, connection: asyncssh.SSHClientConnection
    ) -> None:
        if self._connection is connection:
            self._connection = None
        connection.close()
        await connection.wait_closed()

    async def _close_connection(self) -> None:
        connection = self._connection
        self._connection = None
        if connection is not None:
            connection.close()
            await connection.wait_closed()

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
    ) -> tuple[bool, float | None]:
        try:
            await self.refresh()
        except ConnectionError:
            now = time.monotonic()
            started = now if failing_since is None else failing_since
            if now - started >= OUTAGE_GRACE_SECONDS:
                raise
            if deadline is not None and now >= deadline:
                raise
            return False, started
        return True, None

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
        while True:
            ok, failing_since = await self._refresh_while_waiting(
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
            await _sleep_until_next_poll(deadline)

    async def wait_until_ready(
        self, *, timeout: float | None = 300
    ) -> "Sandbox":
        deadline = None if timeout is None else time.monotonic() + timeout
        failing_since: float | None = None
        while True:
            ok, failing_since = await self._refresh_while_waiting(
                deadline, failing_since
            )
            if ok:
                if self.status == SandboxStatus.READY:
                    return self
                if self.status in {SandboxStatus.FAILED, SandboxStatus.FINISHED}:
                    raise SandboxFailedError(
                        f"sandbox {self.id} did not become ready (status: {self.status.value})"
                    )
            if deadline is not None and time.monotonic() >= deadline:
                raise SandboxTimeoutError(
                    f"sandbox {self.id} did not become ready within {timeout} seconds"
                )
            await _sleep_until_next_poll(deadline)

    async def terminate(self, *, timeout: float | None = 300) -> None:
        try:
            deadline = None if timeout is None else time.monotonic() + timeout
            failing_since: float | None = None
            while self.status == SandboxStatus.CREATED:
                ok, failing_since = await self._refresh_while_waiting(
                    deadline, failing_since
                )
                if ok and self.status in {SandboxStatus.FAILED, SandboxStatus.FINISHED}:
                    return
                if deadline is not None and time.monotonic() >= deadline:
                    raise SandboxTimeoutError(
                        f"sandbox {self.id} did not become ready to stop within {timeout} seconds"
                    )
                if self.status == SandboxStatus.CREATED:
                    await _sleep_until_next_poll(deadline)
            if self.status in {SandboxStatus.FAILED, SandboxStatus.FINISHED}:
                return
            await self._client._request(
                "POST", f"/sandboxes/{_path_segment(self.id)}/stop"
            )
            await self.refresh()
        finally:
            await self._close_connection()


async def _generate_key_pair(path: Path) -> None:
    try:
        key = asyncssh.generate_private_key("ssh-ed25519", comment="thunder-sandbox")
        private_key = key.export_private_key().decode("ascii")
        public_key = key.export_public_key().decode("ascii")
    except asyncssh.Error as exc:
        raise SandboxFailedError(f"could not generate sandbox SSH key: {exc}") from exc
    _write_key(path, private_key, 0o600)
    _write_key(path.with_suffix(".pub"), public_key, 0o644)


def _write_key(path: Path, content: str, mode: int) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        os.write(descriptor, content.encode("utf-8"))
    finally:
        os.close(descriptor)


def _path_segment(value: str) -> str:
    if not value:
        raise InvalidRequestError("sandbox ID cannot be empty")
    return quote(value, safe="")


def _datetime(value: object, optional: bool = False) -> datetime | None:
    if value in (None, ""):
        return None if optional else datetime.fromtimestamp(0).astimezone()
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def _info_from_response(paths: ThunderPaths, response: dict[str, object]) -> SandboxInfo:
    sandbox_id = str(response.get("id", ""))
    name = str(response.get("name", ""))
    if not sandbox_id:
        raise SandboxFailedError("Thunder did not return a sandbox ID")
    spec = response.get("spec") if isinstance(response.get("spec"), dict) else {}
    policy = (
        response.get("network_policy")
        if isinstance(response.get("network_policy"), dict)
        else {}
    )
    gpu_type = GPUType(str(spec["gpu_type"])) if spec.get("gpu_type") else None
    ssh_value = response.get("ssh")
    ssh = None
    if isinstance(ssh_value, dict) and ssh_value.get("host"):
        ssh = SSHConnection(
            host=str(ssh_value["host"]),
            port=int(ssh_value.get("port", 22)),
            user=str(ssh_value.get("user", "ubuntu")),
            private_key_path=paths.sandbox_private_key(sandbox_id),
        )
    return SandboxInfo(
        id=sandbox_id,
        name=name,
        status=SandboxStatus(str(response.get("status", "created"))),
        resources=Resources(
            cpu=int(spec.get("cpu_count", 0)),
            memory=int(spec.get("memory_gib", 0)),
            storage=int(spec.get("storage_gib", 0)),
            gpu_type=gpu_type,
            gpu_count=int(spec.get("gpu_count", 0)),
        ),
        network_policy=NetworkPolicy(
            internet_access=str(policy.get("internet_access", "closed")),
            outbound_cidr_allowlist=tuple(policy.get("cidr_allowlist", ()) or ()),
            outbound_domain_allowlist=tuple(policy.get("domain_allowlist", ()) or ()),
        ),
        created_at=_datetime(response.get("created_at")),
        expires_at=_datetime(response.get("expires_at"), optional=True),
        ssh=ssh,
        failure_code=(
            str(response["failure_code"]) if response.get("failure_code") else None
        ),
        failure=str(response["failure"]) if response.get("failure") else None,
    )


def _validate_create_options(
    *,
    timeout: int | None,
    gpu_type: GPUType | None,
    gpu_count: int | None,
    block_network: bool,
    outbound_cidr_allowlist: Sequence[str] | None,
    outbound_domain_allowlist: Sequence[str] | None,
    ssh_public_key: str | None,
    ssh_private_key: str | None,
) -> None:
    if (ssh_public_key is None) != (ssh_private_key is None):
        raise InvalidRequestError(
            "ssh_public_key and ssh_private_key must be provided together"
        )
    if timeout is not None and timeout < 0:
        raise InvalidRequestError("timeout cannot be negative")
    if gpu_type is not None and not isinstance(gpu_type, GPUType):
        raise InvalidRequestError("gpu_type must be a GPUType")
    if (gpu_type is None) != (gpu_count is None):
        raise InvalidRequestError("gpu_type and gpu_count must be provided together")
    if gpu_count is not None and gpu_count not in (1, 2, 4, 8):
        raise InvalidRequestError("gpu_count must be one of 1, 2, 4, or 8")
    if block_network and (outbound_cidr_allowlist or outbound_domain_allowlist):
        raise InvalidRequestError(
            "network allowlists cannot be combined with block_network"
        )


def _network_policy(
    block_network: bool,
    outbound_cidr_allowlist: Sequence[str] | None,
    outbound_domain_allowlist: Sequence[str] | None,
) -> tuple[str, list[str], list[str]]:
    if block_network:
        return "closed", [], []
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


async def _sleep_until_next_poll(deadline: float | None) -> None:
    delay = 1.0
    if deadline is not None:
        delay = max(0.0, min(delay, deadline - time.monotonic()))
    await asyncio.sleep(delay)


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


__all__ = ["Sandbox"]
