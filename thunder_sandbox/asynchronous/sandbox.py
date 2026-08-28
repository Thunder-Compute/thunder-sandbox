"""Native asynchronous sandbox lifecycle, SSH execution, and transfers."""

from __future__ import annotations

import asyncio
import os
import random
import shlex
import time
from collections.abc import Mapping, Sequence
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from typing import Literal, overload
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
)
from .._common.types import (
    GPUType,
    NetworkPolicy,
    Resources,
    SandboxInfo,
    SandboxStatus,
    SSHConnection,
)
from .client import Client
from .process import Process

OUTAGE_GRACE_SECONDS = 30.0


class Sandbox:
    """A native asynchronous handle to an immutable Thunder sandbox."""

    def __init__(
        self, client: Client, info: SandboxInfo, *, owns_client: bool = False
    ) -> None:
        self._client = client
        self._owns_client = owns_client
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
        client: Client | None = None,
    ) -> "Sandbox":
        _validate_create_options(
            timeout=timeout,
            gpu_type=gpu_type,
            gpu_count=gpu_count,
            block_network=block_network,
            outbound_cidr_allowlist=outbound_cidr_allowlist,
            outbound_domain_allowlist=outbound_domain_allowlist,
        )
        owns_client = client is None
        resolved_client = client or Client.from_cli()
        try:
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

    @overload
    async def exec(
        self,
        *args: str,
        timeout: float | None = None,
        workdir: str | None = None,
        env: Mapping[str, str | None] | None = None,
        text: bool,
        pty: bool = False,
    ) -> Process[str] | Process[bytes]: ...

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
        return Process(process, timeout=timeout, text=text)

    async def upload(
        self,
        local_path: str | os.PathLike[str],
        remote_path: str,
        *,
        recursive: bool = False,
    ) -> None:
        connection = await self._connect()
        try:
            await asyncssh.scp(
                os.fspath(local_path), (connection, remote_path), recurse=recursive
            )
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
            await asyncssh.scp(
                (connection, remote_path), os.fspath(local_path), recurse=recursive
            )
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
                # The node reuses forwarded ports across sandboxes, so a pin
                # keyed by host and port would reject the next sandbox that
                # lands on a finished one's port. Pin per sandbox instead, and
                # hand asyncssh the key itself: it reads a known-hosts file
                # only from a string path, never a Path.
                pinned = _pinned_host_key(self.id)
                known_hosts = ([pinned], [], []) if pinned is not None else None
                credential = await self._client._credentials.ensure(self._client)
                connection = await asyncssh.connect(
                    ssh.host,
                    ssh.port,
                    username=ssh.user,
                    # The sandbox trusts the authority that signed this, not
                    # the key itself, so the same credential opens every
                    # sandbox the organization owns.
                    client_keys=[(credential.key, credential.certificate)],
                    known_hosts=known_hosts,
                    agent_path=None,
                    preferred_auth=["publickey"],
                    config=None,
                )
                if pinned is None:
                    try:
                        _remember_host_key(self.id, connection.get_server_host_key())
                    except BaseException:
                        connection.close()
                        await connection.wait_closed()
                        raise
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
        failing_since: float | None = None
        poll_delay = 1.0
        while True:
            ok, failing_since, retry_after = await self._refresh_while_waiting(
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
            await _sleep_until_next_poll(deadline, poll_delay, retry_after)
            poll_delay = min(5.0, poll_delay * 2.0)

    async def terminate(self, *, timeout: float | None = 300) -> None:
        try:
            deadline = None if timeout is None else time.monotonic() + timeout
            failing_since: float | None = None
            poll_delay = 1.0
            while self.status == SandboxStatus.CREATED:
                ok, failing_since, retry_after = await self._refresh_while_waiting(
                    deadline, failing_since
                )
                if ok and self.status in {SandboxStatus.FAILED, SandboxStatus.FINISHED}:
                    return
                if deadline is not None and time.monotonic() >= deadline:
                    raise SandboxTimeoutError(
                        f"sandbox {self.id} did not become ready to stop within {timeout} seconds"
                    )
                if self.status == SandboxStatus.CREATED:
                    await _sleep_until_next_poll(deadline, poll_delay, retry_after)
                    poll_delay = min(5.0, poll_delay * 2.0)
            if self.status in {SandboxStatus.FAILED, SandboxStatus.FINISHED}:
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
    policy: dict[str, object] = (
        policy_value if isinstance(policy_value, dict) else {}
    )
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
        network_policy=NetworkPolicy(
            internet_access=str(policy.get("internet_access", "closed")),
            outbound_cidr_allowlist=_string_tuple(policy.get("cidr_allowlist")),
            outbound_domain_allowlist=_string_tuple(
                policy.get("domain_allowlist")
            ),
        ),
        created_at=_datetime(response.get("created_at")),
        expires_at=_datetime(response.get("expires_at"), optional=True),
        ssh=ssh,
        failure_code=(
            str(response["failure_code"]) if response.get("failure_code") else None
        ),
        failure=str(response["failure"]) if response.get("failure") else None,
    )


def _string_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(str(item) for item in value)


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


__all__ = ["Sandbox"]
