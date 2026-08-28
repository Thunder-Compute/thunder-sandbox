"""Sandbox lifecycle, SSH execution, and SCP transfers."""

from __future__ import annotations

import asyncio
import os
import shlex
import shutil
import subprocess
import time
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Literal, overload

from .client import Client
from .credentials import ensure_credential
from .exceptions import (
    ConflictError,
    ConnectionError,
    InvalidRequestError,
    NotFoundError,
    SandboxFailedError,
    SandboxTimeoutError,
    UnsupportedFeatureError,
)
from .process import AsyncContainerProcess, ContainerProcess
from .types import GPUType, NetworkPolicy, Resources, SandboxInfo, SandboxStatus, SSHConnection

# How long a wait tolerates uninterrupted connection failures before surfacing
# the error. Long enough to absorb a blip, short enough that a real outage does
# not hide behind a multi-minute timeout.
OUTAGE_GRACE_SECONDS = 30.0


class Sandbox:
    """A handle to an immutable Thunder sandbox."""

    def __init__(self, client: Client, info: SandboxInfo) -> None:
        self._client = client
        self._info = info
        self._main_process: ContainerProcess[str] | None = None

    @staticmethod
    def create(
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
        if timeout is not None and timeout < 0:
            raise InvalidRequestError("timeout cannot be negative")
        if gpu_type is not None and not isinstance(gpu_type, GPUType):
            raise InvalidRequestError("gpu_type must be a GPUType")
        if (gpu_type is None) != (gpu_count is None):
            raise InvalidRequestError("gpu_type and gpu_count must be provided together")
        if gpu_count is not None and gpu_count not in (1, 2, 4, 8):
            raise InvalidRequestError("gpu_count must be one of 1, 2, 4, or 8")
        if block_network and (outbound_cidr_allowlist or outbound_domain_allowlist):
            raise InvalidRequestError("network allowlists cannot be combined with block_network")

        resolved_client = client or Client.from_cli()

        if block_network:
            internet_access, cidrs, domains = "closed", [], []
        else:
            # Unrestricted access is expressed as an explicit restricted
            # policy. Domain and CIDR gates remain independent by design.
            internet_access = "restricted"
            cidrs = (
                list(outbound_cidr_allowlist) if outbound_cidr_allowlist is not None else ["0.0.0.0/0"]
            )
            domains = (
                list(outbound_domain_allowlist) if outbound_domain_allowlist is not None else ["*"]
            )
        request = {
            "spec": {
                "cpu_count": cpu if cpu is not None else 4,
                "memory_gib": memory if memory is not None else 32,
                "storage_gib": storage if storage is not None else 50,
                **({"gpu_type": gpu_type.value, "gpu_count": gpu_count} if gpu_type is not None else {}),
            },
            "env": {key: value for key, value in (env or {}).items() if value is not None},
            "lifetime": {"enforce_ttl": timeout is not None, **({"max_ttl_seconds": timeout} if timeout is not None else {})},
            "network_policy": {"internet_access": internet_access, "cidr_allowlist": cidrs, "domain_allowlist": domains},
            **({"name": name} if name is not None else {}),
        }
        response = resolved_client._request("POST", "/sandboxes/start", request)
        sandbox_id = str(response.get("id", ""))
        if not sandbox_id:
            raise SandboxFailedError("Thunder did not return a sandbox ID")

        sandbox = Sandbox.from_id(sandbox_id, client=resolved_client)
        if args:
            sandbox.wait_until_ready(timeout=timeout)
            sandbox._main_process = sandbox.exec(*args)
        return sandbox

    @staticmethod
    def from_id(sandbox_id: str, *, client: Client | None = None) -> "Sandbox":
        resolved_client = client or Client.from_cli()
        response = resolved_client._request("GET", f"/sandboxes/{_path_segment(sandbox_id)}")
        return Sandbox._from_response(resolved_client, response)

    @staticmethod
    def from_name(name: str, *, client: Client | None = None) -> "Sandbox":
        """Find the live sandbox holding this name.

        A name is not an address: it identifies at most one live sandbox but may
        have belonged to any number of finished ones. Prefer from_id.
        """
        resolved_client = client or Client.from_cli()
        matches = [
            sandbox for sandbox in resolved_client.list_sandboxes(status="active")
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
    def _from_response(client: Client, response: dict[str, object]) -> "Sandbox":
        sandbox_id = str(response.get("id", ""))
        name = str(response.get("name", ""))
        if not sandbox_id:
            raise SandboxFailedError("Thunder did not return a sandbox ID")
        spec = response.get("spec") if isinstance(response.get("spec"), dict) else {}
        policy = response.get("network_policy") if isinstance(response.get("network_policy"), dict) else {}
        gpu_type = GPUType(str(spec["gpu_type"])) if spec.get("gpu_type") else None
        gpu_count = int(spec.get("gpu_count", 0))
        ssh_value = response.get("ssh")
        ssh = None
        if isinstance(ssh_value, dict) and ssh_value.get("host"):
            # The credential is minted when a connection is actually made, not
            # here: listing sandboxes must not reach the API for a certificate.
            ssh = SSHConnection(
                host=str(ssh_value["host"]), port=int(ssh_value.get("port", 22)),
                user=str(ssh_value.get("user", "ubuntu")),
                private_key_path=client.config.paths.ssh_key,
                certificate_path=client.config.paths.ssh_certificate,
            )
        info = SandboxInfo(
            id=sandbox_id, name=name, status=SandboxStatus(str(response.get("status", "created"))),
            resources=Resources(cpu=int(spec.get("cpu_count", 0)), memory=int(spec.get("memory_gib", 0)), storage=int(spec.get("storage_gib", 0)), gpu_type=gpu_type, gpu_count=gpu_count),
            network_policy=NetworkPolicy(internet_access=str(policy.get("internet_access", "closed")), outbound_cidr_allowlist=tuple(policy.get("cidr_allowlist", ()) or ()), outbound_domain_allowlist=tuple(policy.get("domain_allowlist", ()) or ())),
            created_at=_datetime(response.get("created_at")), expires_at=_datetime(response.get("expires_at"), optional=True), ssh=ssh,
            failure_code=str(response["failure_code"]) if response.get("failure_code") else None,
            failure=str(response["failure"]) if response.get("failure") else None,
        )
        return Sandbox(client, info)

    @property
    def id(self) -> str: return self._info.id
    @property
    def name(self) -> str: return self._info.name
    @property
    def status(self) -> SandboxStatus: return self._info.status
    @property
    def info(self) -> SandboxInfo: return self._info

    @property
    def ssh(self) -> SSHConnection:
        if self._info.ssh is None:
            raise SandboxFailedError("sandbox SSH connection details are not available")
        if not self._info.ssh.private_key_path.is_file():
            raise SandboxFailedError(f"SSH private key does not exist for immutable sandbox {self.id}")
        return self._info.ssh

    @property
    def ssh_command(self) -> tuple[str, ...]: return self.ssh.command

    @overload
    def exec(self, *args: str, timeout: float | None = None, workdir: str | None = None, env: Mapping[str, str | None] | None = None, text: Literal[True] = True, pty: bool = False) -> ContainerProcess[str]: ...
    @overload
    def exec(self, *args: str, timeout: float | None = None, workdir: str | None = None, env: Mapping[str, str | None] | None = None, text: Literal[False] = False, pty: bool = False) -> ContainerProcess[bytes]: ...

    def exec(self, *args: str, timeout: float | None = None, workdir: str | None = None, env: Mapping[str, str | None] | None = None, text: bool = True, pty: bool = False) -> ContainerProcess[str] | ContainerProcess[bytes]:
        if not args:
            raise InvalidRequestError("exec requires a command")
        ensure_credential(self._client)
        command = list(self.ssh_command)
        if pty:
            command.insert(1, "-tt")
        command.append(_remote_command(args, workdir=workdir, env=env))
        return ContainerProcess(
            subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=text,
            ),
            timeout=timeout,
        )

    def upload(self, local_path: str | os.PathLike[str], remote_path: str, *, recursive: bool = False) -> None:
        ensure_credential(self._client)
        command = _scp_command(self.ssh, recursive)
        command.extend((str(local_path), f"{self.ssh.user}@{self.ssh.host}:{shlex.quote(remote_path)}"))
        _run_transfer(command)

    def download(self, remote_path: str, local_path: str | os.PathLike[str], *, recursive: bool = False) -> None:
        ensure_credential(self._client)
        command = _scp_command(self.ssh, recursive)
        command.extend((f"{self.ssh.user}@{self.ssh.host}:{shlex.quote(remote_path)}", str(local_path)))
        _run_transfer(command)

    def refresh(self) -> "Sandbox":
        self._info = Sandbox.from_id(self.id, client=self._client)._info
        return self

    def poll(self) -> int | None:
        if self._main_process is not None:
            return self._main_process.poll()
        self.refresh()
        return 0 if self.status == SandboxStatus.FINISHED else (1 if self.status == SandboxStatus.FAILED else None)

    def _refresh_while_waiting(
        self, deadline: float | None, failing_since: float | None
    ) -> tuple[bool, float | None]:
        """Refresh, tolerating a brief API failure.

        Returns whether the status is now current, and when the present run of
        failures began. A single dropped response must not end a wait that may
        span minutes, but an outage should not hide behind the caller's timeout
        either: once failures have run uninterrupted for OUTAGE_GRACE_SECONDS the
        underlying error is raised, so the caller sees "could not communicate"
        rather than waiting out a ten-minute deadline for a misleading timeout.
        """
        try:
            self.refresh()
        except ConnectionError:
            now = time.monotonic()
            started = now if failing_since is None else failing_since
            if now - started >= OUTAGE_GRACE_SECONDS:
                raise
            if deadline is not None and now >= deadline:
                raise
            return False, started
        return True, None

    def wait(self, *, timeout: float | None = None) -> int | None:
        if self._main_process is not None:
            try:
                return self._main_process.wait(timeout=timeout)
            except subprocess.TimeoutExpired as exc:
                raise SandboxTimeoutError(f"sandbox command did not finish within {timeout} seconds") from exc
        deadline = None if timeout is None else time.monotonic() + timeout
        failing_since: float | None = None
        while True:
            ok, failing_since = self._refresh_while_waiting(deadline, failing_since)
            if ok:
                if self.status == SandboxStatus.FINISHED:
                    return 0
                if self.status == SandboxStatus.FAILED:
                    return 1
            if deadline is not None and time.monotonic() >= deadline:
                raise SandboxTimeoutError(f"sandbox {self.id} did not stop within {timeout} seconds")
            time.sleep(1)

    def wait_until_ready(self, *, timeout: float | None = 300) -> "Sandbox":
        deadline = None if timeout is None else time.monotonic() + timeout
        failing_since: float | None = None
        while True:
            ok, failing_since = self._refresh_while_waiting(deadline, failing_since)
            if ok:
                if self.status == SandboxStatus.READY:
                    return self
                if self.status in {SandboxStatus.FAILED, SandboxStatus.FINISHED}:
                    raise SandboxFailedError(f"sandbox {self.id} did not become ready (status: {self.status.value})")
            if deadline is not None and time.monotonic() >= deadline:
                raise SandboxTimeoutError(f"sandbox {self.id} did not become ready within {timeout} seconds")
            time.sleep(1)

    def terminate(self, *, timeout: float | None = 300) -> None:
        deadline = None if timeout is None else time.monotonic() + timeout
        failing_since: float | None = None
        while self.status == SandboxStatus.CREATED:
            ok, failing_since = self._refresh_while_waiting(deadline, failing_since)
            if ok and self.status in {SandboxStatus.FAILED, SandboxStatus.FINISHED}:
                return
            if deadline is not None and time.monotonic() >= deadline:
                raise SandboxTimeoutError(f"sandbox {self.id} did not become ready to stop within {timeout} seconds")
            if self.status == SandboxStatus.CREATED:
                time.sleep(1)
        if self.status in {SandboxStatus.FAILED, SandboxStatus.FINISHED}:
            return
        self._client._request("POST", f"/sandboxes/{_path_segment(self.id)}/stop")
        self.refresh()


class AsyncSandbox:
    def __init__(self, sandbox: Sandbox) -> None: self._sandbox = sandbox
    @staticmethod
    async def create(*args: str, **options: object) -> "AsyncSandbox": return AsyncSandbox(await asyncio.to_thread(Sandbox.create, *args, **options))
    @staticmethod
    async def from_id(sandbox_id: str, *, client: Client | None = None) -> "AsyncSandbox": return AsyncSandbox(await asyncio.to_thread(Sandbox.from_id, sandbox_id, client=client))
    @staticmethod
    async def from_name(name: str, *, client: Client | None = None) -> "AsyncSandbox":
        # Must go through Sandbox.from_name, not from_id: a name is a label, not
        # the sandbox's address, so the API cannot resolve one directly.
        return AsyncSandbox(await asyncio.to_thread(Sandbox.from_name, name, client=client))
    @property
    def id(self) -> str: return self._sandbox.id
    @property
    def name(self) -> str: return self._sandbox.name
    @property
    def status(self) -> SandboxStatus: return self._sandbox.status
    @property
    def info(self) -> SandboxInfo: return self._sandbox.info
    @property
    def ssh(self) -> SSHConnection: return self._sandbox.ssh
    async def exec(self, *args: str, **options: object) -> AsyncContainerProcess[str] | AsyncContainerProcess[bytes]: return AsyncContainerProcess(await asyncio.to_thread(self._sandbox.exec, *args, **options))
    async def upload(self, local_path: str | os.PathLike[str], remote_path: str, *, recursive: bool = False) -> None: await asyncio.to_thread(self._sandbox.upload, local_path, remote_path, recursive=recursive)
    async def download(self, remote_path: str, local_path: str | os.PathLike[str], *, recursive: bool = False) -> None: await asyncio.to_thread(self._sandbox.download, remote_path, local_path, recursive=recursive)
    async def refresh(self) -> "AsyncSandbox":
        await asyncio.to_thread(self._sandbox.refresh)
        return self
    async def poll(self) -> int | None: return await asyncio.to_thread(self._sandbox.poll)
    async def wait(self, *, timeout: float | None = None) -> int | None: return await asyncio.to_thread(self._sandbox.wait, timeout=timeout)
    async def wait_until_ready(self, *, timeout: float | None = 300) -> "AsyncSandbox":
        await asyncio.to_thread(self._sandbox.wait_until_ready, timeout=timeout)
        return self
    async def terminate(self, *, timeout: float | None = 300) -> None: await asyncio.to_thread(self._sandbox.terminate, timeout=timeout)


def _path_segment(value: str) -> str:
    from urllib.parse import quote
    if not value: raise InvalidRequestError("sandbox ID cannot be empty")
    return quote(value, safe="")


def _datetime(value: object, optional: bool = False) -> datetime | None:
    if value in (None, ""):
        return None if optional else datetime.fromtimestamp(0).astimezone()
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def _remote_command(args: Sequence[str], *, workdir: str | None, env: Mapping[str, str | None] | None) -> str:
    parts: list[str] = []
    if workdir is not None: parts.extend(("cd", shlex.quote(workdir), "&&"))
    if env:
        parts.append("env")
        for key, value in env.items():
            parts.extend(("-u", shlex.quote(key))) if value is None else parts.append(f"{shlex.quote(key)}={shlex.quote(value)}")
    parts.extend(shlex.quote(arg) for arg in args)
    return " ".join(parts)


def _scp_command(connection: SSHConnection, recursive: bool) -> list[str]:
    command = ["scp", "-i", str(connection.private_key_path), "-P", str(connection.port), "-o", "IdentitiesOnly=yes", "-o", "IdentityAgent=none", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null"]
    # Copies authenticate the same way sessions do.
    if connection.certificate_path is not None:
        command.extend(("-o", f"CertificateFile={connection.certificate_path}"))
    if recursive: command.append("-r")
    return command


def _run_transfer(command: Sequence[str]) -> None:
    try: subprocess.run(command, check=True)
    except FileNotFoundError as exc: raise UnsupportedFeatureError("scp is required for sandbox file transfers") from exc
    except subprocess.CalledProcessError as exc: raise SandboxFailedError(f"scp exited with status {exc.returncode}") from exc
