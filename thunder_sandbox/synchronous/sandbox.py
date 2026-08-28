"""Blocking adapter over the native asynchronous sandbox implementation."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Literal, overload

from .client import Client
from .process import Process
from ..asynchronous.sandbox import Sandbox as NativeSandbox
from .._common.types import GPUType, SandboxInfo, SandboxStatus, SSHConnection


class Sandbox:
    def __init__(
        self, client: Client, sandbox: NativeSandbox, *, owns_client: bool = False
    ) -> None:
        self._client = client
        self._sandbox = sandbox
        self._owns_client = owns_client

    @staticmethod
    def _from_async(client: Client, sandbox: NativeSandbox) -> "Sandbox":
        return client._wrap_sandbox(sandbox)

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
        ssh_public_key: str | None = None,
        ssh_private_key: str | None = None,
        client: Client | None = None,
    ) -> "Sandbox":
        owns_client = client is None
        resolved_client = client or Client.from_cli()
        try:
            sandbox = resolved_client._bridge.run(
                NativeSandbox.create(
                    *args,
                    name=name,
                    env=env,
                    timeout=timeout,
                    cpu=cpu,
                    memory=memory,
                    storage=storage,
                    gpu_type=gpu_type,
                    gpu_count=gpu_count,
                    block_network=block_network,
                    outbound_cidr_allowlist=outbound_cidr_allowlist,
                    outbound_domain_allowlist=outbound_domain_allowlist,
                    ssh_public_key=ssh_public_key,
                    ssh_private_key=ssh_private_key,
                    client=resolved_client._client,
                )
            )
        except BaseException:
            if owns_client:
                resolved_client.close()
            raise
        wrapped = resolved_client._wrap_sandbox(sandbox)
        wrapped._owns_client = owns_client
        return wrapped

    @staticmethod
    def from_id(sandbox_id: str, *, client: Client | None = None) -> "Sandbox":
        owns_client = client is None
        resolved_client = client or Client.from_cli()
        try:
            sandbox = resolved_client._bridge.run(
                NativeSandbox.from_id(sandbox_id, client=resolved_client._client)
            )
        except BaseException:
            if owns_client:
                resolved_client.close()
            raise
        wrapped = resolved_client._wrap_sandbox(sandbox)
        wrapped._owns_client = owns_client
        return wrapped

    @staticmethod
    def from_name(name: str, *, client: Client | None = None) -> "Sandbox":
        owns_client = client is None
        resolved_client = client or Client.from_cli()
        try:
            sandbox = resolved_client._bridge.run(
                NativeSandbox.from_name(name, client=resolved_client._client)
            )
        except BaseException:
            if owns_client:
                resolved_client.close()
            raise
        wrapped = resolved_client._wrap_sandbox(sandbox)
        wrapped._owns_client = owns_client
        return wrapped

    @property
    def id(self) -> str:
        return self._sandbox.id

    @property
    def name(self) -> str:
        return self._sandbox.name

    @property
    def status(self) -> SandboxStatus:
        return self._sandbox.status

    @property
    def info(self) -> SandboxInfo:
        return self._sandbox.info

    @property
    def ssh(self) -> SSHConnection:
        return self._sandbox.ssh

    @property
    def ssh_command(self) -> tuple[str, ...]:
        return self._sandbox.ssh_command

    @overload
    def exec(
        self, *args: str, timeout: float | None = None,
        workdir: str | None = None, env: Mapping[str, str | None] | None = None,
        text: Literal[True] = True, pty: bool = False,
    ) -> Process[str]: ...

    @overload
    def exec(
        self, *args: str, timeout: float | None = None,
        workdir: str | None = None, env: Mapping[str, str | None] | None = None,
        text: Literal[False] = False, pty: bool = False,
    ) -> Process[bytes]: ...

    def exec(
        self, *args: str, timeout: float | None = None,
        workdir: str | None = None, env: Mapping[str, str | None] | None = None,
        text: bool = True, pty: bool = False,
    ) -> Process[str] | Process[bytes]:
        process = self._client._bridge.run(
            self._sandbox.exec(
                *args,
                timeout=timeout,
                workdir=workdir,
                env=env,
                text=text,
                pty=pty,
            )
        )
        return Process(self._client._bridge, process)

    def upload(
        self, local_path: str | os.PathLike[str], remote_path: str, *,
        recursive: bool = False,
    ) -> None:
        self._client._bridge.run(
            self._sandbox.upload(local_path, remote_path, recursive=recursive)
        )

    def download(
        self, remote_path: str, local_path: str | os.PathLike[str], *,
        recursive: bool = False,
    ) -> None:
        self._client._bridge.run(
            self._sandbox.download(remote_path, local_path, recursive=recursive)
        )

    def refresh(self) -> "Sandbox":
        self._client._bridge.run(self._sandbox.refresh())
        return self

    def poll(self) -> int | None:
        return self._client._bridge.run(self._sandbox.poll())

    def wait(self, *, timeout: float | None = None) -> int | None:
        return self._client._bridge.run(self._sandbox.wait(timeout=timeout))

    def wait_until_ready(self, *, timeout: float | None = 300) -> "Sandbox":
        self._client._bridge.run(self._sandbox.wait_until_ready(timeout=timeout))
        return self

    def terminate(self, *, timeout: float | None = 300) -> None:
        try:
            self._client._bridge.run(self._sandbox.terminate(timeout=timeout))
        finally:
            if self._owns_client:
                self._client.close()


__all__ = ["Sandbox"]
