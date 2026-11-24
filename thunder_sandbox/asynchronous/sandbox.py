"""Native asynchronous sandbox lifecycle; commands and files go through its channel."""

from __future__ import annotations

import asyncio
import math
import os
import posixpath
import random
import time
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Literal, overload
from urllib.parse import quote

import asyncssh

from .._channels._spec import (
    DownloadRequest,
    ExecRequest,
    Identity,
    OutputMode as _OutputMode,
    Retention,
    TransferOptions,
    UploadRequest,
)
from .._channels._ssh_channel import (
    SSHChannel,
    SSHDockerChannel,
    SSHHostChannel,
    _SSHChannelBase,
)
from .._common.config import ThunderPaths
from .._common.exceptions import (
    ConflictError,
    ConnectionError,
    InvalidRequestError,
    NotFoundError,
    RetryableError,
    SandboxFailedError,
    SandboxTimeoutError,
    ThunderError,
    _WaitWindowElapsedError,
)
from .._common.types import (
    MAIN_WORKSPACE_ID,
    GPUType,
    NetworkPolicy,
    OutputMode,
    Resources,
    SandboxInfo,
    SandboxStatus,
    SSHConnection,
    SSHProxyJump,
)
from ..image import Image
from ._jobs import new_job_id, validate_job_id
from ._remote import _remember_host_key
from ._transfer import parse_transfer_path as _parse_transfer_path
from .client import Client
from .process import Process

OUTAGE_GRACE_SECONDS = 30.0
# The API refuses a readiness wait held open longer than this per request.
WAIT_WINDOW_MAX_SECONDS = 30.0
# How much longer than its window one wait request may take to answer before
# the client gives up on it, covering transit and a server that has stalled.
WAIT_REPLY_GRACE_SECONDS = 15.0
# Linux's MAX_ARG_STRLEN, including the terminating NUL.
MAX_ARGUMENT_BYTES = 128 * 1024


def _upload_source(local_path: str | os.PathLike[str]) -> tuple[Path, bool]:
    """The upload source, and whether only its contents are copied.

    A source ending in ``/.`` copies a directory's contents, as with ``cp`` and
    ``scp``, and so does ``.`` itself, as it did before channels.
    """
    text = os.fspath(local_path)
    trimmed = text.rstrip(os.sep) or text
    if trimmed == ".":
        return Path("."), True
    source = _parse_transfer_path(trimmed, separator=os.sep)
    return Path(source.path), source.contents_only


class Sandbox:
    """A native asynchronous handle to a Thunder sandbox."""

    def __init__(
        self, client: Client, info: SandboxInfo, *, owns_client: bool = False
    ) -> None:
        self._client = client
        self._owns_client = owns_client
        self._info = info
        self._main_process: Process[str] | None = None
        # Commands and files go through the channel for this kind of sandbox.
        self._ssh_channel: _SSHChannelBase | None = None

    @staticmethod
    async def create(
        *args: str,
        name: str | None = None,
        env: Mapping[str, str | None] | None = None,
        gated_features: Mapping[str, object] | None = None,
        timeout: int | None = 300,
        queue: bool = False,
        cpu: int | None = None,
        memory: int | None = None,
        storage: int | None = None,
        gpu_type: GPUType | None = None,
        gpu_count: int | None = None,
        image: Image | None = None,
        vm: bool | Mapping[str, object] = False,
        timing_sensitive: bool = False,
        block_network: bool = False,
        outbound_cidr_allowlist: Sequence[str] | None = None,
        outbound_domain_allowlist: Sequence[str] | None = None,
        workspace: str | None = None,
        client: Client | None = None,
    ) -> "Sandbox":
        """Create a sandbox, resolving an optional image before allocation.

        The image's initial command is attempted once. The sandbox remains
        available for commands and file transfers after it exits, including
        after container restarts. Images without ENTRYPOINT or CMD stay alive
        without running an initial command. Positional arguments launch a
        separate ``exec`` command after readiness; they do not override image
        defaults.

        ``queue=True`` admits image builds and waits for sandbox capacity without
        a scheduling timeout. Creation without positional commands returns a
        handle while queued; use ``wait_until_ready(timeout=None)`` to wait
        indefinitely. ``timeout`` remains the runtime TTL and, with positional
        commands, the readiness wait limit. CPU-only sandboxes omit GPU options
        or use ``GPUType.NONE`` with ``gpu_count=0``. ``cpu=1`` and ``cpu=2``
        are supported, subject to the API resource catalog.

        ``workspace`` is a workspace name or ID. Without one, ``TNR_WORKSPACE``
        applies, and then main. The image is built or imported in that
        workspace too, and listed there.

        ``vm=True`` asks for a VM sandbox: the image, built for it with
        ``resolve_image(vm=True)``, boots as the whole VM with its own systemd,
        and commands and transfers reach it over SSH as root with no container
        in between. Without ``image`` it boots Thunder's default VM image.
        ``vm={"imageCommand": {...}}`` also overrides the image's
        ``entrypoint``, ``cmd`` and ``workdir`` and sets a ``restart`` policy.
        Where Thunder does not offer VM sandboxes the request starts the
        sandbox it would start without ``vm``; ``Sandbox.vm`` says which kind
        started.

        ``timing_sensitive=True`` runs the image's container in gVisor on the
        sandbox node instead of in a Thunder VM. GPUs are the node's physical
        GPUs, reserved for the sandbox, rather than virtualized, so GPU timing
        matches bare metal. It needs ``image`` and cannot be combined with
        ``vm``. Such a sandbox reports ``Sandbox.vm`` as true: commands and
        transfers reach the container directly over SSH. The API refuses the
        option where Thunder does not offer it.
        """
        _validate_create_options(
            timeout=timeout,
            queue=queue,
            gpu_type=gpu_type,
            gpu_count=gpu_count,
            image=image,
            block_network=block_network,
            outbound_cidr_allowlist=outbound_cidr_allowlist,
            outbound_domain_allowlist=outbound_domain_allowlist,
        )
        if not isinstance(vm, (bool, Mapping)):
            raise InvalidRequestError("vm must be a boolean or a mapping")
        if not isinstance(timing_sensitive, bool):
            raise InvalidRequestError("timing_sensitive must be a boolean")
        if timing_sensitive and image is None:
            raise InvalidRequestError("timing_sensitive requires an image")
        # vm is the sandbox start's own field. gated_features={"vm": ...}
        # asks for the same through the gated feature.
        vm_config = dict(vm) if isinstance(vm, Mapping) else {} if vm else None
        if timing_sensitive and (vm_config is not None or "vm" in (gated_features or {})):
            raise InvalidRequestError("timing_sensitive cannot be combined with vm")
        owns_client = client is None
        resolved_client = client or Client.from_cli()
        try:
            workspace_id = await resolved_client._resolve_workspace(workspace)
            resource_spec: dict[str, object] = {
                "cpu_count": cpu if cpu is not None else 4,
                "memory_gib": memory if memory is not None else 32,
                "storage_gib": storage if storage is not None else 50,
                **(
                    {"gpu_type": gpu_type.value, "gpu_count": gpu_count}
                    if gpu_type is not None and gpu_type is not GPUType.NONE
                    else {}
                ),
            }
            resolved_image = (
                await resolved_client.resolve_image(
                    image, gated_features=gated_features,
                    spec=resource_spec, queue=queue, workspace=workspace_id,
                    vm=vm_config is not None,
                )
                if image is not None
                else None
            )
            internet_access, cidrs, domains = _network_policy_request(
                block_network,
                outbound_cidr_allowlist,
                outbound_domain_allowlist,
            )
            request = {
                "client_capabilities": {"proxy_jump": True},
                **({"queue": True} if queue else {}),
                **(
                    {"workspace_id": workspace_id}
                    if workspace_id != MAIN_WORKSPACE_ID
                    else {}
                ),
                "spec": resource_spec,
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
                **({"vm": vm_config} if vm_config is not None else {}),
                **({"timing_sensitive": True} if timing_sensitive else {}),
                **(
                    {"gatedFeatures": dict(gated_features)}
                    if gated_features is not None
                    else {}
                ),
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
            # An API from before workspaces ignores the selector and starts the
            # sandbox in main. Stop it (below) rather than leave it where the
            # caller did not ask for it.
            if (
                workspace_id != MAIN_WORKSPACE_ID
                and sandbox.workspace_id != workspace_id
            ):
                raise SandboxFailedError(
                    f"this Thunder API did not start the sandbox in workspace "
                    f"{workspace_id!r}; it may not support workspaces"
                )
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
        name: str, *, workspace: str | None = None, client: Client | None = None
    ) -> "Sandbox":
        owns_client = client is None
        resolved_client = client or Client.from_cli()
        try:
            matches = [
                sandbox
                async for sandbox in resolved_client.list_sandboxes(
                    status="active", workspace=workspace
                )
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
    def vm(self) -> bool:
        """Whether the sandbox boots its image as the whole VM.

        A VM sandbox has no container: commands and transfers run directly on
        the VM over SSH. A sandbox without an image has none either.
        """
        return self._info.vm

    @property
    def _has_container(self) -> bool:
        return self._info.image_id is not None and not self._info.vm

    @property
    def workspace_id(self) -> str:
        return self._info.workspace_id

    @property
    def queued(self) -> bool:
        """Whether the last observed state is waiting for capacity."""
        return self._info.queued

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
        user: str | None = None,
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
        user: str | None = None,
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
        user: str | None = None,
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
        user: str | None = None,
        env: Mapping[str, str | None] | None = None,
        text: bool = True,
        pty: bool = False,
        stdout: OutputMode = "capture",
        stderr: OutputMode = "capture",
        retain: bool = False,
    ) -> Process[str] | Process[bytes]:
        if user is not None:
            if not isinstance(user, str) or not user or "\x00" in user:
                raise InvalidRequestError(
                    "user must be a non-empty string without NUL bytes"
                )
            # A sandbox without an image logs in as an account that can start
            # a command as another user but not signal it, so terminate() and
            # the job's liveness checks would fail on it.
            if not self._has_container and not self._info.vm:
                raise InvalidRequestError("user requires a container or VM sandbox")
        if not args:
            raise InvalidRequestError("exec requires a command")
        for text_value in (*args, *(f"{k}={v}" for k, v in (env or {}).items())):
            if len(text_value.encode("utf-8", "surrogateescape")) >= MAX_ARGUMENT_BYTES:
                raise InvalidRequestError(
                    f"one exec argument or environment value must be under "
                    f"{MAX_ARGUMENT_BYTES} bytes, the Linux limit; upload larger "
                    "input as a file"
                )
        if stdout not in {"capture", "discard"}:
            raise InvalidRequestError("stdout must be 'capture' or 'discard'")
        if stderr not in {"capture", "discard"}:
            raise InvalidRequestError("stderr must be 'capture' or 'discard'")
        if pty and (stdout != "capture" or stderr != "capture" or retain):
            raise InvalidRequestError(
                "stdout, stderr, and retain durable-job options require pty=False"
            )
        channel = self._channel()
        if workdir is not None and not workdir.startswith("/"):
            # A sandbox without an image ran commands from the SSH user's home,
            # where a relative working directory resolved; docker exec has
            # always refused one.
            if self._info.vm or self._info.image_id is not None:
                raise InvalidRequestError(f"workdir must be an absolute path, got {workdir!r}")
            workdir = await self._absolute(workdir)
        unset = tuple(key for key, value in (env or {}).items() if value is None)
        request = ExecRequest(
            exec_id=new_job_id(),
            executable=args[0],
            args=tuple(args[1:]),
            working_dir=workdir,
            env={key: value for key, value in (env or {}).items() if value is not None},
            unset_env=unset,
            identity=_identity(user),
            stdout=_OutputMode(stdout),
            stderr=_OutputMode(stderr),
            pty=pty or None,
            retention=Retention.RETAIN if retain else None,
        )
        handle = await channel.exec(request)
        return Process.from_handle(handle, timeout=timeout, text=text, durable=not pty)

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

        handle = await self._channel().attach(process_id)
        return Process.from_handle(handle, timeout=None, text=text, durable=True)

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

    async def upload(
        self,
        local_path: str | os.PathLike[str],
        remote_path: str,
        *,
        recursive: bool = False,
        timeout: float | None = None,
    ) -> None:
        """Upload a file or directory, raising after ``timeout`` seconds if given."""
        source_path, contents_only = _upload_source(local_path)
        if not source_path.exists():
            raise InvalidRequestError(f"upload source does not exist: {os.fspath(local_path)}")
        if source_path.is_dir() and not recursive:
            raise InvalidRequestError("uploading a directory requires recursive=True")
        if not remote_path or "\x00" in remote_path:
            raise InvalidRequestError("upload destination must be a non-empty remote path")
        await self._channel().upload(UploadRequest(
            local_path=source_path,
            destination=await self._absolute(remote_path),
            options=TransferOptions(
                recursive=recursive,
                contents_only=contents_only,
                timeout_ms=_timeout_ms(timeout),
            ),
        ))

    async def download(
        self,
        remote_path: str,
        local_path: str | os.PathLike[str],
        *,
        recursive: bool = False,
        timeout: float | None = None,
    ) -> None:
        """Download a file or directory, raising after ``timeout`` seconds if given."""
        if not remote_path or "\x00" in remote_path:
            raise InvalidRequestError("download source must be a non-empty remote path")
        source = _parse_transfer_path(await self._absolute(remote_path), separator="/")
        await self._channel().download(DownloadRequest(
            source=source.path,
            local_path=Path(local_path),
            options=TransferOptions(
                recursive=recursive,
                contents_only=source.contents_only,
                timeout_ms=_timeout_ms(timeout),
            ),
        ))

    async def update_network_policy(
        self,
        *,
        block_network: bool = False,
        outbound_cidr_allowlist: Sequence[str] | None = None,
        outbound_domain_allowlist: Sequence[str] | None = None,
        timeout: float | None = 300,
    ) -> None:
        """Replace the outbound policy and wait until its generation is enforced.

        HTTP requests and polling yield to the event loop. ``timeout`` bounds
        the whole operation; ``None`` waits indefinitely. Timeout or cancellation
        does not undo an accepted update.

        Each call replaces the complete policy; previous rules are not
        preserved. Unless ``block_network=True``, passing both allowlists
        as ``None`` allows unrestricted outbound access. Otherwise, an
        omitted or empty allowlist permits nothing through that dimension.
        """
        _validate_network_policy_options(
            block_network=block_network,
            outbound_cidr_allowlist=outbound_cidr_allowlist,
            outbound_domain_allowlist=outbound_domain_allowlist,
        )
        if timeout is not None and (not math.isfinite(timeout) or timeout <= 0):
            raise InvalidRequestError("network policy timeout must be positive and finite")
        internet_access, cidrs, domains = _network_policy_request(
            block_network,
            outbound_cidr_allowlist,
            outbound_domain_allowlist,
        )
        try:
            await asyncio.wait_for(
                self._update_network_policy({
                    "network_policy": {
                        "internet_access": internet_access,
                        "cidr_allowlist": cidrs,
                        "domain_allowlist": domains,
                    }
                }),
                timeout=timeout,
            )
        except asyncio.TimeoutError as exc:
            raise SandboxTimeoutError(
                f"network policy update for sandbox {self.id} did not complete "
                f"within {timeout} seconds; an accepted update may still be applied"
            ) from exc

    async def _update_network_policy(self, request: dict[str, object]) -> None:
        while True:
            try:
                response = await self._client._request(
                    "PATCH",
                    f"/sandboxes/{_path_segment(self.id)}/network-policy",
                    request,
                )
                break
            except ConflictError as exc:
                if exc.code != "sandbox_update_conflict":
                    raise
                # Another update is pending. Pace retries below the policy
                # endpoint's rate limit. A transport failure is resent by
                # Client._request: the API accepts an identical policy again
                # without allocating another generation.
                await _sleep_until_next_poll(None, 5.0, exc.retry_after)
        policy = response.get("network_policy")
        if not isinstance(policy, dict) or not policy.get("internet_access"):
            raise SandboxFailedError(
                "Thunder did not return the accepted sandbox network policy"
            )
        accepted_policy = _network_policy_from_response(policy)
        generation = _network_policy_generation(response.get("generation"))
        if generation is None:
            self._info = replace(
                self._info,
                network_policy=accepted_policy,
                network_policy_generation=None,
                desired_network_policy_generation=None,
            )
        else:
            await self._wait_for_network_policy(accepted_policy, generation)
        await self._channel().after_network_policy_change()

    async def _wait_for_network_policy(
        self, policy: NetworkPolicy, generation: int
    ) -> None:
        failing_since: float | None = None
        delay = 1.0
        while True:
            try:
                response = await self._client._request(
                    "GET", f"/sandboxes/{_path_segment(self.id)}"
                )
            except (ConnectionError, RetryableError) as exc:
                now = time.monotonic()
                failing_since = now if failing_since is None else failing_since
                if now - failing_since >= OUTAGE_GRACE_SECONDS:
                    raise
                await _sleep_until_next_poll(None, delay, exc.retry_after)
                delay = min(5.0, delay * 2.0)
                continue
            failing_since = None
            delay = 1.0
            if response.get("id") != self.id:
                raise ConnectionError(
                    "Thunder returned a different sandbox during policy enforcement"
                )
            self._info = _info_from_response(self._client.config.paths, response)
            if self.status.terminal:
                raise SandboxFailedError(
                    f"sandbox {self.id} stopped before policy enforcement was confirmed "
                    f"(status: {self.status.value}): {self.info.failure or 'sandbox is no longer running'}"
                )
            desired = self.info.desired_network_policy_generation
            enforced = self.info.network_policy_generation
            if any(
                value is not None and value > generation
                for value in (desired, enforced)
            ):
                raise ConflictError(
                    f"network policy generation {generation} was superseded",
                    code="sandbox_update_conflict",
                )
            if desired == enforced == generation and self.status == SandboxStatus.READY:
                if self.info.network_policy != policy:
                    raise ConflictError(
                        "enforced network policy differs from the accepted policy",
                        code="sandbox_update_conflict",
                    )
                return
            await _sleep_until_next_poll(None, delay)

    def _channel(self) -> _SSHChannelBase:
        """The channel for this kind of sandbox, made on first use."""
        if self._ssh_channel is None:
            if self._info.vm:
                kind: type[_SSHChannelBase] = SSHChannel
            elif self._info.image_id is not None:
                kind = SSHDockerChannel
            else:
                kind = SSHHostChannel
            self._ssh_channel = kind(self._client, lambda: self._info)
        return self._ssh_channel

    async def _absolute(self, path: str) -> str:
        """A sandbox path made absolute the way each kind of sandbox always has.

        VM and container sandboxes resolve a relative path from /, as docker
        cp does; a sandbox without an image from the SSH user's home.
        """
        if path.startswith("/"):
            return path
        if self._info.vm or self._info.image_id is not None:
            return "/" + path
        return posixpath.join(await self._channel().home(), path)

    async def _close_connection(self) -> None:
        if self._ssh_channel is not None:
            await self._ssh_channel.close()

    async def get_price_per_second(self) -> float:
        """Return the configured resources' aggregate USD/second at current rates.

        Uses stored resource quantities regardless of lifecycle status; this is
        a rate, not accrued charges. No SDK pricing cache is used.
        """
        pricing = await self._client.get_pricing()
        resources = self.info.resources
        total = (
            resources.cpu * pricing.vcpu
            + resources.memory * pricing.memory_gb
            + resources.storage * pricing.storage_gb
        )
        if resources.gpu_count:
            if resources.gpu_type is None or resources.gpu_type not in pricing.gpu:
                raise ThunderError(
                    f"no published rate for GPU type: {resources.gpu_type}",
                    code="pricing_unavailable",
                )
            total += resources.gpu_count * pricing.gpu[resources.gpu_type]
        # Published rates are per hour.
        return total / 3600

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
        """Wait for the positional create command, or for VM termination.

        This does not wait for the image's ENTRYPOINT/CMD or return its exit code.
        """
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
        """Wait for guest initialization, without checking application health.

        A short image ENTRYPOINT/CMD may already have exited when this returns.
        """
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
            if self.status.terminal:
                return
            try:
                await _stop_sandbox(self._client, self.id, deadline=deadline)
            except ConflictError as exc:
                # Queued sandboxes can stop before readiness. Only fall back
                # to waiting if the API explicitly requires startup to finish.
                if exc.code != "sandbox_not_ready" or self.status != SandboxStatus.CREATED:
                    raise
                if not await self._wait_for_startup(deadline):
                    raise SandboxTimeoutError(
                        f"sandbox {self.id} did not become ready to stop within {timeout} seconds"
                    ) from exc
                if self.status.terminal:
                    return
                await _stop_sandbox(self._client, self.id, deadline=deadline)
            await self.refresh()
        finally:
            await self._close_connection()
            if self._owns_client:
                await self._client.close()


def _identity(user: str | None) -> Identity | None:
    """``user`` as ``docker exec --user`` takes it: a name or ID, then an optional ``:group``."""
    if user is None:
        return None
    name, separator, group = user.partition(":")
    return Identity(user=name, group=group if separator else None)


def _timeout_ms(timeout: float | None) -> int | None:
    if timeout is None:
        return None
    if not math.isfinite(timeout) or timeout <= 0:
        raise InvalidRequestError("transfer timeout must be positive and finite")
    return max(1, math.ceil(timeout * 1000))


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
    gpu_type = GPUType(str(spec["gpu_type"])) if spec.get("gpu_type") else None
    if gpu_type is GPUType.NONE:
        gpu_type = None
    workspace_id = str(response.get("workspace_id") or MAIN_WORKSPACE_ID)
    ssh_value = response.get("ssh")
    ssh = None
    if isinstance(ssh_value, dict) and ssh_value.get("host"):
        proxy_jump = None
        proxy = ssh_value.get("proxy_jump")
        if proxy is not None:
            try:
                if not isinstance(proxy, dict):
                    raise ValueError("expected an object")
                host, user, host_key = (proxy[key] for key in ("host", "user", "host_key"))
                port = int(proxy["port"])
                if any(not isinstance(value, str) or not value or any(c.isspace() for c in value)
                       or value.startswith("-") for value in (host, user)):
                    raise ValueError("invalid jump host or user")
                if not 1 <= port <= 65535 or not isinstance(host_key, str):
                    raise ValueError("invalid jump port or host key")
                key = asyncssh.import_public_key(host_key)
                proxy_jump = SSHProxyJump(
                    host, port, user, key.export_public_key().decode("ascii").strip()
                )
            except (KeyError, TypeError, ValueError, asyncssh.Error) as exc:
                raise SandboxFailedError("Thunder returned invalid SSH ProxyJump metadata") from exc
        ssh = SSHConnection(
            host=str(ssh_value["host"]),
            port=int(ssh_value.get("port", 22)),
            user=str(ssh_value.get("user", "ubuntu")),
            private_key_path=paths.ssh_key,
            certificate_path=paths.workspace_ssh_certificate(workspace_id),
            proxy_jump=proxy_jump,
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
        network_policy_generation=_network_policy_generation(
            response.get("network_policy_generation")
        ),
        desired_network_policy_generation=_network_policy_generation(
            response.get("desired_network_policy_generation")
        ),
        queued=response.get("queued") is True,
        workspace_id=workspace_id,
        created_at=_datetime(response.get("created_at")),
        expires_at=_datetime(response.get("expires_at"), optional=True),
        ssh=ssh,
        failure_code=(
            str(response["failure_code"]) if response.get("failure_code") else None
        ),
        failure=str(response["failure"]) if response.get("failure") else None,
        image_id=str(response["image_id"]) if response.get("image_id") else None,
        stopped_at=_datetime(response.get("stopped_at"), optional=True),
        ready_at=_datetime(response.get("ready_at"), optional=True),
        vm=response.get("vm") is True,
    )


def _string_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(str(item) for item in value)


def _network_policy_generation(value: object) -> int | None:
    if value is None:
        return None
    if type(value) is not int or not 0 <= value <= 0xFFFFFFFF:
        raise ConnectionError("Thunder returned an invalid network policy generation")
    return value


def _network_policy_from_response(value: object) -> NetworkPolicy:
    policy: dict[str, object] = value if isinstance(value, dict) else {}
    return NetworkPolicy(
        internet_access=str(policy.get("internet_access", "closed")),
        outbound_cidr_allowlist=_string_tuple(policy.get("cidr_allowlist")),
        outbound_domain_allowlist=_string_tuple(policy.get("domain_allowlist")),
    )


def _validate_create_options(
    *,
    timeout: int | None,
    queue: bool,
    gpu_type: GPUType | None,
    gpu_count: int | None,
    image: Image | None,
    block_network: bool,
    outbound_cidr_allowlist: Sequence[str] | None,
    outbound_domain_allowlist: Sequence[str] | None,
) -> None:

    if not isinstance(queue, bool):
        raise InvalidRequestError("queue must be a boolean")
    if timeout is not None and timeout < 0:
        raise InvalidRequestError("timeout cannot be negative")
    if gpu_type is not None and not isinstance(gpu_type, GPUType):
        raise InvalidRequestError("gpu_type must be a GPUType")
    if gpu_type in (None, GPUType.NONE):
        if gpu_count not in (None, 0):
            raise InvalidRequestError("CPU-only sandboxes require gpu_count=0 or None")
    else:
        if gpu_count not in (1, 2, 4, 8):
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
    # Naming one allowlist restricts egress to it. Leaving the other open would
    # silently undo the restriction the caller asked for.
    return (
        "restricted",
        list(outbound_cidr_allowlist or ()),
        list(outbound_domain_allowlist or ()),
    )


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


__all__ = ["Sandbox"]
