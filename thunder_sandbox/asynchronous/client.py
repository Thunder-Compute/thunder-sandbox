"""Native asynchronous HTTP client for Thunder sandboxes."""

from __future__ import annotations

import asyncio
import json
import math
import os
import time
from collections.abc import AsyncGenerator, Mapping, Sequence
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import aiohttp
import asyncssh

from .._common.config import WORKSPACE_ENV_VAR, ClientConfig
from .._common.exceptions import (
    AuthenticationError,
    CapacityError,
    ConflictError,
    ConnectionError,
    InvalidRequestError,
    NotFoundError,
    RateLimitError,
    SandboxFailedError,
    SandboxTimeoutError,
    ServiceUnavailableError,
    ThunderError,
    _WaitWindowElapsedError,
)
from .._common.types import MAIN_WORKSPACE_ID, GPUType, Pricing, SandboxStatus, Workspace
from .._version import __version__
from .credentials import CredentialStore

if TYPE_CHECKING:
    from ..image import Image, ResolvedImage, _CanonicalBuildContext
    from .sandbox import Sandbox

USER_AGENT = f"thunder-python-sdk/{__version__}"

WORKSPACE_ID_PREFIX = "ws-"
WORKSPACE_CACHE_SECONDS = 60.0

# Pauses before each retry of a request whose connection failed. A proxy in
# front of the API (Cloudflare) occasionally resets a pooled connection; the
# next attempt opens a fresh one, so a short pause is enough.
CONNECTION_RETRY_DELAYS = (0.2, 1.0)
# Methods safe to send again when a connection fails after the request may
# have reached the server.
IDEMPOTENT_METHODS = frozenset({"GET", "HEAD", "OPTIONS", "PUT", "DELETE"})

_ERROR_CODES: dict[str, type[ThunderError]] = {
    "sandbox_capacity_unavailable": CapacityError,
    "sandbox_scheduler_unavailable": ServiceUnavailableError,
    "sandbox_scheduler_timeout": ServiceUnavailableError,
    "sandbox_already_exists": ConflictError,
    "sandbox_name_in_use": ConflictError,
    "sandbox_scheduler_rejected_request": InvalidRequestError,
    "sandbox_wait_timeout": _WaitWindowElapsedError,
    "rate_limit_exceeded": RateLimitError,
    "sandbox_image_invalid_source": InvalidRequestError,
    "sandbox_image_source_collision": ConflictError,
    "sandbox_image_registry_unavailable": ServiceUnavailableError,
    "sandbox_image_control_plane_unavailable": ServiceUnavailableError,
    "sandbox_image_job_creation_failed": ServiceUnavailableError,
    "sandbox_image_stale_attempt": ConflictError,
    "sandbox_image_not_ready": ConflictError,
    "sandbox_image_failed": SandboxFailedError,
}

_ERROR_STATUSES: dict[int, type[ThunderError]] = {
    400: InvalidRequestError,
    401: AuthenticationError,
    403: AuthenticationError,
    404: NotFoundError,
    409: ConflictError,
    429: RateLimitError,
    502: ServiceUnavailableError,
    503: ServiceUnavailableError,
    504: ServiceUnavailableError,
}


def _api_error(
    status: int, code: str | None, message: str, headers: object
) -> ThunderError:
    error_type = _ERROR_CODES.get(code or "") or _ERROR_STATUSES.get(status, ThunderError)
    retry_after = None
    getter = getattr(headers, "get", None)
    if getter is not None:
        try:
            retry_after = float(getter("Retry-After"))
        except (TypeError, ValueError):
            retry_after = None
    return error_type(message, code=code, status=status, retry_after=retry_after)


def _idempotent(method: str, path: str) -> bool:
    """Whether sending this request twice has the same effect as once.

    Besides the idempotent methods: minting an SSH certificate only issues
    another one, stopping a stopped sandbox does nothing, and a network-policy
    update replaces the complete policy. Starting a sandbox is not.
    """
    method = method.upper()
    if method in IDEMPOTENT_METHODS:
        return True
    if method == "POST":
        return path == "/sandboxes/ssh-certificate" or (
            path.startswith("/sandboxes/") and path.endswith("/stop")
        )
    if method == "PATCH":
        return path.startswith("/sandboxes/") and path.endswith("/network-policy")
    return False


def _resendable(exc: BaseException, idempotent: bool) -> bool:
    """Whether a request that failed with ``exc`` may be sent again.

    A failure to connect means the request never left, so any request can be
    resent. A reset or disconnect afterwards may hit a request the server
    already acted on, so only an idempotent one is resent. A timeout is never
    resent: the server may still be working on it. Neither is a TLS failure,
    which a new connection repeats.
    """
    # aiohttp's ServerTimeoutError is also a ClientConnectionError.
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError, aiohttp.ClientSSLError)):
        return False
    if isinstance(exc, aiohttp.ClientConnectorError):
        return True
    # ClientConnectionError covers every reset and disconnect, including
    # ClientConnectionResetError (aiohttp >= 3.10.6), which is not a
    # ClientOSError.
    return idempotent and isinstance(
        exc,
        (
            aiohttp.ClientConnectionError,
            aiohttp.ClientPayloadError,
            ConnectionResetError,
            BrokenPipeError,
        ),
    )


def _describe(exc: BaseException) -> str:
    # asyncio.TimeoutError and some resets have an empty message.
    return f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__


class Client:
    """An asynchronous Thunder API client backed by :mod:`aiohttp`."""

    def __init__(self, config: ClientConfig | None = None) -> None:
        self.config = config or ClientConfig.from_cli()
        if not self.config.api_token:
            raise AuthenticationError("no authentication token found; run 'tnr login'")
        self._session: aiohttp.ClientSession | None = None
        self._credential_lock = asyncio.Lock()
        self._credentials = CredentialStore(self.config.paths, lock=self._credential_lock)
        self._workspace_credentials = {MAIN_WORKSPACE_ID: self._credentials}
        self._workspace_ids: dict[str, str] = {}
        self._workspace_ids_at = 0.0
        self._closed = False
        # Whether the API offers the blocking readiness wait. Assumed until a
        # request proves otherwise, and remembered here because it is a
        # property of the API this client talks to: an older API then costs
        # one extra read per client rather than one per wait.
        self._wait_endpoint_available = True

    @classmethod
    def from_cli(cls) -> "Client":
        return cls(ClientConfig.from_cli())

    def _get_session(self) -> aiohttp.ClientSession:
        if self._closed:
            raise ConnectionError("client is closed")
        if self._session is None:
            self._session = aiohttp.ClientSession(
                headers={
                    "Authorization": f"Bearer {self.config.api_token}",
                    "Content-Type": "application/json",
                    "Thunder-Client": "PYTHON-SDK",
                    "User-Agent": USER_AGENT,
                },
                timeout=aiohttp.ClientTimeout(total=30),
            )
        return self._session

    async def _request(
        self,
        method: str,
        path: str,
        body: object | None = None,
        query: dict[str, object] | None = None,
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        result = await self._request_json(method, path, body, query, timeout=timeout)
        if not isinstance(result, dict):
            raise ConnectionError("Thunder returned an unexpected response")
        return result

    async def _request_json(
        self,
        method: str,
        path: str,
        body: object | None = None,
        query: dict[str, object] | None = None,
        *,
        timeout: float | None = None,
    ) -> Any:
        """Send one API request and return its decoded JSON.

        ``timeout`` bounds the whole call, in seconds, retries included, for
        calls the server deliberately holds open longer than the session
        default allows. Callers size it to their own deadline.

        A request whose connection fails is sent again, up to
        ``len(CONNECTION_RETRY_DELAYS)`` times: always when the connection
        could not be opened, and after a reset only when the request is
        idempotent (see :func:`_idempotent`). A retry gets only the time
        left of ``timeout`` and is not sent once none is left.
        """
        idempotent = _idempotent(method, path)
        expires_at = None if timeout is None else time.monotonic() + timeout
        attempt_timeout = timeout
        for delay in (*CONNECTION_RETRY_DELAYS, None):
            try:
                payload = await self._send(method, path, body, query, attempt_timeout)
                break
            except ThunderError:
                raise
            except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as exc:
                failure = ConnectionError(
                    f"could not communicate with Thunder: {_describe(exc)}"
                )
                if delay is None or not _resendable(exc, idempotent):
                    raise failure from exc
                await asyncio.sleep(delay)
                if expires_at is not None:
                    attempt_timeout = expires_at - time.monotonic()
                    if attempt_timeout <= 0:
                        raise failure from exc

        if not payload:
            return {}
        try:
            return json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ConnectionError("Thunder returned an invalid JSON response") from exc

    async def _send(
        self,
        method: str,
        path: str,
        body: object | None,
        query: dict[str, object] | None,
        timeout: float | None,
    ) -> bytes:
        """Send one attempt and return the response body, raising API errors."""
        session = self._get_session()
        url = f"{self.config.api_url}/v1{path}"
        params = (
            {key: str(value) for key, value in query.items() if value not in (None, "")}
            if query
            else None
        )
        # None means the session default; aiohttp treats an explicit None as
        # "no timeout at all", which is never what a caller here wants.
        assert timeout is None or timeout > 0, timeout
        options = (
            {"timeout": aiohttp.ClientTimeout(total=timeout)} if timeout is not None else {}
        )
        async with session.request(
            method, url, json=body, params=params, **options
        ) as response:
            payload = await response.read()
            if response.status >= 400:
                code = None
                try:
                    parsed = json.loads(payload)
                    message = (
                        parsed.get("message")
                        or f"Thunder API returned HTTP {response.status}"
                    )
                    code = parsed.get("error")
                except (json.JSONDecodeError, AttributeError):
                    message = f"Thunder API returned HTTP {response.status}"
                raise _api_error(response.status, code, message, response.headers)
        return payload

    async def list_workspaces(self) -> list[Workspace]:
        """List the workspaces this credential can access, main first."""
        response = await self._request_json("GET", "/workspaces")
        if not isinstance(response, list):
            raise ConnectionError("Thunder returned an unexpected response")
        workspaces = [
            Workspace(
                id=str(item.get("id", "")),
                name=str(item.get("name", "")),
                role=str(item["role"]) if item.get("role") else None,
            )
            for item in response
            if isinstance(item, dict) and item.get("id")
        ]
        self._workspace_ids = {item.name.lower(): item.id for item in workspaces}
        self._workspace_ids.update({item.id.lower(): item.id for item in workspaces})
        self._workspace_ids_at = time.monotonic()
        return workspaces

    async def _resolve_workspace(self, reference: str | None) -> str:
        """Turn a workspace name or ID into its ID.

        The argument wins, then ``TNR_WORKSPACE``, then main. Main and ``ws-`` IDs
        need no request, so a client that never names a workspace makes none. The
        list is shared by the whole organization's rate limit, so it is cached, and
        reread once when a name is missing in case it is new.
        """
        chosen = (reference or os.environ.get(WORKSPACE_ENV_VAR) or "").strip()
        key = chosen.lower()
        if not chosen or key == MAIN_WORKSPACE_ID:
            return MAIN_WORKSPACE_ID
        if key.startswith(WORKSPACE_ID_PREFIX):
            return chosen
        if time.monotonic() - self._workspace_ids_at < WORKSPACE_CACHE_SECONDS:
            cached = self._workspace_ids.get(key)
            if cached is not None:
                return cached
        workspaces = await self.list_workspaces()
        resolved = self._workspace_ids.get(key)
        if resolved is None:
            available = ", ".join(item.name for item in workspaces) or "none"
            raise NotFoundError(
                f"no workspace named {chosen!r} is available to this credential; "
                f"available workspaces: {available}"
            )
        return resolved

    def _credentials_for(self, workspace_id: str) -> CredentialStore:
        store = self._workspace_credentials.get(workspace_id)
        if store is None:
            store = CredentialStore(
                self.config.paths, workspace_id=workspace_id, lock=self._credential_lock
            )
            self._workspace_credentials[workspace_id] = store
        return store

    async def get_pricing(self) -> Pricing:
        """Fetch current USD/hour resource rates, subject to API-side caching.

        Memory and storage rates are per GiB. Unpublished GPU rates are omitted.
        """
        response = await self._request("GET", "/pricing")
        rates = response.get("pricing")
        if not isinstance(rates, dict):
            raise ThunderError("invalid pricing catalog", code="invalid_pricing")

        def rate(key: str) -> float:
            value = rates.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ThunderError(f"invalid or missing rate: {key}", code="invalid_pricing")
            try:
                result = float(value)
            except OverflowError as exc:
                raise ThunderError(f"invalid rate: {key}", code="invalid_pricing") from exc
            if not math.isfinite(result) or result < 0:
                raise ThunderError(f"invalid rate: {key}", code="invalid_pricing")
            return result

        return Pricing(
            vcpu=rate("sandbox_vcpu"),
            storage_gb=rate("sandbox_storage_gb"),
            memory_gb=rate("sandbox_memory_gb"),
            gpu={
                gpu: rate(f"sandbox_{gpu.value.lower()}")
                for gpu in GPUType
                if gpu not in (GPUType.NONE, GPUType.UNKNOWN)
                and f"sandbox_{gpu.value.lower()}" in rates
            },
        )

    async def resolve_image(
        self,
        image: "Image",
        *,
        timeout: float | None = 7200,
        gated_features: Mapping[str, object] | None = None,
        spec: Mapping[str, object] | None = None,
        queue: bool = False,
        workspace: str | None = None,
        vm: bool = False,
    ) -> "ResolvedImage":
        """Build or import an image and wait until its managed image is ready.

        Optional ``spec`` supplies sandbox resources for a capacity check; it
        does not reserve resources. Omit it for standalone image preparation.
        Set ``queue=True`` to admit builds without a sandbox capacity check.
        Image preparation still waits for the build within ``timeout``.

        The image is listed in ``workspace``, a workspace name or ID. Without
        one, ``TNR_WORKSPACE`` applies, and then main. Building or importing
        takes the developer role there. The organization builds each recipe
        once, so a Dockerfile or registry digest already built in another
        workspace returns the same image, ready at once. An API from before
        workspaces ignores the selector and lists images organization-wide.

        ``vm=True`` asks for the image built to boot as the whole VM, with the
        image's own systemd as PID 1, for a sandbox created with ``vm``. It is a
        separate image from the same source's container image, and only images
        of a supported distribution (Debian, Ubuntu, Fedora, CentOS Stream,
        Rocky Linux, AlmaLinux, Arch Linux or openSUSE) can be built this way.
        Where Thunder does not offer VM sandboxes it builds the container
        image instead, and the result's ``vm`` is false.
        """
        from ..image import Image, _create_canonical_build_context

        if not isinstance(image, Image):
            raise InvalidRequestError("image must be created with Image.from_*()")
        if not isinstance(queue, bool):
            raise InvalidRequestError("queue must be a boolean")
        if not isinstance(vm, bool):
            raise InvalidRequestError("vm must be a boolean")
        if timeout is not None and timeout <= 0:
            raise InvalidRequestError("image timeout must be positive or None")
        workspace_id = await self._resolve_workspace(workspace)
        scope = {} if workspace_id == MAIN_WORKSPACE_ID else {"workspace_id": workspace_id}
        deadline = None if timeout is None else time.monotonic() + timeout
        # gatedFeatures.vm starts a sandbox from an image built in the VM mode:
        # the build takes it as its mode, not as a build feature. The vm
        # argument asks for the same image through its own field.
        build_features = None if gated_features is None else dict(gated_features)
        gated_vm = build_features is not None and "vm" in build_features
        if build_features is not None:
            build_features.pop("vm", None)
        mode_fields: dict[str, object] = {
            **({"mode": "vm"} if gated_vm else {}),
            **({"vm": True} if vm else {}),
        }
        if build_features is not None and (build_features or not gated_vm):
            mode_fields["gatedFeatures"] = build_features

        context = None
        try:
            if image._source == "registry":
                request_body: dict[str, object] = {
                    "reference": image._registry_url,
                    "username": image._registry_username,
                    "password": image._registry_password,
                    **scope,
                }
                if image._display_name is not None:
                    request_body["display_name"] = image._display_name
                request_body.update(mode_fields)
                if queue:
                    request_body["queue"] = True
                if spec is not None:
                    request_body["spec"] = dict(spec)
                response = await self._start_image(
                    "/sandbox-images/from-registry",
                    request_body,
                    deadline=deadline,
                    workspace_id=workspace_id,
                )
            else:
                if image._context_directory is None:
                    raise InvalidRequestError("Dockerfile image has no build context")
                upload_key = asyncssh.generate_private_key(
                    "ssh-ed25519", comment="thunder-image-upload"
                )
                upload_public_key = (
                    upload_key.export_public_key().decode("utf-8").strip()
                )
                context = await asyncio.to_thread(
                    _create_canonical_build_context,
                    image._context_directory,
                    self.config.paths.image_build_contexts,
                )
                request_body = {
                    "recipe_hash": context.recipe_hash,
                    "context_hash": context.context_hash,
                    "dockerfile_hash": context.dockerfile_hash,
                    "build_options_hash": context.build_options_hash,
                    "archive_bytes": context.archive_bytes,
                    "ssh_public_key": upload_public_key,
                    **scope,
                }
                if image._display_name is not None:
                    request_body["display_name"] = image._display_name
                if image._registry_host is not None:
                    request_body["registry_host"] = image._registry_host
                    request_body["registry_username"] = image._registry_username
                    request_body["registry_password"] = image._registry_password
                request_body.update(mode_fields)
                if queue:
                    request_body["queue"] = True
                if spec is not None:
                    request_body["spec"] = dict(spec)
                response = await self._start_image(
                    "/sandbox-images/from-dockerfile",
                    request_body,
                    deadline=deadline,
                    workspace_id=workspace_id,
                )
                upload = response.get("upload")
                if upload is not None:
                    if not isinstance(upload, dict):
                        raise ConnectionError("Thunder returned an invalid image upload")
                    await self._upload_image_context(
                        context,
                        upload,
                        private_key=upload_key,
                        deadline=deadline,
                    )
                del upload_key
            return await self._wait_for_image(
                response, deadline=deadline, workspace_id=workspace_id
            )
        finally:
            if context is not None:
                await asyncio.to_thread(context.close)

    async def _start_image(
        self,
        path: str,
        body: dict[str, object],
        *,
        deadline: float | None,
        workspace_id: str,
    ) -> dict[str, Any]:
        try:
            return await self._request(
                "POST", path, body=body, timeout=self._request_timeout(deadline)
            )
        except AuthenticationError as exc:
            # A role that cannot build is not a bad token, which is what this
            # error type otherwise suggests. Gated features are refused with
            # the same code for a different reason, so that denial stays as is.
            if exc.code != "forbidden" or "gatedFeatures" in body:
                raise
            raise AuthenticationError(
                f"developer access to workspace {workspace_id!r} is required "
                "to build or import its images",
                code=exc.code,
                status=exc.status,
            ) from exc

    def _request_timeout(self, deadline: float | None) -> float:
        if deadline is None:
            return 60
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise SandboxTimeoutError("image preparation timed out")
        return min(remaining, 60)

    async def _upload_image_context(
        self,
        context: "_CanonicalBuildContext",
        upload: dict[str, Any],
        *,
        private_key: asyncssh.SSHKey,
        deadline: float | None,
    ) -> None:
        try:
            host = upload["host"]
            port = upload["port"]
            username = upload["username"]
            path = upload["path"]
            host_public_key = upload["host_public_key"]
            expires_at = upload["expires_at"]
        except (KeyError, TypeError) as error:
            raise ConnectionError("Thunder returned an incomplete image upload") from error
        if (
            not isinstance(host, str)
            or not host
            or host != host.strip()
            or not isinstance(port, int)
            or isinstance(port, bool)
            or not 1 <= port <= 65535
            or username != "image-upload"
            or path != "/build-context.tar"
            or not isinstance(host_public_key, str)
            or not isinstance(expires_at, str)
        ):
            raise ConnectionError("Thunder returned an invalid image upload")
        try:
            expiry = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
            if expiry.tzinfo is None:
                raise ValueError
            expiry_seconds = expiry.astimezone(timezone.utc).timestamp()
        except ValueError as error:
            raise ConnectionError("Thunder returned an invalid image upload expiry") from error
        try:
            host_key = asyncssh.import_public_key(host_public_key)
            canonical_host_key = host_key.export_public_key().decode("utf-8").strip()
        except (asyncssh.Error, UnicodeError, ValueError) as error:
            raise ConnectionError("Thunder returned an invalid image upload host key") from error
        if canonical_host_key != host_public_key.strip():
            raise ConnectionError("Thunder returned an invalid image upload host key")

        while True:
            remaining = expiry_seconds - datetime.now(timezone.utc).timestamp()
            if deadline is not None:
                remaining = min(remaining, deadline - time.monotonic())
            if remaining <= 0:
                raise SandboxTimeoutError("image upload expired before it completed")
            try:
                await asyncio.wait_for(
                    self._sftp_image_context(
                        context,
                        host=host,
                        port=port,
                        username=username,
                        path=path,
                        private_key=private_key,
                        host_key=host_key,
                    ),
                    timeout=remaining,
                )
                return
            except asyncio.TimeoutError as error:
                raise SandboxTimeoutError(
                    "image upload expired before it completed"
                ) from error
            except (
                asyncssh.PermissionDenied,
                asyncssh.HostKeyNotVerifiable,
                asyncssh.SFTPError,
            ) as error:
                raise ConnectionError(
                    f"the image builder rejected the SFTP upload: {error}"
                ) from error
            except (OSError, asyncssh.Error) as error:
                remaining = expiry_seconds - datetime.now(timezone.utc).timestamp()
                if deadline is not None:
                    remaining = min(remaining, deadline - time.monotonic())
                if remaining <= 1:
                    raise ConnectionError(
                        f"could not upload the image build context over SFTP: {error}"
                    ) from error
                await asyncio.sleep(min(1.0, remaining))

    async def _sftp_image_context(
        self,
        context: "_CanonicalBuildContext",
        *,
        host: str,
        port: int,
        username: str,
        path: str,
        private_key: asyncssh.SSHKey,
        host_key: asyncssh.SSHKey,
    ) -> None:
        connection = await asyncssh.connect(
            host,
            port,
            username=username,
            client_keys=[private_key],
            known_hosts=([host_key], [], []),
            agent_path=None,
            preferred_auth=["publickey"],
            config=None,
        )
        try:
            async with connection.start_sftp_client() as sftp:
                async with sftp.open(path, "wb") as remote:
                    with context.archive_path.open("rb") as archive:
                        while chunk := archive.read(1024 * 1024):
                            await remote.write(chunk)
        finally:
            connection.close()
            await connection.wait_closed()

    async def _wait_for_image(
        self,
        response: dict[str, Any],
        *,
        deadline: float | None,
        workspace_id: str,
    ) -> "ResolvedImage":
        from ..image import ResolvedImage

        # Without the selector, an image only a private workspace lists is not
        # found while it builds.
        scope: dict[str, object] = (
            {} if workspace_id == MAIN_WORKSPACE_ID else {"workspace_id": workspace_id}
        )
        while True:
            state = response.get("state")
            image_id = response.get("id")
            if not isinstance(image_id, str) or not image_id:
                raise ConnectionError("Thunder returned an invalid sandbox image ID")
            if state == "READY":
                reference = response.get("managed_reference")
                digest = response.get("managed_digest")
                if (
                    not isinstance(reference, str)
                    or not reference
                    or not isinstance(digest, str)
                    or not digest
                ):
                    raise ConnectionError("Thunder returned an incomplete ready image")
                return ResolvedImage(
                    image_id,
                    reference,
                    digest,
                    str(response.get("workspace_id") or workspace_id),
                    vm=response.get("mode") == "vm",
                )
            if state == "FAILED":
                code = response.get("failure_code")
                detail = response.get("failure") or "image preparation failed"
                prefix = f"[{code}] " if code else ""
                raise SandboxFailedError(
                    prefix + str(detail), code=str(code) if code else None
                )
            if state != "BUILDING":
                raise ConnectionError(f"Thunder returned an unknown image state: {state!r}")
            delay = 1.0
            if deadline is not None:
                delay = min(delay, max(0, deadline - time.monotonic()))
            await asyncio.sleep(delay)
            response = await self._request(
                "GET",
                f"/sandbox-images/{image_id}",
                query=scope,
                timeout=self._request_timeout(deadline),
            )

    async def create_sandbox(
        self,
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
        image: "Image | None" = None,
        vm: bool | Mapping[str, object] = False,
        timing_sensitive: bool = False,
        block_network: bool = False,
        outbound_cidr_allowlist: Sequence[str] | None = None,
        outbound_domain_allowlist: Sequence[str] | None = None,
        workspace: str | None = None,
    ) -> "Sandbox":
        from .sandbox import Sandbox

        return await Sandbox.create(
            *args,
            name=name,
            env=env,
            gated_features=gated_features,
            timeout=timeout,
            queue=queue,
            cpu=cpu,
            memory=memory,
            storage=storage,
            gpu_type=gpu_type,
            gpu_count=gpu_count,
            image=image,
            vm=vm,
            timing_sensitive=timing_sensitive,
            block_network=block_network,
            outbound_cidr_allowlist=outbound_cidr_allowlist,
            outbound_domain_allowlist=outbound_domain_allowlist,
            workspace=workspace,
            client=self,
        )

    async def get_sandbox(self, sandbox_id: str) -> "Sandbox":
        from .sandbox import Sandbox

        return await Sandbox.from_id(sandbox_id, client=self)

    async def get_sandbox_by_name(
        self, name: str, *, workspace: str | None = None
    ) -> "Sandbox":
        from .sandbox import Sandbox

        return await Sandbox.from_name(name, workspace=workspace, client=self)

    async def list_sandboxes(
        self,
        *,
        status: str | SandboxStatus = "active",
        workspace: str | None = None,
    ) -> AsyncGenerator["Sandbox", None]:
        """List one workspace's sandboxes: ``workspace``, ``TNR_WORKSPACE``, or main."""
        from .sandbox import Sandbox

        wanted = status.value if isinstance(status, SandboxStatus) else str(status)
        workspace_id = await self._resolve_workspace(workspace)
        scope = {} if workspace_id == MAIN_WORKSPACE_ID else {"workspace_id": workspace_id}
        page_token = ""
        while True:
            response = await self._request(
                "GET",
                "/sandboxes",
                query={"limit": 100, "status": wanted, "page_token": page_token, **scope},
            )
            for item in response.get("sandboxes", []):
                if isinstance(item, dict):
                    yield Sandbox._from_response(self, item)
            page_token = str(response.get("next_page_token", ""))
            if not page_token:
                return

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._session is not None:
            await self._session.close()

    async def __aenter__(self) -> "Client":
        if self._closed:
            raise ConnectionError("client is closed")
        return self

    async def __aexit__(
        self, exc_type: object, exc: object, traceback: object
    ) -> None:
        await self.close()
