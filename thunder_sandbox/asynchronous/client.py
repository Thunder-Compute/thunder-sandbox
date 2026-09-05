"""Native asynchronous HTTP client for Thunder sandboxes."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncGenerator, Mapping, Sequence
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import aiohttp
import asyncssh

from .._common.config import ClientConfig
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
)
from .._common.types import GPUType, SandboxStatus
from .._version import __version__
from .credentials import CredentialStore

if TYPE_CHECKING:
    from ..image import Image, ResolvedImage, _CanonicalBuildContext
    from .sandbox import Sandbox

USER_AGENT = f"thunder-python-sdk/{__version__}"

_ERROR_CODES: dict[str, type[ThunderError]] = {
    "sandbox_capacity_unavailable": CapacityError,
    "sandbox_scheduler_unavailable": ServiceUnavailableError,
    "sandbox_scheduler_timeout": ServiceUnavailableError,
    "sandbox_already_exists": ConflictError,
    "sandbox_name_in_use": ConflictError,
    "sandbox_scheduler_rejected_request": InvalidRequestError,
    "rate_limit_exceeded": RateLimitError,
    "sandbox_image_invalid_source": InvalidRequestError,
    "sandbox_image_source_collision": ConflictError,
    "sandbox_image_registry_unavailable": ServiceUnavailableError,
    "sandbox_image_control_plane_unavailable": ServiceUnavailableError,
    "sandbox_image_job_creation_failed": ServiceUnavailableError,
    "sandbox_image_stale_attempt": ConflictError,
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


class Client:
    """An asynchronous Thunder API client backed by :mod:`aiohttp`."""

    def __init__(self, config: ClientConfig | None = None) -> None:
        self.config = config or ClientConfig.from_cli()
        if not self.config.api_token:
            raise AuthenticationError("no authentication token found; run 'tnr login'")
        self._session: aiohttp.ClientSession | None = None
        self._credentials = CredentialStore(self.config.paths)
        self._closed = False

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
        timeout: aiohttp.ClientTimeout | None = None,
    ) -> dict[str, Any]:
        session = self._get_session()
        url = f"{self.config.api_url}/v1{path}"
        params = (
            {key: str(value) for key, value in query.items() if value not in (None, "")}
            if query
            else None
        )
        try:
            async with session.request(
                method, url, json=body, params=params, timeout=timeout
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
                    raise _api_error(
                        response.status, code, message, response.headers
                    )
        except ThunderError:
            raise
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as exc:
            raise ConnectionError(f"could not communicate with Thunder: {exc}") from exc

        if not payload:
            return {}
        try:
            result = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ConnectionError("Thunder returned an invalid JSON response") from exc
        if not isinstance(result, dict):
            raise ConnectionError("Thunder returned an unexpected response")
        return result

    async def resolve_image(
        self, image: "Image", *, timeout: float | None = 7200
    ) -> "ResolvedImage":
        """Build or import an image and wait until its managed image is ready."""
        from ..image import Image, _create_canonical_build_context

        if not isinstance(image, Image):
            raise InvalidRequestError("image must be created with Image.from_*()")
        if timeout is not None and timeout <= 0:
            raise InvalidRequestError("image timeout must be positive or None")
        deadline = None if timeout is None else time.monotonic() + timeout

        context = None
        try:
            if image._source == "registry":
                response = await self._request(
                    "POST",
                    "/sandbox-images/from-registry",
                    body={
                        "reference": image._registry_url,
                        "username": image._registry_username,
                        "password": image._registry_password,
                    },
                    timeout=self._request_timeout(deadline),
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
                response = await self._request(
                    "POST",
                    "/sandbox-images/from-dockerfile",
                    body={
                        "recipe_hash": context.recipe_hash,
                        "context_hash": context.context_hash,
                        "dockerfile_hash": context.dockerfile_hash,
                        "build_options_hash": context.build_options_hash,
                        "archive_bytes": context.archive_bytes,
                        "ssh_public_key": upload_public_key,
                    },
                    timeout=self._request_timeout(deadline),
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
            return await self._wait_for_image(response, deadline=deadline)
        finally:
            if context is not None:
                await asyncio.to_thread(context.close)

    def _request_timeout(
        self, deadline: float | None
    ) -> aiohttp.ClientTimeout | None:
        if deadline is None:
            return aiohttp.ClientTimeout(total=60)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise SandboxTimeoutError("image preparation timed out")
        return aiohttp.ClientTimeout(total=min(remaining, 60))

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
        self, response: dict[str, Any], *, deadline: float | None
    ) -> "ResolvedImage":
        from ..image import ResolvedImage

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
                return ResolvedImage(image_id, reference, digest)
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
                timeout=self._request_timeout(deadline),
            )

    async def create_sandbox(
        self,
        *args: str,
        name: str | None = None,
        env: Mapping[str, str | None] | None = None,
        timeout: int | None = 300,
        cpu: int | None = None,
        memory: int | None = None,
        storage: int | None = None,
        gpu_type: GPUType | None = None,
        gpu_count: int | None = None,
        image: "Image | None" = None,
        block_network: bool = False,
        outbound_cidr_allowlist: Sequence[str] | None = None,
        outbound_domain_allowlist: Sequence[str] | None = None,
    ) -> "Sandbox":
        from .sandbox import Sandbox

        return await Sandbox.create(
            *args,
            name=name,
            env=env,
            timeout=timeout,
            cpu=cpu,
            memory=memory,
            storage=storage,
            gpu_type=gpu_type,
            gpu_count=gpu_count,
            image=image,
            block_network=block_network,
            outbound_cidr_allowlist=outbound_cidr_allowlist,
            outbound_domain_allowlist=outbound_domain_allowlist,
            client=self,
        )

    async def get_sandbox(self, sandbox_id: str) -> "Sandbox":
        from .sandbox import Sandbox

        return await Sandbox.from_id(sandbox_id, client=self)

    async def get_sandbox_by_name(self, name: str) -> "Sandbox":
        from .sandbox import Sandbox

        return await Sandbox.from_name(name, client=self)

    async def list_sandboxes(
        self, *, status: str | SandboxStatus = "active"
    ) -> AsyncGenerator["Sandbox", None]:
        from .sandbox import Sandbox

        wanted = status.value if isinstance(status, SandboxStatus) else str(status)
        page_token = ""
        while True:
            response = await self._request(
                "GET",
                "/sandboxes",
                query={"limit": 100, "status": wanted, "page_token": page_token},
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
